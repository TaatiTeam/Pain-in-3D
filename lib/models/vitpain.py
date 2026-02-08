import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, get_peft_model, TaskType

from .pspi_evaluator_mixin import PSPIEvaluatorMixin


class ViTPain(PSPIEvaluatorMixin, pl.LightningModule):
    def __init__(
        self,
        model_name="timm/vit_large_patch16_dinov3.sat493m",
        num_classes=1,
        num_au_features=6,
        learning_rate=1e-4,
        weight_decay=1e-2,
        max_epochs=100,
        au_loss_weight=1.0,
        pspi_loss_weight=1.0,
        dropout_rate=0.5,
        lora_rank=8,
        lora_alpha=16,
        test_stage_name="test",
        use_neutral_reference=False,
    ):
        """
        Vision Transformer for PSPI (Pain Score) regression and AU regression.

        Always uses:
        - DinoV3 backbone (always frozen)
        - LoRA adapters (only trainable backbone parameters)
        - AU query head (cross-attention for AU prediction)
        - Binary classification heads (pain/no-pain at thresholds 1, 2, 3)

        Args:
            model_name: Pretrained ViT model name (must be a DinoV3 timm model)
            num_classes: Number of output dimensions
            num_au_features: Number of Action Unit features (6 for AU4,6,7,9,10,43)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            max_epochs: Maximum training epochs
            au_loss_weight: Weight for AU regression loss
            pspi_loss_weight: Weight for PSPI regression loss
            dropout_rate: Dropout rate for regularization
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            use_neutral_reference: Whether to use a neutral reference image
        """
        super().__init__()
        self.save_hyperparameters()

        self.dropout_rate = dropout_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.test_stage_name = test_stage_name
        self.use_neutral_reference = use_neutral_reference

        import timm
        timm_model_name = model_name.replace("timm/", "")
        # Normalize potential outdated weight tags
        timm_model_name = timm_model_name.replace("sat_493m", "sat493m").replace("lvd_1689m", "lvd1689m")
        self.vit_model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)

        hidden_size = getattr(self.vit_model, "num_features", None)
        if hidden_size is None:
            hidden_size = getattr(self.vit_model, "embed_dim", None)
        if hidden_size is None:
            raise RuntimeError("Could not determine hidden size from timm model.")

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["qkv", "proj"],
            lora_dropout=0.1,
        )
        self.vit_model = get_peft_model(self.vit_model, lora_config)

        if self.use_neutral_reference:
            self.neutral_encoder = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
            self.neutral_encoder = get_peft_model(self.neutral_encoder, lora_config)
            # Cross-Attention: Pain (Q) attends to Neutral (K/V) to highlight pain-specific features
            self.neutral_cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
            self.neutral_norm = nn.LayerNorm(hidden_size)
        else:
            self.neutral_encoder = None
            self.neutral_cross_attn = None
            self.neutral_norm = None

        self.pspi_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 1),
        )

        self.binary_pspi_thresholds = (1.0, 2.0, 3.0)
        self.pspi_binary_heads = nn.ModuleDict(
            {
                str(int(thr)): nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_size, 1),
                )
                for thr in self.binary_pspi_thresholds
            }
        )

        self.au_shared_features = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
        )

        num_aus = 6  # AU4, AU6, AU7, AU9, AU10, AU43
        self.au_queries = nn.Parameter(torch.randn(num_aus, hidden_size))
        self.au_head = nn.Linear(512, 1)

        self.pspi_loss_weight = pspi_loss_weight
        self.au_loss_weight = au_loss_weight
        self._init_pspi_metrics()
        self.current_epoch_count = 0

    def _extract_tokens(self, images, model=None):
        """Return (cls_features, patch_features) from the timm DinoV3 backbone."""
        if model is None:
            model = self.vit_model

        x = model.patch_embed(images)
        B = x.shape[0]

        if x.dim() == 4:
            if hasattr(model, 'embed_dim'):
                embed_dim = model.embed_dim
                if x.shape[-1] == embed_dim:
                    x = x.reshape(B, -1, embed_dim)
                elif x.shape[1] == embed_dim:
                    x = x.flatten(2).transpose(1, 2)
                else:
                    x = x.flatten(2).transpose(1, 2)
            else:
                x = x.flatten(2).transpose(1, 2)

        if hasattr(model, "cls_token") and model.cls_token is not None:
            cls_tokens = model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            pe = model.pos_embed
            if x.shape[1] != pe.shape[1]:
                if hasattr(model, "pos_embed") and hasattr(model, "pos_drop"):
                    cls_pe = pe[:, :1]
                    spatial_pe = pe[:, 1:]
                    h = w = int(spatial_pe.shape[1] ** 0.5)
                    spatial_pe = spatial_pe.reshape(1, h, w, -1).permute(0, 3, 1, 2)
                    new_hw = int((x.shape[1] - (1 if hasattr(model, "cls_token") and model.cls_token is not None else 0)) ** 0.5)
                    spatial_pe = torch.nn.functional.interpolate(spatial_pe, size=(new_hw, new_hw), mode="bicubic", align_corners=False)
                    spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, -1)
                    pe = torch.cat([cls_pe, spatial_pe], dim=1) if (hasattr(model, "cls_token") and model.cls_token is not None) else spatial_pe
            x = x + pe
        if hasattr(model, "pos_drop"):
            x = model.pos_drop(x)
        for blk in model.blocks:
            x = blk(x)
        if hasattr(model, "norm") and model.norm is not None:
            x = model.norm(x)
        cls_features = x[:, 0]
        patch_features = x[:, 1:, :]
        return cls_features, patch_features

    def forward(self, pixel_values, neutral_pixel_values=None):
        cls_features, patch_features = self._extract_tokens(pixel_values)

        if self.use_neutral_reference and neutral_pixel_values is not None and self.neutral_encoder is not None:
            neutral_cls, neutral_patch = self._extract_tokens(neutral_pixel_values, model=self.neutral_encoder)

            pain_tokens = torch.cat([cls_features.unsqueeze(1), patch_features], dim=1)       # Query
            neutral_tokens = torch.cat([neutral_cls.unsqueeze(1), neutral_patch], dim=1)       # Key/Value

            attn_output, _ = self.neutral_cross_attn(pain_tokens, neutral_tokens, neutral_tokens)
            enhanced_tokens = self.neutral_norm(pain_tokens + attn_output)

            cls_features = enhanced_tokens[:, 0]
            patch_features = enhanced_tokens[:, 1:]

        # Sigmoid ensures output in [0, 1] range
        pspi_pred = F.sigmoid(self.pspi_head(cls_features).squeeze(-1))

        pspi_binary_logits = {k: head(cls_features).squeeze(-1) for k, head in self.pspi_binary_heads.items()}
        pspi_binary_probs = {k: torch.sigmoid(v) for k, v in pspi_binary_logits.items()}

        # AU prediction via cross-attention query tokens
        B = cls_features.shape[0]
        N, D = patch_features.shape[1], patch_features.shape[2]
        queries = self.au_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_aus, D]
        attn_scores = torch.matmul(queries, patch_features.transpose(1, 2)) / (D ** 0.5)  # [B, num_aus, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        au_pooled = torch.matmul(attn_weights, patch_features)  # [B, num_aus, D]

        au_features = self.au_shared_features(au_pooled)  # [B, num_aus, 512]
        au_preds = F.relu(self.au_head(au_features).squeeze(-1))  # [B, num_aus]

        return {
            'au_preds': au_preds,
            'au_expected': au_preds,
            'pspi_pred': pspi_pred,
            'pspi_expected': pspi_pred,
            'pspi_binary_logits': pspi_binary_logits,
            'pspi_binary_probs': pspi_binary_probs,
            'features': cls_features,
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage=self.test_stage_name)

    def _shared_step(self, batch, batch_idx, stage):
        images = batch['image']
        pspi_target_raw = batch['pspi_score'].float()  # Could be [0, 16] or [0, 1]
        au_target = batch['au_vector'].float()
        neutral_images = batch.get('neutral_image', None)

        # Multi-shot neutral references: average predictions across N neutral refs at inference time
        outputs = None
        if (
            neutral_images is not None
            and isinstance(neutral_images, torch.Tensor)
            and neutral_images.dim() == 5
            and stage != "train"
        ):
            num_shots = neutral_images.shape[1]  # [B, N, C, H, W]
            au_sum = None
            pspi_head_sum = None
            pspi_from_au_sum = None
            for shot_idx in range(num_shots):
                outputs_shot = self(images, neutral_pixel_values=neutral_images[:, shot_idx])
                au_shot = outputs_shot['au_preds']
                pspi_head_shot = outputs_shot['pspi_pred']
                pspi_from_au_shot = self._calculate_pspi_from_au(au_shot)

                au_sum = au_shot if au_sum is None else (au_sum + au_shot)
                pspi_head_sum = pspi_head_shot if pspi_head_sum is None else (pspi_head_sum + pspi_head_shot)
                pspi_from_au_sum = (
                    pspi_from_au_shot if pspi_from_au_sum is None else (pspi_from_au_sum + pspi_from_au_shot)
                )

            au_preds = au_sum / float(num_shots)
            pspi_pred_head = pspi_head_sum / float(num_shots)
            pspi_from_au = pspi_from_au_sum / float(num_shots)
            outputs = outputs_shot  # Keep last shot's outputs for binary logits
        else:
            # If 5D tensor provided during training, just use the first neutral
            if (
                neutral_images is not None
                and isinstance(neutral_images, torch.Tensor)
                and neutral_images.dim() == 5
            ):
                neutral_images = neutral_images[:, 0]

            outputs = self(images, neutral_pixel_values=neutral_images)
            au_preds = outputs['au_preds']
            pspi_pred_head = outputs['pspi_pred']
            pspi_from_au = self._calculate_pspi_from_au(au_preds)

        # Optionally derive PSPI from AU head instead of direct PSPI head at inference
        use_derived_pspi = (stage != "train") and bool(getattr(self.hparams, "no_inference_pspi_head", False))
        if use_derived_pspi:
            pspi_pred = self._normalize_pspi(pspi_from_au, pspi_max=16.0)
        else:
            pspi_pred = pspi_pred_head  # Already in [0, 1]

        # Targets from dataloader are denormalized [0, 16]; normalize for loss
        pspi_target_normalized = self._normalize_pspi(pspi_target_raw, pspi_max=16.0)

        pspi_loss = F.mse_loss(pspi_pred, pspi_target_normalized)

        pspi_binary_loss = torch.tensor(0.0, device=self.device)
        if outputs is not None:
            logits_dict = outputs.get('pspi_binary_logits') or {}
            per_thr_losses = []
            for thr in self.binary_pspi_thresholds:
                key = str(int(thr))
                if key not in logits_dict:
                    continue
                logits = logits_dict[key]
                targets = (pspi_target_raw >= thr).float()
                per_thr_losses.append(F.binary_cross_entropy_with_logits(logits, targets))

            if len(per_thr_losses) > 0:
                pspi_binary_loss = torch.stack(per_thr_losses).mean()

        au_loss = F.mse_loss(au_preds, au_target)

        total_loss = (
            self.pspi_loss_weight * pspi_loss
            + self.au_loss_weight * au_loss
            + pspi_binary_loss
        )

        batch_size = images.size(0)
        self._update_pspi_metrics(pspi_pred, pspi_target_raw, au_preds, au_target, pspi_from_au, stage, targets_already_denormalized=True)
        self._log_pspi_metrics(
            stage=stage,
            batch_size=batch_size,
            total_loss=total_loss,
            pspi_loss=pspi_loss,
            au_loss=au_loss,
            batch_idx=batch_idx,
            pspi_pred=pspi_pred,
            pspi_target=pspi_target_raw
        )
        self.log(f"{stage}/loss/pspi_binary", pspi_binary_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return total_loss

    def on_train_epoch_end(self):
        self.current_epoch_count += 1
        self._compute_epoch_end_metrics("train", batch_size=1)

    def on_validation_epoch_end(self):
        self._compute_epoch_end_metrics("val", batch_size=1)

    def on_test_epoch_end(self):
        self._compute_epoch_end_metrics(self.test_stage_name, batch_size=1)

    def on_train_epoch_start(self):
        """Freeze backbone, re-enable only LoRA adapter parameters."""
        for param in self.vit_model.parameters():
            param.requires_grad = False
        for name, param in self.vit_model.named_parameters():
            if 'lora' in name.lower() or 'modules_to_save' in name.lower():
                param.requires_grad = True

    def configure_optimizers(self):
        backbone_params = list(self.vit_model.parameters())
        head_params = (list(self.pspi_head.parameters()) +
              list(self.au_shared_features.parameters()) +
              list(self.au_head.parameters()) +
              list(self.pspi_binary_heads.parameters()))

        if self.neutral_cross_attn is not None:
            head_params.extend(list(self.neutral_cross_attn.parameters()))
        if self.neutral_norm is not None:
            head_params.extend(list(self.neutral_norm.parameters()))
        if self.neutral_encoder is not None:
             for name, param in self.neutral_encoder.named_parameters():
                if 'lora' in name.lower() or 'modules_to_save' in name.lower():
                    backbone_params.append(param)

        optimizer = AdamW([
            {'params': backbone_params, 'lr': self.hparams.learning_rate * 0.1},
            {'params': head_params, 'lr': self.hparams.learning_rate}
        ], weight_decay=self.hparams.weight_decay)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss/total",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_pspi(self, images):
        """Predict PSPI scores for a batch of images."""
        self.eval()
        with torch.no_grad():
            outputs = self(images)
        return {
            'au_expected_values': outputs['au_expected'].cpu().numpy(),
            'pspi_predicted': outputs['pspi_pred'].cpu().numpy(),
        }


def create_vitpain_model(
    model_size="large_dinov3",
    lora_rank=8,
    lora_alpha=16,
    **kwargs
):
    """
    Create a PSPI ViT model with DinoV3 backbone, LoRA adapters, AU query head,
    and binary classification heads.

    Args:
        model_size: Model size ("small_dinov3", "base_dinov3", "large_dinov3")
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        **kwargs: Additional arguments passed to ViTPain
    """
    model_configs = {
        "small_dinov3": "timm/vit_small_patch16_dinov3.sat493m",
        "base_dinov3": "timm/vit_base_patch16_dinov3.sat493m",
        "large_dinov3": "timm/vit_large_patch16_dinov3.sat493m",
    }

    if model_size not in model_configs:
        raise ValueError(f"model_size must be one of {list(model_configs.keys())}")

    model_name = model_configs[model_size]

    # Pop legacy kwargs for backward compatibility with old calling conventions
    kwargs.pop('use_lora', None)
    kwargs.pop('use_binary_classification_head', None)
    kwargs.pop('use_contrastive_loss', None)
    kwargs.pop('contrastive_loss_weight', None)
    kwargs.pop('freeze_backbone_epochs', None)
    kwargs.pop('warmup_steps', None)

    return ViTPain(
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        **kwargs
    )

def load_pretrained_heatmap_model(checkpoint_path, distilled_model_size="large_dinov3", **kwargs):
    """Load a pretrained heatmap model from a checkpoint, freezing all parameters."""
    kwargs['model_size'] = distilled_model_size
    model = create_vitpain_model(**kwargs)

    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    model_sd = model.state_dict()

    compatible = {k: v for k, v in state_dict.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    model.load_state_dict(compatible, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    return model

def load_pretrained_synthetic_data_model(checkpoint_path, synthetic_model_size="large_dinov3", **kwargs):
    """Load a pretrained synthetic-data model from a checkpoint."""
    kwargs.setdefault('model_size', synthetic_model_size)
    model = create_vitpain_model(**kwargs)

    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    model_sd = model.state_dict()

    compatible = {k: v for k, v in state_dict.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    model.load_state_dict(compatible, strict=False)

    return model
