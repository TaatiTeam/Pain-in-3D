import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

from .pspi_evaluator_mixin import PSPIEvaluatorMixin

# Import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT library not available. LoRA functionality will be disabled.")


class PSPIViTRegressor(PSPIEvaluatorMixin, pl.LightningModule):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        num_classes=1,  # Number of output dimensions (1 for regression)
        num_au_features=6,
        learning_rate=1e-4,
        weight_decay=1e-2,
        warmup_steps=1000,
        max_epochs=100,
        freeze_backbone_epochs=9999,  # Always frozen (only train LoRA)
        au_loss_weight=1.0,  # Weight for AU regression loss
        pspi_loss_weight=1.0,  # Weight for PSPI regression loss
        use_binary_classification_head=True,  # Default: enabled for pain/no-pain classification
        dropout_rate=0.5,  # Default dropout
        # LoRA parameters (always enabled)
        use_lora=True,  # Always use LoRA for fine-tuning
        lora_rank=8,  # LoRA rank
        lora_alpha=16,  # LoRA alpha
        test_stage_name="test",  # Stage name for test step (can be overridden to "test_unbc")
        use_neutral_reference=False,  # Whether to use a neutral reference image
    ):
        """
        Vision Transformer for PSPI (Pain Score) regression and AU regression.
        
        Always uses DinoV3 backbone with LoRA adapters and AU query head.
        Binary classification head is enabled by default for pain/no-pain classification.
        
        Args:
            model_name: Pretrained ViT model name (default: DinoV3)
            num_classes: Number of output dimensions (1 for regression)
            num_au_features: Number of Action Unit features (6 for AU4,6,7,9,10,43)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_epochs: Maximum training epochs
            freeze_backbone_epochs: Number of epochs to freeze backbone (default: 9999, always frozen)
            au_loss_weight: Weight for AU regression loss
            pspi_loss_weight: Weight for PSPI regression loss
            use_binary_classification_head: Whether to use binary classification head (default: True)
            dropout_rate: Dropout rate for regularization
            lora_rank: LoRA rank (default: 8)
            lora_alpha: LoRA alpha (default: 16)
            use_neutral_reference: Whether to use a neutral reference image
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.dropout_rate = dropout_rate
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.test_stage_name = test_stage_name
        self.use_neutral_reference = use_neutral_reference
        self.use_binary_classification_head = use_binary_classification_head

        # Load pretrained backbone (HuggingFace ViT or timm for DinoV3)
        self.is_timm = ("dinov3" in model_name) or (model_name.startswith("timm/"))
        if self.is_timm:
            import timm
            timm_model_name = model_name.replace("timm/", "")
            # normalize potential outdated weight tags
            timm_model_name = timm_model_name.replace("sat_493m", "sat493m").replace("lvd_1689m", "lvd1689m")
            print(f"Loading DinoV3 pretrained weights from timm: {timm_model_name}")
            self.vit_model = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0
            )
            print(f"✓ Successfully loaded DinoV3 pretrained weights")
            hidden_size = getattr(self.vit_model, "num_features", None)
            if hidden_size is None:
                # Fallback to embed dim attr used by some timm ViTs
                hidden_size = getattr(self.vit_model, "embed_dim", None)
            if hidden_size is None:
                raise RuntimeError("Could not determine hidden size from timm model.")
            
            # Apply LoRA if requested
            if use_lora:
                if not PEFT_AVAILABLE:
                    raise ImportError("PEFT library is required for LoRA. Please install it: pip install peft")
                if not self.is_timm:
                    raise ValueError("LoRA is currently only supported for timm models (DinoV3)")
                
                # Configure LoRA for attention layers
                # For timm ViT models, attention layers are typically named 'attn.qkv' and 'attn.proj'
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["qkv", "proj"],  # Target attention layers
                    lora_dropout=0.1,
                )
                self.vit_model = get_peft_model(self.vit_model, lora_config)
                print(f"Applied LoRA to model with rank={lora_rank}, alpha={lora_alpha}")
        else:
            print(f"Loading HuggingFace ViT pretrained weights: {model_name}")
            from transformers import ViTModel, ViTConfig
            self.vit_config = ViTConfig.from_pretrained(model_name)
            self.vit_model = ViTModel.from_pretrained(model_name)
            print(f"✓ Successfully loaded HuggingFace ViT pretrained weights")
            hidden_size = self.vit_config.hidden_size
        
        # Initialize Neutral Face Encoder and Cross-Attention if requested
        if self.use_neutral_reference:
            print("Initializing Neutral Face Encoder and Cross-Attention...")
            if self.is_timm:
                import timm
                # Re-create timm model for neutral encoder
                self.neutral_encoder = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=0
                )
                # Apply LoRA to neutral encoder if main model uses LoRA
                if use_lora:
                    self.neutral_encoder = get_peft_model(self.neutral_encoder, lora_config)
            else:
                # Re-create HF model for neutral encoder
                from transformers import ViTModel
                self.neutral_encoder = ViTModel.from_pretrained(model_name)
            
            # Cross-Attention: Pain (Query) attends to Neutral (Key/Value)
            self.neutral_cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
            self.neutral_norm = nn.LayerNorm(hidden_size)
        else:
            self.neutral_encoder = None
            self.neutral_cross_attn = None
            self.neutral_norm = None

        # Regression head for PSPI prediction (uses configurable dropout)
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
            nn.Linear(128, 1),  # Output 1 value for regression
        )

        # Optional additional binary classifier heads for UNBC pain/no-pain objectives.
        # These are auxiliary losses meant to directly optimize binary classification at PSPI thresholds 1/2/3.
        self.binary_pspi_thresholds = (1.0, 2.0, 3.0)
        if self.use_binary_classification_head:
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
        else:
            self.pspi_binary_heads = None
        
        # AU prediction heads (uses configurable dropout)
        self.au_shared_features = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
        )

        # AU query head (always enabled)
        num_aus = 6  # AU4, AU6, AU7, AU9, AU10, AU43
        self.au_queries = nn.Parameter(torch.randn(num_aus, hidden_size))

        # Individual regression heads for each AU (single output each)
        self.au_head = nn.Linear(512, 1)
        
        # Loss weights
        self.pspi_loss_weight = pspi_loss_weight
        self.au_loss_weight = au_loss_weight
        
        # Initialize metrics using mixin
        self._init_pspi_metrics()

        # Track current epoch for backbone freezing
        self.current_epoch_count = 0
        
    def _extract_tokens(self, images, model=None):
        """Return (cls_features, patch_features) for both HF and timm backbones."""
        if model is None:
            model = self.vit_model
            
        if not self.is_timm:
            outputs = model(pixel_values=images)
            last_hidden = outputs.last_hidden_state
            return last_hidden[:, 0], last_hidden[:, 1:, :]
        # timm path
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
        """Forward pass through the model."""
        cls_features, patch_features = self._extract_tokens(pixel_values)
        
        neutral_cls_raw = None
        
        # Apply cross-attention with neutral reference if available
        if self.use_neutral_reference and neutral_pixel_values is not None and self.neutral_encoder is not None:
            # Extract neutral features
            neutral_cls, neutral_patch = self._extract_tokens(neutral_pixel_values, model=self.neutral_encoder)
            neutral_cls_raw = neutral_cls
            
            # Concatenate CLS and Patch tokens for attention
            # Pain tokens (Query)
            pain_tokens = torch.cat([cls_features.unsqueeze(1), patch_features], dim=1)
            # Neutral tokens (Key/Value)
            neutral_tokens = torch.cat([neutral_cls.unsqueeze(1), neutral_patch], dim=1)
            
            # Cross-Attention: Pain attends to Neutral
            # This highlights differences from neutral state
            attn_output, _ = self.neutral_cross_attn(pain_tokens, neutral_tokens, neutral_tokens)
            
            # Residual connection + Norm
            enhanced_tokens = self.neutral_norm(pain_tokens + attn_output)
            
            # Split back into CLS and Patch features
            cls_features = enhanced_tokens[:, 0]
            patch_features = enhanced_tokens[:, 1:]
        
        # PSPI prediction (regression) - apply sigmoid to ensure [0, 1] range
        pspi_pred = F.sigmoid(self.pspi_head(cls_features).squeeze(-1))  # [batch_size]

        pspi_binary_logits = None
        pspi_binary_probs = None
        if self.pspi_binary_heads is not None:
            pspi_binary_logits = {k: head(cls_features).squeeze(-1) for k, head in self.pspi_binary_heads.items()}
            pspi_binary_probs = {k: torch.sigmoid(v) for k, v in pspi_binary_logits.items()}
        
        # AU prediction: use query head cross-attention (always enabled)
        B = cls_features.shape[0]
        # AU attention pooling with query tokens
        N, D = patch_features.shape[1], patch_features.shape[2]
        queries = self.au_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_aus, D]
        keys = patch_features
        values = patch_features
        attn_scores = torch.matmul(queries, keys.transpose(1, 2)) / (D ** 0.5)  # [B, num_aus, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        au_pooled = torch.matmul(attn_weights, values)  # [B, num_aus, D]

        # Apply ReLU to ensure non-negative values (as in reference)
        au_features = self.au_shared_features(au_pooled)  # [B, num_aus, 512]
        au_preds = F.relu(self.au_head(au_features).squeeze(-1))  # [B, num_aus]
        
        return {
            'au_preds': au_preds,              # AU continuous predictions [batch_size, 6]
            'au_expected': au_preds,           # Same as au_preds for compatibility
            'pspi_pred': pspi_pred,            # PSPI regression prediction [batch_size]
            'pspi_expected': pspi_pred,        # Alias for consistency
            'pspi_binary_logits': pspi_binary_logits,  # Dict[str -> Tensor[batch_size]] (logits)
            'pspi_binary_probs': pspi_binary_probs,    # Dict[str -> Tensor[batch_size]] (sigmoid)
            'features': cls_features,
            # Explicit name for the embedding AFTER neutral-reference cross-attention (if enabled)
            # This is what we want to distill against when using neutral-reference learning.
            'features_post_neutral_crossattn': cls_features,
            'pain_cls': pain_cls_raw,
            'neutral_cls': neutral_cls_raw
        }
    
    def _apply_selective_layer_unfreezing(self):
        """Apply selective layer unfreezing: freeze all layers except the last N."""
        if not self.is_timm or self.unfreeze_last_n_layers <= 0:
            return

        def _ensure_lora_trainable():
            """Keep LoRA adapters trainable regardless of backbone freezing."""
            if not self.use_lora:
                return
            for name, param in self.vit_model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
        
        # Handle PEFT-wrapped models (LoRA)
        # PEFT wraps the model, so we need to access the base model to get blocks
        base_model = self.vit_model
        if hasattr(self.vit_model, 'base_model'):
            # PEFT model structure: vit_model.base_model contains the actual timm model
            base_model = self.vit_model.base_model
        elif hasattr(self.vit_model, 'model'):
            # Alternative PEFT structure
            base_model = self.vit_model.model
        
        # Get the number of blocks
        if not hasattr(base_model, 'blocks'):
            print("Warning: Model does not have 'blocks' attribute. Cannot apply selective unfreezing.")
            return
        
        blocks = base_model.blocks
        num_blocks = len(blocks)
        
        if self.unfreeze_last_n_layers > num_blocks:
            print(f"Warning: unfreeze_last_n_layers ({self.unfreeze_last_n_layers}) > num_blocks ({num_blocks}). Unfreezing all blocks.")
            self.unfreeze_last_n_layers = num_blocks
        
        # Freeze all blocks first
        for block in blocks:
            # Freeze base weights but never freeze LoRA adapter params
            for name, param in block.named_parameters(recurse=True):
                if 'lora' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Unfreeze the last N blocks
        for i in range(num_blocks - self.unfreeze_last_n_layers, num_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True

        # Defensive: ensure LoRA stays trainable even if some modules were frozen above
        _ensure_lora_trainable()
        
        print(f"Applied selective layer unfreezing: last {self.unfreeze_last_n_layers} layers unfrozen out of {num_blocks} total layers")
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._shared_step(batch, batch_idx, stage="train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_step(batch, batch_idx, stage="val")
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_step(batch, batch_idx, stage=self.test_stage_name)
    
    def _shared_step(self, batch, batch_idx, stage):
        """Shared step for train/val/test."""
        images = batch['image']
        pspi_target_raw = batch['pspi_score'].float()  # Regression target (could be [0, 16] or [0, 1])
        au_target = batch['au_vector'].float()      # AU features for regression
        
        # Get neutral image if available
        neutral_images = batch.get('neutral_image', None)

        # Forward pass (supports multi-shot neutral references at inference time)
        if (
            neutral_images is not None
            and isinstance(neutral_images, torch.Tensor)
            and neutral_images.dim() == 5
            and stage != "train"
        ):
            # neutral_images: [B, N, C, H, W]
            num_shots = neutral_images.shape[1]
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
            pspi_from_au = pspi_from_au_sum / float(num_shots)  # denormalized
        else:
            # If a 5D tensor is accidentally provided during training, just use the first neutral.
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

        # Inference-time option: ignore PSPI head and derive PSPI from AU head
        # Training still uses the PSPI head as auxiliary supervision.
        use_derived_pspi = (stage != "train") and bool(getattr(self.hparams, "no_inference_pspi_head", False))
        if use_derived_pspi:
            pspi_pred = self._normalize_pspi(pspi_from_au, pspi_max=16.0)
        else:
            pspi_pred = pspi_pred_head  # Model output is in [0, 1] range
        
        # Normalize targets to [0, 1] for loss computation
        # We assume targets from dataloader are always denormalized [0, 16]
        pspi_target_normalized = self._normalize_pspi(pspi_target_raw, pspi_max=16.0)
        
        # PSPI loss (MSE) - use normalized values
        pspi_loss = F.mse_loss(pspi_pred, pspi_target_normalized)

        # Optional auxiliary binary classification losses (UNBC): thresholds 1/2/3
        pspi_binary_loss = torch.tensor(0.0, device=self.device)
        if self.pspi_binary_heads is not None:
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
        
        # AU regression loss (MSE)
        au_loss = F.mse_loss(au_preds, au_target)
        
        # Contrastive Loss
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if self.use_contrastive_loss and self.use_neutral_reference and outputs.get('pain_cls') is not None and outputs.get('neutral_cls') is not None:
            pain_cls = outputs['pain_cls']
            neutral_cls = outputs['neutral_cls']
            
            # Normalize embeddings
            pain_norm = F.normalize(pain_cls, p=2, dim=1)
            neutral_norm = F.normalize(neutral_cls, p=2, dim=1)
            
            # Compute cosine similarity
            cosine_sim = torch.sum(pain_norm * neutral_norm, dim=1)
            
            # Target distance: PSPI normalized [0, 1]
            # Actual distance: 0.5 * (1 - cosine_sim) [0, 1]
            # This encourages embedding distance to correlate with pain intensity
            target_dist = pspi_target_normalized
            actual_dist = 0.5 * (1.0 - cosine_sim)
            
            contrastive_loss = F.mse_loss(actual_dist, target_dist)
        
        # Total loss
        total_loss = (
            self.pspi_loss_weight * pspi_loss
            + self.au_loss_weight * au_loss
            + pspi_binary_loss
        )

        # Update metrics using mixin method
        # Pass raw target (will be handled by _update_pspi_metrics)
        batch_size = images.size(0)
        self._update_pspi_metrics(pspi_pred, pspi_target_raw, au_preds, au_target, pspi_from_au, stage, targets_already_denormalized=True)
        
        # Log metrics using mixin method
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

        if self.pspi_binary_heads is not None:
            self.log(f"{stage}/loss/pspi_binary", pspi_binary_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        
        if self.use_contrastive_loss:
            self.log(f"{stage}/contrastive_loss", contrastive_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return total_loss
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        self.current_epoch_count += 1
        backbone_frozen = not any(p.requires_grad for p in self.vit_model.parameters())
        self.log("backbone_frozen", float(backbone_frozen), sync_dist=True)
        
        # Compute epoch-end metrics (tolerance accuracies, CCC, ICC)
        self._compute_epoch_end_metrics("train", batch_size=1)
        
        if self.trainer.is_global_zero:
            print(f"Epoch {self.current_epoch} completed:", flush=True)
            print(f"  Backbone Frozen: {backbone_frozen}", flush=True)
        
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        # Compute epoch-end metrics (tolerance accuracies, CCC, ICC)
        # This will store _last_val_mae and _last_val_corr before resetting metrics
        self._compute_epoch_end_metrics("val", batch_size=1)
        
        # Print validation metrics
        # Use stored values from _compute_and_log_torchmetrics if available,
        # otherwise fall back to logged_metrics
        if self.trainer.is_global_zero:
            try:
                # Get loss from logged_metrics (this is logged during validation_step)
                val_loss = self.trainer.logged_metrics.get('val/loss/total', 0.0)
                
                # Try to get MAE and correlation from stored values (computed before reset)
                val_mae = getattr(self, '_last_val_mae', None)
                val_corr = getattr(self, '_last_val_corr', None)
                
                # Fall back to logged_metrics if stored values not available
                if val_mae is None:
                    val_mae_metric = self.trainer.logged_metrics.get('val/regression/mae', 0.0)
                    if isinstance(val_mae_metric, torch.Tensor):
                        val_mae = val_mae_metric.item()
                    else:
                        val_mae = float(val_mae_metric)
                
                if val_corr is None:
                    val_corr_metric = self.trainer.logged_metrics.get('val/regression/corr', 0.0)
                    if isinstance(val_corr_metric, torch.Tensor):
                        val_corr = val_corr_metric.item() if not torch.isnan(val_corr_metric) else 0.0
                    else:
                        val_corr = float(val_corr_metric) if not (isinstance(val_corr_metric, float) and (val_corr_metric != val_corr_metric)) else 0.0
                
                # Convert loss to float if it's a tensor
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()
                else:
                    val_loss = float(val_loss)
                
                print(f"  Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | Corr: {val_corr:.4f}", flush=True)
            except (RuntimeError, AttributeError, Exception) as e:
                # If accessing metrics fails (e.g., due to distributed sync issues),
                # just skip printing but don't crash
                print(f"  Could not retrieve validation metrics: {e}", flush=True)
    
    def on_test_epoch_end(self):
        """Called at the end of each test epoch."""
        # Debug: Check if predictions were accumulated
        if self.test_stage_name == 'test_unbc':
            preds_list_name = 'test_unbc_preds'
        elif self.test_stage_name == 'last_epoch_test':
            preds_list_name = 'last_epoch_test_preds'
        else:
            preds_list_name = 'test_preds'
        
        if hasattr(self, preds_list_name):
            preds_list = getattr(self, preds_list_name)
            print(f"on_test_epoch_end: {preds_list_name} has {len(preds_list)} batches")
            if len(preds_list) > 0:
                total_samples = sum(pred.shape[0] for pred in preds_list)
                print(f"on_test_epoch_end: Total test samples: {total_samples}")
        else:
            print(f"WARNING: {preds_list_name} does not exist in on_test_epoch_end!")
        
        # Compute epoch-end metrics
        self._compute_epoch_end_metrics(self.test_stage_name, batch_size=1)
    
    def on_train_epoch_start(self):
        """Handle backbone freezing logic at the start of each epoch."""
        freeze_epochs = self.hparams.freeze_backbone_epochs

        # If we are in the freezing period
        if self.current_epoch < freeze_epochs:
            # Freeze backbone completely
            # Note: This iterates over all parameters including LoRA adapters if present
            for param in self.vit_model.parameters():
                param.requires_grad = False
            
            # If using LoRA, we must ensure LoRA adapters remain trainable
            if self.use_lora:
                for name, param in self.vit_model.named_parameters():
                    if 'lora' in name.lower() or 'modules_to_save' in name.lower():
                        param.requires_grad = True
                        
        # If we just finished the freezing period (transition point)
        elif self.current_epoch == freeze_epochs:
            print(f"Unfreezing backbone at epoch {self.current_epoch}")
            # Unfreeze everything first
            for param in self.vit_model.parameters():
                param.requires_grad = True
                
            # With LoRA, the base model remains frozen - only LoRA adapters are trained
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        backbone_params = list(self.vit_model.parameters())
        head_params = (list(self.pspi_head.parameters()) + 
              list(self.au_shared_features.parameters()) +
              list(self.au_head.parameters()))

        if self.pspi_binary_heads is not None:
            head_params.extend(list(self.pspi_binary_heads.parameters()))
        
        # Add neutral reference attention parameters if they exist
        if self.neutral_cross_attn is not None:
            head_params.extend(list(self.neutral_cross_attn.parameters()))
        if self.neutral_norm is not None:
            head_params.extend(list(self.neutral_norm.parameters()))
            
        # Add neutral encoder LoRA parameters if they exist
        if self.neutral_encoder is not None and self.use_lora:
             for name, param in self.neutral_encoder.named_parameters():
                if 'lora' in name.lower() or 'modules_to_save' in name.lower():
                    # Add to backbone_params to match the learning rate of the main backbone LoRA
                    backbone_params.append(param)
            
        # Note: self.neutral_encoder parameters are intentionally NOT added to optimizer
        # This ensures the neutral encoder remains frozen as a fixed reference
        
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
            au_expected = outputs['au_expected']
            pspi_pred = outputs['pspi_pred']
            
        return {
            'au_expected_values': au_expected.cpu().numpy(),
            'pspi_predicted': pspi_pred.cpu().numpy(),
        }


def create_pspi_vit_model(
    model_size="large_dinov3",
    lora_rank=8,
    lora_alpha=16,
    **kwargs
):
    """
    Create a PSPI ViT model with DinoV3 backbone, LoRA adapters, and AU query head.
    
    Fixed features (always enabled):
    - DinoV3 backbone (always frozen)
    - LoRA adapters (only trainable parameters)
    - AU query head (cross-attention for AU prediction)
    - Binary classification head (pain/no-pain)
    
    Args:
        model_size: Model size ("small_dinov3", "base_dinov3", "large_dinov3")
        lora_rank: LoRA rank (default: 8)
        lora_alpha: LoRA alpha (default: 16)
        **kwargs: Additional arguments passed to PSPIViTRegressor
    """
    model_configs = {
        "small_dinov3": "timm/vit_small_patch14_dinov2.lvd142m",
        "base_dinov3": "timm/vit_base_patch14_dinov2.lvd142m",
        "large_dinov3": "timm/vit_large_patch14_dinov2.lvd142m",
    }
    
    if model_size not in model_configs:
        raise ValueError(f"model_size must be one of {list(model_configs.keys())}")
    
    model_name = model_configs[model_size]
    
    # All these are always enabled
    return PSPIViTRegressor(
        model_name=model_name,
        use_lora=True,  # Always use LoRA
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_binary_classification_head=True,  # Always use binary classification
        **kwargs
    )

def load_pretrained_heatmap_model(checkpoint_path, distilled_model_size="large", **kwargs):
    kwargs['model_size'] = distilled_model_size
    model = create_pspi_vit_model(**kwargs)

    def _summarize_prefixes(keys, max_items=15):
        counts = {}
        for key in keys:
            prefix = key.split('.', 1)[0]
            counts[prefix] = counts.get(prefix, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_items]

    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    model_sd = model.state_dict()

    ckpt_keys = set(state_dict.keys())
    model_keys = set(model_sd.keys())
    matched = ckpt_keys & model_keys
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatch = sorted([k for k in matched if tuple(model_sd[k].shape) != tuple(state_dict[k].shape)])
    compatible = {k: v for k, v in state_dict.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}

    incompatible = model.load_state_dict(compatible, strict=False)
    print(f"Loaded pretrained heatmap model from: {checkpoint_path}")
    print(f"  Checkpoint tensors: {len(ckpt_keys)} | Model tensors: {len(model_keys)}")
    print(f"  Matched keys: {len(matched)}")
    print(f"  Shape-mismatched keys (skipped): {len(shape_mismatch)}")
    if len(shape_mismatch) > 0:
        print(f"    {shape_mismatch[:20]}{' ...' if len(shape_mismatch) > 20 else ''}")
    print(f"  Loaded tensors: {len(compatible)}")
    print(f"  Missing keys: {len(missing)}")
    if len(missing) > 0:
        print(f"    {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print(f"    {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
    print(f"  Matched prefixes (top): {_summarize_prefixes(matched)}")
    print(f"  Missing prefixes (top): {_summarize_prefixes(missing)}")
    print(f"  Unexpected prefixes (top): {_summarize_prefixes(unexpected)}")
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_pretrained_synthetic_data_model(checkpoint_path, synthetic_model_size="large_dinov3", **kwargs):
    # Default to DinoV3 unless caller explicitly requests a different backbone.
    # Important: do NOT overwrite kwargs['model_size'] if provided.
    kwargs.setdefault('model_size', synthetic_model_size)
    model = create_pspi_vit_model(**kwargs)

    def _summarize_prefixes(keys, max_items=15):
        counts = {}
        for key in keys:
            prefix = key.split('.', 1)[0]
            counts[prefix] = counts.get(prefix, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_items]

    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    model_sd = model.state_dict()

    ckpt_keys = set(state_dict.keys())
    model_keys = set(model_sd.keys())
    matched = ckpt_keys & model_keys
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatch = sorted([k for k in matched if tuple(model_sd[k].shape) != tuple(state_dict[k].shape)])

    compatible = {k: v for k, v in state_dict.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    # This avoids hard crashes on size mismatches while still loading everything compatible.
    model.load_state_dict(compatible, strict=False)

    ckpt_lora = sorted([k for k in ckpt_keys if 'lora' in k.lower()])
    loaded_lora = sorted([k for k in compatible.keys() if 'lora' in k.lower()])

    print(f"Loaded pretrained synthetic model from: {checkpoint_path}")
    print(f"  Requested model_size: {kwargs.get('model_size')}")
    print(f"  Checkpoint tensors: {len(ckpt_keys)} | Model tensors: {len(model_keys)}")
    print(f"  Matched keys: {len(matched)}")
    print(f"  Shape-mismatched keys (skipped): {len(shape_mismatch)}")
    if len(shape_mismatch) > 0:
        print(f"    {shape_mismatch[:20]}{' ...' if len(shape_mismatch) > 20 else ''}")
    print(f"  Loaded tensors: {len(compatible)}")
    print(f"  Missing keys: {len(missing)}")
    if len(missing) > 0:
        print(f"    {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print(f"    {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
    print(f"  Matched prefixes (top): {_summarize_prefixes(matched)}")
    print(f"  Missing prefixes (top): {_summarize_prefixes(missing)}")
    print(f"  Unexpected prefixes (top): {_summarize_prefixes(unexpected)}")
    print(f"  LoRA tensors in checkpoint: {len(ckpt_lora)}")
    print(f"  LoRA tensors loaded: {len(loaded_lora)}")
    if len(ckpt_lora) > 0 and len(loaded_lora) == 0:
        print("  Warning: checkpoint contains LoRA tensors but none matched this model. Model/backbone/PEFT config may differ.")

    return model
