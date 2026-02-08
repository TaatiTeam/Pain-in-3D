"""
PSPI Evaluator Mixin for shared evaluation and logging logic.

Simplified version with core metrics only:
- Regression: MAE, Pearson Correlation
- Classification: Binary F1, Precision, Recall, AUROC, AUPR
"""

import torch
import torchmetrics
import numpy as np


class PSPIEvaluatorMixin:
    """
    Mixin class providing essential evaluation and logging functionality for PSPI regression models.
    
    Core Metrics:
    - Regression: MAE, Pearson Correlation
    - Classification: F1, Precision, Recall, AUROC, AUPR (for pain vs. no-pain)
    """
    
    def _denormalize_pspi(self, normalized_values, pspi_max=16.0):
        """Denormalize model outputs from [0, 1] to [0, pspi_max]."""
        return normalized_values * pspi_max
    
    def _normalize_pspi(self, pspi_values, pspi_max=16.0):
        """Normalize PSPI values from [0, pspi_max] to [0, 1]."""
        return pspi_values / pspi_max
    
    def _ensure_denormalized(self, pspi_values, pspi_max=16.0):
        """Ensure PSPI values are in denormalized [0, pspi_max] range."""
        if torch.all(pspi_values <= 1.1):
            return self._denormalize_pspi(pspi_values, pspi_max)
        return pspi_values
    
    def _pspi_to_binary(self, pspi_values, threshold=2.0):
        """Convert denormalized PSPI to binary (pain/no-pain) using threshold."""
        return (pspi_values >= threshold).long()
    
    def _calculate_pspi_from_au(self, au_preds):
        """
        Calculate PSPI from AU predictions using the standard formula.
        Formula: PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
        
        Args:
            au_preds: Tensor of shape [batch_size, 6] containing AU predictions
            
        Returns:
            pspi_from_au: Tensor of shape [batch_size] containing PSPI calculated from AUs
        """
        pspi_from_au = (
            au_preds[:, 0] +  # AU4
            torch.max(au_preds[:, 1], au_preds[:, 2]) +  # max(AU6, AU7)
            torch.max(au_preds[:, 3], au_preds[:, 4]) +  # max(AU9, AU10)
            au_preds[:, 5]  # AU43
        )
        # Safeguard against NaN
        pspi_from_au = torch.where(torch.isnan(pspi_from_au), torch.zeros_like(pspi_from_au), pspi_from_au)
        return pspi_from_au
    
    def _init_pspi_metrics(self):
        """Initialize core torchmetrics for PSPI evaluation."""
        # Initialize for train, val, test, test_unbc stages
        stages = ['train', 'val', 'test', 'last_epoch_test', 'test_unbc']
        
        for stage in stages:
            # Regression metrics
            setattr(self, f"{stage}_mae", torchmetrics.MeanAbsoluteError())
            setattr(self, f"{stage}_corr", torchmetrics.PearsonCorrCoef())
            
            # Binary classification metrics (pain vs. no-pain, threshold=2)
            setattr(self, f"{stage}_binary_precision", torchmetrics.Precision(task='binary'))
            setattr(self, f"{stage}_binary_recall", torchmetrics.Recall(task='binary'))
            setattr(self, f"{stage}_binary_f1", torchmetrics.F1Score(task='binary'))
            setattr(self, f"{stage}_binary_auroc", torchmetrics.AUROC(task='binary'))
            setattr(self, f"{stage}_binary_aupr", torchmetrics.AveragePrecision(task='binary'))
            
            # AU metrics
            setattr(self, f"{stage}_au_mae", torchmetrics.MeanAbsoluteError())
            setattr(self, f"{stage}_pspi_au_mae", torchmetrics.MeanAbsoluteError())
            setattr(self, f"{stage}_pspi_au_corr", torchmetrics.PearsonCorrCoef())
        
        # Storage for predictions/targets (for epoch-end processing)
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        self.last_epoch_test_preds = []
        self.last_epoch_test_targets = []
        self.test_unbc_preds = []
        self.test_unbc_targets = []
    
    def _update_pspi_metrics(self, pspi_pred, pspi_target, au_preds, au_target, pspi_from_au, stage, 
                           pspi_max=16.0, targets_already_denormalized=False):
        """
        Update all metrics for the given stage.
        
        Args:
            pspi_pred: PSPI predictions [batch_size] - assumed to be in [0, 1] range
            pspi_target: PSPI targets [batch_size]
            au_preds: AU predictions [batch_size, 6]
            au_target: AU targets [batch_size, 6]
            pspi_from_au: PSPI calculated from AU predictions [batch_size]
            stage: Stage name ("train", "val", "test", "last_epoch_test", or "test_unbc")
            pspi_max: Maximum PSPI value for denormalization (default 16)
            targets_already_denormalized: If True, skip auto-detection
        """
        # Denormalize predictions from [0, 1] to [0, pspi_max]
        pspi_pred_denorm = self._denormalize_pspi(pspi_pred, pspi_max)
        
        # Ensure targets are in denormalized range
        if targets_already_denormalized:
            pspi_target_denorm = pspi_target
        else:
            pspi_target_denorm = self._ensure_denormalized(pspi_target, pspi_max)
        
        # Store predictions/targets for epoch-end processing
        if stage == "train":
            self.train_preds.append(pspi_pred_denorm.detach().cpu())
            self.train_targets.append(pspi_target_denorm.detach().cpu())
        elif stage == "val":
            self.val_preds.append(pspi_pred_denorm.detach().cpu())
            self.val_targets.append(pspi_target_denorm.detach().cpu())
        elif stage == "test":
            self.test_preds.append(pspi_pred_denorm.detach().cpu())
            self.test_targets.append(pspi_target_denorm.detach().cpu())
        elif stage == "last_epoch_test":
            self.last_epoch_test_preds.append(pspi_pred_denorm.detach().cpu())
            self.last_epoch_test_targets.append(pspi_target_denorm.detach().cpu())
        elif stage == "test_unbc":
            self.test_unbc_preds.append(pspi_pred_denorm.detach().cpu())
            self.test_unbc_targets.append(pspi_target_denorm.detach().cpu())
        
        # Update regression metrics (MAE, Correlation)
        getattr(self, f"{stage}_mae")(pspi_pred_denorm, pspi_target_denorm)
        getattr(self, f"{stage}_corr")(pspi_pred_denorm, pspi_target_denorm)
        
        # Update binary classification metrics (pain vs. no-pain, threshold=2)
        pred_binary = self._pspi_to_binary(pspi_pred_denorm, threshold=2.0)
        target_binary = self._pspi_to_binary(pspi_target_denorm, threshold=2.0)
        
        getattr(self, f"{stage}_binary_precision")(pred_binary, target_binary)
        getattr(self, f"{stage}_binary_recall")(pred_binary, target_binary)
        getattr(self, f"{stage}_binary_f1")(pred_binary, target_binary)
        # For AUROC/AUPR, use continuous predictions as scores
        getattr(self, f"{stage}_binary_auroc")(pspi_pred_denorm, target_binary)
        getattr(self, f"{stage}_binary_aupr")(pspi_pred_denorm, target_binary)
        
        # Update AU metrics
        getattr(self, f"{stage}_au_mae")(au_preds, au_target)
        getattr(self, f"{stage}_pspi_au_mae")(pspi_from_au, pspi_target_denorm)
        getattr(self, f"{stage}_pspi_au_corr")(pspi_from_au, pspi_target_denorm)
    
    def _log_pspi_metrics(self, stage, batch_size, total_loss, pspi_loss, au_loss, 
                         batch_idx=0, pspi_pred=None, pspi_target=None, additional_losses=None):
        """
        Log losses to PyTorch Lightning logger.
        
        Args:
            stage: Stage name ("train", "val", or "test")
            batch_size: Batch size for proper logging
            total_loss: Total loss value
            pspi_loss: PSPI regression loss value
            au_loss: AU regression loss value
            batch_idx: Batch index
            pspi_pred: PSPI predictions (optional)
            pspi_target: PSPI targets (optional)
            additional_losses: Dict of additional losses to log
        """
        # Log main losses
        self.log(f"{stage}/loss/total", total_loss, prog_bar=True, on_step=(stage=="train"), 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/loss/pspi", pspi_loss, on_step=(stage=="train"), 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/loss/au", au_loss, on_step=(stage=="train"), 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Log additional losses if provided
        if additional_losses is not None:
            for loss_name, loss_value in additional_losses.items():
                if isinstance(loss_value, dict):
                    loss_tensor = loss_value.get('value', loss_value)
                    weight = loss_value.get('weight', 1.0)
                    self.log(f"{stage}/loss/{loss_name}", loss_tensor, 
                            on_step=(stage=="train"), on_epoch=True, 
                            sync_dist=True, batch_size=batch_size)
                else:
                    self.log(f"{stage}/loss/{loss_name}", loss_value, 
                            on_step=(stage=="train"), on_epoch=True, 
                            sync_dist=True, batch_size=batch_size)
    
    def _compute_epoch_end_metrics(self, stage, batch_size, pspi_max=16.0):
        """
        Compute and log epoch-end metrics.
        
        Args:
            stage: Stage name
            batch_size: Batch size
            pspi_max: Maximum PSPI value
        """
        # Get predictions from this rank
        if stage == "train":
            if len(self.train_preds) == 0: return
            preds = torch.cat(self.train_preds, dim=0)
            targets = torch.cat(self.train_targets, dim=0)
        elif stage == "val":
            if len(self.val_preds) == 0: return
            preds = torch.cat(self.val_preds, dim=0)
            targets = torch.cat(self.val_targets, dim=0)
        elif stage == "test":
            if len(self.test_preds) == 0: return
            preds = torch.cat(self.test_preds, dim=0)
            targets = torch.cat(self.test_targets, dim=0)
        elif stage == "last_epoch_test":
            if len(self.last_epoch_test_preds) == 0: return
            preds = torch.cat(self.last_epoch_test_preds, dim=0)
            targets = torch.cat(self.last_epoch_test_targets, dim=0)
        elif stage == "test_unbc":
            if len(self.test_unbc_preds) == 0: return
            preds = torch.cat(self.test_unbc_preds, dim=0)
            targets = torch.cat(self.test_unbc_targets, dim=0)
        else:
            return
        
        device = self.device
        
        # Gather predictions from all ranks if using DDP
        if self.trainer.world_size > 1:
            preds_device = preds.to(device)
            targets_device = targets.to(device)
            preds_gathered = self.all_gather(preds_device)
            targets_gathered = self.all_gather(targets_device)
            preds = preds_gathered.view(-1).cpu()
            targets = targets_gathered.view(-1).cpu()
        
        # Compute and log metrics
        self._compute_and_log_torchmetrics(stage, device, batch_size)
        
        # Clear storage
        if stage == "train":
            self.train_preds.clear()
            self.train_targets.clear()
        elif stage == "val":
            self.val_preds.clear()
            self.val_targets.clear()
        elif stage == "test":
            self.test_preds.clear()
            self.test_targets.clear()
        elif stage == "last_epoch_test":
            self.last_epoch_test_preds.clear()
            self.last_epoch_test_targets.clear()
        elif stage == "test_unbc":
            self.test_unbc_preds.clear()
            self.test_unbc_targets.clear()
    
    def _compute_and_log_torchmetrics(self, stage, device, batch_size):
        """Compute all torchmetrics and log them."""
        
        def _compute_metric_safe(metric_obj):
            try:
                value = metric_obj.compute()
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32, device=device)
                else:
                    value = value.to(device)
                metric_obj.reset()
                return value
            except Exception:
                return torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # Regression metrics (MAE, Correlation)
        mae_value = _compute_metric_safe(getattr(self, f"{stage}_mae"))
        corr_value = _compute_metric_safe(getattr(self, f"{stage}_corr"))
        
        # Store values for validation (used by checkpointing)
        if stage == "val":
            self._last_val_mae = mae_value.item() if isinstance(mae_value, torch.Tensor) else float(mae_value)
            self._last_val_corr = corr_value.item() if isinstance(corr_value, torch.Tensor) else float(corr_value)
        
        # Log regression metrics
        self.log(f"{stage}/regression/mae", mae_value, prog_bar=True, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/regression/corr", corr_value, prog_bar=True, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Log aliases for checkpoint filenames (underscore format)
        self.log(f"{stage}_regression_mae", mae_value, logger=False, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}_regression_corr", corr_value, logger=False, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Binary classification metrics
        binary_precision = _compute_metric_safe(getattr(self, f"{stage}_binary_precision"))
        binary_recall = _compute_metric_safe(getattr(self, f"{stage}_binary_recall"))
        binary_f1 = _compute_metric_safe(getattr(self, f"{stage}_binary_f1"))
        binary_auroc = _compute_metric_safe(getattr(self, f"{stage}_binary_auroc"))
        binary_aupr = _compute_metric_safe(getattr(self, f"{stage}_binary_aupr"))
        
        self.log(f"{stage}/binary/precision", binary_precision, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/binary/recall", binary_recall, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/binary/f1", binary_f1, prog_bar=(stage=="val"), on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/binary/auroc", binary_auroc, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/binary/aupr", binary_aupr, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Log alias for checkpoint filename
        self.log(f"{stage}_regression_binary_f1", binary_f1, logger=False, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # AU metrics
        au_mae = _compute_metric_safe(getattr(self, f"{stage}_au_mae"))
        pspi_au_mae = _compute_metric_safe(getattr(self, f"{stage}_pspi_au_mae"))
        pspi_au_corr = _compute_metric_safe(getattr(self, f"{stage}_pspi_au_corr"))
        
        self.log(f"{stage}/au/mae", au_mae, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/au/pspi_mae", pspi_au_mae, on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/au/pspi_corr", pspi_au_corr, prog_bar=(stage=="val"), on_step=False, 
                 on_epoch=True, sync_dist=True, batch_size=batch_size)
