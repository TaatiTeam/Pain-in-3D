"""
PSPI Evaluator Mixin for shared evaluation and logging logic.

Metrics:
- Regression: MAE, Pearson Correlation
- Classification: F1, Accuracy, AUROC, Macro AUROC (binary pain vs. no-pain)
"""

import torch
import torchmetrics


class PSPIEvaluatorMixin:
    """
    Mixin providing evaluation and logging for PSPI regression models.

    Metrics tracked per stage (train / val / test / test_unbc / last_epoch_test):
      Regression  -- MAE, Pearson Correlation
      Binary cls  -- F1, Accuracy, AUROC, Macro AUROC  (pain vs. no-pain, threshold=2)
    """

    # ------------------------------------------------------------------ #
    #  Utility helpers
    # ------------------------------------------------------------------ #

    def _denormalize_pspi(self, normalized_values, pspi_max=16.0):
        return normalized_values * pspi_max

    def _normalize_pspi(self, pspi_values, pspi_max=16.0):
        return pspi_values / pspi_max

    def _ensure_denormalized(self, pspi_values, pspi_max=16.0):
        if torch.all(pspi_values <= 1.1):
            return self._denormalize_pspi(pspi_values, pspi_max)
        return pspi_values

    def _pspi_to_binary(self, pspi_values, threshold=2.0):
        return (pspi_values >= threshold).long()

    def _calculate_pspi_from_au(self, au_preds):
        """PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43"""
        pspi_from_au = (
            au_preds[:, 0]
            + torch.max(au_preds[:, 1], au_preds[:, 2])
            + torch.max(au_preds[:, 3], au_preds[:, 4])
            + au_preds[:, 5]
        )
        return torch.where(torch.isnan(pspi_from_au), torch.zeros_like(pspi_from_au), pspi_from_au)

    # ------------------------------------------------------------------ #
    #  Metric initialisation
    # ------------------------------------------------------------------ #

    def _init_pspi_metrics(self):
        stages = ["train", "val", "test", "last_epoch_test", "test_unbc"]

        for stage in stages:
            setattr(self, f"{stage}_mae", torchmetrics.MeanAbsoluteError())
            setattr(self, f"{stage}_corr", torchmetrics.PearsonCorrCoef())

            # Binary classification (pain vs. no-pain, threshold=2)
            setattr(self, f"{stage}_binary_f1", torchmetrics.F1Score(task="binary"))
            setattr(self, f"{stage}_binary_acc", torchmetrics.Accuracy(task="binary"))
            setattr(self, f"{stage}_binary_auroc", torchmetrics.AUROC(task="binary"))
            setattr(self, f"{stage}_binary_macro_auroc", torchmetrics.AUROC(task="binary"))

            # AU-derived PSPI (for comparison with direct head)
            setattr(self, f"{stage}_pspi_au_mae", torchmetrics.MeanAbsoluteError())
            setattr(self, f"{stage}_pspi_au_corr", torchmetrics.PearsonCorrCoef())

        # Per-stage prediction/target storage for DDP gathering at epoch-end
        for stage in stages:
            setattr(self, f"{stage}_preds", [])
            setattr(self, f"{stage}_targets", [])

    # ------------------------------------------------------------------ #
    #  Per-batch metric update
    # ------------------------------------------------------------------ #

    def _update_pspi_metrics(
        self, pspi_pred, pspi_target, au_preds, au_target, pspi_from_au, stage,
        pspi_max=16.0, targets_already_denormalized=False,
    ):
        pspi_pred_denorm = self._denormalize_pspi(pspi_pred, pspi_max)

        if targets_already_denormalized:
            pspi_target_denorm = pspi_target
        else:
            pspi_target_denorm = self._ensure_denormalized(pspi_target, pspi_max)

        getattr(self, f"{stage}_preds").append(pspi_pred_denorm.detach().cpu())
        getattr(self, f"{stage}_targets").append(pspi_target_denorm.detach().cpu())

        getattr(self, f"{stage}_mae")(pspi_pred_denorm, pspi_target_denorm)
        getattr(self, f"{stage}_corr")(pspi_pred_denorm, pspi_target_denorm)

        pred_binary = self._pspi_to_binary(pspi_pred_denorm, threshold=2.0)
        target_binary = self._pspi_to_binary(pspi_target_denorm, threshold=2.0)

        getattr(self, f"{stage}_binary_f1")(pred_binary, target_binary)
        getattr(self, f"{stage}_binary_acc")(pred_binary, target_binary)
        # AUROC uses continuous scores rather than binary predictions
        getattr(self, f"{stage}_binary_auroc")(pspi_pred_denorm, target_binary)
        getattr(self, f"{stage}_binary_macro_auroc")(pspi_pred_denorm, target_binary)

        getattr(self, f"{stage}_pspi_au_mae")(pspi_from_au, pspi_target_denorm)
        getattr(self, f"{stage}_pspi_au_corr")(pspi_from_au, pspi_target_denorm)

    # ------------------------------------------------------------------ #
    #  Per-batch loss logging
    # ------------------------------------------------------------------ #

    def _log_pspi_metrics(
        self, stage, batch_size, total_loss, pspi_loss, au_loss,
        batch_idx=0, pspi_pred=None, pspi_target=None,
    ):
        self.log(f"{stage}/loss/total", total_loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/loss/pspi", pspi_loss,
                 on_step=(stage == "train"), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/loss/au", au_loss,
                 on_step=(stage == "train"), on_epoch=True, sync_dist=True, batch_size=batch_size)

    # ------------------------------------------------------------------ #
    #  Epoch-end metric computation + logging
    # ------------------------------------------------------------------ #

    def _compute_epoch_end_metrics(self, stage, batch_size, pspi_max=16.0):
        preds_list = getattr(self, f"{stage}_preds")
        targets_list = getattr(self, f"{stage}_targets")

        if len(preds_list) == 0:
            return

        preds = torch.cat(preds_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        if self.trainer.world_size > 1:
            device = self.device
            preds = self.all_gather(preds.to(device)).view(-1).cpu()
            targets = self.all_gather(targets.to(device)).view(-1).cpu()

        self._compute_and_log_torchmetrics(stage, self.device, batch_size)

        preds_list.clear()
        targets_list.clear()

    def _compute_and_log_torchmetrics(self, stage, device, batch_size):
        def _safe(metric_obj):
            try:
                v = metric_obj.compute()
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, dtype=torch.float32, device=device)
                else:
                    v = v.to(device)
                metric_obj.reset()
                return v
            except Exception:
                metric_obj.reset()
                return torch.tensor(0.0, dtype=torch.float32, device=device)

        mae_val = _safe(getattr(self, f"{stage}_mae"))
        corr_val = _safe(getattr(self, f"{stage}_corr"))

        if stage == "val":
            self._last_val_mae = mae_val.item()
            self._last_val_corr = corr_val.item()

        self.log(f"{stage}/regression/mae", mae_val, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/regression/corr", corr_val, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        # Underscore aliases used by ModelCheckpoint filename templates
        self.log(f"{stage}_regression_mae", mae_val, logger=False,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}_regression_corr", corr_val, logger=False,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        f1_val = _safe(getattr(self, f"{stage}_binary_f1"))
        acc_val = _safe(getattr(self, f"{stage}_binary_acc"))
        auroc_val = _safe(getattr(self, f"{stage}_binary_auroc"))
        macro_auroc_val = _safe(getattr(self, f"{stage}_binary_macro_auroc"))

        self.log(f"{stage}/binary/f1", f1_val, prog_bar=(stage == "val"),
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/binary/accuracy", acc_val,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/binary/auroc", auroc_val,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/binary/macro_auroc", macro_auroc_val,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        pspi_au_mae = _safe(getattr(self, f"{stage}_pspi_au_mae"))
        pspi_au_corr = _safe(getattr(self, f"{stage}_pspi_au_corr"))

        self.log(f"{stage}/au/pspi_mae", pspi_au_mae,
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/au/pspi_corr", pspi_au_corr, prog_bar=(stage == "val"),
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
