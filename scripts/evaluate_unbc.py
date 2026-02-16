#!/usr/bin/env python3
"""
Combined 5-fold UNBC evaluation script for PainGeneration_clean.

Evaluates cross-validated performance on UNBC-McMaster dataset:
1. Loads the best checkpoint from each of the 5 folds
2. Evaluates each checkpoint on its respective held-out fold
3. Computes and reports key cross-validated metrics:
   - Pearson correlation (mean across folds with 95% CI)
   - AUROC at thresholds 1, 2, and 3 (mean across folds with 95% CI)
   - Train-calibrated F1 scores (threshold tuned on train, applied to test)
     * Most realistic evaluation: simulates production usage
     * Reported for PSPI thresholds 1, 2, 3, and macro average

Note: A verbose version with all metrics is saved as evaluate_unbc_verbose.py

Usage:
  # Evaluate a completed 5-fold experiment (auto-finds best checkpoints):
  python scripts/evaluate_unbc.py experiment/unbc_5fold_cv_neutralref_v3

  # Evaluate with a specific checkpoint metric (corr or f1):
  python scripts/evaluate_unbc.py experiment/unbc_5fold_cv_neutralref_v3 --checkpoint_metric corr

  # Evaluate with multi-shot inference:
  python scripts/evaluate_unbc.py experiment/unbc_5fold_cv_neutralref_v3 --use_neutral_reference --multi_shot_inference 3

  # Evaluate a single pretraining checkpoint across all folds (zero-shot):
  python scripts/evaluate_unbc.py --single_checkpoint path/to/checkpoint.ckpt --use_neutral_reference
"""

import os
import sys
import re
import argparse
import numpy as np
import torch
import torchmetrics
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, precision_score, recall_score

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lib.models.vitpain import ViTPain, create_vitpain_model
from data.unbc_loader import UNBCDataModule


# ---------------------------------------------------------------------------
#  Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_checkpoint(fold_dir: str, metric: str = "corr") -> str:
    """
    Find the best checkpoint in a fold directory.

    The clean codebase saves checkpoints with names like:
      pspi-vit-fold{fold}-epoch=XX-val_regression_corr=0.XXX.ckpt
    """
    checkpoint_dir = os.path.join(fold_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    fold_id = None
    m = re.search(r"fold_(\d+)", os.path.basename(os.path.normpath(fold_dir)))
    if m:
        fold_id = int(m.group(1))

    checkpoint_files: List[str] = []
    for filename in os.listdir(checkpoint_dir):
        if not filename.endswith(".ckpt"):
            continue
        if "last" in filename:
            continue
        # Match clean naming: pspi-vit-fold{X}-epoch=...
        if fold_id is not None and f"fold{fold_id}" not in filename:
            continue
        if metric == "corr" and "val_regression_corr" in filename:
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))
        elif metric == "f1" and "f1" in filename:
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))
        elif metric == "corr" and "corr" in filename:
            # Also match original naming: ...-corrval_regression_corr=...
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))

    if not checkpoint_files:
        # Fall back: any non-last checkpoint
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith(".ckpt") and "last" not in filename:
                checkpoint_files.append(os.path.join(checkpoint_dir, filename))

    if not checkpoint_files:
        # Last resort: last.ckpt
        last = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last):
            return last
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    # Pick the one with the highest metric value in filename
    score_re = re.compile(r"(?:corr|f1)[^=]*=(-?\d+(?:\.\d+)?)")

    def _score(path: str) -> float:
        m2 = score_re.search(os.path.basename(path))
        if not m2:
            return float("-inf")
        try:
            return float(m2.group(1))
        except Exception:
            return float("-inf")

    scored = [(p, _score(p)) for p in checkpoint_files]
    best_path, best_score = max(scored, key=lambda t: (t[1], os.path.getmtime(t[0])))
    if best_score == float("-inf"):
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        return checkpoint_files[0]
    return best_path


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, use_neutral_reference: bool = False) -> ViTPain:
    """Load a ViTPain model from checkpoint."""
    # Try Lightning load_from_checkpoint first
    try:
        model = ViTPain.load_from_checkpoint(checkpoint_path, map_location="cpu")
        print(f"  Loaded model via load_from_checkpoint")
        return model
    except Exception as e:
        print(f"  load_from_checkpoint failed ({e}), trying manual load...")

    # Fallback: create model and load state dict
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model = create_vitpain_model(
        model_size="large_dinov3",
        use_neutral_reference=use_neutral_reference,
        learning_rate=1e-4,
        weight_decay=1e-1,
        max_epochs=50,
        au_loss_weight=0.1,
        pspi_loss_weight=1.0,
    )
    model_sd = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    missing = set(model_sd.keys()) - set(compatible.keys())
    if missing:
        print(f"  Warning: {len(missing)} keys not loaded from checkpoint")
    model.load_state_dict(compatible, strict=False)
    return model


# ---------------------------------------------------------------------------
#  Per-fold evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint_on_fold(
    checkpoint_path: str,
    data_dir: str,
    fold: int,
    batch_size: int = 100,
    num_workers: int = 4,
    use_neutral_reference: bool = True,
    multi_shot_inference: int = 1,
    return_train_preds: bool = False,
) -> Tuple[np.ndarray, np.ndarray, ...]:
    """
    Evaluate a single checkpoint on its held-out fold.
    Returns (predictions, targets) as numpy arrays in [0, 16] scale.
    If return_train_preds=True, also returns (train_predictions, train_targets).
    """
    print(f"\n{'='*80}")
    print(f"  Evaluating Fold {fold}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*80}")

    model = load_model(checkpoint_path, use_neutral_reference=use_neutral_reference)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Infer neutral reference from checkpoint hparams
    ckpt_use_neutral = bool(getattr(model.hparams, "use_neutral_reference", False))
    effective_use_neutral = ckpt_use_neutral or use_neutral_reference
    effective_multi_shot = multi_shot_inference if effective_use_neutral else 1
    print(f"  Neutral ref (ckpt): {ckpt_use_neutral} | eval: {effective_use_neutral} | shots: {effective_multi_shot}")

    dm = UNBCDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=224,
        fold=fold,
        cv_protocol="5fold",
        return_aus=True,
        pin_memory=(device.type == "cuda"),
        use_neutral_reference=effective_use_neutral,
        multi_shot_inference=effective_multi_shot,
    )
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # Helper function to evaluate a dataloader
    def _evaluate_loader(loader, split_name="test"):
        preds_list = []
        targets_list = []
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            pspi_targets = batch["pspi_score"].float()
            neutral_images = batch.get("neutral_image", None)

            if neutral_images is not None and neutral_images.dim() == 5:
                num_shots = neutral_images.shape[1]
                pspi_sum = None
                for s in range(num_shots):
                    out = model(images, neutral_pixel_values=neutral_images[:, s].to(device))
                    pspi_s = out["pspi_pred"]
                    pspi_sum = pspi_s if pspi_sum is None else pspi_sum + pspi_s
                pspi_pred = pspi_sum / num_shots
            elif neutral_images is not None:
                out = model(images, neutral_pixel_values=neutral_images.to(device))
                pspi_pred = out["pspi_pred"]
            else:
                out = model(images)
                pspi_pred = out["pspi_pred"]

            # Denormalize: model outputs [0, 1] -> [0, 16]
            pspi_pred_denorm = pspi_pred.cpu() * 16.0
            preds_list.append(pspi_pred_denorm.numpy())
            targets_list.append(pspi_targets.numpy())  # Already [0, 16]

            if (batch_idx + 1) % 20 == 0:
                print(f"    Processed {batch_idx + 1}/{len(loader)} batches ({split_name})")
        
        return np.concatenate(preds_list).flatten(), np.concatenate(targets_list).flatten()

    # Evaluate test set
    predictions, targets = _evaluate_loader(test_loader, "test")
    print(f"  Fold {fold} test: {len(predictions)} samples, "
          f"pred range [{predictions.min():.2f}, {predictions.max():.2f}], "
          f"target range [{targets.min():.2f}, {targets.max():.2f}]")

    # Optionally evaluate train set
    if return_train_preds:
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        train_predictions, train_targets = _evaluate_loader(train_loader, "train")
        print(f"  Fold {fold} train: {len(train_predictions)} samples, "
              f"pred range [{train_predictions.min():.2f}, {train_predictions.max():.2f}], "
              f"target range [{train_targets.min():.2f}, {train_targets.max():.2f}]")
        return predictions, targets, train_predictions, train_targets

    return predictions, targets


# ---------------------------------------------------------------------------
#  Metric helpers (matching original PainGeneration methodology)
# ---------------------------------------------------------------------------

def compute_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Pearson correlation with NaN/edge-case handling."""
    valid = ~(np.isnan(predictions) | np.isnan(targets))
    p, t = predictions[valid], targets[valid]
    if len(p) < 2 or np.std(p) == 0 or np.std(t) == 0:
        return 0.0
    try:
        corr, _ = pearsonr(p, t)
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def compute_auroc_at_threshold(predictions: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    """AUROC for binary pain classification at a PSPI threshold."""
    binary_targets = (targets >= threshold).astype(np.int32)
    if len(np.unique(binary_targets)) < 2:
        return 0.0
    try:
        metric = torchmetrics.AUROC(task="binary")
        metric.update(torch.from_numpy(predictions).float(), torch.from_numpy(binary_targets).long())
        val = metric.compute()
        return float(val.item()) if torch.isfinite(val) else 0.0
    except Exception:
        return 0.0


def compute_macro_f1_with_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    thresholds: List[float] = [1.0, 2.0, 3.0],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Find the best prediction threshold for each GT threshold (1, 2, 3)
    and compute calibrated F1. Returns per-threshold and macro F1.
    """
    results = {}
    f1_scores = []

    for gt_thr in thresholds:
        binary_targets = (targets >= gt_thr).astype(np.int32)
        if len(np.unique(binary_targets)) < 2:
            results[f"f1_threshold_{int(gt_thr)}"] = 0.0
            results[f"best_pred_threshold_{int(gt_thr)}"] = 0.0
            results[f"precision_threshold_{int(gt_thr)}"] = 0.0
            results[f"recall_threshold_{int(gt_thr)}"] = 0.0
            continue

        best_f1, best_thr, best_prec, best_rec = -1.0, 0.0, 0.0, 0.0
        for pred_thr in np.linspace(0, 6, 200):
            binary_preds = (predictions >= pred_thr).astype(np.int32)
            try:
                f1 = f1_score(binary_targets, binary_preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = pred_thr
                    best_prec = precision_score(binary_targets, binary_preds, zero_division=0)
                    best_rec = recall_score(binary_targets, binary_preds, zero_division=0)
            except Exception:
                continue

        results[f"f1_threshold_{int(gt_thr)}"] = best_f1
        results[f"best_pred_threshold_{int(gt_thr)}"] = best_thr
        results[f"precision_threshold_{int(gt_thr)}"] = best_prec
        results[f"recall_threshold_{int(gt_thr)}"] = best_rec
        f1_scores.append(best_f1)

        if verbose:
            print(f"    GT>={gt_thr}: F1={best_f1:.4f}, pred_thr={best_thr:.4f}, "
                  f"prec={best_prec:.4f}, recall={best_rec:.4f}")

    results["macro_f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
    return results


def compute_f1_with_train_calibrated_thresholds(
    train_predictions: np.ndarray,
    train_targets: np.ndarray,
    test_predictions: np.ndarray,
    test_targets: np.ndarray,
    thresholds: List[float] = [1.0, 2.0, 3.0],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Calibrate prediction thresholds on TRAIN set, then apply to TEST set.
    This simulates realistic evaluation where thresholds are tuned on training data.
    
    Returns per-threshold F1 scores and macro F1 on the test set.
    """
    results = {}
    f1_scores = []

    for gt_thr in thresholds:
        # Find best threshold on train set
        train_binary_targets = (train_targets >= gt_thr).astype(np.int32)
        if len(np.unique(train_binary_targets)) < 2:
            results[f"f1_threshold_{int(gt_thr)}"] = 0.0
            results[f"train_calibrated_threshold_{int(gt_thr)}"] = 0.0
            results[f"precision_threshold_{int(gt_thr)}"] = 0.0
            results[f"recall_threshold_{int(gt_thr)}"] = 0.0
            continue

        best_f1_train, best_thr = -1.0, 0.0
        for pred_thr in np.linspace(0, 6, 200):
            train_binary_preds = (train_predictions >= pred_thr).astype(np.int32)
            try:
                f1 = f1_score(train_binary_targets, train_binary_preds, zero_division=0)
                if f1 > best_f1_train:
                    best_f1_train = f1
                    best_thr = pred_thr
            except Exception:
                continue

        # Apply calibrated threshold to test set
        test_binary_targets = (test_targets >= gt_thr).astype(np.int32)
        test_binary_preds = (test_predictions >= best_thr).astype(np.int32)
        
        try:
            test_f1 = f1_score(test_binary_targets, test_binary_preds, zero_division=0)
            test_prec = precision_score(test_binary_targets, test_binary_preds, zero_division=0)
            test_rec = recall_score(test_binary_targets, test_binary_preds, zero_division=0)
        except Exception:
            test_f1 = 0.0
            test_prec = 0.0
            test_rec = 0.0

        results[f"f1_threshold_{int(gt_thr)}"] = test_f1
        results[f"train_calibrated_threshold_{int(gt_thr)}"] = best_thr
        results[f"precision_threshold_{int(gt_thr)}"] = test_prec
        results[f"recall_threshold_{int(gt_thr)}"] = test_rec
        f1_scores.append(test_f1)

        if verbose:
            print(f"    GT>={gt_thr}: Test F1={test_f1:.4f} (train-calibrated pred_thr={best_thr:.4f}), "
                  f"prec={test_prec:.4f}, recall={test_rec:.4f}")

    results["macro_f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
    return results


def _mean_and_95ci(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) with t-interval for n=5."""
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    # t_{0.975, df=4} = 2.776 for n=5
    t_crit = 2.776 if arr.size == 5 else 1.96
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    half = t_crit * se
    return mean, mean - half, mean + half


def _binary_f1_at_threshold(predictions: np.ndarray, targets: np.ndarray, threshold: float = 2.0) -> float:
    """Simple binary F1 using same threshold for predictions and targets."""
    pred_bin = (predictions >= threshold).astype(np.int32)
    tgt_bin = (targets >= threshold).astype(np.int32)
    tp = int(np.sum((pred_bin == 1) & (tgt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (tgt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (tgt_bin == 1)))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------

def run_combined_evaluation(
    experiment_dir: str,
    data_dir: str = "datasets/UNBC-McMaster",
    batch_size: int = 100,
    checkpoint_metric: str = "corr",
    use_neutral_reference: bool = True,
    multi_shot_inference: int = 1,
    single_checkpoint: str = None,
    folds: List[int] = None,
):
    """Run combined 5-fold evaluation matching original PainGeneration methodology."""
    if folds is None:
        folds = [0, 1, 2, 3, 4]

    print("=" * 80)
    print("  UNBC 5-Fold Combined Evaluation (PainGeneration_clean)")
    print("=" * 80)
    if single_checkpoint:
        print(f"  Single checkpoint: {single_checkpoint}")
    else:
        print(f"  Experiment dir: {experiment_dir}")
    print(f"  Checkpoint metric: {checkpoint_metric}")
    print(f"  Neutral reference: {use_neutral_reference}")
    print(f"  Multi-shot: {multi_shot_inference}")
    print("=" * 80)

    all_predictions: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_train_predictions: List[np.ndarray] = []
    all_train_targets: List[np.ndarray] = []
    per_fold_corr: List[Tuple[int, float]] = []
    per_fold_auroc: Dict[float, List[Tuple[int, float]]] = {1.0: [], 2.0: [], 3.0: []}
    per_fold_macro_f1: List[Tuple[int, float]] = []
    per_fold_calibrated_f1: Dict[float, List[Tuple[int, float]]] = {1.0: [], 2.0: [], 3.0: []}
    per_fold_default_f1: Dict[float, List[Tuple[int, float]]] = {1.0: [], 2.0: [], 3.0: []}
    per_fold_train_calibrated_macro_f1: List[Tuple[int, float]] = []
    per_fold_train_calibrated_f1: Dict[float, List[Tuple[int, float]]] = {1.0: [], 2.0: [], 3.0: []}

    for fold in folds:
        if single_checkpoint:
            ckpt_path = single_checkpoint
        else:
            fold_dir = os.path.join(experiment_dir, f"fold_{fold}")
            if not os.path.isdir(fold_dir):
                print(f"  Fold directory not found: {fold_dir}, skipping...")
                continue
            try:
                ckpt_path = find_best_checkpoint(fold_dir, metric=checkpoint_metric)
            except FileNotFoundError as e:
                print(f"  {e}, skipping fold {fold}...")
                continue

        try:
            result = evaluate_checkpoint_on_fold(
                checkpoint_path=ckpt_path,
                data_dir=data_dir,
                fold=fold,
                batch_size=batch_size,
                use_neutral_reference=use_neutral_reference,
                multi_shot_inference=multi_shot_inference,
                return_train_preds=True,
            )
            predictions, targets, train_predictions, train_targets = result
        except Exception as e:
            print(f"  Error evaluating fold {fold}: {e}")
            import traceback; traceback.print_exc()
            continue

        all_predictions.append(predictions)
        all_targets.append(targets)
        all_train_predictions.append(train_predictions)
        all_train_targets.append(train_targets)

        # Per-fold metrics
        fold_corr = compute_correlation(predictions, targets)
        per_fold_corr.append((fold, fold_corr))

        for thr in (1.0, 2.0, 3.0):
            auroc = compute_auroc_at_threshold(predictions, targets, thr)
            per_fold_auroc[thr].append((fold, auroc))

            default_f1 = _binary_f1_at_threshold(predictions, targets, thr)
            per_fold_default_f1[thr].append((fold, default_f1))

        fold_f1_results = compute_macro_f1_with_calibration(predictions, targets, verbose=False)
        per_fold_macro_f1.append((fold, fold_f1_results.get("macro_f1", 0.0)))
        for thr in (1.0, 2.0, 3.0):
            per_fold_calibrated_f1[thr].append(
                (fold, fold_f1_results.get(f"f1_threshold_{int(thr)}", 0.0))
            )

        # Train-calibrated F1 (train thresholds -> test predictions)
        fold_train_cal_f1_results = compute_f1_with_train_calibrated_thresholds(
            train_predictions, train_targets, predictions, targets, verbose=False
        )
        per_fold_train_calibrated_macro_f1.append((fold, fold_train_cal_f1_results.get("macro_f1", 0.0)))
        for thr in (1.0, 2.0, 3.0):
            per_fold_train_calibrated_f1[thr].append(
                (fold, fold_train_cal_f1_results.get(f"f1_threshold_{int(thr)}", 0.0))
            )

        print(f"  -> Fold {fold}: corr={fold_corr:.4f}")

    if not all_predictions:
        print("ERROR: No predictions collected!")
        return

    # Combined predictions
    combined_preds = np.concatenate(all_predictions)
    combined_targets = np.concatenate(all_targets)
    combined_train_preds = np.concatenate(all_train_predictions)
    combined_train_targets = np.concatenate(all_train_targets)

    # ---- Save scatter plot ----
    output_dir = experiment_dir or os.path.dirname(single_checkpoint)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_path = os.path.join(output_dir, f"combined_pred_vs_gt_{checkpoint_metric}.png")
        x, y = combined_targets, combined_preds
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        lo = float(min(x.min(), y.min())) - 0.5
        hi = float(max(x.max(), y.max())) + 0.5

        plt.figure(figsize=(7, 7))
        plt.scatter(x, y, s=6, alpha=0.25)
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        plt.xlim(lo, hi); plt.ylim(lo, hi)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Ground truth PSPI")
        plt.ylabel("Predicted PSPI")
        plt.title(f"UNBC combined pred vs GT ({checkpoint_metric} ckpt)")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"\nSaved scatter plot: {fig_path}")
    except Exception as e:
        print(f"Warning: could not save scatter plot: {e}")

    # ---- Print results ----
    print("\n" + "=" * 80)
    print("  COMBINED RESULTS (All Folds)")
    print("=" * 80)
    print(f"  Total samples: {len(combined_preds)}")

    # ========================================================================
    # KEY METRICS (Cross-Validated)
    # ========================================================================
    
    # 1. Correlation
    corr_values = [v for _, v in per_fold_corr]
    mean_corr, ci_lo, ci_hi = _mean_and_95ci(corr_values)
    pm = 0.5 * (ci_hi - ci_lo)
    
    # 2. AUROC
    auroc_means = {}
    auroc_cis = {}
    for thr in (1.0, 2.0, 3.0):
        vals = [v for _, v in per_fold_auroc[thr]]
        mean_a, lo_a, hi_a = _mean_and_95ci(vals)
        pm_a = 0.5 * (hi_a - lo_a)
        auroc_means[thr] = mean_a
        auroc_cis[thr] = pm_a

    # 3. Train-Calibrated F1 (threshold tuned on TRAIN, applied to TEST)
    fold_train_cal_f1_vals = [v for _, v in per_fold_train_calibrated_macro_f1]
    mean_train_f1, lo_train_f1, hi_train_f1 = _mean_and_95ci(fold_train_cal_f1_vals)
    pm_train_f1 = 0.5 * (hi_train_f1 - lo_train_f1)
    
    train_cal_f1_means = {}
    train_cal_f1_cis = {}
    for thr in (1.0, 2.0, 3.0):
        vals = [v for _, v in per_fold_train_calibrated_f1[thr]]
        m_t, lo_t, hi_t = _mean_and_95ci(vals)
        pm_t = 0.5 * (hi_t - lo_t)
        train_cal_f1_means[thr] = m_t
        train_cal_f1_cis[thr] = pm_t

    # Print key metrics
    print(f"\n{'='*80}")
    print(f"  KEY METRICS (Mean across folds ± 95% CI)")
    print(f"{'='*80}")
    print(f"\n  Pearson Correlation:  {mean_corr:.4f} ± {pm:.4f}")
    print(f"\n  AUROC:")
    print(f"    PSPI >= 1:          {auroc_means[1.0]:.4f} ± {auroc_cis[1.0]:.4f}")
    print(f"    PSPI >= 2:          {auroc_means[2.0]:.4f} ± {auroc_cis[2.0]:.4f}")
    print(f"    PSPI >= 3:          {auroc_means[3.0]:.4f} ± {auroc_cis[3.0]:.4f}")
    print(f"\n  Train-Calibrated F1 (threshold tuned on train):")
    print(f"    PSPI >= 1:          {train_cal_f1_means[1.0]:.4f} ± {train_cal_f1_cis[1.0]:.4f}")
    print(f"    PSPI >= 2:          {train_cal_f1_means[2.0]:.4f} ± {train_cal_f1_cis[2.0]:.4f}")
    print(f"    PSPI >= 3:          {train_cal_f1_means[3.0]:.4f} ± {train_cal_f1_cis[3.0]:.4f}")
    print(f"    Macro F1:           {mean_train_f1:.4f} ± {pm_train_f1:.4f}")
    
    # Per-fold details
    print(f"\n{'='*80}")
    print(f"  PER-FOLD DETAILS")
    print(f"{'='*80}")
    print(f"\n  Correlation (per fold):")
    for fid, cv in per_fold_corr:
        print(f"    Fold {fid}: {cv:.4f}")
    
    print(f"\n  AUROC @ PSPI>=2 (per fold):")
    for fid, av in per_fold_auroc[2.0]:
        print(f"    Fold {fid}: {av:.4f}")
    
    print(f"\n  Train-Calibrated Macro F1 (per fold):")
    for fid, f1v in per_fold_train_calibrated_macro_f1:
        print(f"    Fold {fid}: {f1v:.4f}")

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    header = f"  {'Fold':<8} {'Corr':>8} {'AUROC@2':>10} {'F1_macro':>10}"
    print(header)
    print(f"  {'-'*40}")

    # Build lookup dicts for easy access
    fold_corr_dict = {fid: cv for fid, cv in per_fold_corr}
    fold_auroc_dict = {thr: {fid: av for fid, av in pairs} for thr, pairs in per_fold_auroc.items()}
    fold_train_f1_dict = {fid: fv for fid, fv in per_fold_train_calibrated_macro_f1}

    evaluated_folds = [fid for fid, _ in per_fold_corr]
    for fid in evaluated_folds:
        fc = fold_corr_dict.get(fid, 0)
        fa2 = fold_auroc_dict.get(2.0, {}).get(fid, 0)
        ftf1 = fold_train_f1_dict.get(fid, 0)
        print(f"  Fold {fid:<3} {fc:>8.4f} {fa2:>10.4f} {ftf1:>10.4f}")

    print(f"  {'-'*40}")
    print(f"  {'Mean':<8} {mean_corr:>8.4f} {auroc_means[2.0]:>10.4f} {mean_train_f1:>10.4f}")
    print(f"  {'±95%CI':<8} {pm:>8.4f} {auroc_cis[2.0]:>10.4f} {pm_train_f1:>10.4f}")
    print("=" * 80)

    # ---- Save to file ----
    results_file = os.path.join(output_dir, f"combined_evaluation_results_{checkpoint_metric}.txt")
    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("UNBC 5-Fold Cross-Validation Results (PainGeneration_clean)\n")
        f.write("=" * 80 + "\n")
        if single_checkpoint:
            f.write(f"Single checkpoint: {single_checkpoint}\n")
        else:
            f.write(f"Experiment directory: {experiment_dir}\n")
        f.write(f"Total samples: {len(combined_preds)}\n")
        f.write(f"Checkpoint metric: {checkpoint_metric}\n\n")

        f.write("KEY METRICS (Mean ± 95% CI):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Pearson Correlation:     {mean_corr:.4f} ± {pm:.4f}\n")
        f.write(f"AUROC @ PSPI>=1:         {auroc_means[1.0]:.4f} ± {auroc_cis[1.0]:.4f}\n")
        f.write(f"AUROC @ PSPI>=2:         {auroc_means[2.0]:.4f} ± {auroc_cis[2.0]:.4f}\n")
        f.write(f"AUROC @ PSPI>=3:         {auroc_means[3.0]:.4f} ± {auroc_cis[3.0]:.4f}\n")
        f.write(f"Train-cal F1 @ PSPI>=1:  {train_cal_f1_means[1.0]:.4f} ± {train_cal_f1_cis[1.0]:.4f}\n")
        f.write(f"Train-cal F1 @ PSPI>=2:  {train_cal_f1_means[2.0]:.4f} ± {train_cal_f1_cis[2.0]:.4f}\n")
        f.write(f"Train-cal F1 @ PSPI>=3:  {train_cal_f1_means[3.0]:.4f} ± {train_cal_f1_cis[3.0]:.4f}\n")
        f.write(f"Train-cal Macro F1:      {mean_train_f1:.4f} ± {pm_train_f1:.4f}\n")
        f.write("\n")

        # Per-fold results
        f.write("PER-FOLD RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Fold':<8} {'Corr':>8} {'AUROC@2':>10} {'F1_macro':>10}\n")
        f.write("-" * 40 + "\n")
        for fid in evaluated_folds:
            fc = fold_corr_dict.get(fid, 0)
            fa2 = fold_auroc_dict.get(2.0, {}).get(fid, 0)
            ftf1 = fold_train_f1_dict.get(fid, 0)
            f.write(f"Fold {fid:<3} {fc:>8.4f} {fa2:>10.4f} {ftf1:>10.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Mean':<8} {mean_corr:>8.4f} {auroc_means[2.0]:>10.4f} {mean_train_f1:>10.4f}\n")
        f.write(f"{'±95%CI':<8} {pm:>8.4f} {auroc_cis[2.0]:>10.4f} {pm_train_f1:>10.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="UNBC 5-fold combined evaluation (matching original PainGeneration methodology)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "experiment_dir", type=str, nargs="?", default=None,
        help="Path to experiment dir with fold_0/, fold_1/, ... subdirectories",
    )
    parser.add_argument(
        "--single_checkpoint", type=str, default=None,
        help="Evaluate a single checkpoint across all folds (e.g., pretraining checkpoint)",
    )
    parser.add_argument("--data_dir", type=str, default="datasets/UNBC-McMaster")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--checkpoint_metric", type=str, default="corr", choices=["corr", "f1"],
        help="Which checkpoint metric to select (default: corr)",
    )
    parser.add_argument("--use_neutral_reference", action="store_true")
    parser.add_argument("--multi_shot_inference", type=int, default=1)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    args = parser.parse_args()

    if args.experiment_dir is None and args.single_checkpoint is None:
        parser.error("Either experiment_dir or --single_checkpoint is required")

    run_combined_evaluation(
        experiment_dir=args.experiment_dir or os.path.dirname(args.single_checkpoint),
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        checkpoint_metric=args.checkpoint_metric,
        use_neutral_reference=args.use_neutral_reference,
        multi_shot_inference=args.multi_shot_inference,
        single_checkpoint=args.single_checkpoint,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
