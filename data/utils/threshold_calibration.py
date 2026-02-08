"""Threshold calibration utilities for pain detection."""

import numpy as np


def find_optimal_threshold(y_true, y_pred_scores, min_threshold=0.0, max_threshold=6.0, num_steps=200):
    """
    Find optimal threshold for binary classification by grid search on F1.

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_pred_scores: Continuous prediction scores
        min_threshold: Minimum threshold to search
        max_threshold: Maximum threshold to search
        num_steps: Number of thresholds to try

    Returns:
        Dictionary with best_threshold, best_f1, best_precision, best_recall
    """
    thresholds = np.linspace(min_threshold, max_threshold, num_steps)

    best_f1 = -1
    best_threshold = min_threshold
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        y_pred_binary = (y_pred_scores >= threshold).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall,
    }
