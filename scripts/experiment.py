# Built with AI assistance (Claude). Selective-prediction idea follows Geifman &
# El-Yaniv, "Selective Classification for Deep Neural Networks", NeurIPS 2017.
"""
Analysis utilities for the experiments and results section.

  * confidence_rejection  - selective-prediction accuracy/coverage trade-off
                            (the previous version compared y_true to itself and
                            always returned 1.0; this is the corrected version
                            that actually uses the model's predictions)
  * per_class_accuracy    - accuracy per class, used for head/tail analysis
  * head_tail_gap         - accuracy gap between frequent and rare classes
"""

from typing import List

import numpy as np


def confidence_rejection(y_true: np.ndarray, proba: np.ndarray,
                         thresholds: List[float] | None = None) -> List[dict]:
    """
    Selective prediction: accept a prediction only when its top softmax
    probability is >= threshold; report accuracy on the accepted subset and
    the coverage (fraction accepted).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.0, 0.95, 20)]

    y_pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = y_pred == y_true
    n = len(y_true)

    rows = []
    for thr in thresholds:
        accept = conf >= thr
        cov = float(accept.mean())
        acc = float(correct[accept].mean()) if accept.any() else 0.0
        rows.append(
            {"threshold": float(thr), "accuracy": acc, "coverage": cov,
             "rejected": int(n - accept.sum())}
        )
    return rows


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """Return an array of per-class accuracy (NaN for classes absent in y_true)."""
    accs = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            accs[c] = (y_pred[mask] == c).mean()
    return accs


def head_tail_gap(y_true: np.ndarray, y_pred: np.ndarray,
                  train_counts: np.ndarray, num_classes: int) -> dict:
    """Compare accuracy on the most-frequent vs least-frequent training classes."""
    median = float(np.median(train_counts))
    head = np.where(train_counts >= median)[0]
    tail = np.where(train_counts < median)[0]

    head_mask = np.isin(y_true, head)
    tail_mask = np.isin(y_true, tail)
    head_acc = float((y_pred[head_mask] == y_true[head_mask]).mean()) if head_mask.any() else 0.0
    tail_acc = float((y_pred[tail_mask] == y_true[tail_mask]).mean()) if tail_mask.any() else 0.0

    return {
        "median_train_count": median,
        "num_head_classes": int(len(head)),
        "num_tail_classes": int(len(tail)),
        "head_accuracy": head_acc,
        "tail_accuracy": tail_acc,
        "gap": head_acc - tail_acc,
    }


def summarize_robustness(clean_acc: float, corruption_accs: dict) -> dict:
    """Summarize accuracy drop under each corruption relative to the clean set."""
    rows = {}
    for name, acc in corruption_accs.items():
        rows[name] = {
            "accuracy": float(acc),
            "absolute_drop": float(clean_acc - acc),
            "relative_drop_pct": float(100.0 * (clean_acc - acc) / clean_acc) if clean_acc else 0.0,
        }
    if corruption_accs:
        rows["mean_corruption_accuracy"] = float(np.mean(list(corruption_accs.values())))
    return rows
