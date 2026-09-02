"""Classification + boundary metrics (spec section 21).

Naive accuracy is never reported alone: every classification metric call
also returns macro/weighted F1, precision/recall and a confusion matrix,
so a model that just predicts the majority class is visibly bad on macro
F1 even under severe class imbalance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from lithology.constants import IGNORE_INDEX


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    mask = y_true != IGNORE_INDEX
    y_true, y_pred = y_true[mask], y_pred[mask]
    labels = list(range(num_classes))
    if len(y_true) == 0:
        return {"n_points": 0}

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n_points": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def boundary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    mask = y_true != IGNORE_INDEX
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) == 0 or y_true.sum() == 0:
        return {"n_points": int(len(y_true)), "note": "no positive boundary points to score"}

    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        "n_points": int(len(y_true)),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "f1_positive": float(f1[1]),
        "pr_auc": float(pr_auc),
        "threshold": threshold,
    }
