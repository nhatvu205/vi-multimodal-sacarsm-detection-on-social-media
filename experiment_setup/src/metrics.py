from __future__ import annotations

from typing import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(labels: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float] | None = None) -> dict:
    metrics = {
        'accuracy': round(float(accuracy_score(labels, predictions)), 4),
        'f1_macro': round(float(f1_score(labels, predictions, average='macro', zero_division=0)), 4),
        'f1_weighted': round(float(f1_score(labels, predictions, average='weighted', zero_division=0)), 4),
        'precision_weighted': round(float(precision_score(labels, predictions, average='weighted', zero_division=0)), 4),
        'recall_weighted': round(float(recall_score(labels, predictions, average='weighted', zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
        'num_samples': len(labels),
    }
    if probabilities is not None:
        try:
            metrics['auc'] = round(float(roc_auc_score(labels, probabilities)), 4)
        except ValueError:
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    return metrics
