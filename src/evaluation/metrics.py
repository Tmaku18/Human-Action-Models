"""
Evaluation: balanced accuracy, macro F1, confusion matrix.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return dict with balanced_accuracy, macro_f1, accuracy, confusion_matrix (list)."""
    ba = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "balanced_accuracy": float(ba),
        "macro_f1": float(f1_macro),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
