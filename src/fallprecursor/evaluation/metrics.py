import numpy as np
from sklearn.metrics import f1_score


def macro_f1(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Unweighted mean of per-class F1, so the rare precursor/fall classes count as much as normal."""
    return f1_score(labels, predictions, labels=list(range(num_classes)), average="macro", zero_division=0)


def per_class_f1(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return the (num_classes,) array of per-class F1 scores."""
    return f1_score(labels, predictions, labels=list(range(num_classes)), average=None, zero_division=0)