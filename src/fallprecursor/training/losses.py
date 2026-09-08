import numpy as np
import torch
from torch import nn


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Return inverse-frequency class weights, normalized to sum to num_classes.

    Classes with zero occurrences get a weight of 0 rather than dividing by
    zero, since a class that never appears in this split cannot contribute
    to the loss regardless.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_cross_entropy(labels: np.ndarray, num_classes: int) -> nn.CrossEntropyLoss:
    """Cross entropy weighted by inverse class frequency, since normal windows dominate."""
    weights = compute_class_weights(labels, num_classes)
    return nn.CrossEntropyLoss(weight=weights)