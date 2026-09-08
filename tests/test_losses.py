import numpy as np
import torch

from fallprecursor.training.losses import compute_class_weights, build_weighted_cross_entropy


def test_rare_class_gets_higher_weight():
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
    weights = compute_class_weights(labels, num_classes=3)
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]


def test_weights_sum_to_num_classes():
    labels = np.array([0, 0, 1, 2])
    weights = compute_class_weights(labels, num_classes=3)
    assert torch.isclose(weights.sum(), torch.tensor(3.0), atol=1e-5)


def test_absent_class_gets_zero_weight():
    labels = np.array([0, 0, 1])
    weights = compute_class_weights(labels, num_classes=3)
    assert weights[2] == 0.0


def test_build_weighted_cross_entropy_runs_on_a_batch():
    labels = np.array([0, 0, 1, 2])
    loss_fn = build_weighted_cross_entropy(labels, num_classes=3)
    logits = torch.rand(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0