import numpy as np
import torch
from torch.utils.data import DataLoader

from fallprecursor.data.dataset import SkeletonWindowDataset
from fallprecursor.models.lstm import LSTMClassifier
from fallprecursor.evaluation.reporting import compute_precursor_scores


def test_scores_are_valid_probabilities_with_correct_length():
    num_samples = 10
    windows = np.random.rand(num_samples, 5, 33, 3).astype(np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)
    dataset = SkeletonWindowDataset(windows, labels, indices=np.arange(num_samples))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=8, num_classes=3)
    scores = compute_precursor_scores(model, loader, torch.device("cpu"))

    assert scores.shape == (num_samples,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_scores_preserve_dataloader_order():
    num_samples = 8
    windows = np.random.rand(num_samples, 5, 33, 3).astype(np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)
    dataset = SkeletonWindowDataset(windows, labels, indices=np.arange(num_samples))
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=8, num_classes=3)
    scores_full_batch = compute_precursor_scores(
        model, DataLoader(dataset, batch_size=num_samples, shuffle=False), torch.device("cpu")
    )
    scores_small_batches = compute_precursor_scores(model, loader, torch.device("cpu"))

    assert np.allclose(scores_full_batch, scores_small_batches, atol=1e-5)