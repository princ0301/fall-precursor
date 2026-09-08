from pathlib import Path

import numpy as np
import torch

from fallprecursor.data.dataset import SkeletonWindowDataset, load_processed_dataset

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "processed_sample"


def test_load_processed_dataset_shapes_and_group_ids():
    windows, labels, group_ids = load_processed_dataset(FIXTURES_DIR)
    assert windows.shape == (4, 5, 33, 3)
    assert labels.shape == (4,)
    assert list(group_ids) == [
        "Coffee_room_01/video (1)",
        "Coffee_room_01/video (2)",
        "Home_01/video (1)",
        "Home_01/video (2)",
    ]


def test_dataset_selects_only_given_indices():
    windows, labels, _ = load_processed_dataset(FIXTURES_DIR)
    dataset = SkeletonWindowDataset(windows, labels, indices=np.array([1, 3]))
    assert len(dataset) == 2
    _, first_label = dataset[0]
    _, second_label = dataset[1]
    assert first_label.item() == 1
    assert second_label.item() == 2


def test_dataset_item_types_and_shapes():
    windows, labels, _ = load_processed_dataset(FIXTURES_DIR)
    dataset = SkeletonWindowDataset(windows, labels, indices=np.array([0]))
    window, label = dataset[0]
    assert isinstance(window, torch.Tensor)
    assert window.dtype == torch.float32
    assert window.shape == (5, 33, 3)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long