from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SkeletonWindowDataset(Dataset):
    """Wrap a subset of a windows array and its labels for use with a DataLoader."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.windows = windows[indices]
        self.labels = labels[indices]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(self.windows[item]).float()
        label = torch.tensor(self.labels[item], dtype=torch.long)
        return window, label


def load_processed_dataset(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load windows, labels, and room/video group ids from a processed data directory."""
    windows = np.load(processed_dir / "windows.npy")
    labels_df = pd.read_csv(processed_dir / "labels.csv")
    labels = labels_df["label"].to_numpy()
    group_ids = (labels_df["room"] + "/" + labels_df["video"]).to_numpy()
    return windows, labels, group_ids