from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fallprecursor.data.dataset import SkeletonWindowDataset, load_processed_dataset
from fallprecursor.data.splits import split_by_subject_two_way
from fallprecursor.evaluation.grouping import group_scores_by_video
from fallprecursor.evaluation.lead_time import lead_time_vs_false_alarm_curve
from fallprecursor.evaluation.metrics import macro_f1
from fallprecursor.evaluation.reporting import compute_precursor_scores
from fallprecursor.models.factory import build_model
from fallprecursor.training.losses import build_weighted_cross_entropy
from fallprecursor.training.trainer import train_model

NORMAL_ONLY_ROOMS = {"Office", "Lecture_room"}


def run_cross_dataset_lead_time(
    model_name: str,
    held_out_room: str,
    processed_dir: Path,
    train_frac: float,
    seed: int,
    hidden_size: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    thresholds: np.ndarray,
    verbose: bool = True,
) -> list[dict]:
    """Train on all rooms except held_out_room and the normal-only rooms, then return the lead-time curve.

    The normal-only rooms are always excluded from training and always
    added to the evaluation pool, regardless of which room is held out, so
    every run has a real false-alarm denominator even for rooms with few or
    no no-fall videos of their own.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows, labels, group_ids = load_processed_dataset(processed_dir)
    labels_df = pd.read_csv(processed_dir / "labels.csv")
    rooms = np.array([group_id.split("/")[0] for group_id in group_ids])

    if (rooms == held_out_room).sum() == 0:
        raise ValueError(f"no windows found for held-out room '{held_out_room}'")

    held_out_mask = (rooms == held_out_room) | np.isin(rooms, list(NORMAL_ONLY_ROOMS))
    train_val_mask = ~held_out_mask
    train_val_global_indices = np.flatnonzero(train_val_mask)
    held_out_indices = np.flatnonzero(held_out_mask)

    train_idx_local, val_idx_local = split_by_subject_two_way(group_ids[train_val_mask], train_frac, seed)
    train_idx = train_val_global_indices[train_idx_local]
    val_idx = train_val_global_indices[val_idx_local]

    train_loader = DataLoader(SkeletonWindowDataset(windows, labels, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SkeletonWindowDataset(windows, labels, val_idx), batch_size=batch_size)
    held_out_loader = DataLoader(SkeletonWindowDataset(windows, labels, held_out_indices), batch_size=batch_size, shuffle=False)

    model = build_model(model_name, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_weighted_cross_entropy(labels[train_idx], num_classes=3)

    def scoring_fn(predictions, true_labels):
        return macro_f1(predictions, true_labels, num_classes=3)

    train_model(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        max_epochs, patience, scoring_fn=scoring_fn,
    )

    precursor_scores = compute_precursor_scores(model, held_out_loader, device)
    held_out_rows = labels_df.iloc[held_out_indices].reset_index(drop=True).copy()
    held_out_rows["precursor_score"] = precursor_scores

    fall_scores, fall_starts, fall_onsets, normal_scores, fps = group_scores_by_video(held_out_rows)
    if verbose:
        print(f"seed {seed}: {len(fall_scores)} fall videos, {len(normal_scores)} normal videos, fps={fps:.2f}")

    return lead_time_vs_false_alarm_curve(fall_scores, fall_starts, fall_onsets, normal_scores, thresholds, fps)