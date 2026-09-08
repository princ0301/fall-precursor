import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fallprecursor.data.dataset import SkeletonWindowDataset, load_processed_dataset
from fallprecursor.data.splits import split_by_subject
from fallprecursor.evaluation.grouping import group_scores_by_video
from fallprecursor.evaluation.lead_time import lead_time_vs_false_alarm_curve
from fallprecursor.evaluation.metrics import macro_f1
from fallprecursor.evaluation.reporting import compute_precursor_scores, print_lead_time_curve
from fallprecursor.models.factory import build_model
from fallprecursor.training.losses import build_weighted_cross_entropy
from fallprecursor.training.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "stgcn"], required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--alert-position", choices=["start", "center", "end"], default="start",
        help="which frame within each window counts as its alert timestamp for lead-time purposes; "
             "'center' is recommended when comparing across different window lengths",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows, labels, group_ids = load_processed_dataset(args.processed_dir)
    labels_df = pd.read_csv(args.processed_dir / "labels.csv")
    train_idx, val_idx, test_idx = split_by_subject(group_ids, args.train_frac, args.val_frac, args.seed)

    train_loader = DataLoader(SkeletonWindowDataset(windows, labels, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SkeletonWindowDataset(windows, labels, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(SkeletonWindowDataset(windows, labels, test_idx), batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_weighted_cross_entropy(labels[train_idx], num_classes=3)

    def scoring_fn(predictions, true_labels):
        return macro_f1(predictions, true_labels, num_classes=3)

    train_model(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        args.max_epochs, args.patience, scoring_fn=scoring_fn,
    )

    precursor_scores = compute_precursor_scores(model, test_loader, device)
    test_rows = labels_df.iloc[test_idx].reset_index(drop=True).copy()
    test_rows["precursor_score"] = precursor_scores

    fall_scores, fall_starts, fall_onsets, normal_scores, fps = group_scores_by_video(test_rows, args.alert_position)
    print(f"test set: {len(fall_scores)} fall videos, {len(normal_scores)} normal videos, fps={fps:.2f}")

    thresholds = np.linspace(0.05, 0.95, 19)
    curve = lead_time_vs_false_alarm_curve(fall_scores, fall_starts, fall_onsets, normal_scores, thresholds, fps)
    print_lead_time_curve(curve)


if __name__ == "__main__":
    main()