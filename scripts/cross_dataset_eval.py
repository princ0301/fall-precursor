import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from fallprecursor.data.dataset import SkeletonWindowDataset, load_processed_dataset
from fallprecursor.data.splits import split_by_subject_two_way
from fallprecursor.evaluation.metrics import macro_f1
from fallprecursor.models.factory import build_model
from fallprecursor.training.losses import build_weighted_cross_entropy
from fallprecursor.training.trainer import train_model, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "stgcn"], required=True)
    parser.add_argument("--held-out-room", required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows, labels, group_ids = load_processed_dataset(args.processed_dir)
    rooms = np.array([group_id.split("/")[0] for group_id in group_ids])

    held_out_mask = rooms == args.held_out_room
    if held_out_mask.sum() == 0:
        raise ValueError(f"no windows found for held-out room '{args.held_out_room}'")

    train_val_mask = ~held_out_mask
    train_val_global_indices = np.flatnonzero(train_val_mask)
    held_out_indices = np.flatnonzero(held_out_mask)

    train_idx_local, val_idx_local = split_by_subject_two_way(
        group_ids[train_val_mask], args.train_frac, args.seed
    )
    train_idx = train_val_global_indices[train_idx_local]
    val_idx = train_val_global_indices[val_idx_local]

    print(f"held-out room: {args.held_out_room} ({len(held_out_indices)} windows)")
    print(f"train: {len(train_idx)} windows, val: {len(val_idx)} windows")

    train_loader = DataLoader(SkeletonWindowDataset(windows, labels, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SkeletonWindowDataset(windows, labels, val_idx), batch_size=args.batch_size)
    held_out_loader = DataLoader(SkeletonWindowDataset(windows, labels, held_out_indices), batch_size=args.batch_size)

    model = build_model(args.model, args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_weighted_cross_entropy(labels[train_idx], num_classes=3)

    def scoring_fn(predictions, labels):
        return macro_f1(predictions, labels, num_classes=3)

    train_model(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        args.max_epochs, args.patience, scoring_fn=scoring_fn,
    )

    target_names = ["normal", "precursor", "fall"]

    print(f"held-out room ({args.held_out_room}) performance:")
    _, held_out_predictions, held_out_labels = evaluate(model, held_out_loader, loss_fn, device)
    print(classification_report(held_out_labels, held_out_predictions, target_names=target_names))


if __name__ == "__main__":
    main()