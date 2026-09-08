import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from fallprecursor.data.dataset import SkeletonWindowDataset, load_processed_dataset
from fallprecursor.data.splits import split_by_subject
from fallprecursor.evaluation.metrics import macro_f1
from fallprecursor.models.factory import build_model
from fallprecursor.training.losses import build_weighted_cross_entropy
from fallprecursor.training.trainer import train_model, evaluate


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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows, labels, group_ids = load_processed_dataset(args.processed_dir)
    train_idx, val_idx, test_idx = split_by_subject(group_ids, args.train_frac, args.val_frac, args.seed)

    train_dataset = SkeletonWindowDataset(windows, labels, train_idx)
    val_dataset = SkeletonWindowDataset(windows, labels, val_idx)
    test_dataset = SkeletonWindowDataset(windows, labels, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

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

    print("validation set performance:")
    _, val_predictions, val_labels = evaluate(model, val_loader, loss_fn, device)
    print(classification_report(val_labels, val_predictions, target_names=target_names))

    print("test set performance:")
    _, test_predictions, test_labels = evaluate(model, test_loader, loss_fn, device)
    print(classification_report(test_labels, test_predictions, target_names=target_names))


if __name__ == "__main__":
    main()