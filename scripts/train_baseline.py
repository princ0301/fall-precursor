import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from fallprecursor.data.splits import split_by_subject
from fallprecursor.features.kinematics import extract_features
from fallprecursor.models.baseline import train_random_forest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    windows = np.load(args.processed_dir / "windows.npy")
    labels_df = pd.read_csv(args.processed_dir / "labels.csv")
    labels = labels_df["label"].to_numpy()

    group_ids = (labels_df["room"] + "/" + labels_df["video"]).to_numpy()
    train_idx, val_idx, test_idx = split_by_subject(group_ids, args.train_frac, args.val_frac, args.seed)

    features = np.stack([extract_features(window) for window in windows], axis=0)

    model = train_random_forest(features[train_idx], labels[train_idx], args.seed)

    print("validation set performance:")
    val_predictions = model.predict(features[val_idx])
    print(classification_report(labels[val_idx], val_predictions, target_names=["normal", "precursor", "fall"]))

    print("test set performance:")
    test_predictions = model.predict(features[test_idx])
    print(classification_report(labels[test_idx], test_predictions, target_names=["normal", "precursor", "fall"]))


if __name__ == "__main__":
    main()