import argparse
from pathlib import Path

import numpy as np

from fallprecursor.evaluation.experiments import run_cross_dataset_lead_time
from fallprecursor.evaluation.reporting import print_lead_time_curve


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

    thresholds = np.linspace(0.05, 0.95, 19)
    curve = run_cross_dataset_lead_time(
        args.model, args.held_out_room, args.processed_dir, args.train_frac, args.seed,
        args.hidden_size, args.batch_size, args.max_epochs, args.patience, thresholds,
    )
    print_lead_time_curve(curve)


if __name__ == "__main__":
    main()