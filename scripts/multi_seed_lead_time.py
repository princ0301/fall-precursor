import argparse
from pathlib import Path

import numpy as np

from fallprecursor.evaluation.aggregation import aggregate_curves
from fallprecursor.evaluation.experiments import run_cross_dataset_lead_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "stgcn"], required=True)
    parser.add_argument("--held-out-room", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    thresholds = np.linspace(0.05, 0.95, 19)
    curves = []

    for seed in args.seeds:
        print(f"--- seed {seed} ---")
        curve = run_cross_dataset_lead_time(
            args.model, args.held_out_room, args.processed_dir, args.train_frac, seed,
            args.hidden_size, args.batch_size, args.max_epochs, args.patience, thresholds,
        )
        curves.append(curve)

    aggregated = aggregate_curves(curves)
    print_aggregated_curve(aggregated, num_seeds=len(args.seeds))


def print_aggregated_curve(aggregated: list[dict], num_seeds: int) -> None:
    print(f"\naggregated over {num_seeds} seeds:")
    header = f"{'threshold':>10} {'false_alarm_rate':>20} {'detection_rate':>18} {'lead_time_s':>18} {'n_lt':>5}"
    print(header)
    for point in aggregated:
        far = f"{point['false_alarm_rate_mean']:.2f}±{point['false_alarm_rate_std']:.2f}"
        detection = f"{point['detection_rate_mean']:.2f}±{point['detection_rate_std']:.2f}"
        if point["lead_time_mean"] is not None:
            lead_time = f"{point['lead_time_mean']:.2f}±{point['lead_time_std']:.2f}"
        else:
            lead_time = "n/a"
        print(f"{point['threshold']:>10.2f} {far:>20} {detection:>18} {lead_time:>18} {point['num_seeds_with_lead_time']:>5}")


if __name__ == "__main__":
    main()