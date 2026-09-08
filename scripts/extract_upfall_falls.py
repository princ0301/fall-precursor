import argparse
from pathlib import Path

import numpy as np

from fallprecursor.data.labeling import label_frames
from fallprecursor.data.upfall import load_upfall_fall_csv

NO_FALL_SENTINEL = -1
PRECURSOR_WINDOW = 30
ASSUMED_FPS = 18.9  # measured from real UP-Fall image timestamps; see scripts/extract_upfall_adl.py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True, help="e.g. subject1")
    parser.add_argument("--room-name", required=True, help="e.g. UpFall_Subject1_Falls")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()

    csv_paths = sorted(args.csv_dir.glob("*.csv"))
    print(f"found {len(csv_paths)} fall CSVs in {args.csv_dir}")

    output_dir = args.output_dir / args.room_name
    output_dir.mkdir(parents=True, exist_ok=True)

    num_processed = 0
    num_skipped = 0

    for csv_path in csv_paths:
        try:
            keypoints, onset_frame, impact_frame = load_upfall_fall_csv(csv_path)
            frame_labels = label_frames(
                num_frames=keypoints.shape[0],
                onset_frame=onset_frame,
                impact_frame=impact_frame,
                precursor_window=PRECURSOR_WINDOW,
            )
        except ValueError as error:
            print(f"skipping {csv_path.name}: {error}")
            num_skipped += 1
            continue

        output_path = output_dir / f"{csv_path.stem}.npz"
        np.savez(
            output_path,
            keypoints=keypoints,
            frame_labels=frame_labels,
            onset_frame=onset_frame if onset_frame is not None else NO_FALL_SENTINEL,
            impact_frame=impact_frame if impact_frame is not None else NO_FALL_SENTINEL,
            fps=ASSUMED_FPS,
        )
        num_processed += 1
        print(f"processed {csv_path.name}: {keypoints.shape[0]} frames, onset={onset_frame} -> {output_path}")

    print(f"done: {num_processed} processed, {num_skipped} skipped")


if __name__ == "__main__":
    main()