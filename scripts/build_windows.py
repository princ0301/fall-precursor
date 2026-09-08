import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fallprecursor.data.windowing import make_windows

NO_FALL_SENTINEL = -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_rows = []
    num_dropped_for_nan = 0

    npz_paths = sorted(args.interim_dir.glob("*/*.npz"))
    print(f"found {len(npz_paths)} per-video files")

    for npz_path in npz_paths:
        room_name = npz_path.parent.name
        video_name = npz_path.stem

        data = np.load(npz_path)
        keypoints, frame_labels = data["keypoints"], data["frame_labels"]
        onset_frame = int(data["onset_frame"])
        fps = float(data["fps"])

        windows, window_labels, start_frames = make_windows(
            keypoints, frame_labels, args.window_length, args.stride
        )
        clean_mask = ~np.isnan(windows).any(axis=(1, 2, 3))
        num_dropped_for_nan += np.sum(~clean_mask)

        for window, label, start_frame in zip(windows[clean_mask], window_labels[clean_mask], start_frames[clean_mask]):
            all_windows.append(window)
            all_rows.append({
                "room": room_name,
                "video": video_name,
                "label": int(label),
                "start_frame": int(start_frame),
                "window_length": args.window_length,
                "onset_frame": onset_frame,
                "fps": fps,
            })

    windows_array = np.stack(all_windows, axis=0)
    labels_df = pd.DataFrame(all_rows)
    labels_df.insert(0, "window_id", range(len(labels_df)))

    np.save(args.output_dir / "windows.npy", windows_array)
    labels_df.to_csv(args.output_dir / "labels.csv", index=False)

    print(f"windows kept: {len(labels_df)}")
    print(f"windows dropped for containing a missing detection: {num_dropped_for_nan}")
    print("label distribution:")
    print(labels_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()