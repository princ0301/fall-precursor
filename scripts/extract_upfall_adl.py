import argparse
from pathlib import Path

import numpy as np

from fallprecursor.data.labeling import label_frames
from fallprecursor.data.timestamps import make_strictly_increasing
from fallprecursor.data.upfall import parse_upfall_image_timestamp
from fallprecursor.pose.extractor import extract_pose_sequence_from_images

NO_FALL_SENTINEL = -1
PRECURSOR_WINDOW = 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--room-name", required=True, help="e.g. UpFall_Subject1")
    parser.add_argument("--video-name", required=True, help="e.g. A6T1")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()

    image_paths = sorted(args.image_dir.glob("*.png"))
    if len(image_paths) == 0:
        raise ValueError(f"no PNG images found in {args.image_dir}")

    timestamps_sec = [parse_upfall_image_timestamp(p.name) for p in image_paths]
    start_time = timestamps_sec[0]
    elapsed_seconds = timestamps_sec[-1] - timestamps_sec[0]
    fps = (len(image_paths) - 1) / elapsed_seconds if elapsed_seconds > 0 else 30.0

    timestamps_ms = [int(round((t - start_time) * 1000)) for t in timestamps_sec]
    timestamps_ms = make_strictly_increasing(timestamps_ms)

    print(f"found {len(image_paths)} images, spanning {elapsed_seconds:.2f}s, fps={fps:.2f}")

    keypoints = extract_pose_sequence_from_images(image_paths, args.model, timestamps_ms)
    frame_labels = label_frames(
        num_frames=keypoints.shape[0], onset_frame=None, impact_frame=None, precursor_window=PRECURSOR_WINDOW
    )

    output_dir = args.output_dir / args.room_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.video_name}.npz"
    np.savez(
        output_path,
        keypoints=keypoints,
        frame_labels=frame_labels,
        onset_frame=NO_FALL_SENTINEL,
        impact_frame=NO_FALL_SENTINEL,
        fps=fps,
    )
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()