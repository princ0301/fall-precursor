import argparse
from pathlib import Path

import numpy as np

from fallprecursor.pose.extractor import extract_pose_sequence, get_video_fps
from fallprecursor.data.le2i import parse_le2i_annotation
from fallprecursor.data.labeling import label_frames

NO_FALL_SENTINEL = -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--precursor-window", type=int, default=30)
    parser.add_argument(
        "--no-fall-room", action="store_true",
        help="treat every video in this room as fall-free; used for rooms with no Annotation_files folder",
    )
    args = parser.parse_args()

    videos_dir = args.room_dir / "Videos" if (args.room_dir / "Videos").exists() else args.room_dir
    annotations_dir = args.room_dir / "Annotation_files"
    output_dir = args.output_dir / args.room_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(videos_dir.glob("*.avi"))
    print(f"found {len(video_paths)} videos in {videos_dir}")

    num_processed = 0
    num_skipped = 0

    for video_path in video_paths:
        if args.no_fall_room:
            onset_frame, impact_frame = None, None
        else:
            annotation_path = annotations_dir / f"{video_path.stem}.txt"
            if not annotation_path.exists():
                print(f"skipping {video_path.name}: no matching annotation file")
                num_skipped += 1
                continue
            onset_frame, impact_frame = parse_le2i_annotation(annotation_path)

        try:
            keypoints = extract_pose_sequence(video_path, args.model)
            fps = get_video_fps(video_path)
            frame_labels = label_frames(
                num_frames=keypoints.shape[0],
                onset_frame=onset_frame,
                impact_frame=impact_frame,
                precursor_window=args.precursor_window,
            )
        except (ValueError, RuntimeError) as error:
            print(f"skipping {video_path.name}: {error}")
            num_skipped += 1
            continue

        output_path = output_dir / f"{video_path.stem}.npz"
        np.savez(
            output_path,
            keypoints=keypoints,
            frame_labels=frame_labels,
            onset_frame=onset_frame if onset_frame is not None else NO_FALL_SENTINEL,
            impact_frame=impact_frame if impact_frame is not None else NO_FALL_SENTINEL,
            fps=fps,
        )
        num_processed += 1
        print(f"processed {video_path.name}: {keypoints.shape[0]} frames -> {output_path}")

    print(f"done: {num_processed} processed, {num_skipped} skipped")


if __name__ == "__main__":
    main()