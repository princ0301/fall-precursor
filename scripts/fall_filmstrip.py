import argparse
from pathlib import Path

import cv2
import numpy as np

from fallprecursor.pose.visualization import draw_skeleton

NO_FALL_SENTINEL = -1
FRAME_OFFSETS = [-45, -30, -15, 0, 15, 30]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--video", required=True, help="video stem, e.g. 'video (5)'")
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/filmstrips"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.interim_dir / args.room / f"{args.video}.npz"
    data = np.load(npz_path)
    keypoints = data["keypoints"]
    onset_frame = int(data["onset_frame"])

    if onset_frame == NO_FALL_SENTINEL:
        raise ValueError(f"{args.video} in {args.room} has no fall onset; pick a fall video")

    video_dir = args.raw_dir / args.room / "Videos"
    video_dir = video_dir if video_dir.exists() else args.raw_dir / args.room
    video_path = video_dir / f"{args.video}.avi"

    panels = build_panels(video_path, keypoints, onset_frame)
    filmstrip = np.hstack(panels)

    output_path = args.output_dir / f"{args.room}_{args.video.replace(' ', '_')}_filmstrip.png"
    cv2.imwrite(str(output_path), filmstrip)
    print(f"saved filmstrip to {output_path}")


def build_panels(video_path: Path, keypoints: np.ndarray, onset_frame: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    panels = []

    for offset in FRAME_OFFSETS:
        frame_index = onset_frame + offset
        if frame_index < 0 or frame_index >= keypoints.shape[0]:
            continue

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            continue

        if not np.isnan(keypoints[frame_index]).any():
            height, width = frame.shape[0], frame.shape[1]
            points = [(int(x * width), int(y * height)) for x, y in keypoints[frame_index, :, :2]]
            draw_skeleton(frame, points)

        label = f"onset{offset:+d}"
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(cv2.resize(frame, (240, 180)))

    capture.release()
    return panels


if __name__ == "__main__":
    main()