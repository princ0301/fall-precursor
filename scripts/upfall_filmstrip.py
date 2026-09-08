import argparse
from pathlib import Path

import cv2
import numpy as np

from fallprecursor.pose.visualization import draw_skeleton

NUM_PANELS = 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/filmstrips"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.image_dir.glob("*.png"))
    data = np.load(args.npz_path)
    keypoints = data["keypoints"]

    if len(image_paths) != keypoints.shape[0]:
        raise ValueError(
            f"image count ({len(image_paths)}) does not match keypoint frame count ({keypoints.shape[0]})"
        )

    frame_indices = np.linspace(0, len(image_paths) - 1, NUM_PANELS, dtype=int)
    panels = []

    for frame_index in frame_indices:
        frame = cv2.imread(str(image_paths[frame_index]))
        if not np.isnan(keypoints[frame_index]).any():
            height, width = frame.shape[0], frame.shape[1]
            points = [(int(x * width), int(y * height)) for x, y in keypoints[frame_index, :, :2]]
            draw_skeleton(frame, points)

        label = f"frame {frame_index}"
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(cv2.resize(frame, (200, 200)))

    filmstrip = np.hstack(panels)
    output_path = args.output_dir / f"{args.npz_path.stem}_filmstrip.png"
    cv2.imwrite(str(output_path), filmstrip)
    print(f"saved filmstrip to {output_path}")


if __name__ == "__main__":
    main()