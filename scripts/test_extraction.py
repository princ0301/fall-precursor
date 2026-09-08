import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from fallprecursor.pose.extractor import extract_pose_sequence
from fallprecursor.pose.schema import POSE_CONNECTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sequence = extract_pose_sequence(args.video, args.model)
    report_detection_stats(sequence)
    save_annotated_frame(args.video, args.model, args.output_dir / f"{args.video.stem}_annotated.png")


def report_detection_stats(sequence: np.ndarray) -> None:
    num_frames = sequence.shape[0]
    frames_with_detection = np.sum(~np.isnan(sequence).any(axis=(1, 2)))
    detection_rate = frames_with_detection / num_frames if num_frames > 0 else 0.0

    print(f"total frames: {num_frames}")
    print(f"frames with a detected pose: {frames_with_detection}")
    print(f"detection rate: {detection_rate:.1%}")


def save_annotated_frame(video_path: Path, model_path: Path, output_path: Path) -> None:
    """Save the first frame with a detected pose, landmarks drawn on top, for visual inspection."""
    capture = cv2.VideoCapture(str(video_path))
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                print("no frame with a detected pose was found in this video")
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, frame_index)

            if len(result.pose_landmarks) > 0:
                draw_landmarks(frame, result.pose_landmarks[0])
                cv2.imwrite(str(output_path), frame)
                print(f"saved annotated frame to {output_path}")
                return

            frame_index += 1

    capture.release()


def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    height, width = frame.shape[0], frame.shape[1]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for start_index, end_index in POSE_CONNECTIONS:
        cv2.line(frame, points[start_index], points[end_index], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


if __name__ == "__main__":
    main()