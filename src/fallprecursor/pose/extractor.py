from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from fallprecursor.pose.schema import NUM_LANDMARKS, NUM_CHANNELS


def get_video_fps(video_path: Path) -> float:
    """Return the video's frames-per-second, or 30.0 if the metadata is missing or invalid."""
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps > 0 else 30.0


def extract_pose_sequence(
    video_path: Path,
    model_path: Path,
    min_detection_confidence: float = 0.5,
) -> np.ndarray:
    """Run MediaPipe's PoseLandmarker (Tasks API) over a video.

    Returns an array of shape (F, NUM_LANDMARKS, NUM_CHANNELS) where F is the
    number of frames read from the video. Channels are (x, y, visibility),
    both x and y normalized to [0, 1]. Frames where no pose is detected are
    filled with NaN so downstream code can decide how to handle missing
    detections rather than silently receiving zeros. Only the first detected
    person per frame is used; multi-person clips are not currently supported.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    landmarker = _build_landmarker(model_path, min_detection_confidence)

    frames = []
    with landmarker:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp_ms = int(frame_index * (1000.0 / fps))
            frames.append(_extract_single_frame(landmarker, frame, timestamp_ms))
            frame_index += 1

    capture.release()

    if len(frames) == 0:
        return np.empty((0, NUM_LANDMARKS, NUM_CHANNELS), dtype=np.float32)
    return np.stack(frames, axis=0)


def _build_landmarker(model_path: Path, min_detection_confidence: float) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_detection_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _extract_single_frame(
    landmarker: vision.PoseLandmarker,
    frame: np.ndarray,
    timestamp_ms: int,
) -> np.ndarray:
    """Return landmarks for one BGR frame, or NaN if no pose is detected."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if len(result.pose_landmarks) == 0:
        return np.full((NUM_LANDMARKS, NUM_CHANNELS), np.nan, dtype=np.float32)

    landmarks = result.pose_landmarks[0]
    return np.array(
        [[lm.x, lm.y, getattr(lm, "visibility", 1.0)] for lm in landmarks],
        dtype=np.float32,
    )