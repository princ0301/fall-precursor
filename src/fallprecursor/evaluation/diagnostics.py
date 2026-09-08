import numpy as np

from fallprecursor.features.kinematics import joint_center


def compute_video_stats(keypoints: np.ndarray) -> dict:
    """Summarize one video's skeleton sequence into quality/scale/motion statistics.

    keypoints has shape (F, K, C). detection_rate is the fraction of frames
    with no missing landmarks. avg_torso_length is a proxy for camera
    distance/zoom (larger means the person appears bigger in frame).
    avg_motion is the mean frame-to-frame hip displacement across the whole
    video, a proxy for general movement pace rather than fall-specific
    dynamics, since most of a video's duration is not the fall itself. The
    latter two are computed only over frames with a valid detection, and
    are None if there are fewer than two such frames to compare.
    """
    valid_mask = ~np.isnan(keypoints).any(axis=(1, 2))
    detection_rate = float(np.mean(valid_mask)) if len(valid_mask) > 0 else 0.0

    valid_keypoints = keypoints[valid_mask]
    if valid_keypoints.shape[0] < 2:
        return {"detection_rate": detection_rate, "avg_torso_length": None, "avg_motion": None}

    hip_center = joint_center(valid_keypoints, "left_hip", "right_hip")
    shoulder_center = joint_center(valid_keypoints, "left_shoulder", "right_shoulder")
    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=1)

    frame_to_frame_motion = np.linalg.norm(np.diff(hip_center, axis=0), axis=1)

    return {
        "detection_rate": detection_rate,
        "avg_torso_length": float(np.mean(torso_length)),
        "avg_motion": float(np.mean(frame_to_frame_motion)),
    }


def compute_fall_window_motion(
    keypoints: np.ndarray,
    onset_frame: int,
    frames_before: int = 45,
    frames_after: int = 30,
) -> float | None:
    """Mean frame-to-frame hip displacement in the window around a fall's onset.

    Unlike compute_video_stats's avg_motion, which dilutes fall dynamics
    across an entire video's normal activity, this isolates motion
    specifically in the frames most likely to contain the fall itself.
    Returns None if fewer than two valid frames fall within the window.
    """
    start = max(0, onset_frame - frames_before)
    end = min(keypoints.shape[0], onset_frame + frames_after + 1)
    window = keypoints[start:end]

    valid_mask = ~np.isnan(window).any(axis=(1, 2))
    valid_window = window[valid_mask]
    if valid_window.shape[0] < 2:
        return None

    hip_center = joint_center(valid_window, "left_hip", "right_hip")
    frame_to_frame_motion = np.linalg.norm(np.diff(hip_center, axis=0), axis=1)
    return float(np.mean(frame_to_frame_motion))