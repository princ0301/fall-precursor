import numpy as np

from fallprecursor.pose.schema import NAME_TO_INDEX


def normalize_window(window: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Make a (T, K, C) skeleton window translation- and scale-invariant.

    Every frame's x, y coordinates are re-centered on that frame's hip
    center, then divided by a single scale factor for the whole window (the
    mean torso length across its frames). This removes camera position and
    zoom level as a signal a model could rely on, which matters because a
    model trained on one room's camera framing should not depend on where
    in the frame or at what scale a fall happens when applied to a
    different room. Using one scale for the whole window, rather than
    normalizing every frame to a fixed torso length, preserves within
    window changes in apparent body size (e.g. foreshortening as a person
    falls), which is itself part of the motion signal. The visibility
    channel is left unchanged. After this, hip center is always (0, 0) by
    construction, so motion features must be computed relative to the hip,
    not from the hip's own trajectory.
    """
    left_hip = window[:, NAME_TO_INDEX["left_hip"], :2]
    right_hip = window[:, NAME_TO_INDEX["right_hip"], :2]
    left_shoulder = window[:, NAME_TO_INDEX["left_shoulder"], :2]
    right_shoulder = window[:, NAME_TO_INDEX["right_shoulder"], :2]

    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=1)
    scale = max(np.mean(torso_length), epsilon)

    normalized = window.copy()
    normalized[:, :, :2] = (window[:, :, :2] - hip_center[:, None, :]) / scale
    return normalized