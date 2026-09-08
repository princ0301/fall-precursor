import numpy as np

from fallprecursor.pose.schema import NAME_TO_INDEX


def joint_center(window: np.ndarray, joint_a: str, joint_b: str) -> np.ndarray:
    """Return the (T, 2) midpoint trajectory of two named joints' x, y coordinates."""
    index_a = NAME_TO_INDEX[joint_a]
    index_b = NAME_TO_INDEX[joint_b]
    return (window[:, index_a, :2] + window[:, index_b, :2]) / 2.0


def torso_angle(window: np.ndarray) -> np.ndarray:
    """Return the (T,) angle in radians between the torso and vertical, 0 when upright."""
    hip_center = joint_center(window, "left_hip", "right_hip")
    shoulder_center = joint_center(window, "left_shoulder", "right_shoulder")
    torso_vector = shoulder_center - hip_center
    return np.arctan2(torso_vector[:, 0], -torso_vector[:, 1])


def velocity(points: np.ndarray) -> np.ndarray:
    """Return the (T,) frame-to-frame displacement magnitude of a (T, 2) trajectory.

    The first frame has no previous frame to compare against, so its velocity
    is set to 0 rather than shortening the output array.
    """
    displacement = np.diff(points, axis=0)
    magnitude = np.linalg.norm(displacement, axis=1)
    return np.concatenate([[0.0], magnitude])


def extract_features(window: np.ndarray) -> np.ndarray:
    """Summarize one (T, K, C) skeleton window into a fixed-length feature vector.

    Returns 8 features: mean and std of torso angle, mean and max of hip
    center velocity, mean and min of vertical hip velocity (large positive
    values indicate rapid downward motion in image coordinates), and the
    total range of hip height and torso angle across the window.
    """
    hip_center = joint_center(window, "left_hip", "right_hip")
    hip_velocity = velocity(hip_center)
    vertical_hip_velocity = np.concatenate([[0.0], np.diff(hip_center[:, 1])])
    angles = torso_angle(window)

    return np.array([
        np.mean(angles),
        np.std(angles),
        np.mean(hip_velocity),
        np.max(hip_velocity),
        np.mean(vertical_hip_velocity),
        np.min(vertical_hip_velocity),
        np.ptp(hip_center[:, 1]),
        np.ptp(angles),
    ])