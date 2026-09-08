import numpy as np

from fallprecursor.features.kinematics import joint_center, torso_angle, velocity, extract_features
from fallprecursor.pose.schema import NUM_LANDMARKS, NAME_TO_INDEX


def make_window(num_frames: int) -> np.ndarray:
    return np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float32)


def test_joint_center_is_midpoint():
    window = make_window(2)
    window[:, NAME_TO_INDEX["left_hip"], :2] = [0.0, 0.0]
    window[:, NAME_TO_INDEX["right_hip"], :2] = [1.0, 1.0]
    center = joint_center(window, "left_hip", "right_hip")
    assert np.allclose(center, [[0.5, 0.5], [0.5, 0.5]])


def test_torso_angle_zero_when_upright():
    window = make_window(1)
    window[0, NAME_TO_INDEX["left_hip"], :2] = [0.0, 1.0]
    window[0, NAME_TO_INDEX["right_hip"], :2] = [0.2, 1.0]
    window[0, NAME_TO_INDEX["left_shoulder"], :2] = [0.0, 0.0]
    window[0, NAME_TO_INDEX["right_shoulder"], :2] = [0.2, 0.0]
    angles = torso_angle(window)
    assert np.isclose(angles[0], 0.0, atol=1e-6)


def test_torso_angle_quarter_turn_when_horizontal():
    window = make_window(1)
    window[0, NAME_TO_INDEX["left_hip"], :2] = [0.0, 0.5]
    window[0, NAME_TO_INDEX["right_hip"], :2] = [0.0, 0.5]
    window[0, NAME_TO_INDEX["left_shoulder"], :2] = [1.0, 0.5]
    window[0, NAME_TO_INDEX["right_shoulder"], :2] = [1.0, 0.5]
    angles = torso_angle(window)
    assert np.isclose(abs(angles[0]), np.pi / 2, atol=1e-6)


def test_velocity_first_frame_is_zero():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    speeds = velocity(points)
    assert speeds[0] == 0.0
    assert np.isclose(speeds[1], 5.0)
    assert np.isclose(speeds[2], 0.0)


def test_extract_features_shape_and_finiteness():
    window = make_window(10)
    window[:, NAME_TO_INDEX["left_hip"], :2] = [0.0, 0.5]
    window[:, NAME_TO_INDEX["right_hip"], :2] = [0.2, 0.5]
    window[:, NAME_TO_INDEX["left_shoulder"], :2] = [0.0, 0.0]
    window[:, NAME_TO_INDEX["right_shoulder"], :2] = [0.2, 0.0]
    features = extract_features(window)
    assert features.shape == (8,)
    assert np.all(np.isfinite(features))