import numpy as np

from fallprecursor.data.normalization import normalize_window
from fallprecursor.pose.schema import NUM_LANDMARKS, NAME_TO_INDEX


def make_window_with_torso(num_frames: int, hip_xy, shoulder_xy, other_offset=(0.3, 0.3)) -> np.ndarray:
    window = np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float64)
    window[:, :, :2] = hip_xy  # every joint starts at the hip position, so translation moves the whole skeleton together
    window[:, NAME_TO_INDEX["left_hip"], :2] = hip_xy
    window[:, NAME_TO_INDEX["right_hip"], :2] = hip_xy
    window[:, NAME_TO_INDEX["left_shoulder"], :2] = shoulder_xy
    window[:, NAME_TO_INDEX["right_shoulder"], :2] = shoulder_xy
    window[:, NAME_TO_INDEX["nose"], :2] = np.array(shoulder_xy) + np.array(other_offset)
    return window


def test_hip_center_is_origin_after_normalization():
    window = make_window_with_torso(5, hip_xy=(2.0, 3.0), shoulder_xy=(2.0, 1.0))
    normalized = normalize_window(window)
    hip_center = (normalized[:, NAME_TO_INDEX["left_hip"], :2] + normalized[:, NAME_TO_INDEX["right_hip"], :2]) / 2.0
    assert np.allclose(hip_center, 0.0, atol=1e-6)


def test_translation_invariance():
    window_a = make_window_with_torso(5, hip_xy=(0.0, 0.0), shoulder_xy=(0.0, -2.0))
    window_b = make_window_with_torso(5, hip_xy=(10.0, 10.0), shoulder_xy=(10.0, 8.0))
    normalized_a = normalize_window(window_a)
    normalized_b = normalize_window(window_b)
    assert np.allclose(normalized_a, normalized_b, atol=1e-6)


def test_scale_invariance():
    window_small = make_window_with_torso(5, hip_xy=(0.0, 0.0), shoulder_xy=(0.0, -1.0))
    window_large = make_window_with_torso(5, hip_xy=(0.0, 0.0), shoulder_xy=(0.0, -3.0), other_offset=(0.9, 0.9))
    normalized_small = normalize_window(window_small)
    normalized_large = normalize_window(window_large)
    # both have a torso length of 1x their own scale, so their normalized torso vectors should match.
    nose_small = normalized_small[:, NAME_TO_INDEX["nose"], :2]
    nose_large = normalized_large[:, NAME_TO_INDEX["nose"], :2]
    assert np.allclose(nose_small, nose_large, atol=1e-6)


def test_degenerate_zero_torso_length_does_not_produce_nan_or_inf():
    window = make_window_with_torso(5, hip_xy=(1.0, 1.0), shoulder_xy=(1.0, 1.0))
    normalized = normalize_window(window)
    assert np.all(np.isfinite(normalized))


def test_visibility_channel_is_unchanged():
    window = make_window_with_torso(3, hip_xy=(0.0, 0.0), shoulder_xy=(0.0, -1.0))
    window[:, :, 2] = 0.77
    normalized = normalize_window(window)
    assert np.allclose(normalized[:, :, 2], 0.77)