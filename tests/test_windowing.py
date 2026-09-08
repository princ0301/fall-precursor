import numpy as np
import pytest

from fallprecursor.data.windowing import make_windows
from fallprecursor.data.labeling import NORMAL, PRECURSOR, FALL


def make_dummy_keypoints(num_frames: int) -> np.ndarray:
    return np.arange(num_frames * 4 * 3, dtype=np.float32).reshape(num_frames, 4, 3)


def test_window_count_and_shape_no_remainder():
    keypoints = make_dummy_keypoints(20)
    labels = np.full(20, NORMAL, dtype=np.int64)
    windows, window_labels, start_frames = make_windows(keypoints, labels, window_length=5, stride=5)
    assert windows.shape == (4, 5, 4, 3)
    assert window_labels.shape == (4,)
    assert start_frames.shape == (4,)


def test_start_frames_match_expected_positions():
    keypoints = make_dummy_keypoints(20)
    labels = np.full(20, NORMAL, dtype=np.int64)
    _, _, start_frames = make_windows(keypoints, labels, window_length=5, stride=5)
    assert list(start_frames) == [0, 5, 10, 15]


def test_partial_trailing_window_is_dropped():
    keypoints = make_dummy_keypoints(12)
    labels = np.full(12, NORMAL, dtype=np.int64)
    windows, window_labels, start_frames = make_windows(keypoints, labels, window_length=5, stride=5)
    assert windows.shape[0] == 2
    assert window_labels.shape[0] == 2
    assert list(start_frames) == [0, 5]


def test_sequence_shorter_than_window_returns_empty():
    keypoints = make_dummy_keypoints(3)
    labels = np.full(3, NORMAL, dtype=np.int64)
    windows, window_labels, start_frames = make_windows(keypoints, labels, window_length=5, stride=5)
    assert windows.shape == (0, 5, 4, 3)
    assert window_labels.shape == (0,)
    assert start_frames.shape == (0,)


def test_window_content_matches_source_slice():
    keypoints = make_dummy_keypoints(10)
    labels = np.full(10, NORMAL, dtype=np.int64)
    windows, _, start_frames = make_windows(keypoints, labels, window_length=4, stride=2)
    assert np.array_equal(windows[0], keypoints[0:4])
    assert np.array_equal(windows[1], keypoints[2:6])
    assert start_frames[0] == 0
    assert start_frames[1] == 2


def test_clear_majority_label():
    keypoints = make_dummy_keypoints(5)
    labels = np.array([NORMAL, NORMAL, PRECURSOR, PRECURSOR, PRECURSOR], dtype=np.int64)
    _, window_labels, _ = make_windows(keypoints, labels, window_length=5, stride=5)
    assert window_labels[0] == PRECURSOR


def test_tie_breaks_toward_more_urgent_class():
    keypoints = make_dummy_keypoints(4)
    labels = np.array([NORMAL, NORMAL, FALL, FALL], dtype=np.int64)
    _, window_labels, _ = make_windows(keypoints, labels, window_length=4, stride=4)
    assert window_labels[0] == FALL


def test_three_way_tie_breaks_toward_fall():
    keypoints = make_dummy_keypoints(3)
    labels = np.array([NORMAL, PRECURSOR, FALL], dtype=np.int64)
    _, window_labels, _ = make_windows(keypoints, labels, window_length=3, stride=3)
    assert window_labels[0] == FALL


def test_mismatched_lengths_raise():
    keypoints = make_dummy_keypoints(10)
    labels = np.full(8, NORMAL, dtype=np.int64)
    with pytest.raises(ValueError):
        make_windows(keypoints, labels, window_length=4, stride=2)


def test_non_positive_window_length_raises():
    keypoints = make_dummy_keypoints(10)
    labels = np.full(10, NORMAL, dtype=np.int64)
    with pytest.raises(ValueError):
        make_windows(keypoints, labels, window_length=0, stride=2)


def test_non_positive_stride_raises():
    keypoints = make_dummy_keypoints(10)
    labels = np.full(10, NORMAL, dtype=np.int64)
    with pytest.raises(ValueError):
        make_windows(keypoints, labels, window_length=4, stride=0)