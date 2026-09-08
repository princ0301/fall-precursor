import pandas as pd
import numpy as np
import pytest

from fallprecursor.evaluation.grouping import group_scores_by_video


def make_test_rows() -> pd.DataFrame:
    return pd.DataFrame([
        # fall video: 3 windows, onset at frame 80, window_length=30
        {"room": "Home_01", "video": "video (1)", "start_frame": 50, "window_length": 30, "onset_frame": 80, "fps": 25.0, "precursor_score": 0.2},
        {"room": "Home_01", "video": "video (1)", "start_frame": 0, "window_length": 30, "onset_frame": 80, "fps": 25.0, "precursor_score": 0.1},
        {"room": "Home_01", "video": "video (1)", "start_frame": 100, "window_length": 30, "onset_frame": 80, "fps": 25.0, "precursor_score": 0.9},
        # normal (no-fall) video
        {"room": "Home_01", "video": "video (2)", "start_frame": 0, "window_length": 30, "onset_frame": -1, "fps": 30.0, "precursor_score": 0.1},
        {"room": "Home_01", "video": "video (2)", "start_frame": 10, "window_length": 30, "onset_frame": -1, "fps": 30.0, "precursor_score": 0.15},
    ])


def test_separates_fall_and_normal_videos():
    fall_scores, fall_starts, fall_onsets, normal_scores, fps = group_scores_by_video(make_test_rows())
    assert len(fall_scores) == 1
    assert len(normal_scores) == 1


def test_fall_video_rows_are_sorted_by_start_frame():
    fall_scores, fall_starts, fall_onsets, _, _ = group_scores_by_video(make_test_rows())
    assert list(fall_starts[0]) == [0, 50, 100]
    assert list(fall_scores[0]) == [0.1, 0.2, 0.9]


def test_onset_frame_recovered_correctly():
    _, _, fall_onsets, _, _ = group_scores_by_video(make_test_rows())
    assert fall_onsets[0] == 80


def test_fps_is_mean_across_all_rows():
    _, _, _, _, fps = group_scores_by_video(make_test_rows())
    assert np.isclose(fps, (25.0 * 3 + 30.0 * 2) / 5)


def test_default_alert_position_is_start():
    _, fall_starts, _, _, _ = group_scores_by_video(make_test_rows())
    assert list(fall_starts[0]) == [0, 50, 100]


def test_alert_position_start_explicit():
    _, fall_starts, _, _, _ = group_scores_by_video(make_test_rows(), alert_position="start")
    assert list(fall_starts[0]) == [0, 50, 100]


def test_alert_position_end_shifts_by_window_length_minus_one():
    _, fall_starts, _, _, _ = group_scores_by_video(make_test_rows(), alert_position="end")
    # window_length=30, so end frame = start_frame + 29.
    assert list(fall_starts[0]) == [29, 79, 129]


def test_alert_position_center_shifts_by_half_window_length():
    _, fall_starts, _, _, _ = group_scores_by_video(make_test_rows(), alert_position="center")
    # window_length=30, so center frame = start_frame + 15.
    assert list(fall_starts[0]) == [15, 65, 115]


def test_unknown_alert_position_raises():
    with pytest.raises(ValueError):
        group_scores_by_video(make_test_rows(), alert_position="middle")