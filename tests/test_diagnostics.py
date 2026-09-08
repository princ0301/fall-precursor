import numpy as np

from fallprecursor.evaluation.diagnostics import compute_video_stats, compute_fall_window_motion
from fallprecursor.pose.schema import NUM_LANDMARKS, NAME_TO_INDEX


def make_keypoints(num_frames: int) -> np.ndarray:
    return np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float64)


def test_detection_rate_reflects_missing_frames():
    keypoints = make_keypoints(10)
    keypoints[3] = np.nan
    keypoints[7] = np.nan
    stats = compute_video_stats(keypoints)
    assert np.isclose(stats["detection_rate"], 0.8)


def test_perfect_detection_rate_when_no_missing_frames():
    keypoints = make_keypoints(5)
    stats = compute_video_stats(keypoints)
    assert stats["detection_rate"] == 1.0


def test_torso_length_and_motion_are_none_with_fewer_than_two_valid_frames():
    keypoints = make_keypoints(3)
    keypoints[0] = np.nan
    keypoints[1] = np.nan
    stats = compute_video_stats(keypoints)
    assert stats["avg_torso_length"] is None
    assert stats["avg_motion"] is None


def test_torso_length_matches_known_geometry():
    keypoints = make_keypoints(2)
    keypoints[:, NAME_TO_INDEX["left_hip"], :2] = [0.0, 1.0]
    keypoints[:, NAME_TO_INDEX["right_hip"], :2] = [0.2, 1.0]
    keypoints[:, NAME_TO_INDEX["left_shoulder"], :2] = [0.0, 0.5]
    keypoints[:, NAME_TO_INDEX["right_shoulder"], :2] = [0.2, 0.5]
    stats = compute_video_stats(keypoints)
    assert np.isclose(stats["avg_torso_length"], 0.5)


def test_motion_is_zero_for_a_static_skeleton():
    keypoints = make_keypoints(4)
    keypoints[:, NAME_TO_INDEX["left_hip"], :2] = [0.0, 1.0]
    keypoints[:, NAME_TO_INDEX["right_hip"], :2] = [0.2, 1.0]
    stats = compute_video_stats(keypoints)
    assert np.isclose(stats["avg_motion"], 0.0)


def test_fall_window_motion_ignores_activity_outside_the_window():
    keypoints = make_keypoints(200)
    # huge motion far from onset (outside the window) - should not affect the result.
    for i in range(100):
        keypoints[i, NAME_TO_INDEX["left_hip"], :2] = [i * 0.1, 0.0]
        keypoints[i, NAME_TO_INDEX["right_hip"], :2] = [i * 0.1, 0.0]
    # small, consistent motion right at onset.
    onset_frame = 150
    for offset in range(-5, 6):
        keypoints[onset_frame + offset, NAME_TO_INDEX["left_hip"], :2] = [offset * 0.01, 0.0]
        keypoints[onset_frame + offset, NAME_TO_INDEX["right_hip"], :2] = [offset * 0.01, 0.0]

    motion = compute_fall_window_motion(keypoints, onset_frame, frames_before=10, frames_after=10)
    assert motion is not None
    assert motion < 0.05  # far smaller than the 0.1-per-frame motion outside the window


def test_fall_window_clips_at_video_start():
    keypoints = make_keypoints(20)
    motion = compute_fall_window_motion(keypoints, onset_frame=5, frames_before=45, frames_after=30)
    assert motion == 0.0  # static skeleton, but should not crash from window extending before frame 0


def test_fall_window_clips_at_video_end():
    keypoints = make_keypoints(20)
    motion = compute_fall_window_motion(keypoints, onset_frame=15, frames_before=45, frames_after=30)
    assert motion == 0.0


def test_fall_window_none_with_insufficient_valid_frames():
    keypoints = make_keypoints(10)
    keypoints[:] = np.nan
    motion = compute_fall_window_motion(keypoints, onset_frame=5, frames_before=45, frames_after=30)
    assert motion is None