import numpy as np

from fallprecursor.evaluation.lead_time import (
    first_alert_frame,
    lead_time_frames,
    false_alarm_rate,
    mean_lead_time_seconds,
    lead_time_vs_false_alarm_curve,
)


def test_first_alert_frame_returns_first_frame_above_threshold():
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.3])
    starts = np.array([0, 10, 20, 30, 40])
    assert first_alert_frame(scores, starts, threshold=0.5) == 20


def test_first_alert_frame_none_when_nothing_crosses_threshold():
    scores = np.array([0.1, 0.2, 0.3])
    starts = np.array([0, 10, 20])
    assert first_alert_frame(scores, starts, threshold=0.9) is None


def test_lead_time_frames_correct_when_alert_precedes_onset():
    scores = np.array([0.1, 0.9, 0.9])
    starts = np.array([0, 50, 100])
    lead_time = lead_time_frames(scores, starts, onset_frame=80, threshold=0.5)
    assert lead_time == 30


def test_lead_time_frames_none_when_alert_fires_at_or_after_onset():
    scores = np.array([0.1, 0.1, 0.9])
    starts = np.array([0, 50, 100])
    lead_time = lead_time_frames(scores, starts, onset_frame=100, threshold=0.5)
    assert lead_time is None


def test_lead_time_frames_none_when_no_alert_fires():
    scores = np.array([0.1, 0.1, 0.1])
    starts = np.array([0, 50, 100])
    lead_time = lead_time_frames(scores, starts, onset_frame=100, threshold=0.5)
    assert lead_time is None


def test_false_alarm_rate_counts_videos_not_windows():
    # video 1 has 3 windows above threshold, video 2 has 0 - both count as one false alarm each or none.
    normal_video_scores = [
        np.array([0.9, 0.9, 0.9]),
        np.array([0.1, 0.1]),
    ]
    rate = false_alarm_rate(normal_video_scores, threshold=0.5)
    assert rate == 0.5


def test_false_alarm_rate_empty_list_is_zero():
    assert false_alarm_rate([], threshold=0.5) == 0.0


def test_mean_lead_time_seconds_converts_frames_to_seconds():
    lead_times = [30, 60, None]
    mean_seconds = mean_lead_time_seconds(lead_times, fps=30.0)
    assert np.isclose(mean_seconds, 1.5)  # mean(30, 60) = 45 frames = 1.5s at 30fps


def test_mean_lead_time_seconds_none_when_no_video_warned():
    assert mean_lead_time_seconds([None, None], fps=30.0) is None


def test_curve_has_one_entry_per_threshold_with_expected_keys():
    fall_scores = [np.array([0.9, 0.9])]
    fall_starts = [np.array([0, 50])]
    fall_onsets = [80]
    normal_scores = [np.array([0.1, 0.1])]
    thresholds = np.array([0.3, 0.7])

    curve = lead_time_vs_false_alarm_curve(
        fall_scores, fall_starts, fall_onsets, normal_scores, thresholds, fps=30.0
    )

    assert len(curve) == 2
    assert set(curve[0].keys()) == {"threshold", "false_alarm_rate", "mean_lead_time_seconds", "detection_rate"}


def test_curve_detection_rate_and_lead_time_are_consistent_at_a_given_threshold():
    fall_scores = [np.array([0.9, 0.1]), np.array([0.1, 0.1])]
    fall_starts = [np.array([0, 50]), np.array([0, 50])]
    fall_onsets = [80, 80]
    normal_scores = []
    thresholds = np.array([0.5])

    curve = lead_time_vs_false_alarm_curve(
        fall_scores, fall_starts, fall_onsets, normal_scores, thresholds, fps=30.0
    )

    # only the first fall video crosses the threshold, so detection rate is 1/2.
    assert curve[0]["detection_rate"] == 0.5
    assert np.isclose(curve[0]["mean_lead_time_seconds"], 80 / 30.0)