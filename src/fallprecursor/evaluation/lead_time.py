import numpy as np


def first_alert_frame(scores: np.ndarray, window_start_frames: np.ndarray, threshold: float) -> int | None:
    """Return the start frame of the first window whose score meets threshold, or None if none do."""
    above_threshold = np.flatnonzero(scores >= threshold)
    if len(above_threshold) == 0:
        return None
    return int(window_start_frames[above_threshold[0]])


def lead_time_frames(
    scores: np.ndarray,
    window_start_frames: np.ndarray,
    onset_frame: int,
    threshold: float,
) -> int | None:
    """Return onset_frame minus the first alert frame, or None if no alert fired before onset.

    An alert that fires at or after onset counts as a miss for lead-time
    purposes, since it provides no warning: the fall has already begun.
    """
    alert_frame = first_alert_frame(scores, window_start_frames, threshold)
    if alert_frame is None or alert_frame >= onset_frame:
        return None
    return onset_frame - alert_frame


def false_alarm_rate(scores_per_normal_video: list[np.ndarray], threshold: float) -> float:
    """Fraction of no-fall videos where at least one window scores above threshold."""
    if len(scores_per_normal_video) == 0:
        return 0.0
    num_false_alarms = sum(1 for scores in scores_per_normal_video if np.any(scores >= threshold))
    return num_false_alarms / len(scores_per_normal_video)


def mean_lead_time_seconds(lead_times_in_frames: list[int | None], fps: float) -> float | None:
    """Mean lead time in seconds across fall videos that received a warning.

    Returns None if no fall video received a warning at this threshold,
    since a mean over zero warned videos is undefined, not zero.
    """
    warned = [frames for frames in lead_times_in_frames if frames is not None]
    if len(warned) == 0:
        return None
    return float(np.mean(warned)) / fps


def lead_time_vs_false_alarm_curve(
    fall_video_scores: list[np.ndarray],
    fall_video_start_frames: list[np.ndarray],
    fall_video_onset_frames: list[int],
    normal_video_scores: list[np.ndarray],
    thresholds: np.ndarray,
    fps: float,
) -> list[dict]:
    """Sweep threshold, returning false alarm rate, mean lead time, and detection rate at each.

    detection_rate is the fraction of fall videos that received any warning
    before onset at that threshold, independent of how much lead time it
    gave; mean_lead_time_seconds is computed only over those detected.
    """
    curve = []
    for threshold in thresholds:
        lead_times = [
            lead_time_frames(scores, starts, onset, threshold)
            for scores, starts, onset in zip(fall_video_scores, fall_video_start_frames, fall_video_onset_frames)
        ]
        num_fall_videos = len(lead_times)
        detection_rate = (
            sum(1 for frames in lead_times if frames is not None) / num_fall_videos
            if num_fall_videos > 0 else 0.0
        )

        curve.append({
            "threshold": float(threshold),
            "false_alarm_rate": false_alarm_rate(normal_video_scores, threshold),
            "mean_lead_time_seconds": mean_lead_time_seconds(lead_times, fps),
            "detection_rate": detection_rate,
        })
    return curve