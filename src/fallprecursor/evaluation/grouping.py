import numpy as np
import pandas as pd

NO_FALL_SENTINEL = -1


def group_scores_by_video(test_rows: pd.DataFrame, alert_position: str = "start") -> tuple[list, list, list, list, float]:
    """Group per-window precursor scores into per-video sequences, sorted in time order.

    test_rows must have columns: room, video, start_frame, onset_frame, fps,
    precursor_score, and (if alert_position is not "start") window_length.
    Videos where onset_frame equals NO_FALL_SENTINEL are treated as no-fall
    videos; all others are treated as fall videos. Returns (fall_video_scores,
    fall_video_start_frames, fall_video_onset_frames, normal_video_scores,
    fps), where fps is the mean fps across all test videos, since
    lead_time_vs_false_alarm_curve needs a single fps value.

    alert_position controls which frame within each window is used as its
    alert timestamp for lead-time purposes. "start" credits a window's
    detection as happening at its first frame, which inflates apparent lead
    time for longer windows whose informative content may only live near
    the end. "end" credits it at the window's last frame, which
    structurally under-credits correctly-classified precursor windows,
    since a window that exactly matches the true precursor period ends at
    onset by construction, near-zero end-frame lead time regardless of
    whether the detection is genuinely useful. "center" (the midpoint)
    avoids both distortions and is the recommended default for comparing
    across different window lengths.
    """
    fps = float(test_rows["fps"].mean())

    fall_video_scores, fall_video_start_frames, fall_video_onset_frames = [], [], []
    normal_video_scores = []

    for _, video_rows in test_rows.groupby(["room", "video"]):
        video_rows = video_rows.sort_values("start_frame")
        scores = video_rows["precursor_score"].to_numpy()
        onset_frame = int(video_rows["onset_frame"].iloc[0])
        frame_positions = _compute_frame_positions(video_rows, alert_position)

        if onset_frame == NO_FALL_SENTINEL:
            normal_video_scores.append(scores)
        else:
            fall_video_scores.append(scores)
            fall_video_start_frames.append(frame_positions)
            fall_video_onset_frames.append(onset_frame)

    return fall_video_scores, fall_video_start_frames, fall_video_onset_frames, normal_video_scores, fps


def _compute_frame_positions(video_rows: pd.DataFrame, alert_position: str) -> np.ndarray:
    if alert_position == "start":
        return video_rows["start_frame"].to_numpy()
    if alert_position == "center":
        return (video_rows["start_frame"] + video_rows["window_length"] // 2).to_numpy()
    if alert_position == "end":
        return (video_rows["start_frame"] + video_rows["window_length"] - 1).to_numpy()
    raise ValueError(f"unknown alert_position: {alert_position}")