import numpy as np

from fallprecursor.data.labeling import NORMAL, PRECURSOR, FALL


def make_windows(
    keypoints: np.ndarray,
    frame_labels: np.ndarray,
    window_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a per-frame sequence into fixed-length windows with a stride.

    keypoints has shape (F, K, C), frame_labels has shape (F,). Returns
    (windows, window_labels, start_frames) where windows has shape
    (N, window_length, K, C), window_labels has shape (N,), and
    start_frames has shape (N,) giving each window's first frame index
    within the original sequence, needed to place a window in time for
    lead-time evaluation. A window's label is the majority class among its
    frames, with ties broken toward the more urgent class
    (fall > precursor > normal). Windows that would run past the end of the
    sequence are dropped rather than padded, so N can be smaller than what a
    naive (F - window_length) / stride formula would suggest whenever F does
    not divide evenly.
    """
    if keypoints.shape[0] != frame_labels.shape[0]:
        raise ValueError("keypoints and frame_labels must have the same number of frames")
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    num_frames = keypoints.shape[0]
    starts = range(0, num_frames - window_length + 1, stride)

    windows = np.stack([keypoints[start:start + window_length] for start in starts], axis=0) \
        if len(starts) > 0 else np.empty((0, window_length, *keypoints.shape[1:]), dtype=keypoints.dtype)
    window_labels = np.array(
        [_majority_label(frame_labels[start:start + window_length]) for start in starts],
        dtype=np.int64,
    )
    start_frames = np.array(list(starts), dtype=np.int64)

    return windows, window_labels, start_frames


def _majority_label(labels: np.ndarray) -> int:
    """Return the majority class in labels, ties broken toward FALL over PRECURSOR over NORMAL."""
    counts = np.bincount(labels, minlength=FALL + 1)
    max_count = counts.max()
    tied_classes = np.flatnonzero(counts == max_count)
    return int(tied_classes.max())