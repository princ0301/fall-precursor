import numpy as np

NORMAL = 0
PRECURSOR = 1
FALL = 2


def label_frames(
    num_frames: int,
    onset_frame: int | None,
    impact_frame: int | None,
    precursor_window: int,
) -> np.ndarray:
    """Assign a normal/precursor/fall label to every frame of a clip.

    onset_frame is the first frame of loss-of-balance, impact_frame is the
    frame of ground impact. Both are None for clips containing no fall, in
    which case every frame is labeled normal. Frames in
    [onset_frame - precursor_window, onset_frame) are labeled precursor,
    frames in [onset_frame, impact_frame] are labeled fall, everything else
    is normal. The precursor window is clipped at the start of the clip
    rather than raising an error, since real recordings do not guarantee a
    full window of lead-up footage.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    labels = np.full(num_frames, NORMAL, dtype=np.int64)

    if onset_frame is None and impact_frame is None:
        return labels

    if onset_frame is None or impact_frame is None:
        raise ValueError("onset_frame and impact_frame must both be set or both be None")
    if not (0 <= onset_frame <= impact_frame < num_frames):
        raise ValueError("expected 0 <= onset_frame <= impact_frame < num_frames")
    if precursor_window <= 0:
        raise ValueError("precursor_window must be positive")

    precursor_start = max(0, onset_frame - precursor_window)
    labels[precursor_start:onset_frame] = PRECURSOR
    labels[onset_frame:impact_frame + 1] = FALL
    return labels