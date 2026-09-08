from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

NUM_JOINTS = 33


def parse_upfall_image_timestamp(filename: str) -> float:
    """Parse a UP-Fall image filename into a timestamp in seconds since the epoch.

    Filenames look like '2018-07-04T12_09_23.419628.png' - ISO 8601 with
    underscores in place of colons in the time portion. The absolute epoch
    is not meaningful here; only differences between timestamps are used,
    to recover frame ordering and real elapsed time (and hence fps) from
    filenames alone.
    """
    stem = Path(filename).stem
    date_part, time_part = stem.split("T")
    time_part = time_part.replace("_", ":")
    return datetime.fromisoformat(f"{date_part}T{time_part}").timestamp()


def load_upfall_fall_csv(csv_path: Path) -> tuple[np.ndarray, int | None, int | None]:
    """Load one of Koffi et al.'s improved UP-Fall skeleton CSVs.

    Returns (keypoints, onset_frame, impact_frame). keypoints has shape
    (F, 33, 3) with channels (x, y, visibility). The source data provides
    real (x, y, z) coordinates and no visibility/confidence score; z is
    dropped and visibility is set to a constant 1.0 throughout, to match
    the (x, y, visibility) format used elsewhere in this project without
    requiring architecture changes. onset_frame is the first frame labeled
    impact (LABEL == 1) and impact_frame is the last; both are None if no
    frame is labeled impact at all, treating the trial as fall-free.
    """
    df = pd.read_csv(csv_path)
    if "LABEL" not in df.columns:
        raise ValueError(f"no LABEL column found in {csv_path}")

    num_frames = len(df)

    keypoints = np.ones((num_frames, NUM_JOINTS, 3), dtype=np.float32)
    for joint_index in range(NUM_JOINTS):
        keypoints[:, joint_index, 0] = df[f"Joint{joint_index + 1}_X"].to_numpy()
        keypoints[:, joint_index, 1] = df[f"Joint{joint_index + 1}_Y"].to_numpy()

    impact_indices = np.flatnonzero(df["LABEL"].to_numpy() == 1)
    if len(impact_indices) == 0:
        return keypoints, None, None
    return keypoints, int(impact_indices[0]), int(impact_indices[-1])