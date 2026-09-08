import argparse
from pathlib import Path

import numpy as np

from fallprecursor.evaluation.diagnostics import compute_video_stats, compute_fall_window_motion

NO_FALL_SENTINEL = -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()

    room_dirs = sorted(p for p in args.interim_dir.iterdir() if p.is_dir())

    header = (
        f"{'room':<15} {'videos':>7} {'fall_videos':>12} {'avg_frames':>11} {'detection_rate':>15} "
        f"{'avg_torso_len':>14} {'avg_motion':>11} {'fall_motion':>12} {'fall_motion_norm':>17}"
    )
    print(header)

    for room_dir in room_dirs:
        stats = summarize_room(room_dir)
        print(
            f"{room_dir.name:<15} {stats['num_videos']:>7} {stats['num_fall_videos']:>12} "
            f"{stats['avg_frames']:>11.1f} {stats['detection_rate']:>14.1%} "
            f"{stats['avg_torso_length']:>14.4f} {stats['avg_motion']:>11.4f} "
            f"{stats['fall_motion']:>12.4f} {stats['fall_motion_normalized']:>17.4f}"
        )


def summarize_room(room_dir: Path) -> dict:
    npz_paths = sorted(room_dir.glob("*.npz"))

    num_fall_videos = 0
    frame_counts = []
    detection_rates = []
    torso_lengths = []
    motions = []
    fall_motions = []
    fall_motions_normalized = []

    for npz_path in npz_paths:
        data = np.load(npz_path)
        keypoints = data["keypoints"]
        onset_frame = int(data["onset_frame"])

        frame_counts.append(keypoints.shape[0])
        video_stats = compute_video_stats(keypoints)
        detection_rates.append(video_stats["detection_rate"])
        if video_stats["avg_torso_length"] is not None:
            torso_lengths.append(video_stats["avg_torso_length"])
            motions.append(video_stats["avg_motion"])

        if onset_frame != NO_FALL_SENTINEL:
            num_fall_videos += 1
            fall_motion = compute_fall_window_motion(keypoints, onset_frame)
            if fall_motion is not None and video_stats["avg_torso_length"]:
                fall_motions.append(fall_motion)
                fall_motions_normalized.append(fall_motion / video_stats["avg_torso_length"])

    return {
        "num_videos": len(npz_paths),
        "num_fall_videos": num_fall_videos,
        "avg_frames": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "detection_rate": float(np.mean(detection_rates)) if detection_rates else 0.0,
        "avg_torso_length": float(np.mean(torso_lengths)) if torso_lengths else 0.0,
        "avg_motion": float(np.mean(motions)) if motions else 0.0,
        "fall_motion": float(np.mean(fall_motions)) if fall_motions else 0.0,
        "fall_motion_normalized": float(np.mean(fall_motions_normalized)) if fall_motions_normalized else 0.0,
    }


if __name__ == "__main__":
    main()