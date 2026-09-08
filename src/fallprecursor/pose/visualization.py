import cv2
import numpy as np

from fallprecursor.pose.schema import POSE_CONNECTIONS


def draw_skeleton(frame: np.ndarray, points_xy_pixels: list[tuple[int, int]]) -> None:
    """Draw skeleton connections and joints onto frame in place, given pixel-space (x, y) points."""
    for start_index, end_index in POSE_CONNECTIONS:
        cv2.line(frame, points_xy_pixels[start_index], points_xy_pixels[end_index], (0, 255, 0), 2)
    for point in points_xy_pixels:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)