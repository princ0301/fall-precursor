import numpy as np

from fallprecursor.pose.visualization import draw_skeleton
from fallprecursor.pose.schema import NUM_LANDMARKS


def test_draw_skeleton_modifies_the_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    points = [(50, 50)] * NUM_LANDMARKS
    draw_skeleton(frame, points)
    assert np.any(frame != 0)


def test_draw_skeleton_does_not_crash_on_edge_coordinates():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    points = [(0, 0)] * NUM_LANDMARKS
    draw_skeleton(frame, points)