import numpy as np
import pytest

from fallprecursor.data.labeling import label_frames, NORMAL, PRECURSOR, FALL


def test_no_fall_clip_is_all_normal():
    labels = label_frames(num_frames=50, onset_frame=None, impact_frame=None, precursor_window=10)
    assert np.all(labels == NORMAL)
    assert labels.shape == (50,)


def test_standard_fall_window_boundaries():
    labels = label_frames(num_frames=100, onset_frame=40, impact_frame=45, precursor_window=10)
    assert np.all(labels[0:30] == NORMAL)
    assert np.all(labels[30:40] == PRECURSOR)
    assert np.all(labels[40:46] == FALL)
    assert np.all(labels[46:100] == NORMAL)


def test_precursor_window_clips_at_clip_start():
    labels = label_frames(num_frames=100, onset_frame=5, impact_frame=8, precursor_window=20)
    assert np.all(labels[0:5] == PRECURSOR)
    assert np.all(labels[5:9] == FALL)


def test_onset_equal_impact_is_single_fall_frame():
    labels = label_frames(num_frames=20, onset_frame=10, impact_frame=10, precursor_window=5)
    assert labels[10] == FALL
    assert labels[9] == PRECURSOR
    assert labels[11] == NORMAL


@pytest.mark.parametrize("onset_frame,impact_frame", [(10, None), (None, 10)])
def test_only_one_of_onset_impact_set_raises(onset_frame, impact_frame):
    with pytest.raises(ValueError):
        label_frames(num_frames=20, onset_frame=onset_frame, impact_frame=impact_frame, precursor_window=5)


def test_impact_before_onset_raises():
    with pytest.raises(ValueError):
        label_frames(num_frames=20, onset_frame=10, impact_frame=5, precursor_window=5)


def test_impact_at_or_past_clip_end_raises():
    with pytest.raises(ValueError):
        label_frames(num_frames=20, onset_frame=15, impact_frame=20, precursor_window=5)


def test_non_positive_precursor_window_raises():
    with pytest.raises(ValueError):
        label_frames(num_frames=20, onset_frame=10, impact_frame=12, precursor_window=0)


def test_non_positive_num_frames_raises():
    with pytest.raises(ValueError):
        label_frames(num_frames=0, onset_frame=None, impact_frame=None, precursor_window=5)