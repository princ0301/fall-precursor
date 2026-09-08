import numpy as np
import pytest

from fallprecursor.data.splits import split_by_subject, split_by_subject_two_way, assert_no_subject_leakage


def make_subject_ids(subjects_and_counts: dict[str, int]) -> np.ndarray:
    ids = []
    for subject, count in subjects_and_counts.items():
        ids.extend([subject] * count)
    return np.array(ids)


def test_split_covers_every_window_exactly_once():
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(20)})
    train_idx, val_idx, test_idx = split_by_subject(subject_ids, train_frac=0.6, val_frac=0.2, seed=0)
    all_indices = np.concatenate([train_idx, val_idx, test_idx])
    assert sorted(all_indices.tolist()) == list(range(len(subject_ids)))


def test_no_subject_appears_in_more_than_one_split():
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(20)})
    train_idx, val_idx, test_idx = split_by_subject(subject_ids, train_frac=0.6, val_frac=0.2, seed=0)
    assert_no_subject_leakage(subject_ids, train_idx, val_idx, test_idx)


def test_same_seed_gives_same_split():
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(20)})
    result_a = split_by_subject(subject_ids, train_frac=0.6, val_frac=0.2, seed=7)
    result_b = split_by_subject(subject_ids, train_frac=0.6, val_frac=0.2, seed=7)
    for a, b in zip(result_a, result_b):
        assert np.array_equal(a, b)


def test_different_seeds_can_give_different_splits():
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(20)})
    result_a = split_by_subject(subject_ids, train_frac=0.6, val_frac=0.2, seed=1)
    result_b = split_by_subject(subject_ids, train_frac=0.6, val_frac=0.2, seed=2)
    assert not np.array_equal(result_a[0], result_b[0])


def test_too_few_subjects_for_three_splits_raises():
    subject_ids = make_subject_ids({"S01": 5, "S02": 5})
    with pytest.raises(ValueError):
        split_by_subject(subject_ids, train_frac=0.6, val_frac=0.2, seed=0)


@pytest.mark.parametrize("train_frac,val_frac", [(0.0, 0.2), (0.6, 0.0), (0.7, 0.4), (1.0, 0.0)])
def test_invalid_fractions_raise(train_frac, val_frac):
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(20)})
    with pytest.raises(ValueError):
        split_by_subject(subject_ids, train_frac=train_frac, val_frac=val_frac, seed=0)


def test_assert_no_leakage_raises_on_overlapping_indices():
    subject_ids = np.array(["S01", "S01", "S02", "S02"])
    train_idx = np.array([0, 1])
    val_idx = np.array([1, 2])
    with pytest.raises(ValueError):
        assert_no_subject_leakage(subject_ids, train_idx, val_idx)


def test_assert_no_leakage_passes_on_disjoint_indices():
    subject_ids = np.array(["S01", "S01", "S02", "S02"])
    train_idx = np.array([0, 1])
    val_idx = np.array([2, 3])
    assert_no_subject_leakage(subject_ids, train_idx, val_idx)


def test_two_way_split_covers_every_window_exactly_once():
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(10)})
    train_idx, val_idx = split_by_subject_two_way(subject_ids, train_frac=0.7, seed=0)
    all_indices = np.concatenate([train_idx, val_idx])
    assert sorted(all_indices.tolist()) == list(range(len(subject_ids)))


def test_two_way_split_has_no_leakage():
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(10)})
    train_idx, val_idx = split_by_subject_two_way(subject_ids, train_frac=0.7, seed=0)
    assert_no_subject_leakage(subject_ids, train_idx, val_idx)


def test_two_way_split_too_few_subjects_raises():
    subject_ids = make_subject_ids({"S01": 5})
    with pytest.raises(ValueError):
        split_by_subject_two_way(subject_ids, train_frac=0.7, seed=0)


def test_two_way_split_invalid_train_frac_raises():
    subject_ids = make_subject_ids({f"S{i:02d}": 5 for i in range(10)})
    with pytest.raises(ValueError):
        split_by_subject_two_way(subject_ids, train_frac=0.0, seed=0)