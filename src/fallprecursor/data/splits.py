import numpy as np


def split_by_subject(
    subject_ids: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split window indices into train/val/test by subject, not by window.

    subject_ids has shape (N,), one subject id per window. Splitting is done
    on the set of unique subjects first, then windows are assigned according
    to which split their subject fell into, so a single subject's windows
    never appear in more than one split. Returns three index arrays into the
    original subject_ids array.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must each be in (0, 1) and sum to less than 1")

    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(unique_subjects)

    num_subjects = len(shuffled_subjects)
    num_train = round(num_subjects * train_frac)
    num_val = round(num_subjects * val_frac)
    num_test = num_subjects - num_train - num_val

    if num_train == 0 or num_val == 0 or num_test == 0:
        raise ValueError(
            f"not enough unique subjects ({num_subjects}) to form three non-empty splits "
            f"with train_frac={train_frac}, val_frac={val_frac}"
        )

    train_subjects = shuffled_subjects[:num_train]
    val_subjects = shuffled_subjects[num_train:num_train + num_val]
    test_subjects = shuffled_subjects[num_train + num_val:]

    train_indices = np.flatnonzero(np.isin(subject_ids, train_subjects))
    val_indices = np.flatnonzero(np.isin(subject_ids, val_subjects))
    test_indices = np.flatnonzero(np.isin(subject_ids, test_subjects))

    assert_no_subject_leakage(subject_ids, train_indices, val_indices, test_indices)
    return train_indices, val_indices, test_indices


def split_by_subject_two_way(
    subject_ids: np.ndarray,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split window indices into train/val by subject, with no test split.

    Used when the test set is not a subset of subject_ids at all, e.g. an
    entirely separate held-out room in a cross-dataset generalization test.
    """
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0, 1)")

    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(unique_subjects)

    num_subjects = len(shuffled_subjects)
    num_train = round(num_subjects * train_frac)

    if num_train == 0 or num_train == num_subjects:
        raise ValueError(
            f"not enough unique subjects ({num_subjects}) to form two non-empty splits "
            f"with train_frac={train_frac}"
        )

    train_subjects = shuffled_subjects[:num_train]
    val_subjects = shuffled_subjects[num_train:]

    train_indices = np.flatnonzero(np.isin(subject_ids, train_subjects))
    val_indices = np.flatnonzero(np.isin(subject_ids, val_subjects))

    assert_no_subject_leakage(subject_ids, train_indices, val_indices)
    return train_indices, val_indices


def assert_no_subject_leakage(subject_ids: np.ndarray, *index_groups: np.ndarray) -> None:
    """Raise if any subject appears in more than one of the given index groups."""
    subject_sets = [set(subject_ids[indices]) for indices in index_groups]
    for i in range(len(subject_sets)):
        for j in range(i + 1, len(subject_sets)):
            overlap = subject_sets[i] & subject_sets[j]
            if overlap:
                raise ValueError(f"subject leakage detected across splits: {overlap}")