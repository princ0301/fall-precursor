import numpy as np

from fallprecursor.evaluation.metrics import macro_f1, per_class_f1


def test_macro_f1_is_one_for_perfect_predictions():
    predictions = np.array([0, 1, 2, 0, 1, 2])
    labels = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(predictions, labels, num_classes=3) == 1.0


def test_macro_f1_treats_rare_and_common_classes_equally():
    # class 0 dominates but is predicted perfectly; class 1 is rare and always wrong.
    predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2])
    score = macro_f1(predictions, labels, num_classes=3)
    # class 0 f1=1.0, class 1 f1=0.0 (predicted but never true), class 2 f1=0.0 (true but never predicted)
    assert np.isclose(score, 1.0 / 3.0)


def test_absent_class_does_not_crash_and_scores_zero():
    predictions = np.array([0, 0, 1, 1])
    labels = np.array([0, 0, 1, 1])
    scores = per_class_f1(predictions, labels, num_classes=3)
    assert scores.shape == (3,)
    assert scores[2] == 0.0


def test_per_class_f1_shape():
    predictions = np.array([0, 1, 2])
    labels = np.array([0, 1, 1])
    scores = per_class_f1(predictions, labels, num_classes=3)
    assert scores.shape == (3,)