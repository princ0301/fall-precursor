import numpy as np

from fallprecursor.evaluation.aggregation import aggregate_curves


def make_curve(false_alarm_rates, detection_rates, lead_times):
    return [
        {"threshold": 0.5, "false_alarm_rate": far, "detection_rate": det, "mean_lead_time_seconds": lt}
        for far, det, lt in zip(false_alarm_rates, detection_rates, lead_times)
    ]


def test_mean_and_std_computed_correctly_across_seeds():
    curve_a = make_curve([0.1], [0.5], [2.0])
    curve_b = make_curve([0.3], [0.7], [4.0])
    aggregated = aggregate_curves([curve_a, curve_b])

    assert np.isclose(aggregated[0]["false_alarm_rate_mean"], 0.2)
    assert np.isclose(aggregated[0]["detection_rate_mean"], 0.6)
    assert np.isclose(aggregated[0]["lead_time_mean"], 3.0)
    assert aggregated[0]["num_seeds_with_lead_time"] == 2


def test_none_lead_times_excluded_from_average_not_treated_as_zero():
    curve_a = make_curve([0.1], [0.0], [None])
    curve_b = make_curve([0.1], [0.5], [4.0])
    aggregated = aggregate_curves([curve_a, curve_b])

    assert np.isclose(aggregated[0]["lead_time_mean"], 4.0)
    assert aggregated[0]["num_seeds_with_lead_time"] == 1


def test_lead_time_mean_is_none_when_no_seed_detected_anything():
    curve_a = make_curve([0.0], [0.0], [None])
    curve_b = make_curve([0.0], [0.0], [None])
    aggregated = aggregate_curves([curve_a, curve_b])

    assert aggregated[0]["lead_time_mean"] is None
    assert aggregated[0]["num_seeds_with_lead_time"] == 0


def test_preserves_number_of_thresholds():
    curve_a = [
        {"threshold": t, "false_alarm_rate": 0.1, "detection_rate": 0.5, "mean_lead_time_seconds": 1.0}
        for t in [0.1, 0.5, 0.9]
    ]
    curve_b = [
        {"threshold": t, "false_alarm_rate": 0.2, "detection_rate": 0.6, "mean_lead_time_seconds": 2.0}
        for t in [0.1, 0.5, 0.9]
    ]
    aggregated = aggregate_curves([curve_a, curve_b])
    assert len(aggregated) == 3
    assert [point["threshold"] for point in aggregated] == [0.1, 0.5, 0.9]


def test_mismatched_curve_lengths_raise():
    curve_a = make_curve([0.1], [0.5], [1.0])
    curve_b = make_curve([0.1, 0.2], [0.5, 0.6], [1.0, 2.0])
    try:
        aggregate_curves([curve_a, curve_b])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_empty_curve_list_raises():
    try:
        aggregate_curves([])
        assert False, "expected ValueError"
    except ValueError:
        pass