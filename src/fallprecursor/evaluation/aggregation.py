import numpy as np


def aggregate_curves(curves: list[list[dict]]) -> list[dict]:
    """Aggregate lead-time-vs-false-alarm curves from multiple seeds into mean/std per threshold.

    Every curve must have the same length and the same threshold at each
    index, which holds when all curves were produced with the same
    thresholds array. mean_lead_time_seconds is averaged only over seeds
    where a warning actually occurred at that threshold, since a missed
    detection is not zero lead time, it is undefined; num_seeds_with_lead_time
    reports how many seeds contributed to that average, since a mean over
    very few seeds should be read with caution.
    """
    if len(curves) == 0:
        raise ValueError("at least one curve is required")

    num_thresholds = len(curves[0])
    for curve in curves:
        if len(curve) != num_thresholds:
            raise ValueError("all curves must have the same number of thresholds")

    aggregated = []
    for i in range(num_thresholds):
        threshold = curves[0][i]["threshold"]
        false_alarm_rates = [curve[i]["false_alarm_rate"] for curve in curves]
        detection_rates = [curve[i]["detection_rate"] for curve in curves]
        lead_times = [curve[i]["mean_lead_time_seconds"] for curve in curves if curve[i]["mean_lead_time_seconds"] is not None]

        aggregated.append({
            "threshold": threshold,
            "false_alarm_rate_mean": float(np.mean(false_alarm_rates)),
            "false_alarm_rate_std": float(np.std(false_alarm_rates)),
            "detection_rate_mean": float(np.mean(detection_rates)),
            "detection_rate_std": float(np.std(detection_rates)),
            "lead_time_mean": float(np.mean(lead_times)) if lead_times else None,
            "lead_time_std": float(np.std(lead_times)) if lead_times else None,
            "num_seeds_with_lead_time": len(lead_times),
        })

    return aggregated