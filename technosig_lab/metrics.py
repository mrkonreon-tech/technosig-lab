from __future__ import annotations


def detection_probability(detections: int, trials: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    return detections / trials


def false_alarm_rate(false_alarms: int, trials: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    return false_alarms / trials

