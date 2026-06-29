from __future__ import annotations

from typing import Any

import numpy as np


def score_excess_ratios(top_score: float, quantiles: dict[str, float]) -> dict[str, float | None]:
    ratios: dict[str, float | None] = {}
    for name, threshold in quantiles.items():
        ratios[name] = (top_score / threshold) if threshold else None
    return ratios


def score_margins(top_score: float, quantiles: dict[str, float]) -> dict[str, float]:
    return {name: top_score - threshold for name, threshold in quantiles.items()}


def quantile_stability_gate(
    margins: dict[str, float],
    p9999_min_margin: float = 0.5,
    p999_min_margin: float = 1.0,
) -> str:
    if margins.get("p9999", float("-inf")) < p9999_min_margin:
        return "fail_borderline_tail"
    if margins.get("p999", float("-inf")) < p999_min_margin:
        return "weak"
    return "pass"


def score_gate_from_margin(top_score: float, p99: float) -> str:
    if top_score < p99:
        return "fail"
    margin = top_score - p99
    relative_margin = margin / abs(p99) if p99 else float("inf")
    if margin < 5.0 or relative_margin < 0.10:
        return "weak_pass"
    return "pass"


def drift_static_gate(top_drift_hz_per_sec: float, static_abs_hz_per_sec: float) -> str:
    if abs(top_drift_hz_per_sec) < static_abs_hz_per_sec:
        return "fail_static"
    return "pass_nonstatic"


def morphology_gate(candidate_summary: dict[str, Any]) -> str:
    if candidate_summary.get("has_narrowband_drifting", False):
        return "pass"
    return "fail_no_narrowband_drifting"


def candidate_strength(gates: dict[str, str]) -> str:
    if (
        gates.get("morphology", "").startswith("fail")
        or gates.get("drift", "").startswith("fail")
        or gates.get("quantile_stability") == "fail_borderline_tail"
    ):
        return "low"
    if gates.get("score_p99") == "weak_pass" or gates.get("neighbor_window", "").startswith("weak"):
        return "medium"
    return "high"


def summarize_score_distribution(scores: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize empty score distribution")
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def neighbor_window_gate(
    event_rank: int | None,
    event_score: float | None,
    neighbor_max_score: float | None,
    similar_fraction: float = 0.8,
) -> str:
    if event_rank is None or event_score is None or neighbor_max_score is None:
        return "not_run"
    if event_rank > 1:
        return "fail_neighbor_stronger"
    if neighbor_max_score >= event_score * similar_fraction:
        return "weak_neighbor_similar"
    return "pass_locally_distinct"
