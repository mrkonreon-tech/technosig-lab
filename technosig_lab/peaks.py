from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .spectrogram import detrend_waterfall, robust_zscore


@dataclass
class Cluster:
    points: list[tuple[int, int]]


def local_peak_mask(z: np.ndarray, threshold_z: float) -> np.ndarray:
    mask = z >= threshold_z
    if z.shape[0] < 3 or z.shape[1] < 3:
        return mask

    center = z[1:-1, 1:-1]
    local = np.ones_like(center, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            local &= center >= z[1 + di : z.shape[0] - 1 + di, 1 + dj : z.shape[1] - 1 + dj]

    out = np.zeros_like(mask, dtype=bool)
    out[1:-1, 1:-1] = mask[1:-1, 1:-1] & local
    return out


def connected_peak_clusters(mask: np.ndarray, adjacent_frame_bin_gap: int = 8) -> list[Cluster]:
    visited = np.zeros_like(mask, dtype=bool)
    clusters: list[Cluster] = []
    rows, cols = mask.shape

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            points: list[tuple[int, int]] = []
            while queue:
                pr, pc = queue.popleft()
                points.append((pr, pc))
                for dr in (-1, 0, 1):
                    dc_range = range(-1, 2) if dr == 0 else range(-adjacent_frame_bin_gap, adjacent_frame_bin_gap + 1)
                    for dc in dc_range:
                        if dr == 0 and dc == 0:
                            continue
                        nr = pr + dr
                        nc = pc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and mask[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            clusters.append(Cluster(points=points))
    return clusters


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float]:
    if len(np.unique(x)) < 2:
        return None, float("inf")
    coeff = np.polyfit(x.astype(np.float64), y.astype(np.float64), deg=1)
    pred = coeff[0] * x + coeff[1]
    residual_std = float(np.std(y - pred))
    return float(coeff[0]), residual_std


def classify_morphology(
    frames: np.ndarray,
    bins: np.ndarray,
    frame_times_sec: np.ndarray,
    freq_bins_hz: np.ndarray,
) -> tuple[str, float | None, float]:
    frame_span = int(np.max(frames) - np.min(frames) + 1)
    bin_span = int(np.max(bins) - np.min(bins) + 1)

    if len(frames) < 2:
        return "sparse", None, float("inf")
    if frame_span <= 2 and bin_span >= 8:
        return "broadband", None, 0.0

    slope_bins_per_frame, residual_bins = _linear_fit(frames, bins)
    if slope_bins_per_frame is None:
        if bin_span >= 8:
            return "broadband", None, residual_bins
        return "sparse", None, residual_bins

    time_span = float(frame_times_sec[np.max(frames)] - frame_times_sec[np.min(frames)])
    frame_span_steps = max(frame_span - 1, 1)
    dt_per_frame = time_span / frame_span_steps if abs(time_span) > 1e-12 else 0.0
    df = float(freq_bins_hz[1] - freq_bins_hz[0]) if len(freq_bins_hz) > 1 else 0.0
    drift_est = (
        slope_bins_per_frame * df / dt_per_frame
        if abs(dt_per_frame) > 1e-12
        else 0.0
    )

    line_like = residual_bins <= 2.0
    if line_like and bin_span <= 4:
        return (
            "narrowband_static" if abs(drift_est) < abs(df) * 0.5 else "narrowband_drifting",
            drift_est,
            residual_bins,
        )
    if line_like:
        return "narrowband_drifting", drift_est, residual_bins
    if bin_span >= max(8, frame_span):
        return "broadband", drift_est, residual_bins
    return "sparse", drift_est, residual_bins


def summarize_cluster(
    cluster_id: int,
    cluster: Cluster,
    z: np.ndarray,
    frame_times_sec: np.ndarray,
    freq_bins_hz: np.ndarray,
) -> dict:
    frames = np.asarray([p[0] for p in cluster.points], dtype=np.int64)
    bins = np.asarray([p[1] for p in cluster.points], dtype=np.int64)
    values = z[frames, bins]
    morphology, drift_est, residual_bins = classify_morphology(
        frames,
        bins,
        frame_times_sec,
        freq_bins_hz,
    )
    score = float(np.sum(values) * np.sqrt(len(values)))
    residual_value = residual_bins if math.isfinite(residual_bins) else None
    return {
        "id": cluster_id,
        "n_peaks": int(len(cluster.points)),
        "frame_min": int(np.min(frames)),
        "frame_max": int(np.max(frames)),
        "bin_min": int(np.min(bins)),
        "bin_max": int(np.max(bins)),
        "time_min_sec": float(frame_times_sec[np.min(frames)]),
        "time_max_sec": float(frame_times_sec[np.max(frames)]),
        "freq_min_hz": float(freq_bins_hz[np.min(bins)]),
        "freq_max_hz": float(freq_bins_hz[np.max(bins)]),
        "z_max": float(np.max(values)),
        "z_mean": float(np.mean(values)),
        "score": score,
        "drift_est_hz_per_sec": drift_est,
        "morphology": morphology,
        "line_residual_bins": residual_value,
    }


def find_candidate_clusters(
    power_db: np.ndarray,
    frame_times_sec: np.ndarray,
    freq_bins_hz: np.ndarray,
    peak_threshold_z: float = 6.0,
    top_n: int = 20,
) -> dict:
    detrended = detrend_waterfall(power_db)
    z = robust_zscore(detrended)
    peaks = z >= peak_threshold_z
    clusters = connected_peak_clusters(peaks)
    summaries = [
        summarize_cluster(i, cluster, z, frame_times_sec, freq_bins_hz)
        for i, cluster in enumerate(clusters)
    ]
    summaries.sort(key=lambda item: item["score"], reverse=True)
    return {
        "n_peaks": int(np.count_nonzero(peaks)),
        "n_clusters": int(len(clusters)),
        "peak_threshold_z": peak_threshold_z,
        "top_clusters": summaries[:top_n],
    }
