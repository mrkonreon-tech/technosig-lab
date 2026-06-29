from __future__ import annotations

import math

import numpy as np

from .models import DetectionResult, DriftSearchConfig, SpectrogramConfig
from .spectrogram import make_spectrogram, robust_zscore


def drift_candidates(
    cfg: DriftSearchConfig,
    exclude_abs_lt: float | None = None,
) -> np.ndarray:
    candidates = np.arange(
        cfg.min_drift_hz_per_sec,
        cfg.max_drift_hz_per_sec + 0.5 * cfg.drift_step_hz_per_sec,
        cfg.drift_step_hz_per_sec,
        dtype=np.float64,
    )
    if exclude_abs_lt is not None:
        candidates = candidates[np.abs(candidates) >= exclude_abs_lt]
    return candidates


def search_drifting_tracks(
    power_db: np.ndarray,
    frame_times_sec: np.ndarray,
    freq_bins_hz: np.ndarray,
    cfg: DriftSearchConfig,
    exclude_drift_abs_lt: float | None = None,
) -> tuple[float, float, float]:
    if power_db.ndim != 2:
        raise ValueError("power_db must be frames x bins")
    if len(frame_times_sec) != power_db.shape[0]:
        raise ValueError("frame_times_sec length must match power_db frames")
    if len(freq_bins_hz) != power_db.shape[1]:
        raise ValueError("freq_bins_hz length must match power_db bins")
    if len(freq_bins_hz) < 2:
        raise ValueError("need at least two frequency bins")

    z = robust_zscore(power_db)
    usable_bins = np.where(
        (freq_bins_hz >= cfg.min_freq_hz) & (freq_bins_hz <= cfg.max_freq_hz)
    )[0]
    if len(usable_bins) == 0:
        raise ValueError("no usable frequency bins in search range")

    df = float(freq_bins_hz[1] - freq_bins_hz[0])
    if df == 0.0:
        raise ValueError("frequency bins are not monotonic")

    t0 = float(frame_times_sec[0])
    rel_t = frame_times_sec - t0
    frame_indices = np.arange(len(frame_times_sec))
    min_freq = float(np.min(freq_bins_hz))
    max_freq = float(np.max(freq_bins_hz))

    best_score = -np.inf
    best_f0 = 0.0
    best_drift = 0.0

    f0_candidates = freq_bins_hz[usable_bins]
    norm = math.sqrt(len(frame_indices))

    drifts = drift_candidates(cfg, exclude_abs_lt=exclude_drift_abs_lt)
    if drifts.size == 0:
        raise ValueError("no drift candidates remain after exclusion")

    for drift in drifts:
        track_freqs = f0_candidates[None, :] + drift * rel_t[:, None]
        valid = (track_freqs >= min_freq) & (track_freqs <= max_freq)
        bin_indices = np.round((track_freqs - freq_bins_hz[0]) / df).astype(np.int64)
        valid &= (bin_indices >= 0) & (bin_indices < len(freq_bins_hz))
        valid_cols = np.all(valid, axis=0)
        if not np.any(valid_cols):
            continue

        usable_f0 = f0_candidates[valid_cols]
        usable_bins_for_drift = bin_indices[:, valid_cols]
        gathered = z[frame_indices[:, None], usable_bins_for_drift]
        scores = np.sum(gathered, axis=0) / norm
        local_i = int(np.argmax(scores))
        local_score = float(scores[local_i])
        if local_score > best_score:
            best_score = local_score
            best_f0 = float(usable_f0[local_i])
            best_drift = float(drift)

    return best_score, best_f0, best_drift


def detect_from_waterfall(
    power_db: np.ndarray,
    frame_times_sec: np.ndarray,
    freq_bins_hz: np.ndarray,
    search_cfg: DriftSearchConfig,
) -> DetectionResult:
    if search_cfg.threshold is None:
        raise ValueError("search_cfg.threshold must be set before detection")
    score, f0, drift = search_drifting_tracks(
        power_db,
        frame_times_sec,
        freq_bins_hz,
        search_cfg,
    )
    detected = score >= search_cfg.threshold
    return DetectionResult(
        detected=detected,
        score=score,
        estimated_f0_hz=f0 if detected else None,
        estimated_drift_hz_per_sec=drift if detected else None,
    )


def detect_drifting_tone(
    x: np.ndarray,
    sample_rate_hz: float,
    spec_cfg: SpectrogramConfig,
    search_cfg: DriftSearchConfig,
) -> DetectionResult:
    power_db, times, freqs = make_spectrogram(x, sample_rate_hz, spec_cfg)
    return detect_from_waterfall(power_db, times, freqs, search_cfg)
