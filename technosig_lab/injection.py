from __future__ import annotations

import numpy as np


def inject_drifting_track_into_waterfall(
    power_db: np.ndarray,
    frame_times_sec: np.ndarray,
    freq_bins_hz: np.ndarray,
    f0_hz: float,
    drift_hz_per_sec: float,
    snr_like_db: float,
    width_bins: int = 1,
) -> np.ndarray:
    if len(freq_bins_hz) < 2:
        raise ValueError("need at least two frequency bins")
    y = np.asarray(power_db, dtype=np.float64).copy()
    noise_floor = float(np.median(y))
    injection_power_db = noise_floor + snr_like_db
    df = float(freq_bins_hz[1] - freq_bins_hz[0])
    t0 = float(frame_times_sec[0])
    rel_t = frame_times_sec - t0

    for i, t in enumerate(rel_t):
        f = f0_hz + drift_hz_per_sec * t
        bin_idx = int(round((f - freq_bins_hz[0]) / df))
        for k in range(-width_bins, width_bins + 1):
            j = bin_idx + k
            if 0 <= j < y.shape[1]:
                y[i, j] = max(y[i, j], injection_power_db)
    return y

