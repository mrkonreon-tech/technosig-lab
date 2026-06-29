from __future__ import annotations

import numpy as np

from .models import SignalTruth


def time_axis(sample_rate_hz: float, duration_sec: float) -> np.ndarray:
    n = int(round(sample_rate_hz * duration_sec))
    return np.arange(n, dtype=np.float64) / sample_rate_hz


def generate_drifting_tone(
    sample_rate_hz: float,
    duration_sec: float,
    f0_hz: float,
    drift_hz_per_sec: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, SignalTruth]:
    t = time_axis(sample_rate_hz, duration_sec)
    phase = 2.0 * np.pi * (f0_hz * t + 0.5 * drift_hz_per_sec * t * t)
    signal = amplitude * np.exp(1j * phase)
    truth = SignalTruth(
        kind="drifting_tone",
        sample_rate_hz=sample_rate_hz,
        duration_sec=duration_sec,
        f0_hz=f0_hz,
        drift_hz_per_sec=drift_hz_per_sec,
        amplitude=amplitude,
    )
    return signal.astype(np.complex128), truth

