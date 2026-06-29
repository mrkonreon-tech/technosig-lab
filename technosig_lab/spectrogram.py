from __future__ import annotations

import numpy as np

from .models import SpectrogramConfig


def make_spectrogram(
    x: np.ndarray,
    sample_rate_hz: float,
    cfg: SpectrogramConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fft_size = cfg.fft_size
    hop = cfg.hop_size
    if len(x) < fft_size:
        raise ValueError("signal shorter than fft_size")

    window = np.hanning(fft_size)
    coherent_gain = max(float(np.sum(window) / fft_size), 1e-12)
    frames = []
    times = []

    for start in range(0, len(x) - fft_size + 1, hop):
        frame = x[start : start + fft_size] * window
        spectrum = np.fft.fftshift(np.fft.fft(frame))
        power = (np.abs(spectrum) / (fft_size * coherent_gain)) ** 2
        frames.append(power)
        times.append((start + fft_size / 2.0) / sample_rate_hz)

    power_linear = np.asarray(frames, dtype=np.float64)
    power_db = 10.0 * np.log10(power_linear + 1e-30)
    freq_bins = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    return power_db, np.asarray(times, dtype=np.float64), freq_bins


def robust_zscore(power_db: np.ndarray) -> np.ndarray:
    median = np.median(power_db)
    mad = np.median(np.abs(power_db - median)) + 1e-12
    sigma = 1.4826 * mad
    return (power_db - median) / sigma


def detrend_waterfall(power_db: np.ndarray) -> np.ndarray:
    """Remove simple per-frequency and per-frame medians before peak finding."""
    x = np.asarray(power_db, dtype=np.float64)
    freq_med = np.median(x, axis=0, keepdims=True)
    y = x - freq_med
    time_med = np.median(y, axis=1, keepdims=True)
    return y - time_med

