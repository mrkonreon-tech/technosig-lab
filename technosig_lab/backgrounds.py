from __future__ import annotations

import math

import numpy as np

from .models import BackgroundConfig, ChannelConfig, SignalTruth
from .signals import time_axis


def complex_awgn(n: int, power: float, rng: np.random.Generator) -> np.ndarray:
    return (
        rng.normal(0.0, math.sqrt(power / 2.0), size=n)
        + 1j * rng.normal(0.0, math.sqrt(power / 2.0), size=n)
    ).astype(np.complex128)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.abs(x) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-30)
    return x + complex_awgn(len(x), noise_power, rng)


def apply_phase_noise(
    x: np.ndarray,
    std_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if std_rad <= 0.0:
        return x
    phase = rng.normal(0.0, std_rad, size=x.shape)
    return x * np.exp(1j * phase)


def apply_multipath(x: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    if delay_samples <= 0 or gain == 0.0:
        return x
    delayed = np.zeros_like(x)
    delayed[delay_samples:] = x[:-delay_samples]
    return x + gain * delayed


def add_rfi_tone(
    x: np.ndarray,
    sample_rate_hz: float,
    freq_hz: float,
    amplitude: float,
    drift_hz_per_sec: float,
) -> np.ndarray:
    if amplitude <= 0.0:
        return x
    t = time_axis(sample_rate_hz, len(x) / sample_rate_hz)
    phase = 2.0 * np.pi * (freq_hz * t + 0.5 * drift_hz_per_sec * t * t)
    return x + amplitude * np.exp(1j * phase)


def add_impulse_rfi(
    x: np.ndarray,
    probability: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if probability <= 0.0 or amplitude <= 0.0:
        return x
    y = x.copy()
    mask = rng.random(size=x.shape) < probability
    impulses = amplitude * complex_awgn(len(x), power=1.0, rng=rng)
    y[mask] += impulses[mask]
    return y


def add_band_limited_rfi(
    x: np.ndarray,
    sample_rate_hz: float,
    center_hz: float,
    width_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if width_hz <= 0.0 or amplitude <= 0.0:
        return x
    n = len(x)
    noise = complex_awgn(n, power=1.0, rng=rng)
    spectrum = np.fft.fftshift(np.fft.fft(noise))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate_hz))
    mask = np.abs(freqs - center_hz) <= width_hz / 2.0
    spectrum[~mask] = 0.0
    band = np.fft.ifft(np.fft.ifftshift(spectrum))
    power = float(np.mean(np.abs(band) ** 2))
    if power > 0.0:
        band = band / math.sqrt(power)
    return x + amplitude * band


def apply_channel(
    x: np.ndarray,
    truth: SignalTruth,
    cfg: ChannelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    y = x.copy()
    y = apply_phase_noise(y, cfg.phase_noise_std_rad, rng)
    if cfg.multipath_enabled:
        y = apply_multipath(y, cfg.multipath_delay_samples, cfg.multipath_gain)
    if cfg.rfi_tone_enabled:
        y = add_rfi_tone(
            y,
            truth.sample_rate_hz,
            cfg.rfi_tone_freq_hz,
            cfg.rfi_tone_amplitude,
            cfg.rfi_tone_drift_hz_per_sec,
        )
    if cfg.impulse_rfi_enabled:
        y = add_impulse_rfi(
            y,
            cfg.impulse_probability,
            cfg.impulse_amplitude,
            rng,
        )
    return add_awgn(y, cfg.snr_db, rng)


def generate_noise_only(
    sample_rate_hz: float,
    duration_sec: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(round(sample_rate_hz * duration_sec))
    return complex_awgn(n, power=1.0, rng=rng)


def generate_background(
    sample_rate_hz: float,
    duration_sec: float,
    cfg: BackgroundConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(round(sample_rate_hz * duration_sec))
    x = complex_awgn(n, power=cfg.awgn_power, rng=rng)
    if cfg.rfi_tone_enabled:
        x = add_rfi_tone(
            x,
            sample_rate_hz,
            cfg.rfi_tone_freq_hz,
            cfg.rfi_tone_amplitude,
            cfg.rfi_tone_drift_hz_per_sec,
        )
    if cfg.impulse_rfi_enabled:
        x = add_impulse_rfi(
            x,
            cfg.impulse_probability,
            cfg.impulse_amplitude,
            rng,
        )
    if cfg.band_rfi_enabled:
        x = add_band_limited_rfi(
            x,
            sample_rate_hz,
            cfg.band_rfi_center_hz,
            cfg.band_rfi_width_hz,
            cfg.band_rfi_amplitude,
            rng,
        )
    return x.astype(np.complex128)

