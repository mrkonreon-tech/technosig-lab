from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SignalTruth:
    kind: str
    sample_rate_hz: float
    duration_sec: float
    f0_hz: float
    drift_hz_per_sec: float
    amplitude: float


@dataclass(frozen=True)
class ChannelConfig:
    snr_db: float
    phase_noise_std_rad: float = 0.0
    multipath_enabled: bool = False
    multipath_delay_samples: int = 0
    multipath_gain: float = 0.0
    rfi_tone_enabled: bool = False
    rfi_tone_freq_hz: float = 0.0
    rfi_tone_amplitude: float = 0.0
    rfi_tone_drift_hz_per_sec: float = 0.0
    impulse_rfi_enabled: bool = False
    impulse_probability: float = 0.0
    impulse_amplitude: float = 0.0


@dataclass(frozen=True)
class BackgroundConfig:
    name: str
    awgn_power: float = 1.0
    rfi_tone_enabled: bool = False
    rfi_tone_freq_hz: float = 0.0
    rfi_tone_amplitude: float = 0.0
    rfi_tone_drift_hz_per_sec: float = 0.0
    impulse_rfi_enabled: bool = False
    impulse_probability: float = 0.0
    impulse_amplitude: float = 0.0
    band_rfi_enabled: bool = False
    band_rfi_center_hz: float = 0.0
    band_rfi_width_hz: float = 0.0
    band_rfi_amplitude: float = 0.0


@dataclass(frozen=True)
class SpectrogramConfig:
    fft_size: int = 512
    hop_size: int = 128


@dataclass(frozen=True)
class DriftSearchConfig:
    min_freq_hz: float = -1500.0
    max_freq_hz: float = 1500.0
    min_drift_hz_per_sec: float = -50.0
    max_drift_hz_per_sec: float = 50.0
    drift_step_hz_per_sec: float = 2.0
    threshold: Optional[float] = None


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    score: float
    estimated_f0_hz: Optional[float]
    estimated_drift_hz_per_sec: Optional[float]


@dataclass(frozen=True)
class WaterfallSlice:
    power_db: object
    frame_times_sec: object
    freq_bins_hz: object
    header: dict

