from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .backgrounds import apply_channel, generate_background, generate_noise_only
from .detectors import detect_drifting_tone, search_drifting_tracks
from .models import (
    BackgroundConfig,
    ChannelConfig,
    DriftSearchConfig,
    SignalTruth,
    SpectrogramConfig,
)
from .signals import generate_drifting_tone
from .spectrogram import make_spectrogram


def without_threshold(cfg: DriftSearchConfig) -> DriftSearchConfig:
    return DriftSearchConfig(
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
        min_drift_hz_per_sec=cfg.min_drift_hz_per_sec,
        max_drift_hz_per_sec=cfg.max_drift_hz_per_sec,
        drift_step_hz_per_sec=cfg.drift_step_hz_per_sec,
        threshold=None,
    )


def with_threshold(cfg: DriftSearchConfig, threshold: float) -> DriftSearchConfig:
    return DriftSearchConfig(
        min_freq_hz=cfg.min_freq_hz,
        max_freq_hz=cfg.max_freq_hz,
        min_drift_hz_per_sec=cfg.min_drift_hz_per_sec,
        max_drift_hz_per_sec=cfg.max_drift_hz_per_sec,
        drift_step_hz_per_sec=cfg.drift_step_hz_per_sec,
        threshold=threshold,
    )


def score_iq(
    x: np.ndarray,
    sample_rate_hz: float,
    spec_cfg: SpectrogramConfig,
    search_cfg: DriftSearchConfig,
) -> float:
    power_db, times, freqs = make_spectrogram(x, sample_rate_hz, spec_cfg)
    score, _, _ = search_drifting_tracks(power_db, times, freqs, without_threshold(search_cfg))
    return float(score)


def score_waterfall(
    power_db: np.ndarray,
    frame_times_sec: np.ndarray,
    freq_bins_hz: np.ndarray,
    search_cfg: DriftSearchConfig,
) -> float:
    score, _, _ = search_drifting_tracks(
        power_db,
        frame_times_sec,
        freq_bins_hz,
        without_threshold(search_cfg),
    )
    return float(score)


def calibrate_threshold_for_pfa(
    sample_rate_hz: float,
    duration_sec: float,
    spec_cfg: SpectrogramConfig,
    search_cfg: DriftSearchConfig,
    target_pfa: float,
    trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    cfg = without_threshold(search_cfg)
    for _ in range(trials):
        noise = generate_noise_only(sample_rate_hz, duration_sec, rng)
        scores.append(score_iq(noise, sample_rate_hz, spec_cfg, cfg))
    return float(np.quantile(np.asarray(scores, dtype=np.float64), 1.0 - target_pfa))


def estimate_false_alarm_rate(
    sample_rate_hz: float,
    duration_sec: float,
    spec_cfg: SpectrogramConfig,
    search_cfg: DriftSearchConfig,
    trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    false_alarms = 0
    for _ in range(trials):
        noise = generate_noise_only(sample_rate_hz, duration_sec, rng)
        result = detect_drifting_tone(noise, sample_rate_hz, spec_cfg, search_cfg)
        if result.detected:
            false_alarms += 1
    return false_alarms / trials


def estimate_detection_probability(
    signal_cfg: SignalTruth,
    channel_cfg: ChannelConfig,
    spec_cfg: SpectrogramConfig,
    search_cfg: DriftSearchConfig,
    trials: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    detections = 0
    f0_errors = []
    drift_errors = []
    scores = []

    for _ in range(trials):
        x, truth = generate_drifting_tone(
            signal_cfg.sample_rate_hz,
            signal_cfg.duration_sec,
            signal_cfg.f0_hz,
            signal_cfg.drift_hz_per_sec,
            signal_cfg.amplitude,
        )
        y = apply_channel(x, truth, channel_cfg, rng)
        result = detect_drifting_tone(y, truth.sample_rate_hz, spec_cfg, search_cfg)
        scores.append(result.score)
        if result.detected:
            detections += 1
            if result.estimated_f0_hz is not None:
                f0_errors.append(result.estimated_f0_hz - truth.f0_hz)
            if result.estimated_drift_hz_per_sec is not None:
                drift_errors.append(
                    result.estimated_drift_hz_per_sec - truth.drift_hz_per_sec
                )

    return {
        "pd": detections / trials,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "mean_abs_f0_error_hz": float(np.mean(np.abs(f0_errors))) if f0_errors else None,
        "mean_abs_drift_error_hz_per_sec": (
            float(np.mean(np.abs(drift_errors))) if drift_errors else None
        ),
        "detections": detections,
        "trials": trials,
    }


def calibrate_background_threshold(
    background_cfg: BackgroundConfig,
    sample_rate_hz: float,
    duration_sec: float,
    spec_cfg: SpectrogramConfig,
    search_cfg: DriftSearchConfig,
    target_pfa: float,
    trials: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(trials):
        bg = generate_background(sample_rate_hz, duration_sec, background_cfg, rng)
        scores.append(score_iq(bg, sample_rate_hz, spec_cfg, search_cfg))
    return summarize_scores(scores, target_pfa)


def estimate_background_pfa(
    background_cfg: BackgroundConfig,
    threshold: float,
    sample_rate_hz: float,
    duration_sec: float,
    spec_cfg: SpectrogramConfig,
    search_cfg: DriftSearchConfig,
    trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    false_alarms = 0
    for _ in range(trials):
        bg = generate_background(sample_rate_hz, duration_sec, background_cfg, rng)
        score = score_iq(bg, sample_rate_hz, spec_cfg, search_cfg)
        if score >= threshold:
            false_alarms += 1
    return false_alarms / trials


def summarize_scores(scores: list[float] | np.ndarray, target_pfa: float) -> dict:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize empty score list")
    threshold = float(np.quantile(arr, 1.0 - target_pfa))
    return {
        "threshold": threshold,
        "target_pfa": target_pfa,
        "trials": int(arr.size),
        "score_mean": float(np.mean(arr)),
        "score_median": float(np.median(arr)),
        "score_std": float(np.std(arr)),
        "score_p90": float(np.quantile(arr, 0.90)),
        "score_p95": float(np.quantile(arr, 0.95)),
        "score_p99": float(np.quantile(arr, 0.99)),
        "score_p999": float(np.quantile(arr, 0.999)),
        "score_p9999": float(np.quantile(arr, 0.9999)),
        "score_max": float(np.max(arr)),
    }


def dataclass_dict(obj: object) -> dict:
    return asdict(obj)
