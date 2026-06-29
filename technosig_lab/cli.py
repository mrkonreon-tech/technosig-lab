from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .blio import iter_frequency_chunks, load_bl_waterfall_slice
from . import candidate_vetting as vet
from .detectors import search_drifting_tracks
from .injection import inject_drifting_track_into_waterfall
from .jsonl import write_jsonl
from .models import (
    BackgroundConfig,
    ChannelConfig,
    DriftSearchConfig,
    SignalTruth,
    SpectrogramConfig,
)
from .peaks import find_candidate_clusters
from .plots import plot_records
from .thresholds import (
    calibrate_background_threshold,
    calibrate_threshold_for_pfa,
    estimate_background_pfa,
    estimate_detection_probability,
    estimate_false_alarm_rate,
    score_waterfall,
    summarize_scores,
    with_threshold,
    without_threshold,
)
from .validation import validate_positive_control


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            item = value.item()
            if item is not value:
                return json_safe(item)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def default_backgrounds() -> list[BackgroundConfig]:
    return [
        BackgroundConfig(name="awgn_only", awgn_power=1.0),
        BackgroundConfig(
            name="awgn_plus_static_rfi_tone",
            awgn_power=1.0,
            rfi_tone_enabled=True,
            rfi_tone_freq_hz=600.0,
            rfi_tone_amplitude=0.8,
            rfi_tone_drift_hz_per_sec=0.0,
        ),
        BackgroundConfig(
            name="awgn_plus_drifting_rfi_tone",
            awgn_power=1.0,
            rfi_tone_enabled=True,
            rfi_tone_freq_hz=600.0,
            rfi_tone_amplitude=0.8,
            rfi_tone_drift_hz_per_sec=-8.0,
        ),
        BackgroundConfig(
            name="awgn_plus_impulse_rfi",
            awgn_power=1.0,
            impulse_rfi_enabled=True,
            impulse_probability=0.002,
            impulse_amplitude=25.0,
        ),
        BackgroundConfig(
            name="awgn_plus_band_limited_rfi",
            awgn_power=1.0,
            band_rfi_enabled=True,
            band_rfi_center_hz=-400.0,
            band_rfi_width_hz=250.0,
            band_rfi_amplitude=1.5,
        ),
        BackgroundConfig(
            name="mixed_rfi",
            awgn_power=1.0,
            rfi_tone_enabled=True,
            rfi_tone_freq_hz=600.0,
            rfi_tone_amplitude=0.8,
            rfi_tone_drift_hz_per_sec=-8.0,
            impulse_rfi_enabled=True,
            impulse_probability=0.002,
            impulse_amplitude=25.0,
            band_rfi_enabled=True,
            band_rfi_center_hz=-400.0,
            band_rfi_width_hz=250.0,
            band_rfi_amplitude=1.5,
        ),
    ]


def synthetic_spec_cfg(args: argparse.Namespace) -> SpectrogramConfig:
    return SpectrogramConfig(fft_size=args.fft_size, hop_size=args.hop_size)


def synthetic_search_cfg(
    args: argparse.Namespace,
    threshold: float | None = None,
) -> DriftSearchConfig:
    return DriftSearchConfig(
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        min_drift_hz_per_sec=args.min_drift_hz_per_sec,
        max_drift_hz_per_sec=args.max_drift_hz_per_sec,
        drift_step_hz_per_sec=args.drift_step_hz_per_sec,
        threshold=threshold,
    )


def real_search_cfg(
    args: argparse.Namespace,
    freq_bins_hz: np.ndarray,
    threshold: float | None = None,
) -> DriftSearchConfig:
    min_freq = args.min_freq_hz
    max_freq = args.max_freq_hz
    if min_freq is None:
        min_freq = float(np.min(freq_bins_hz))
    if max_freq is None:
        max_freq = float(np.max(freq_bins_hz))
    return DriftSearchConfig(
        min_freq_hz=min_freq,
        max_freq_hz=max_freq,
        min_drift_hz_per_sec=args.min_drift_hz_per_sec,
        max_drift_hz_per_sec=args.max_drift_hz_per_sec,
        drift_step_hz_per_sec=args.drift_step_hz_per_sec,
        threshold=threshold,
    )


def run_synthetic_pd(args: argparse.Namespace) -> int:
    records: list[dict] = []
    sample_rate_hz = args.sample_rate_hz
    duration_sec = args.duration_sec
    signal_cfg = SignalTruth(
        kind="drifting_tone",
        sample_rate_hz=sample_rate_hz,
        duration_sec=duration_sec,
        f0_hz=args.signal_f0_hz,
        drift_hz_per_sec=args.signal_drift_hz_per_sec,
        amplitude=1.0,
    )
    spec_cfg = synthetic_spec_cfg(args)
    base_search = synthetic_search_cfg(args, threshold=None)
    threshold = calibrate_threshold_for_pfa(
        sample_rate_hz,
        duration_sec,
        spec_cfg,
        base_search,
        args.target_pfa,
        args.calibration_trials,
        args.seed,
    )
    search_cfg = with_threshold(base_search, threshold)
    measured_pfa = estimate_false_alarm_rate(
        sample_rate_hz,
        duration_sec,
        spec_cfg,
        search_cfg,
        args.validation_trials,
        args.seed + 1,
    )

    scenarios = [
        ("awgn_only", dict(phase_noise_std_rad=0.0, rfi_tone_enabled=False, multipath_enabled=False)),
        (
            "awgn_plus_phase_noise",
            dict(phase_noise_std_rad=0.08, rfi_tone_enabled=False, multipath_enabled=False),
        ),
        (
            "awgn_plus_rfi_tone",
            dict(
                phase_noise_std_rad=0.0,
                rfi_tone_enabled=True,
                rfi_tone_freq_hz=600.0,
                rfi_tone_amplitude=0.8,
                rfi_tone_drift_hz_per_sec=-6.0,
                multipath_enabled=False,
            ),
        ),
        (
            "awgn_plus_multipath",
            dict(
                phase_noise_std_rad=0.0,
                rfi_tone_enabled=False,
                multipath_enabled=True,
                multipath_delay_samples=37,
                multipath_gain=0.35,
            ),
        ),
    ]

    print(f"synthetic-pd threshold={threshold:.3f} measured_pfa={measured_pfa:.3f}")
    for scenario_name, overrides in scenarios:
        print(f"scenario={scenario_name}")
        for snr_db in args.snrs:
            channel_cfg = ChannelConfig(snr_db=float(snr_db), **overrides)
            stats = estimate_detection_probability(
                signal_cfg,
                channel_cfg,
                spec_cfg,
                search_cfg,
                args.trials,
                args.seed + 1000 + int((snr_db + 100) * 10),
            )
            record = {
                "experiment": "synthetic_pd",
                "scenario": scenario_name,
                "signal": asdict(signal_cfg),
                "channel": asdict(channel_cfg),
                "spectrogram": asdict(spec_cfg),
                "detector": {
                    "kind": "drift_search",
                    **asdict(search_cfg),
                    "target_pfa": args.target_pfa,
                    "measured_pfa": measured_pfa,
                },
                "result": stats,
            }
            records.append(record)
            print(f"  snr={snr_db:>6.1f} dB pd={stats['pd']:.2f} score={stats['mean_score']:.2f}")

    write_jsonl(args.out, records)
    print(f"saved {len(records)} records to {args.out}")
    return 0


def run_thresholds_synthetic(args: argparse.Namespace) -> int:
    records: list[dict] = []
    sample_rate_hz = args.sample_rate_hz
    duration_sec = args.duration_sec
    spec_cfg = synthetic_spec_cfg(args)
    search_cfg = synthetic_search_cfg(args, threshold=None)
    backgrounds = default_backgrounds()
    calibrated: dict[str, float] = {}

    print(f"thresholds-synthetic target_pfa={args.target_pfa}")
    for i, bg in enumerate(backgrounds):
        stats = calibrate_background_threshold(
            bg,
            sample_rate_hz,
            duration_sec,
            spec_cfg,
            search_cfg,
            args.target_pfa,
            args.calibration_trials,
            args.seed + i,
        )
        measured_pfa = estimate_background_pfa(
            bg,
            stats["threshold"],
            sample_rate_hz,
            duration_sec,
            spec_cfg,
            search_cfg,
            args.validation_trials,
            args.seed + 100 + i,
        )
        stats["measured_pfa_validation"] = measured_pfa
        calibrated[bg.name] = stats["threshold"]
        records.append(
            {
                "experiment": "thresholds_synthetic",
                "background": asdict(bg),
                "spectrogram": asdict(spec_cfg),
                "detector": {"kind": "drift_search", **asdict(search_cfg)},
                "result": stats,
            }
        )
        print(
            f"  {bg.name:<32} threshold={stats['threshold']:>8.3f} "
            f"p99={stats['score_p99']:>8.3f} pfa={measured_pfa:.3f}"
        )

    for row_i, validation_bg in enumerate(backgrounds):
        for col_name, threshold in calibrated.items():
            pfa = estimate_background_pfa(
                validation_bg,
                threshold,
                sample_rate_hz,
                duration_sec,
                spec_cfg,
                search_cfg,
                args.cross_trials,
                args.seed + 1000 + row_i * 100 + len(col_name),
            )
            records.append(
                {
                    "experiment": "thresholds_synthetic_cross_pfa",
                    "validation_background": asdict(validation_bg),
                    "threshold_source_background": col_name,
                    "threshold": threshold,
                    "measured_pfa": pfa,
                }
            )

    write_jsonl(args.out, records)
    print(f"saved {len(records)} records to {args.out}")
    return 0


def load_real_slice(args: argparse.Namespace):
    return load_bl_waterfall_slice(
        args.input,
        f_start_mhz=args.f_start_mhz,
        f_stop_mhz=args.f_stop_mhz,
        max_frames=args.max_frames,
        max_bins=args.max_bins,
        center_frequency=not args.absolute_frequency,
    )


def chunk_scores(
    args: argparse.Namespace,
    wf,
    exclude_drift_abs_lt: float | None = None,
) -> list[dict]:
    chunks = []
    for chunk_index, chunk_power, chunk_freqs, start, stop in iter_frequency_chunks(
        wf.power_db,
        wf.freq_bins_hz,
        args.chunk_bins,
        args.max_chunks,
    ):
        search_cfg = real_search_cfg(args, chunk_freqs, threshold=None)
        score, f0, drift = search_drifting_tracks(
            chunk_power,
            wf.frame_times_sec,
            chunk_freqs,
            without_threshold(search_cfg),
            exclude_drift_abs_lt=exclude_drift_abs_lt,
        )
        chunks.append(
            {
                "chunk_index": chunk_index,
                "bin_start": start,
                "bin_stop": stop,
                "freq_min_hz": float(np.min(chunk_freqs)),
                "freq_max_hz": float(np.max(chunk_freqs)),
                "score": float(score),
                "estimated_f0_hz": float(f0),
                "estimated_drift_hz_per_sec": float(drift),
            }
        )
    return chunks


def run_real_thresholds(args: argparse.Namespace) -> int:
    wf = load_real_slice(args)
    chunks = chunk_scores(args, wf, exclude_drift_abs_lt=args.exclude_drift_abs_lt)
    if not chunks:
        raise RuntimeError("no usable chunks found")
    stats = summarize_scores([c["score"] for c in chunks], args.target_pfa)
    record = {
        "experiment": "real_thresholds",
        "source": str(args.input),
        "header": json_safe(wf.header),
        "shape": list(np.asarray(wf.power_db).shape),
        "chunk_bins": args.chunk_bins,
        "max_chunks": args.max_chunks,
        "detector": {
            "kind": "drift_search",
            "min_drift_hz_per_sec": args.min_drift_hz_per_sec,
            "max_drift_hz_per_sec": args.max_drift_hz_per_sec,
            "drift_step_hz_per_sec": args.drift_step_hz_per_sec,
            "exclude_drift_abs_lt": args.exclude_drift_abs_lt,
        },
        "result": stats,
        "chunks": chunks,
    }
    write_jsonl(args.out, [record])
    print(f"real threshold={stats['threshold']:.3f} from {len(chunks)} chunks")
    print(f"saved 1 record to {args.out}")
    return 0


def run_real_injection(args: argparse.Namespace) -> int:
    wf = load_real_slice(args)
    chunk_records = []
    chunk_payloads = []
    for chunk_index, chunk_power, chunk_freqs, start, stop in iter_frequency_chunks(
        wf.power_db,
        wf.freq_bins_hz,
        args.chunk_bins,
        args.max_chunks,
    ):
        search_cfg = real_search_cfg(args, chunk_freqs, threshold=None)
        bg_score = score_waterfall(chunk_power, wf.frame_times_sec, chunk_freqs, search_cfg)
        chunk_records.append(
            {
                "chunk_index": chunk_index,
                "score": bg_score,
                "bin_start": start,
                "bin_stop": stop,
                "freq_min_hz": float(np.min(chunk_freqs)),
                "freq_max_hz": float(np.max(chunk_freqs)),
            }
        )
        chunk_payloads.append((chunk_index, chunk_power, chunk_freqs, search_cfg))
    if not chunk_payloads:
        raise RuntimeError("no usable chunks found")

    bg_stats = summarize_scores([c["score"] for c in chunk_records], args.target_pfa)
    threshold = bg_stats["threshold"]
    records: list[dict] = [
        {
            "experiment": "real_injection_background_threshold",
            "source": str(args.input),
            "header": json_safe(wf.header),
            "result": bg_stats,
            "chunks": chunk_records,
        }
    ]

    for snr_like_db in args.snr_like_dbs:
        detections = 0
        trial_scores = []
        for chunk_index, chunk_power, chunk_freqs, search_cfg in chunk_payloads:
            f0_hz = args.inject_f0_hz
            if f0_hz is None:
                f0_hz = float(np.median(chunk_freqs))
            injected = inject_drifting_track_into_waterfall(
                chunk_power,
                wf.frame_times_sec,
                chunk_freqs,
                f0_hz=f0_hz,
                drift_hz_per_sec=args.inject_drift_hz_per_sec,
                snr_like_db=float(snr_like_db),
                width_bins=args.inject_width_bins,
            )
            score = score_waterfall(injected, wf.frame_times_sec, chunk_freqs, search_cfg)
            trial_scores.append({"chunk_index": chunk_index, "score": score})
            if score >= threshold:
                detections += 1

        pd = detections / len(chunk_payloads)
        records.append(
            {
                "experiment": "real_injection",
                "source": str(args.input),
                "snr_like_db": float(snr_like_db),
                "threshold": threshold,
                "injection": {
                    "f0_hz": args.inject_f0_hz,
                    "drift_hz_per_sec": args.inject_drift_hz_per_sec,
                    "width_bins": args.inject_width_bins,
                },
                "result": {
                    "pd": pd,
                    "detections": detections,
                    "trials": len(chunk_payloads),
                },
                "scores": trial_scores,
            }
        )
        print(f"snr_like={snr_like_db:>6.1f} dB pd={pd:.2f}")

    write_jsonl(args.out, records)
    print(f"saved {len(records)} records to {args.out}")
    return 0


def run_real_candidates(args: argparse.Namespace) -> int:
    wf = load_real_slice(args)
    records: list[dict] = []
    all_top: list[dict] = []
    for chunk_index, chunk_power, chunk_freqs, start, stop in iter_frequency_chunks(
        wf.power_db,
        wf.freq_bins_hz,
        args.chunk_bins,
        args.max_chunks,
    ):
        result = find_candidate_clusters(
            chunk_power,
            wf.frame_times_sec,
            chunk_freqs,
            peak_threshold_z=args.peak_threshold_z,
            top_n=args.top_n,
        )
        for cluster in result["top_clusters"]:
            enriched = dict(cluster)
            enriched["chunk_index"] = chunk_index
            all_top.append(enriched)

        record = {
            "experiment": "real_candidates",
            "source": str(args.input),
            "chunk_index": chunk_index,
            "bin_start": start,
            "bin_stop": stop,
            "freq_min_hz": float(np.min(chunk_freqs)),
            "freq_max_hz": float(np.max(chunk_freqs)),
            "header": json_safe(wf.header),
            "result": result,
        }
        if args.positive_control:
            record["positive_control_validation"] = validate_positive_control(
                result["top_clusters"],
                top_n=args.positive_control_top_n,
            )
        records.append(record)
        best = result["top_clusters"][0] if result["top_clusters"] else None
        print(
            f"chunk={chunk_index} peaks={result['n_peaks']} clusters={result['n_clusters']} "
            f"best={best.get('morphology') if best else 'none'}"
        )

    if args.positive_control:
        all_top.sort(key=lambda item: item["score"], reverse=True)
        records.append(
            {
                "experiment": "real_candidates_positive_control_summary",
                "source": str(args.input),
                "positive_control_validation": validate_positive_control(
                    all_top,
                    top_n=args.positive_control_top_n,
                ),
                "top_clusters": all_top[: args.positive_control_top_n],
            }
        )

    write_jsonl(args.out, records)
    print(f"saved {len(records)} records to {args.out}")
    return 0


def run_compare_candidates(args: argparse.Namespace) -> int:
    records: list[dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    chunks: list[dict[str, Any]] = []
    for record in records:
        for chunk in record.get("chunks", []):
            if "score" in chunk and "estimated_f0_hz" in chunk:
                chunks.append(chunk)

    if not chunks:
        raise RuntimeError("no candidate chunks with estimated_f0_hz found")

    chunks.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = chunks[: args.top_n]
    candidates = []
    for rank, chunk in enumerate(selected, start=1):
        f0_mhz = float(chunk["estimated_f0_hz"]) / 1_000_000.0
        candidate = {
            "rank": rank,
            "chunk_index": int(chunk["chunk_index"]),
            "f0_mhz": f0_mhz,
            "delta_khz": (f0_mhz - args.reference_f0_mhz) * 1000.0,
            "score": float(chunk["score"]),
            "drift_hz_per_sec": float(chunk["estimated_drift_hz_per_sec"]),
        }
        if "freq_min_hz" in chunk and "freq_max_hz" in chunk:
            candidate["freq_min_mhz"] = float(chunk["freq_min_hz"]) / 1_000_000.0
            candidate["freq_max_mhz"] = float(chunk["freq_max_hz"]) / 1_000_000.0
        candidates.append(candidate)

    result = {
        "input": str(args.input),
        "reference_f0_mhz": float(args.reference_f0_mhz),
        "top_n": int(args.top_n),
        "candidates": candidates,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as handle:
        json.dump(json_safe(result), handle, indent=2, sort_keys=True)
        handle.write("\n")

    for candidate in candidates:
        print(
            f"rank={candidate['rank']} "
            f"f0={candidate['f0_mhz']:.9f} MHz "
            f"delta={candidate['delta_khz']:+.3f} kHz "
            f"score={candidate['score']:.3f} "
            f"drift={candidate['drift_hz_per_sec']:.3f} Hz/s"
        )
    print(f"saved {len(candidates)} candidates to {args.out}")
    return 0


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def threshold_summary_from_chunks(
    chunks: list[dict[str, Any]],
    path: str | Path | None = None,
) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError("no chunks found for threshold summary")
    chunks_sorted = sorted(chunks, key=lambda item: float(item["score"]), reverse=True)
    scores = np.asarray([float(chunk["score"]) for chunk in chunks], dtype=np.float64)
    top = chunks_sorted[0]
    quantiles = {
        "p99": float(np.quantile(scores, 0.99)),
        "p999": float(np.quantile(scores, 0.999)),
        "p9999": float(np.quantile(scores, 0.9999)),
    }
    score = float(top["score"])
    margins = vet.score_margins(score, quantiles)
    ratios = vet.score_excess_ratios(score, quantiles)
    summary: dict[str, Any] = {
        "trials": int(scores.size),
        "top_score": score,
        "top_frequency_mhz": float(top["estimated_f0_hz"]) / 1_000_000.0,
        "top_drift_hz_per_sec": float(top["estimated_drift_hz_per_sec"]),
        "top_chunk_index": int(top["chunk_index"]),
        "quantiles": quantiles,
        "margins": margins,
        "score_excess_ratio": ratios,
        "passes": {name: score >= threshold for name, threshold in quantiles.items()},
        "quantile_stability_gate": vet.quantile_stability_gate(margins),
        "sample_warnings": [
            warning
            for warning, enabled in [
                ("p999 is interpolated from fewer than 1000 chunks", scores.size < 1000),
                ("p9999 is interpolated from fewer than 10000 chunks", scores.size < 10000),
            ]
            if enabled
        ],
    }
    if path is not None:
        summary["file"] = str(path)
    return summary


def summarize_threshold_file(path: str | Path) -> dict[str, Any]:
    records = read_jsonl_records(path)
    if not records:
        raise RuntimeError(f"no threshold records found in {path}")
    record = records[0]
    chunks = record.get("chunks", [])
    if not chunks:
        raise RuntimeError(f"no chunks found in threshold record {path}")
    return threshold_summary_from_chunks(chunks, path=path)


def summarize_candidate_file(path: str | Path) -> dict[str, Any]:
    records = read_jsonl_records(path)
    clusters: list[dict[str, Any]] = []
    morphology_counts: dict[str, int] = {}
    total_peaks = 0
    total_clusters = 0
    chunks_with_peaks = 0
    for record in records:
        result = record.get("result", {})
        n_peaks = int(result.get("n_peaks", 0))
        n_clusters = int(result.get("n_clusters", 0))
        total_peaks += n_peaks
        total_clusters += n_clusters
        if n_peaks:
            chunks_with_peaks += 1
        for cluster in result.get("top_clusters", []):
            enriched = dict(cluster)
            enriched["chunk_index"] = record.get("chunk_index")
            clusters.append(enriched)
            morphology = str(cluster.get("morphology", "unknown"))
            morphology_counts[morphology] = morphology_counts.get(morphology, 0) + 1
    clusters.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return {
        "file": str(path),
        "records": len(records),
        "total_peaks": total_peaks,
        "total_clusters": total_clusters,
        "chunks_with_peaks": chunks_with_peaks,
        "morphology_counts": morphology_counts,
        "has_narrowband_drifting": morphology_counts.get("narrowband_drifting", 0) > 0,
        "top_clusters": clusters[:5],
    }


def detector_cfg_from_threshold_record(record: dict[str, Any]) -> dict[str, float]:
    detector = record.get("detector", {})
    return {
        "min_drift_hz_per_sec": float(detector.get("min_drift_hz_per_sec", -1.0)),
        "max_drift_hz_per_sec": float(detector.get("max_drift_hz_per_sec", 1.0)),
        "drift_step_hz_per_sec": float(detector.get("drift_step_hz_per_sec", 0.01)),
    }


def score_real_window(
    path: str | Path,
    f_start_mhz: float,
    f_stop_mhz: float,
    max_frames: int,
    max_bins: int,
    chunk_bins: int,
    max_chunks: int,
    detector_cfg: dict[str, float],
    exclude_drift_abs_lt: float | None = None,
) -> list[dict[str, Any]]:
    wf = load_bl_waterfall_slice(
        path,
        f_start_mhz=f_start_mhz,
        f_stop_mhz=f_stop_mhz,
        max_frames=max_frames,
        max_bins=max_bins,
        center_frequency=False,
    )
    chunks = []
    for chunk_index, chunk_power, chunk_freqs, start, stop in iter_frequency_chunks(
        wf.power_db,
        wf.freq_bins_hz,
        chunk_bins,
        max_chunks,
    ):
        search_cfg = DriftSearchConfig(
            min_freq_hz=float(np.min(chunk_freqs)),
            max_freq_hz=float(np.max(chunk_freqs)),
            min_drift_hz_per_sec=detector_cfg["min_drift_hz_per_sec"],
            max_drift_hz_per_sec=detector_cfg["max_drift_hz_per_sec"],
            drift_step_hz_per_sec=detector_cfg["drift_step_hz_per_sec"],
            threshold=None,
        )
        score, f0, drift = search_drifting_tracks(
            chunk_power,
            wf.frame_times_sec,
            chunk_freqs,
            without_threshold(search_cfg),
            exclude_drift_abs_lt=exclude_drift_abs_lt,
        )
        chunks.append(
            {
                "chunk_index": chunk_index,
                "bin_start": start,
                "bin_stop": stop,
                "freq_min_hz": float(np.min(chunk_freqs)),
                "freq_max_hz": float(np.max(chunk_freqs)),
                "score": float(score),
                "estimated_f0_hz": float(f0),
                "estimated_drift_hz_per_sec": float(drift),
            }
        )
    return chunks


def static_rejection_rerun(
    args: argparse.Namespace,
    event_threshold_record: dict[str, Any],
    original_thresholds: dict[str, Any],
) -> dict[str, Any]:
    shape = event_threshold_record.get("shape", [])
    max_frames = args.vetting_max_frames or int(shape[0])
    max_bins = args.vetting_max_bins or int(shape[1])
    chunk_bins = args.vetting_chunk_bins or int(event_threshold_record.get("chunk_bins", 256))
    max_chunks = args.vetting_max_chunks or int(event_threshold_record.get("max_chunks", 128))
    detector_cfg = detector_cfg_from_threshold_record(event_threshold_record)
    chunks = score_real_window(
        args.file,
        args.event_f_start_mhz,
        args.event_f_stop_mhz,
        max_frames,
        max_bins,
        chunk_bins,
        max_chunks,
        detector_cfg,
        exclude_drift_abs_lt=args.static_rejection_exclude_abs_lt,
    )
    summary = threshold_summary_from_chunks(chunks)
    original_score = float(original_thresholds["top_score"])
    nonstatic_score = float(summary["top_score"])
    summary.update(
        {
            "exclude_drift_abs_lt_hz_per_sec": args.static_rejection_exclude_abs_lt,
            "original_score": original_score,
            "nonstatic_score": nonstatic_score,
            "nonstatic_score_drop": original_score - nonstatic_score,
            "nonstatic_score_ratio": (
                nonstatic_score / original_score if original_score else None
            ),
            "nonstatic_top_drift_hz_per_sec": summary["top_drift_hz_per_sec"],
            "nonstatic_top_frequency_mhz": summary["top_frequency_mhz"],
        }
    )
    return summary


def parse_neighbor_windows(spec: str | None) -> list[dict[str, Any]]:
    if not spec:
        return []
    windows = []
    for index, item in enumerate(spec.split(","), start=1):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            window_id, range_spec = item.split("=", 1)
        else:
            window_id = f"window_{index}"
            range_spec = item
        start_text, stop_text = range_spec.split(":", 1)
        windows.append(
            {
                "id": window_id.strip(),
                "f_start_mhz": float(start_text),
                "f_stop_mhz": float(stop_text),
            }
        )
    return windows


def neighbor_window_report(
    args: argparse.Namespace,
    event_threshold_record: dict[str, Any],
) -> dict[str, Any] | None:
    windows = parse_neighbor_windows(args.neighbor_windows)
    if not windows:
        return None

    shape = event_threshold_record.get("shape", [])
    max_frames = args.vetting_max_frames or int(shape[0])
    max_bins = args.vetting_max_bins or int(shape[1])
    chunk_bins = args.vetting_chunk_bins or int(event_threshold_record.get("chunk_bins", 256))
    max_chunks = args.vetting_max_chunks or int(event_threshold_record.get("max_chunks", 128))
    detector_cfg = detector_cfg_from_threshold_record(event_threshold_record)
    reports = []
    for window in windows:
        chunks = score_real_window(
            args.file,
            window["f_start_mhz"],
            window["f_stop_mhz"],
            max_frames,
            max_bins,
            chunk_bins,
            max_chunks,
            detector_cfg,
            exclude_drift_abs_lt=None,
        )
        summary = threshold_summary_from_chunks(chunks)
        summary.update(window)
        summary["is_event_window"] = window["id"] == args.event_window_id
        reports.append(summary)

    ranked = sorted(reports, key=lambda item: float(item["top_score"]), reverse=True)
    event_rank = None
    event_score = None
    for rank, report in enumerate(ranked, start=1):
        report["rank"] = rank
        if report["id"] == args.event_window_id:
            event_rank = rank
            event_score = float(report["top_score"])
    neighbor_scores = [
        float(report["top_score"])
        for report in reports
        if report["id"] != args.event_window_id
    ]
    neighbor_max_score = max(neighbor_scores) if neighbor_scores else None
    top_scores = [float(report["top_score"]) for report in reports]
    return {
        "event_window_id": args.event_window_id,
        "event_rank_among_neighbor_windows": event_rank,
        "event_score": event_score,
        "neighbor_max_score": neighbor_max_score,
        "neighbor_score_distribution": vet.summarize_score_distribution(top_scores),
        "gate": vet.neighbor_window_gate(
            event_rank,
            event_score,
            neighbor_max_score,
            similar_fraction=args.neighbor_similar_fraction,
        ),
        "windows": ranked,
    }


def run_candidate_evidence(args: argparse.Namespace) -> int:
    header = json.loads(Path(args.header_summary).read_text(encoding="utf-8"))
    event_threshold_records = read_jsonl_records(args.event_thresholds)
    if not event_threshold_records:
        raise RuntimeError(f"no event threshold record found in {args.event_thresholds}")
    event_threshold_record = event_threshold_records[0]
    event_thresholds = summarize_threshold_file(args.event_thresholds)
    event_candidates = summarize_candidate_file(args.event_candidates)
    control_thresholds = summarize_threshold_file(args.control_thresholds)
    control_candidates = summarize_candidate_file(args.control_candidates)

    actual_size = Path(args.file).stat().st_size
    size_verified = (
        actual_size == args.expected_size_bytes
        if args.expected_size_bytes is not None
        else None
    )
    event_top_drift = event_thresholds["top_drift_hz_per_sec"]
    drift_gate = vet.drift_static_gate(
        event_top_drift,
        args.static_drift_abs_hz_per_sec,
    )
    morphology_gate = vet.morphology_gate(event_candidates)
    range_mhz = header.get("range_mhz")
    if isinstance(range_mhz, list) and len(range_mhz) == 2:
        lo_mhz = min(float(range_mhz[0]), float(range_mhz[1]))
        hi_mhz = max(float(range_mhz[0]), float(range_mhz[1]))
        frequency_covered = lo_mhz <= args.event_frequency_mhz <= hi_mhz
        requested_2380_covered = lo_mhz <= 2380.0 <= hi_mhz
    else:
        frequency_covered = bool(header.get("contains_2375_931298_mhz", False))
        requested_2380_covered = bool(header.get("contains_2380_000000_mhz", False))
    score_gate = vet.score_gate_from_margin(
        event_thresholds["top_score"],
        event_thresholds["quantiles"]["p99"],
    )
    static_rerun = (
        None
        if args.skip_static_rejection
        else static_rejection_rerun(args, event_threshold_record, event_thresholds)
    )
    neighbor_report = neighbor_window_report(args, event_threshold_record)
    gates = {
        "header": "pass" if frequency_covered and requested_2380_covered else "fail",
        "frequency_coverage": "pass" if frequency_covered else "fail",
        "score_p99": score_gate,
        "quantile_stability": event_thresholds["quantile_stability_gate"],
        "morphology": morphology_gate,
        "drift": drift_gate,
        "static_rejection": (
            "not_run"
            if static_rerun is None
            else (
                "fail_static_supported"
                if static_rerun["nonstatic_score_drop"] > 0
                else "pass_no_static_drop"
            )
        ),
        "neighbor_window": (
            "not_run" if neighbor_report is None else neighbor_report["gate"]
        ),
        "positive_control": "fail_not_known_positive",
    }

    report = {
        "id": args.id,
        "status": "initial_analysis_complete",
        "classification": args.classification,
        "positive_control": False,
        "file": {
            "path": str(args.file),
            "size_bytes": actual_size,
            "expected_size_bytes": args.expected_size_bytes,
            "size_verified": size_verified,
            "sha256": args.sha256,
            "sha256_verified": args.sha256_verified,
        },
        "header": {
            "status": "pass" if frequency_covered and requested_2380_covered else "fail",
            "summary": str(args.header_summary),
            "frequency_range_mhz": header.get("range_mhz"),
            "contains_event_frequency_mhz": frequency_covered,
            "contains_requested_2380_mhz": requested_2380_covered,
        },
        "frequency_coverage": {
            "event_frequency_mhz": args.event_frequency_mhz,
            "covered": frequency_covered,
            "requested_2380_mhz_covered": requested_2380_covered,
        },
        "event_window": {
            "range_mhz": [args.event_f_start_mhz, args.event_f_stop_mhz],
            "source_frequency_mhz": args.event_frequency_mhz,
            **event_thresholds,
            "score_gate": score_gate,
            "morphology": (
                "narrowband_drifting"
                if event_candidates["has_narrowband_drifting"]
                else "no_narrowband_drifting"
            ),
            "sparse_peaks": event_candidates["morphology_counts"].get("sparse", 0),
            "candidate_summary": event_candidates,
        },
        "control_window": {
            "range_mhz": [args.control_f_start_mhz, args.control_f_stop_mhz],
            **control_thresholds,
            "morphology": (
                "narrowband_drifting"
                if control_candidates["has_narrowband_drifting"]
                else "no_narrowband_drifting"
            ),
            "candidate_summary": control_candidates,
        },
        "static_rejection_rerun": static_rerun,
        "neighbor_window_report": neighbor_report,
        "gates": gates,
        "candidate_strength": vet.candidate_strength(gates),
        "interpretation": (
            "Initial detector evidence only: weak score excess, static/near-static "
            "top drift, and no narrowband_drifting morphology recovery."
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"id={args.id}")
    print(
        "event "
        f"score={event_thresholds['top_score']:.3f} "
        f"p99={event_thresholds['quantiles']['p99']:.3f} "
        f"margin={event_thresholds['margins']['p99']:.3f} "
        f"p9999_margin={event_thresholds['margins']['p9999']:.3f} "
        f"drift={event_top_drift:.3f} "
        f"score_gate={score_gate} "
        f"quantile_stability={event_thresholds['quantile_stability_gate']} "
        f"drift_gate={drift_gate} "
        f"morphology_gate={morphology_gate}"
    )
    if static_rerun is not None:
        print(
            "static_rejection "
            f"exclude_abs_lt={static_rerun['exclude_drift_abs_lt_hz_per_sec']:.3f} "
            f"nonstatic_score={static_rerun['nonstatic_score']:.3f} "
            f"drop={static_rerun['nonstatic_score_drop']:.3f} "
            f"nonstatic_drift={static_rerun['nonstatic_top_drift_hz_per_sec']:.3f}"
        )
    if neighbor_report is not None:
        print(
            "neighbor "
            f"event_rank={neighbor_report['event_rank_among_neighbor_windows']} "
            f"neighbor_max={neighbor_report['neighbor_max_score']:.3f} "
            f"gate={neighbor_report['gate']}"
        )
    print(
        "control "
        f"score={control_thresholds['top_score']:.3f} "
        f"p99={control_thresholds['quantiles']['p99']:.3f} "
        f"margin={control_thresholds['margins']['p99']:.3f} "
        f"peaks={control_candidates['total_peaks']} "
        f"clusters={control_candidates['total_clusters']}"
    )
    print(f"saved candidate evidence to {args.out}")
    return 0


def run_plot(args: argparse.Namespace) -> int:
    created = plot_records(args.input, args.out)
    if not created:
        print("no plottable records found")
    for path in created:
        print(f"saved {path}")
    return 0


def add_common_synthetic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate-hz", type=float, default=4096.0)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=128)
    parser.add_argument("--min-freq-hz", type=float, default=-1500.0)
    parser.add_argument("--max-freq-hz", type=float, default=1500.0)
    parser.add_argument("--min-drift-hz-per-sec", type=float, default=-50.0)
    parser.add_argument("--max-drift-hz-per-sec", type=float, default=50.0)
    parser.add_argument("--drift-step-hz-per-sec", type=float, default=4.0)
    parser.add_argument("--target-pfa", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=100)


def add_common_real_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--f-start-mhz", type=float, default=None)
    parser.add_argument("--f-stop-mhz", type=float, default=None)
    parser.add_argument("--absolute-frequency", action="store_true")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-bins", type=int, default=2048)
    parser.add_argument("--chunk-bins", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--min-freq-hz", type=float, default=None)
    parser.add_argument("--max-freq-hz", type=float, default=None)
    parser.add_argument("--min-drift-hz-per-sec", type=float, default=-50.0)
    parser.add_argument("--max-drift-hz-per-sec", type=float, default=50.0)
    parser.add_argument("--drift-step-hz-per-sec", type=float, default=2.0)
    parser.add_argument("--exclude-drift-abs-lt", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="technosig_lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synthetic-pd")
    add_common_synthetic_args(p)
    p.add_argument("--signal-f0-hz", type=float, default=350.0)
    p.add_argument("--signal-drift-hz-per-sec", type=float, default=18.0)
    p.add_argument("--calibration-trials", type=int, default=20)
    p.add_argument("--validation-trials", type=int, default=20)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--snrs", type=float, nargs="+", default=[-30, -25, -20, -15, -10, -5, 0])
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_synthetic_pd)

    p = sub.add_parser("thresholds-synthetic")
    add_common_synthetic_args(p)
    p.add_argument("--calibration-trials", type=int, default=20)
    p.add_argument("--validation-trials", type=int, default=20)
    p.add_argument("--cross-trials", type=int, default=10)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_thresholds_synthetic)

    p = sub.add_parser("real-thresholds")
    add_common_real_args(p)
    p.add_argument("--target-pfa", type=float, default=0.01)
    p.set_defaults(func=run_real_thresholds)

    p = sub.add_parser("real-injection")
    add_common_real_args(p)
    p.add_argument("--target-pfa", type=float, default=0.01)
    p.add_argument("--snr-like-dbs", type=float, nargs="+", default=[-20, -15, -10, -5, 0, 5])
    p.add_argument("--inject-f0-hz", type=float, default=None)
    p.add_argument("--inject-drift-hz-per-sec", type=float, default=18.0)
    p.add_argument("--inject-width-bins", type=int, default=1)
    p.set_defaults(func=run_real_injection)

    p = sub.add_parser("real-candidates")
    add_common_real_args(p)
    p.add_argument("--peak-threshold-z", type=float, default=6.0)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--positive-control", action="store_true")
    p.add_argument("--positive-control-top-n", type=int, default=5)
    p.set_defaults(func=run_real_candidates)

    p = sub.add_parser("compare-candidates")
    p.add_argument("--input", required=True)
    p.add_argument("--reference-f0-mhz", type=float, required=True)
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_compare_candidates)

    p = sub.add_parser("candidate-evidence")
    p.add_argument("--id", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--header-summary", required=True)
    p.add_argument("--event-thresholds", required=True)
    p.add_argument("--event-candidates", required=True)
    p.add_argument("--control-thresholds", required=True)
    p.add_argument("--control-candidates", required=True)
    p.add_argument("--event-frequency-mhz", type=float, required=True)
    p.add_argument("--event-f-start-mhz", type=float, required=True)
    p.add_argument("--event-f-stop-mhz", type=float, required=True)
    p.add_argument("--control-f-start-mhz", type=float, required=True)
    p.add_argument("--control-f-stop-mhz", type=float, required=True)
    p.add_argument("--expected-size-bytes", type=int, default=None)
    p.add_argument("--sha256", default=None)
    p.add_argument("--sha256-verified", action="store_true")
    p.add_argument("--static-drift-abs-hz-per-sec", type=float, default=0.5)
    p.add_argument("--static-rejection-exclude-abs-lt", type=float, default=0.5)
    p.add_argument("--skip-static-rejection", action="store_true")
    p.add_argument("--neighbor-windows", default=None)
    p.add_argument("--event-window-id", default="event")
    p.add_argument("--neighbor-similar-fraction", type=float, default=0.8)
    p.add_argument("--vetting-max-frames", type=int, default=None)
    p.add_argument("--vetting-max-bins", type=int, default=None)
    p.add_argument("--vetting-chunk-bins", type=int, default=None)
    p.add_argument("--vetting-max-chunks", type=int, default=None)
    p.add_argument("--classification", default="low_strength_event_table_hit")
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_candidate_evidence)

    p = sub.add_parser("plot")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
