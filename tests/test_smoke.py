from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from technosig_lab import candidate_vetting as vet
from technosig_lab.cli import main as cli_main
from technosig_lab.detectors import drift_candidates
from technosig_lab.models import DriftSearchConfig, SpectrogramConfig
from technosig_lab.peaks import find_candidate_clusters
from technosig_lab.signals import generate_drifting_tone
from technosig_lab.spectrogram import make_spectrogram
from technosig_lab.detectors import search_drifting_tracks
from technosig_lab.validation import validate_positive_control


class SmokeTests(unittest.TestCase):
    def test_candidate_vetting_quantile_gates(self) -> None:
        margins = {"p99": 1.5978, "p999": 0.1598, "p9999": 0.016}
        ratios = vet.score_excess_ratios(
            48.13189010729773,
            {
                "p99": 46.53408360920945,
                "p999": 47.972109457488905,
                "p9999": 48.115912042316836,
            },
        )
        self.assertEqual(vet.quantile_stability_gate(margins), "fail_borderline_tail")
        self.assertAlmostEqual(ratios["p99"], 1.034336262243962)
        self.assertEqual(vet.drift_static_gate(0.0, 0.5), "fail_static")
        self.assertEqual(
            vet.neighbor_window_gate(1, 48.13, 4.95),
            "pass_locally_distinct",
        )

    def test_drift_candidates_can_exclude_static_region(self) -> None:
        cfg = DriftSearchConfig(
            min_drift_hz_per_sec=-1.0,
            max_drift_hz_per_sec=1.0,
            drift_step_hz_per_sec=0.5,
        )
        drifts = drift_candidates(cfg, exclude_abs_lt=0.5)
        self.assertNotIn(0.0, drifts.tolist())
        self.assertIn(-0.5, drifts.tolist())
        self.assertIn(0.5, drifts.tolist())

    def test_drift_search_scores_injected_tone(self) -> None:
        x, truth = generate_drifting_tone(
            sample_rate_hz=4096.0,
            duration_sec=2.0,
            f0_hz=350.0,
            drift_hz_per_sec=18.0,
            amplitude=1.0,
        )
        power_db, times, freqs = make_spectrogram(x, 4096.0, SpectrogramConfig())
        score, f0, drift = search_drifting_tracks(
            power_db,
            times,
            freqs,
            DriftSearchConfig(
                min_freq_hz=-1500.0,
                max_freq_hz=1500.0,
                min_drift_hz_per_sec=-50.0,
                max_drift_hz_per_sec=50.0,
                drift_step_hz_per_sec=2.0,
            ),
        )
        self.assertGreater(score, 10.0)
        self.assertAlmostEqual(f0, truth.f0_hz, delta=16.0)
        self.assertAlmostEqual(drift, truth.drift_hz_per_sec, delta=2.0)

    def test_positive_control_morphology(self) -> None:
        frames, bins = 32, 128
        rng = np.random.default_rng(1)
        power = rng.normal(0.0, 1.0, size=(frames, bins))
        times = np.arange(frames, dtype=float)
        freqs = np.arange(bins, dtype=float)
        for i in range(frames):
            power[i, 20 + i // 3] = 20.0
        result = find_candidate_clusters(power, times, freqs, peak_threshold_z=5.0)
        self.assertEqual(result["top_clusters"][0]["morphology"], "narrowband_drifting")
        self.assertTrue(
            validate_positive_control(result["top_clusters"], top_n=5)["passed"]
        )

    def test_compare_candidates_offsets(self) -> None:
        record = {
            "chunks": [
                {
                    "chunk_index": 0,
                    "score": 20.0,
                    "estimated_f0_hz": 1_000_000_000.0,
                    "estimated_drift_hz_per_sec": -0.1,
                },
                {
                    "chunk_index": 1,
                    "score": 10.0,
                    "estimated_f0_hz": 1_000_010_000.0,
                    "estimated_drift_hz_per_sec": -0.2,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "candidates.jsonl"
            output_path = Path(temp_dir) / "offsets.json"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                rc = cli_main(
                    [
                        "compare-candidates",
                        "--input",
                        str(input_path),
                        "--reference-f0-mhz",
                        "1000.0",
                        "--out",
                        str(output_path),
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["candidates"]), 2)
            self.assertAlmostEqual(payload["candidates"][0]["delta_khz"], 0.0)
            self.assertAlmostEqual(payload["candidates"][1]["delta_khz"], 10.0)

    def test_candidate_evidence_gates_static_sparse_event(self) -> None:
        event_thresholds = {
            "chunks": [
                {
                    "chunk_index": 0,
                    "score": 10.0,
                    "estimated_f0_hz": 1_000_000_000.0,
                    "estimated_drift_hz_per_sec": 0.0,
                },
                {
                    "chunk_index": 1,
                    "score": 1.0,
                    "estimated_f0_hz": 1_000_001_000.0,
                    "estimated_drift_hz_per_sec": 0.2,
                },
            ]
        }
        control_thresholds = {
            "chunks": [
                {
                    "chunk_index": 0,
                    "score": 2.0,
                    "estimated_f0_hz": 1_010_000_000.0,
                    "estimated_drift_hz_per_sec": 0.7,
                },
                {
                    "chunk_index": 1,
                    "score": 1.5,
                    "estimated_f0_hz": 1_010_001_000.0,
                    "estimated_drift_hz_per_sec": -0.7,
                },
            ]
        }
        event_candidates = {
            "result": {
                "n_peaks": 1,
                "n_clusters": 1,
                "top_clusters": [{"morphology": "sparse", "score": 8.0}],
            }
        }
        control_candidates = {
            "result": {"n_peaks": 0, "n_clusters": 0, "top_clusters": []}
        }
        header = {
            "range_mhz": [900.0, 1100.0],
            "contains_2375_931298_mhz": True,
            "contains_2380_000000_mhz": True,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            data_file = temp / "data.h5"
            data_file.write_bytes(b"abc")
            header_path = temp / "header.json"
            event_thresholds_path = temp / "event_thresholds.jsonl"
            event_candidates_path = temp / "event_candidates.jsonl"
            control_thresholds_path = temp / "control_thresholds.jsonl"
            control_candidates_path = temp / "control_candidates.jsonl"
            output_path = temp / "evidence.json"
            header_path.write_text(json.dumps(header), encoding="utf-8")
            event_thresholds_path.write_text(
                json.dumps(event_thresholds) + "\n",
                encoding="utf-8",
            )
            event_candidates_path.write_text(
                json.dumps(event_candidates) + "\n",
                encoding="utf-8",
            )
            control_thresholds_path.write_text(
                json.dumps(control_thresholds) + "\n",
                encoding="utf-8",
            )
            control_candidates_path.write_text(
                json.dumps(control_candidates) + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                rc = cli_main(
                    [
                        "candidate-evidence",
                        "--id",
                        "test_static",
                        "--file",
                        str(data_file),
                        "--header-summary",
                        str(header_path),
                        "--event-thresholds",
                        str(event_thresholds_path),
                        "--event-candidates",
                        str(event_candidates_path),
                        "--control-thresholds",
                        str(control_thresholds_path),
                        "--control-candidates",
                        str(control_candidates_path),
                        "--event-frequency-mhz",
                        "1000.0",
                        "--event-f-start-mhz",
                        "999.9",
                        "--event-f-stop-mhz",
                        "1000.1",
                        "--control-f-start-mhz",
                        "1009.9",
                        "--control-f-stop-mhz",
                        "1010.1",
                        "--expected-size-bytes",
                        "3",
                        "--skip-static-rejection",
                        "--out",
                        str(output_path),
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["gates"]["score_p99"], "weak_pass")
            self.assertEqual(payload["gates"]["drift"], "fail_static")
            self.assertEqual(
                payload["gates"]["morphology"],
                "fail_no_narrowband_drifting",
            )
            self.assertEqual(payload["candidate_strength"], "low")


if __name__ == "__main__":
    unittest.main()
