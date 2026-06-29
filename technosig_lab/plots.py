from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .jsonl import read_jsonl


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc
    return plt


def load_records(paths: list[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records


def plot_records(input_paths: list[str | Path], out_dir: str | Path) -> list[Path]:
    plt = _require_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(input_paths)
    created: list[Path] = []

    created.extend(_plot_pd_by_snr(records, out, plt))
    created.extend(_plot_thresholds(records, out, plt))
    created.extend(_plot_cross_pfa(records, out, plt))
    created.extend(_plot_real_injection(records, out, plt))
    return created


def _plot_pd_by_snr(records: list[dict], out: Path, plt) -> list[Path]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        if record.get("experiment") != "synthetic_pd":
            continue
        grouped[record["scenario"]].append(
            (float(record["channel"]["snr_db"]), float(record["result"]["pd"]))
        )
    if not grouped:
        return []
    plt.figure(figsize=(10, 6))
    for name, points in grouped.items():
        points = sorted(points)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
    plt.xlabel("SNR, dB")
    plt.ylabel("Detection probability Pd")
    plt.title("Synthetic Pd(SNR)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    target = out / "synthetic_pd_by_snr.png"
    plt.savefig(target, dpi=160)
    plt.close()
    return [target]


def _plot_thresholds(records: list[dict], out: Path, plt) -> list[Path]:
    points = []
    for record in records:
        if record.get("experiment") not in {"thresholds_synthetic", "real_thresholds"}:
            continue
        bg = record.get("background", {}).get("name") or record.get("source", "real")
        result = record["result"]
        if "threshold" in result:
            points.append((bg, float(result["threshold"]), float(result.get("score_p99", result["threshold"]))))
    if not points:
        return []
    names = [p[0] for p in points]
    thresholds = [p[1] for p in points]
    p99s = [p[2] for p in points]
    x = np.arange(len(names))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, thresholds, width=0.4, label="threshold")
    plt.bar(x + 0.2, p99s, width=0.4, label="score p99")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Drift-search score")
    plt.title("Thresholds by background")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    target = out / "thresholds_by_background.png"
    plt.savefig(target, dpi=160)
    plt.close()
    return [target]


def _plot_cross_pfa(records: list[dict], out: Path, plt) -> list[Path]:
    matrix_records = [
        r for r in records if r.get("experiment") == "thresholds_synthetic_cross_pfa"
    ]
    if not matrix_records:
        return []
    validation_names: list[str] = []
    threshold_names: list[str] = []
    for record in matrix_records:
        validation = record["validation_background"]["name"]
        threshold = record["threshold_source_background"]
        if validation not in validation_names:
            validation_names.append(validation)
        if threshold not in threshold_names:
            threshold_names.append(threshold)
    matrix = np.full((len(validation_names), len(threshold_names)), np.nan)
    row_index = {name: i for i, name in enumerate(validation_names)}
    col_index = {name: i for i, name in enumerate(threshold_names)}
    for record in matrix_records:
        i = row_index[record["validation_background"]["name"]]
        j = col_index[record["threshold_source_background"]]
        matrix[i, j] = record["measured_pfa"]
    plt.figure(figsize=(11, 8))
    image = plt.imshow(matrix, aspect="auto")
    plt.colorbar(image, label="Measured Pfa")
    plt.xticks(np.arange(len(threshold_names)), threshold_names, rotation=35, ha="right")
    plt.yticks(np.arange(len(validation_names)), validation_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    plt.xlabel("Threshold calibrated on background")
    plt.ylabel("Validated on background")
    plt.title("Cross-background false alarm matrix")
    plt.tight_layout()
    target = out / "cross_background_pfa_matrix.png"
    plt.savefig(target, dpi=160)
    plt.close()
    return [target]


def _plot_real_injection(records: list[dict], out: Path, plt) -> list[Path]:
    points = []
    for record in records:
        if record.get("experiment") == "real_injection":
            points.append((record["snr_like_db"], record["result"]["pd"]))
    if not points:
        return []
    points = sorted((float(a), float(b)) for a, b in points)
    plt.figure(figsize=(9, 5))
    plt.plot([p[0] for p in points], [p[1] for p in points], marker="o")
    plt.xlabel("Waterfall injection level over median, dB")
    plt.ylabel("Detection probability Pd")
    plt.title("Real-background injection Pd")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()
    target = out / "real_injection_pd.png"
    plt.savefig(target, dpi=160)
    plt.close()
    return [target]

