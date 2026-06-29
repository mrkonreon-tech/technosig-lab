from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .models import WaterfallSlice


def _load_blimpy_waterfall(
    path: str | Path,
    f_start_mhz: Optional[float],
    f_stop_mhz: Optional[float],
):
    try:
        from blimpy import Waterfall
    except ImportError as exc:
        raise RuntimeError(
            "blimpy is required for real BL files. Install with: pip install blimpy h5py"
        ) from exc

    kwargs = {}
    if f_start_mhz is not None and f_stop_mhz is not None:
        kwargs["f_start"] = f_start_mhz
        kwargs["f_stop"] = f_stop_mhz
    return Waterfall(str(path), **kwargs)


def load_bl_waterfall_slice(
    path: str | Path,
    f_start_mhz: Optional[float] = None,
    f_stop_mhz: Optional[float] = None,
    max_frames: Optional[int] = None,
    max_bins: Optional[int] = None,
    center_frequency: bool = True,
) -> WaterfallSlice:
    fb = _load_blimpy_waterfall(path, f_start_mhz, f_stop_mhz)
    data = np.asarray(fb.data)
    if data.ndim == 3:
        data = data[:, 0, :]
    elif data.ndim != 2:
        raise ValueError(f"unexpected waterfall shape: {data.shape}")

    if max_frames is not None:
        data = data[:max_frames, :]
    if max_bins is not None:
        data = data[:, :max_bins]

    power = np.abs(data).astype(np.float64)
    power_db = 10.0 * np.log10(power + 1e-30)

    header = dict(getattr(fb, "header", {}))
    tsamp = float(header.get("tsamp", 1.0))
    frame_times_sec = np.arange(power_db.shape[0], dtype=np.float64) * tsamp

    foff_mhz = float(header.get("foff", 1.0))
    container = getattr(fb, "container", None)
    if container is not None and getattr(container, "f_start", None) is not None:
        f_start_mhz = float(container.f_start)
        f_stop_mhz = float(getattr(container, "f_stop", f_start_mhz))
        if foff_mhz < 0:
            freq_bins_mhz = f_stop_mhz - np.arange(power_db.shape[1], dtype=np.float64) * abs(foff_mhz)
        else:
            freq_bins_mhz = f_start_mhz + np.arange(power_db.shape[1], dtype=np.float64) * abs(foff_mhz)
    else:
        try:
            freq_bins_mhz = np.asarray(fb.get_freqs(), dtype=np.float64)
            if max_bins is not None:
                freq_bins_mhz = freq_bins_mhz[: power_db.shape[1]]
        except Exception:
            fch1_mhz = float(header.get("fch1", 0.0))
            freq_bins_mhz = fch1_mhz + np.arange(power_db.shape[1], dtype=np.float64) * foff_mhz
    freq_bins_hz = freq_bins_mhz * 1e6

    if len(freq_bins_hz) > 1 and freq_bins_hz[1] < freq_bins_hz[0]:
        freq_bins_hz = freq_bins_hz[::-1]
        power_db = power_db[:, ::-1]

    if center_frequency and len(freq_bins_hz):
        freq_bins_hz = freq_bins_hz - float(np.median(freq_bins_hz))

    return WaterfallSlice(
        power_db=power_db,
        frame_times_sec=frame_times_sec,
        freq_bins_hz=freq_bins_hz,
        header=header,
    )


def iter_frequency_chunks(
    power_db: np.ndarray,
    freq_bins_hz: np.ndarray,
    chunk_bins: int,
    max_chunks: Optional[int] = None,
):
    if chunk_bins <= 0:
        raise ValueError("chunk_bins must be positive")
    chunk_index = 0
    for start in range(0, power_db.shape[1], chunk_bins):
        stop = min(start + chunk_bins, power_db.shape[1])
        if stop - start < 2:
            continue
        yield chunk_index, power_db[:, start:stop], freq_bins_hz[start:stop], start, stop
        chunk_index += 1
        if max_chunks is not None and chunk_index >= max_chunks:
            break
