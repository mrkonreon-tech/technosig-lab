$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$input = "data/bl_sband/spliced_blc2021222324252627_guppi_57992_30093_HIP194_0014.gpuspec.0000.h5"
if (-not (Test-Path $input)) {
    throw "Missing $input. Download it from the Breakthrough Listen Open Data Archive before running this script."
}

New-Item -ItemType Directory -Force -Path "data/bl_sband" | Out-Null
New-Item -ItemType Directory -Force -Path "results" | Out-Null

python -c @"
from blimpy import Waterfall
from pathlib import Path
import json

path = Path(r"$input")
wf = Waterfall(str(path), load_data=False)
h = wf.header
fch1 = float(h["fch1"])
foff = float(h["foff"])
nchans = int(h["nchans"])
f_last = fch1 + foff * (nchans - 1)
lo = min(fch1, f_last)
hi = max(fch1, f_last)
summary = {
    "path": str(path),
    "source_name": h.get("source_name"),
    "fch1_mhz": fch1,
    "foff_mhz": foff,
    "nchans": nchans,
    "nbits": h.get("nbits"),
    "nifs": h.get("nifs"),
    "tsamp_sec": h.get("tsamp"),
    "tstart_mjd": h.get("tstart"),
    "range_mhz": [lo, hi],
    "selection_shape": getattr(wf, "selection_shape", None),
    "contains_2375_931298_mhz": lo <= 2375.931298 <= hi,
    "contains_2380_000000_mhz": lo <= 2380.0 <= hi,
}
out = Path(r"data/bl_sband/HIP194_2375p931_HEADER_SUMMARY.json")
out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
print(f"saved {out}")
"@

python -m technosig_lab.cli real-thresholds `
  --input $input `
  --f-start-mhz 2375.90 `
  --f-stop-mhz 2375.96 `
  --absolute-frequency `
  --max-frames 16 `
  --max-bins 32768 `
  --chunk-bins 256 `
  --max-chunks 128 `
  --min-drift-hz-per-sec -1 `
  --max-drift-hz-per-sec 1 `
  --drift-step-hz-per-sec 0.01 `
  --target-pfa 0.01 `
  --out results/hip194_2375p93_thresholds.jsonl

python -m technosig_lab.cli real-candidates `
  --input $input `
  --f-start-mhz 2375.90 `
  --f-stop-mhz 2375.96 `
  --absolute-frequency `
  --max-frames 16 `
  --max-bins 32768 `
  --chunk-bins 256 `
  --max-chunks 128 `
  --peak-threshold-z 8 `
  --positive-control-top-n 20 `
  --out results/hip194_2375p93_candidates.jsonl

python -m technosig_lab.cli real-thresholds `
  --input $input `
  --f-start-mhz 2379.95 `
  --f-stop-mhz 2380.05 `
  --absolute-frequency `
  --max-frames 16 `
  --max-bins 32768 `
  --chunk-bins 256 `
  --max-chunks 128 `
  --min-drift-hz-per-sec -1 `
  --max-drift-hz-per-sec 1 `
  --drift-step-hz-per-sec 0.01 `
  --target-pfa 0.01 `
  --out results/hip194_2380p00_thresholds.jsonl

python -m technosig_lab.cli real-candidates `
  --input $input `
  --f-start-mhz 2379.95 `
  --f-stop-mhz 2380.05 `
  --absolute-frequency `
  --max-frames 16 `
  --max-bins 32768 `
  --chunk-bins 256 `
  --max-chunks 128 `
  --peak-threshold-z 8 `
  --positive-control-top-n 20 `
  --out results/hip194_2380p00_candidates.jsonl

$sha = (Get-FileHash $input -Algorithm SHA256).Hash

python -m technosig_lab.cli candidate-evidence `
  --id hip194_sband_2375p931_bldr1_event `
  --file $input `
  --header-summary data/bl_sband/HIP194_2375p931_HEADER_SUMMARY.json `
  --event-thresholds results/hip194_2375p93_thresholds.jsonl `
  --event-candidates results/hip194_2375p93_candidates.jsonl `
  --control-thresholds results/hip194_2380p00_thresholds.jsonl `
  --control-candidates results/hip194_2380p00_candidates.jsonl `
  --event-frequency-mhz 2375.931298 `
  --event-f-start-mhz 2375.90 `
  --event-f-stop-mhz 2375.96 `
  --control-f-start-mhz 2379.95 `
  --control-f-stop-mhz 2380.05 `
  --expected-size-bytes 15666581851 `
  --sha256 $sha `
  --sha256-verified `
  --static-drift-abs-hz-per-sec 0.5 `
  --static-rejection-exclude-abs-lt 0.5 `
  --neighbor-windows "pre2=2375.70:2375.80,pre1=2375.80:2375.90,event=2375.90:2375.96,post1=2375.96:2376.06,post2=2376.06:2376.16" `
  --event-window-id event `
  --vetting-max-frames 16 `
  --vetting-max-bins 32768 `
  --vetting-chunk-bins 256 `
  --vetting-max-chunks 128 `
  --out results/hip194_sband_candidate_evidence.json

Write-Host "HIP194 vetting complete."
