$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$input = "data/voyager1_bl/voyager_f1032192_t300_v2.fil"
if (-not (Test-Path $input)) {
    throw "Missing $input. Run scripts/fetch_voyager_positive_control.ps1 -DownloadData first."
}

New-Item -ItemType Directory -Force -Path "results" | Out-Null

python -m technosig_lab.cli real-candidates `
  --input $input `
  --f-start-mhz 8420.2163 `
  --f-stop-mhz 8420.2166 `
  --absolute-frequency `
  --max-frames 2 `
  --max-bins 4096 `
  --chunk-bins 256 `
  --max-chunks 1 `
  --peak-threshold-z 8 `
  --positive-control `
  --positive-control-top-n 10 `
  --out results/voyager_candidates_carrier_run.jsonl

python -m technosig_lab.cli real-thresholds `
  --input $input `
  --f-start-mhz 8420.18 `
  --f-stop-mhz 8420.25 `
  --absolute-frequency `
  --max-frames 2 `
  --max-bins 32768 `
  --chunk-bins 256 `
  --max-chunks 128 `
  --min-drift-hz-per-sec -1 `
  --max-drift-hz-per-sec 1 `
  --drift-step-hz-per-sec 0.05 `
  --target-pfa 0.01 `
  --out results/voyager_thresholds_8420_18_25_run.jsonl

python -m technosig_lab.cli real-candidates `
  --input $input `
  --f-start-mhz 8420.3000 `
  --f-stop-mhz 8420.3003 `
  --absolute-frequency `
  --max-frames 2 `
  --max-bins 4096 `
  --chunk-bins 256 `
  --max-chunks 1 `
  --peak-threshold-z 8 `
  --positive-control `
  --positive-control-top-n 10 `
  --out results/voyager_candidates_offtarget_run.jsonl

python -m technosig_lab.cli real-thresholds `
  --input $input `
  --f-start-mhz 8420.30 `
  --f-stop-mhz 8420.37 `
  --absolute-frequency `
  --max-frames 2 `
  --max-bins 32768 `
  --chunk-bins 256 `
  --max-chunks 128 `
  --min-drift-hz-per-sec -1 `
  --max-drift-hz-per-sec 1 `
  --drift-step-hz-per-sec 0.05 `
  --target-pfa 0.01 `
  --out results/voyager_thresholds_offtarget_run.jsonl

Write-Host "Voyager positive/off-target control complete."
