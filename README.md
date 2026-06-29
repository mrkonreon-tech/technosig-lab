# technosig_lab

[![smoke](https://github.com/mrkonreon-tech/technosig-lab/actions/workflows/smoke.yml/badge.svg)](https://github.com/mrkonreon-tech/technosig-lab/actions/workflows/smoke.yml)

Reproducible SETI technosignature benchmark and vetting harness.

This project is not a claim detector. It is a small research tool for
measuring when a narrowband drifting-signal detector works, when it fails, and
when a real-data event should be downgraded instead of over-interpreted.

It generates controlled artificial signals, runs them through simplified
channel/background models, calibrates drift-search thresholds, and logs Pd/Pfa
style results as JSONL. It can also read reduced Breakthrough Listen `.h5` /
`.fil` waterfall data through `blimpy` when that optional dependency is
installed.

Useful public framing:

```text
Technosig Lab is a reproducible benchmark and vetting harness for narrowband
drifting-signal detection in synthetic and real Breakthrough Listen waterfall
backgrounds.
```

## Install

```powershell
pip install -r requirements.txt
```

Only `numpy` is required for synthetic experiments. `matplotlib` is needed for
plots. `blimpy` and `h5py` are needed for real Breakthrough Listen files.

## Repository Scope

This repository should contain code, tests, docs, small examples, and metadata.
It should not contain large Breakthrough Listen data files or generated result
directories. See `docs/DATA.md` for the data policy and
`docs/CANDIDATE_GATES.md` for the candidate gate definitions.

Reproducibility wrappers live under `scripts/`. They write local data to
ignored `data/` and `results/` directories.

## Quick Synthetic Run

```powershell
python -m technosig_lab.cli thresholds-synthetic --out results/thresholds_synthetic.jsonl
python -m technosig_lab.cli synthetic-pd --out results/synthetic_pd.jsonl
python -m technosig_lab.cli plot --input results/thresholds_synthetic.jsonl results/synthetic_pd.jsonl --out figures
```

## Reproducibility Scripts

Run from the repository root:

```powershell
pwsh scripts/fetch_voyager_positive_control.ps1 -DownloadData
pwsh scripts/run_voyager_positive_control.ps1
pwsh scripts/fetch_bldr1_sband_index.ps1
pwsh scripts/run_hip194_vetting.ps1
```

`run_hip194_vetting.ps1` expects the large HIP194 HDF5 file to already exist
locally under `data/bl_sband/`; it does not download that file automatically.

## Real BL Waterfall Run

```powershell
python -m technosig_lab.cli real-candidates ^
  --input path\to\file.h5 ^
  --max-frames 64 ^
  --max-bins 2048 ^
  --chunk-bins 256 ^
  --max-chunks 16 ^
  --peak-threshold-z 6 ^
  --out results/real_candidates.jsonl

python -m technosig_lab.cli real-thresholds ^
  --input path\to\file.h5 ^
  --max-frames 64 ^
  --max-bins 2048 ^
  --chunk-bins 256 ^
  --max-chunks 16 ^
  --out results/real_thresholds.jsonl

python -m technosig_lab.cli real-injection ^
  --input path\to\file.h5 ^
  --max-frames 64 ^
  --max-bins 2048 ^
  --chunk-bins 256 ^
  --max-chunks 16 ^
  --out results/real_injection.jsonl
```

## Voyager Positive Control

Voyager should be treated as a positive control, but not by blindly looking for
`8.4 GHz` inside a locally cut waterfall. Reduced data can use local or relative
frequency axes. The machine criterion here is morphology-first:

```powershell
python -m technosig_lab.cli real-candidates ^
  --input path\to\voyager_file.h5 ^
  --positive-control ^
  --positive-control-top-n 5 ^
  --max-frames 64 ^
  --max-bins 2048 ^
  --chunk-bins 256 ^
  --max-chunks 16 ^
  --peak-threshold-z 6 ^
  --out results/voyager_candidates.jsonl
```

PASS for this command means a `narrowband_drifting` cluster appears in the top-N
candidate clusters. That is only a shape check. A real claim still needs
drift-search scoring, header frequency checks, ON/OFF cadence, and RFI vetting.

## Recorded Voyager Control Results

Machine-readable benchmark entry:

- Catalog: `known_signals_catalog.json`
- Dataset id: `voyager1_bl_gbt_xband`
- Interpretation: known artificial carrier with Voyager-associated secondary
  narrowband-drifting responses

Positive control:

- Input: `data/voyager1_bl/voyager_f1032192_t300_v2.fil`
- Carrier window: `8420.2163-8420.2166 MHz`
- Result file: `results/voyager_candidates_carrier_run.jsonl`
- Result: PASS
- Morphology: `narrowband_drifting`
- Peaks: `6`
- `z_max`: `38.77`
- `z_mean`: `35.03`
- Morphology score: `514.85`
- Estimated drift: `-0.107 Hz/s`

Independent drift-search validation:

- Search window: `8420.18-8420.25 MHz`
- Result file: `results/voyager_thresholds_8420_18_25_run.jsonl`
- Top score: `85.42`
- P99 threshold: `31.77`
- Estimated drift: `-0.10 Hz/s`

Negative/off-target control:

- Candidate window: `8420.3000-8420.3003 MHz`
- Result file: `results/voyager_candidates_offtarget_run.jsonl`
- Result: FAIL as expected, no `narrowband_drifting` cluster
- Off-target drift-search window: `8420.30-8420.37 MHz`
- Result file: `results/voyager_thresholds_offtarget_run.jsonl`
- Top score: `4.66`
- P99 threshold: `4.59`
- Interpretation: marginal near-threshold drift-search score without a
  corresponding morphology cluster

Conclusion: the stand recovers a known artificial narrowband drifting carrier
from real filterbank data and does not label a nearby off-target window as the
same positive control. This validates the stand as a real-data positive-control
detector, not merely a synthetic-signal detector. It does not yet prove blind
survey performance, calibrated false-alarm behavior across observing
conditions, or robust RFI rejection.

Wide carrier-neighborhood scan:

- Search window: `8420.00-8420.60 MHz`
- Result file: `results/voyager_thresholds_wide_8420_00_60_absolute.jsonl`
- Chunks: `826`
- Top score: `93.86`
- P99 threshold: `5.13`
- Top estimated frequency: `8420.216449 MHz`
- Top estimated drift: `-0.10 Hz/s`

Conclusion from the wider scan: the dominant score in the broader Voyager
neighborhood lands at the tutorial carrier frequency, rather than appearing
uniformly across nearby spectrum.

### Interpretation of secondary Voyager responses

The wide-window drift-search produced several secondary responses near the main
Voyager carrier neighborhood. Targeted morphology review showed that these
secondary windows are not empty off-target regions.

```text
8420.23895-8420.23925 MHz:
  morphology: narrowband_drifting
  peaks: 4
  clusters: 1
  z_max: 13.20
  score: 86.95
  drift: -0.107 Hz/s

8420.19395-8420.19425 MHz:
  morphology: narrowband_drifting
  peaks: 5
  clusters: 1
  z_max: 10.17
  score: 109.62
  drift: -0.094 Hz/s

8420.19365-8420.19395 MHz:
  morphology: narrowband_drifting
  peaks: 5
  clusters: 1
  z_max: 12.00
  score: 115.67
  drift: -0.094 Hz/s
```

These responses are not classified as independent blind SETI candidates. They
are recorded as Voyager-associated sideband/neighborhood responses because they
appear in the same Voyager observation context and show drift values consistent
with the main carrier response.

This distinction matters:

```text
main carrier:
  strongest wide-search response
  matches known Voyager carrier neighborhood

secondary responses:
  morphology PASS
  drift-consistent with the Voyager response family
  require separate instrumental/source interpretation

off-target control:
  morphology FAIL
  no peaks
  no clusters
```

The detector therefore distinguishes an empty off-target window from structured
Voyager-associated secondary narrowband-drifting responses.

### Frequency separation analysis

The top wide-search responses form a structured frequency neighborhood around
the main carrier. The table below combines wide-search estimated frequencies
with targeted morphology-review drift estimates for the secondary rows.

| frequency_mhz | delta_from_main_khz | morphology | drift_hz_s |
| ---: | ---: | --- | ---: |
| 8420.216449386 | 0.000 | main carrier | -0.100 |
| 8420.239107666 | +22.658 | narrowband_drifting | -0.107 |
| 8420.194108997 | -22.340 | narrowband_drifting | -0.094 |
| 8420.193791106 | -22.658 | narrowband_drifting | -0.094 |

The symmetric pair at `+22.658 kHz` and `-22.658 kHz` is recorded as a
Voyager-associated secondary response pattern, not as multiple independent
signals.

The comparison JSON is generated with:

```powershell
python -m technosig_lab.cli compare-candidates ^
  --input results\voyager_thresholds_wide_8420_00_60_absolute.jsonl ^
  --reference-f0-mhz 8420.216449386 ^
  --out results\voyager_candidate_frequency_offsets.json
```

The generated JSON stores the wide-search drift estimates from
`results/voyager_thresholds_wide_8420_00_60_absolute.jsonl`.

## BLDR1 S-band HIP194 Initial Evidence

`hip194_sband_2375p931_bldr1_event` is tracked as an initial BLDR1 S-band
event-table candidate, not as a known positive control.

- File: `data/bl_sband/spliced_blc2021222324252627_guppi_57992_30093_HIP194_0014.gpuspec.0000.h5`
- Header summary: `data/bl_sband/HIP194_2375p931_HEADER_SUMMARY.json`
- Evidence report: `results/hip194_sband_candidate_evidence.json`
- Catalog status: `initial_analysis_complete`
- Classification: `low_strength_event_table_hit`
- Candidate strength: `low`

Header checks:

```text
frequency range: 1818.457034-2720.800781 MHz
contains event frequency 2375.931298 MHz: true
contains requested 2380.000000 MHz: true
```

Gate report:

```text
score_p99: weak_pass
quantile_stability: fail_borderline_tail
drift: fail_static
static_rejection: fail_static_supported
morphology: fail_no_narrowband_drifting
neighbor_window: pass_locally_distinct
positive_control: fail_not_known_positive
```

Quantile and static-drift evidence:

```text
score: 48.132
p99 margin: 1.598
p999 margin: 0.160
p9999 margin: 0.016

score/p99: 1.034
score/p999: 1.003
score/p9999: 1.0003

nonstatic rerun, abs(drift) >= 0.5 Hz/s:
  nonstatic score: 7.688
  score drop: 40.443
  top nonstatic drift: -0.55 Hz/s
```

Neighbor-window vetting:

```text
event rank among five neighbor windows: 1
neighbor max score: 4.952
gate: pass_locally_distinct
```

The event window has a weak local score excess, but the top drift is
static/near-static and the morphology gate does not recover a
`narrowband_drifting` cluster. It is therefore recorded as initial detector
evidence only.
