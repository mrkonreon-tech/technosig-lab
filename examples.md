# Examples

## Calibrate Synthetic Background Thresholds

```powershell
python -m technosig_lab.cli thresholds-synthetic ^
  --calibration-trials 80 ^
  --validation-trials 80 ^
  --target-pfa 0.01 ^
  --out results/thresholds_synthetic.jsonl
```

## Measure Synthetic Detection Probability

```powershell
python -m technosig_lab.cli synthetic-pd ^
  --calibration-trials 80 ^
  --trials 40 ^
  --snrs -30 -25 -20 -15 -10 -5 0 ^
  --out results/synthetic_pd.jsonl
```

## Find Real Candidate Clusters

```powershell
python -m technosig_lab.cli real-candidates ^
  --input path\to\file.h5 ^
  --peak-threshold-z 6 ^
  --max-frames 64 ^
  --max-bins 2048 ^
  --chunk-bins 256 ^
  --max-chunks 16 ^
  --out results/real_candidates.jsonl
```

If the file is huge or the target carrier is outside the first slice, set
`--f-start-mhz` and `--f-stop-mhz`, or increase `--max-bins`.

## Voyager Positive Control

```powershell
python -m technosig_lab.cli real-candidates ^
  --input path\to\voyager_file.h5 ^
  --positive-control ^
  --positive-control-top-n 5 ^
  --out results/voyager_candidates.jsonl
```

Then run:

```powershell
python -m technosig_lab.cli real-thresholds ^
  --input path\to\voyager_file.h5 ^
  --out results/voyager_thresholds.jsonl
```

The first command checks morphology. The second checks drift-search score against
the observed background.

