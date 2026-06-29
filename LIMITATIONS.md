# Limitations

This stand is deliberately conservative.

- `real-injection` injects at the waterfall level, not at IQ/baseband level. It
  is useful for testing detector behavior on real background, but it is not a
  full receiver-chain simulation.
- `narrowband_drifting` is a morphology label, not proof of source. RFI can
  produce a narrow drifting-looking cluster.
- Synthetic thresholds calibrated on AWGN are not valid real-radio thresholds.
  Use `thresholds-synthetic` to see this failure mode and `real-thresholds`
  for actual BL background slices.
- Drift search is grid-based. If the true drift sits between grid points, the
  score and parameter estimate can be biased. A later version should add
  coarse-to-fine refinement.
- The BL adapter targets reduced waterfall/filterbank data. Raw/baseband files
  are intentionally out of scope for the first stand.
- A single file is not enough for an unknown SETI candidate. Unknown candidates
  need ON/OFF checks, repeat observations, header validation, and RFI vetting.

