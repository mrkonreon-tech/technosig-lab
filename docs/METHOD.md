# Method

`technosig_lab` is a reproducible benchmark and vetting harness for narrowband
drifting-signal detection in synthetic and real waterfall backgrounds.

It is not a claim detector. Its main purpose is to make candidate strength
auditable.

## Workflow

```text
synthetic signal checks
real-data positive control
real-data off-target control
wide-window scan
candidate gate report
catalog entry
```

## Positive Controls

Known artificial signals, such as the Voyager 1 Breakthrough Listen observation,
are used to check that the detector can recover an expected narrowband drifting
carrier in real data.

Positive-control PASS does not imply blind survey performance.

## Event-Table Candidates

BLDR event-table entries can be used as sanity tests against real backgrounds.
They are not automatically positive controls.

The HIP194 S-band example is intentionally classified as low strength: it passes
file/header/coverage checks and has weak local score evidence, but fails
quantile-stability, static-drift, and morphology gates.
