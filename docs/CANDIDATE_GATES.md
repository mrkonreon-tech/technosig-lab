# Candidate Gates

This project separates "score above a local threshold" from "strong candidate".
The gates are intentionally conservative.

## Score Gate

The drift-search score is compared against quantiles of chunk scores within the
same local window.

```text
score_p99: fail | weak_pass | pass
```

A weak pass means the top score is above p99 but with a small absolute or
relative margin.

## Quantile Stability Gate

The top score must stay meaningfully above deeper tail estimates.

```text
if margin_p9999 < 0.5:
    fail_borderline_tail
elif margin_p999 < 1.0:
    weak
else:
    pass
```

The report also stores score ratios:

```text
score_excess_ratio_p99
score_excess_ratio_p999
score_excess_ratio_p9999
```

## Drift Static Gate

Near-static tracks are downgraded because local, instrumental, or stationary
RFI-like structure can dominate those tracks.

```text
abs(top_drift_hz_per_sec) < 0.5 -> fail_static
```

## Static Rejection Rerun

The same window is rescored with near-static drift tracks excluded:

```text
exclude_drift_abs_lt = 0.5 Hz/s
```

The report stores:

- original score
- nonstatic score
- nonstatic score drop
- nonstatic score ratio
- nonstatic top drift

## Morphology Gate

Candidate clusters must pass the morphology detector as `narrowband_drifting`.
Sparse single-frame peaks do not pass this gate.

## Neighbor Window Gate

Neighbor windows around the event are scored with the same detector settings.
The report stores event rank, neighbor max score, and neighbor score
distribution.

## Positive Control Gate

A known-positive dataset is not the same class as an event-table candidate.
Catalog entries should use:

```text
dataset_class: positive_control
dataset_class: bldr_event_candidate
```
