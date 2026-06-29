# Data

Breakthrough Listen data files are not included in this repository.

The code expects `.h5`, `.fil`, `.raw`, or other waterfall/baseband files to be
downloaded separately into local ignored directories such as `data/`. Large
analysis outputs should be written under `results/`, also ignored by git.

For reproducible records, store metadata instead of data:

- source name
- archive URL
- file size
- hash, when computed
- frequency range from the file header
- tested frequency window
- detector command and output artifact path

## Current Public Metadata Pattern

Catalog entries may point to expected local paths, for example:

```text
data/voyager1_bl/voyager_f1032192_t300_v2.fil
data/bl_sband/spliced_blc2021222324252627_guppi_57992_30093_HIP194_0014.gpuspec.0000.h5
```

Those files are not part of the repository. Reproducers should download them
from the upstream archive and verify header coverage before running detector
commands.

## Scripts

The `scripts/` directory contains PowerShell wrappers for common reproduction
flows:

- `fetch_voyager_positive_control.ps1` writes Voyager metadata and optionally
  downloads the Voyager positive-control filterbank when `-DownloadData` is
  passed.
- `run_voyager_positive_control.ps1` runs the Voyager positive/off-target
  control commands.
- `fetch_bldr1_sband_index.ps1` downloads the public BLDR1 S-band event index
  and extracts nearest rows around 2380 MHz.
- `run_hip194_vetting.ps1` runs the HIP194 header, event-window,
  requested-window, static-rejection, and neighbor-window vetting steps. It
  requires the large HIP194 HDF5 file to already exist locally.

## Do Not Commit

Do not commit:

- large Breakthrough Listen data files
- generated `results/`
- generated `figures/`
- local virtual environments
- local absolute paths
- private notes
