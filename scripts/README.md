# Reproducibility Scripts

These PowerShell scripts wrap the commands documented in the README.

They assume they are run from the repository root.

## Data Policy

The scripts write local data under `data/` and generated outputs under
`results/`. Both directories are ignored by git.

Large upstream files are not downloaded unless the script exposes an explicit
download switch.

## Typical Order

```powershell
pwsh scripts/fetch_voyager_positive_control.ps1 -DownloadData
pwsh scripts/run_voyager_positive_control.ps1
pwsh scripts/run_hip194_vetting.ps1
```

`run_hip194_vetting.ps1` expects the large HIP194 HDF5 file to already exist
locally. It does not download the file automatically.
