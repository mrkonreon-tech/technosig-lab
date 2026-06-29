param(
    [switch]$DownloadData
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$dataDir = Join-Path $repoRoot "data/voyager1_bl"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$manifestPath = Join-Path $dataDir "SOURCE_MANIFEST.json"
$manifest = @"
{
  "dataset": "Breakthrough Listen Voyager 1 GBT filterbank",
  "file": "voyager_f1032192_t300_v2.fil",
  "source_url": "https://storage.googleapis.com/gbt_fil/voyager_f1032192_t300_v2.fil",
  "source_tutorial": "https://github.com/UCBerkeleySETI/breakthrough/blob/master/GBT/voyager/voyager.ipynb",
  "format": "SIGPROC filterbank",
  "target": "VOYAGER1",
  "instrument": "Green Bank Telescope",
  "band": "X-band",
  "known_signal": true,
  "notes": "Real Breakthrough Listen filterbank data. Suitable for positive-control validation."
}
"@
$manifest | Set-Content -Encoding UTF8 $manifestPath
Write-Host "Wrote $manifestPath"

if (-not $DownloadData) {
    Write-Host "Data download skipped. Re-run with -DownloadData to fetch the 504 MB filterbank file."
    exit 0
}

$outFile = Join-Path $dataDir "voyager_f1032192_t300_v2.fil"
$url = "https://storage.googleapis.com/gbt_fil/voyager_f1032192_t300_v2.fil"

Write-Host "Downloading $url"
curl.exe -L -C - -o $outFile $url
Write-Host "Wrote $outFile"
