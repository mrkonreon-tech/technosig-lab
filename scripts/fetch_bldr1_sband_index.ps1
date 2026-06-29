$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

New-Item -ItemType Directory -Force -Path "data/bl_indexes" | Out-Null

$outFile = "data/bl_indexes/sband2019_events.csv"
$url = "https://seti.berkeley.edu/listen2019/sband2019_events.csv"
Invoke-WebRequest -Uri $url -OutFile $outFile
Write-Host "Wrote $outFile"

python -c @"
import csv
import json
from pathlib import Path

path = Path(r"data/bl_indexes/sband2019_events.csv")
target = 2380.0
rows = []
with path.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        try:
            freq = float(row["FreqMid"])
        except Exception:
            continue
        rows.append({
            "source": row.get("Source", ""),
            "freq_mid_mhz": freq,
            "delta_mhz": freq - target,
            "file_id": row.get("FileID", ""),
            "snr": row.get("SNR", ""),
            "drift_rates": row.get("DriftRates", ""),
        })
rows.sort(key=lambda item: abs(item["delta_mhz"]))
out = {
    "source_csv": str(path),
    "target_mhz": target,
    "nearest_count": 20,
    "nearest": rows[:20],
}
out_path = Path(r"results/sband_2380_nearest_fileids.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"saved {out_path}")
for item in rows[:5]:
    print(f"{item['freq_mid_mhz']:.6f} MHz delta={item['delta_mhz']:+.6f} source={item['source']} file={item['file_id']}")
"@
