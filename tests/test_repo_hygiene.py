from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "data",
    "results",
    "figures",
}
FORBIDDEN_SUFFIXES = {".h5", ".fil", ".raw", ".fits"}
TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024


def tracked_or_public_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        return [ROOT / line for line in result.stdout.splitlines() if line.strip()]

    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(ROOT).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts):
            continue
        files.append(path)
    return files


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_large_or_raw_data_files_in_public_tree(self) -> None:
        bad = []
        for path in tracked_or_public_files():
            suffix = path.suffix.lower()
            size = path.stat().st_size
            if suffix in FORBIDDEN_SUFFIXES or size > MAX_TRACKED_FILE_BYTES:
                bad.append(str(path.relative_to(ROOT)))
        self.assertEqual(bad, [])

    def test_no_local_absolute_paths_in_text_files(self) -> None:
        forbidden = [
            "C:" + "\\Users" + "\\",
            "C:" + "/Users/",
            "D:" + "\\",
            "Admin" + "istrator",
            "/mnt/" + "data/",
            "file:" + "//",
        ]
        hits = []
        for path in tracked_or_public_files():
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in forbidden:
                if needle in text:
                    hits.append(f"{path.relative_to(ROOT)} contains {needle}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
