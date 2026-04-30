#!/usr/bin/env python3
"""
Download the Lichess games CSV from Kaggle (datasnaek/chess) into data/raw/games.csv.

Looks for the Kaggle CLI in this order:
  1. .venv/bin/kaggle (macOS / Linux)
  2. .venv/Scripts/kaggle.exe (Windows)
  3. Any `kaggle` on PATH

If the CLI or API credentials are missing, exits with code 0 and a clear message so
`scripts/setup.sh` can continue without failing the whole bootstrap.

Credentials: https://www.kaggle.com/settings → API token → ~/.kaggle/kaggle.json (chmod 600).

Usage (from repository root):
  .venv/bin/python scripts/download_kaggle_dataset.py
  .venv/bin/python scripts/download_kaggle_dataset.py --force   # re-download
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_kaggle_bin(root: Path) -> Path | None:
    candidates = [
        root / ".venv" / "bin" / "kaggle",
        root / ".venv" / "Scripts" / "kaggle.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    which = shutil.which("kaggle")
    return Path(which) if which else None


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    token_file = Path.home() / ".kaggle" / "kaggle.json"
    return token_file.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasnaek/chess to data/raw/games.csv")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if games.csv already exists.",
    )
    args = parser.parse_args()

    root = _repo_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "games.csv"

    if target.exists() and not args.force:
        print(f"[download] Already present: {target}")
        return 0

    kaggle_bin = _find_kaggle_bin(root)
    if not kaggle_bin:
        print(
            "[download] SKIP: Kaggle CLI not found. After `pip install kaggle`, run this script again "
            "or place games.csv in data/raw/ manually.\n"
            "           https://www.kaggle.com/datasets/datasnaek/chess"
        )
        return 0

    if not _has_kaggle_credentials():
        print(
            "[download] SKIP: No Kaggle API credentials.\n"
            "           Create ~/.kaggle/kaggle.json from https://www.kaggle.com/settings "
            "(macOS/Linux: chmod 600 ~/.kaggle/kaggle.json)\n"
            "           Or copy games.csv into data/raw/ by hand."
        )
        return 0

    cmd = [
        str(kaggle_bin),
        "datasets",
        "download",
        "-d",
        "datasnaek/chess",
        "-p",
        str(raw_dir),
        "--unzip",
        "--force",
    ]
    print("[download] Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[download] ERROR: Kaggle command failed (exit {e.returncode}).", file=sys.stderr)
        return 1

    if not target.exists():
        nested = raw_dir / "chess" / "games.csv"
        if nested.exists():
            shutil.move(str(nested), str(target))
            shutil.rmtree(raw_dir / "chess", ignore_errors=True)

    if not target.exists():
        found = list(raw_dir.rglob("games.csv"))
        if found:
            shutil.copy2(found[0], target)
            print(f"[download] Copied {found[0]} -> {target}")
        else:
            print(f"[download] ERROR: games.csv not found under {raw_dir} after unzip.", file=sys.stderr)
            return 1

    print(f"[download] OK: {target} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
