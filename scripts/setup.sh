#!/usr/bin/env bash
# One-shot environment + dependencies (+ optional Kaggle data).
# Usage: from repo root —  bash scripts/setup.sh
# Or:    chmod +x scripts/setup.sh && ./scripts/setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found. Install Python 3.10+ or set PYTHON=/path/to/python3" >&2
  exit 1
fi

echo "==> Using interpreter: $($PYTHON -c 'import sys; print(sys.executable)')"

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment in .venv"
  "$PYTHON" -m venv .venv
else
  echo "==> Virtual environment .venv already exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install -U pip setuptools wheel

echo "==> Installing dependencies (requirements.txt)"
python -m pip install -r requirements.txt

echo "==> Dataset (data/raw/games.csv)"
if [[ "${SKIP_DATA_DOWNLOAD:-}" == "1" ]]; then
  echo "    SKIP_DATA_DOWNLOAD=1 — skipping Kaggle download."
else
  python scripts/download_kaggle_dataset.py
fi

if [[ "${INSTALL_JUPYTER_KERNEL:-}" == "1" ]]; then
  echo "==> Registering Jupyter kernel 'chess-anomaly'"
  python -m ipykernel install --user --name chess-anomaly --display-name "Chess anomaly (.venv)" || {
    echo "    (ipykernel optional — pip install ipykernel if you want this)"
  }
fi

echo ""
echo "Done. Next steps:"
echo "  source .venv/bin/activate"
echo "  python -m pytest"
echo "  python -m jupyter lab notebooks/01_eda.ipynb"
echo ""
echo "Optional: INSTALL_JUPYTER_KERNEL=1 bash scripts/setup.sh  — register Jupyter kernel"
echo "Optional: SKIP_DATA_DOWNLOAD=1 bash scripts/setup.sh     — skip Kaggle step"
