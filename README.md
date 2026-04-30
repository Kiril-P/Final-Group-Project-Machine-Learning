# Unmasking the Board: Behavioral Anomaly Detection in Human Chess

BCSAI — Machine Learning Foundations | Group project

**Status:** Approved by Prof. Matteo Turilli

**Group repository:** [github.com/Kiril-P/Final-Group-Project-Machine-Learning](https://github.com/Kiril-P/Final-Group-Project-Machine-Learning)

Unsupervised anomaly detection on aggregated player behavior (Lichess-style games), framed as **detectable deviations** from rating-group norms — not definitive “smurfing” labels.

---

## Prerequisites

- **Python 3.10+** (3.12 recommended; matches course tooling).
- **Disk space** for the Kaggle sample (~3 MB) and generated figures.

---

## First-time setup (recommended)

Pick **one** of the following; each creates a **`.venv`** in the project root, installs everything from `requirements.txt`, and tries to download **`data/raw/games.csv`** when Kaggle is configured (otherwise it prints skip instructions and continues).

### macOS / Linux

```bash
cd "/path/to/Behavioral Anomaly Detection in Human Chess"
bash scripts/setup.sh
source .venv/bin/activate
```

Optional environment variables:

| Variable | Effect |
|----------|--------|
| `PYTHON=/path/to/python3.12` | Use a specific interpreter to create `.venv`. |
| `SKIP_DATA_DOWNLOAD=1` | Do not run the Kaggle step. |
| `INSTALL_JUPYTER_KERNEL=1` | Register a Jupyter kernel named `chess-anomaly` for this venv. |

### Windows (PowerShell)

```powershell
cd "C:\path\to\Behavioral Anomaly Detection in Human Chess"
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Same optional variables: `PYTHON`, `SKIP_DATA_DOWNLOAD`, `INSTALL_JUPYTER_KERNEL`.

### Conda / Mamba (optional)

If your team prefers Conda instead of `.venv`:

```bash
conda env create -f environment.yml
conda activate chess-anomaly
python scripts/download_kaggle_dataset.py
```

The `pip:` section installs the same packages as `requirements.txt`.

### Make (macOS / Linux)

```bash
make setup      # .venv + pip install + dataset download (same as setup.sh)
make test       # pytest (uses .venv when present)
make lab        # Jupyter Lab → 01_eda.ipynb
```

Run `make help` for all targets.

---

## Manual setup (if you prefer not to use the scripts)

```bash
cd "/path/to/Behavioral Anomaly Detection in Human Chess"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
python scripts/download_kaggle_dataset.py
```

Always **activate `.venv`** before running notebooks, tests, or `python -m src.pipeline` so Jupyter and the shell use the same packages.

---

## Data (`data/raw/games.csv`)

The bootstrap scripts run **`scripts/download_kaggle_dataset.py`**, which:

1. Prefers **`.venv/bin/kaggle`** (or **`.venv\Scripts\kaggle.exe`**) so it works right after `pip install` without relying on global PATH.
2. Exits successfully with a **SKIP** message if the CLI or **`~/.kaggle/kaggle.json`** is missing (teammates without Kaggle are not blocked).

To enable automatic download:

1. [Kaggle → Settings → API](https://www.kaggle.com/settings) → create token.
2. Place `kaggle.json` in **`~/.kaggle/`** and on macOS/Linux run: `chmod 600 ~/.kaggle/kaggle.json`.

**Manual fallback:** download [Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess) and put **`games.csv`** in **`data/raw/`**.

---

## Verify

```bash
source .venv/bin/activate
python -m pytest
python -c "from src.data_loader import load_raw; print(load_raw().shape)"
```

---

## Notebooks

Use the activated **`.venv`** as the Jupyter kernel (or run `INSTALL_JUPYTER_KERNEL=1 bash scripts/setup.sh` once).

```bash
python -m jupyter lab notebooks/01_eda.ipynb
```

Notebooks add the repo root to `sys.path`, so they work whether the process cwd is the repo root or `notebooks/`.

---

## Full pipeline (optional)

```bash
python -m src.pipeline
```

Outputs under `results/` and `models/`.

---

## Optional: Stockfish (ACPL)

```bash
brew install stockfish   # macOS
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish   # Apple Silicon example
```

---

## Repository layout

```
├── data/
│   ├── raw/              # games.csv (download; not committed)
│   └── processed/        # Intermediate tables (optional)
├── docs/                 # Proposal, feedback, course PDFs
├── models/               # Saved artifacts (.pkl / .pt)
├── notebooks/            # Jupyter entry points (run from repo root or notebooks/)
├── results/              # Figures and CSV outputs from notebooks / pipeline
├── scripts/              # setup.sh, setup.ps1, download_kaggle_dataset.py
├── src/                  # Importable library (loaders, features, models, validation)
├── tests/                # pytest
├── Makefile              # make setup | test | lab | …
├── environment.yml       # optional Conda/Mamba env (same deps as requirements.txt)
├── requirements.txt
├── pytest.ini
└── README.md
```

Course requirement: keep **heavy logic in `src/`** and use notebooks as a thin, top-to-bottom runnable interface.

---

## Notebooks (planned)

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | EDA, data quality, distributions |
| `notebooks/03_modeling.ipynb` | Isolation Forest, OCSVM, autoencoder comparison |

Add `02_feature_engineering`, `04_autoencoder`, `05_evaluation` as the project matures.

---

## Team

| Name | Role |
|------|------|
| Christoph | Lead |
| … | … |

---

## References

- [Kaggle: Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess)
- [Lichess open database](https://database.lichess.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Stockfish](https://stockfishchess.org/)
- [SHAP](https://shap.readthedocs.io/)
