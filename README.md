# Unmasking the Board: Behavioral Anomaly Detection in Human Chess

BCSAI - Machine Learning Foundations | Group project | IE University 2026

**Team:** Christoph, Kiril, Ali, Georgy

Status: Approved by Prof. Matteo Turilli

Group repository: [github.com/Kiril-P/Final-Group-Project-Machine-Learning](https://github.com/Kiril-P/Final-Group-Project-Machine-Learning)

---

## Deliverables

| Deliverable | File |
|---|---|
| 📄 Report (PDF) | [`deliverables/report.pdf`](deliverables/report.pdf) |
| 🖼️ Poster (A1 PDF) | [`deliverables/poster.pdf`](deliverables/poster.pdf) |
| 📊 Presentation (PPTX) | [`deliverables/presentation.pptx`](deliverables/presentation.pptx) |
| 📊 Pre-computed results | [`results/`](results/) — all charts, CSVs, and metrics |
| 📓 Notebooks | `notebooks/01_eda.ipynb`, `02_preprocessing.ipynb`, `03_modeling.ipynb` |

---

## Project Summary

This project detects unusual player behavior patterns in online chess games using unsupervised anomaly detection. Game-level Lichess data is reshaped into player-level records, 21 behavioral and engine-accuracy features are engineered, and six detectors (Z-Score baseline, LOF, Isolation Forest, One-Class SVM, Autoencoder, ACPLSubAutoencoder) are compared against synthetic anomaly injections. An ensemble (LOF + Autoencoder + OC-SVM, majority vote ≥ 2/3) flags players for human review. The objective is to identify statistically unusual behavior clusters, not to assign definitive cheating labels.

**Key result:** LOF achieves CV ROC-AUC = 0.959 ± 0.030 and test AUC = 0.971 on subtle synthetic injection. 312 of 17,909 players (1.7%) were flagged by the ensemble.

---

## What Can Be Run

### ✅ 1. Verify the code works — run tests (no dataset needed, ~10 seconds)

```bash
python -m pytest
```

Expected output: `8 passed`. This verifies all feature engineering, splitting, and validation logic without requiring any data download.

### ✅ 2. Verify the small dataset loads (Kaggle mirror, already in repo)

```bash
python -c "from src.data_loader import load_raw; print(load_raw().shape)"
```

Expected output: `(20058, 16)`. The small Kaggle mirror (`data/raw/games.csv`) is included and loads automatically.

### ✅ 3. View pre-computed results (no setup needed)

All pipeline outputs from the full Lichess run are committed to [`results/`](results/):

| File | Contents |
|---|---|
| `results/all_player_results.csv` | Anomaly scores + ensemble flags for all 17,909 players |
| `results/cv_summary.csv` | 5-fold CV ROC-AUC mean ± std per model |
| `results/holdout_evaluation.csv` | Test-set metrics (AUC, AP, Recall@k) per model |
| `results/feature_importance.png` | Permutation importance chart |
| `results/umap_overview.png` | UMAP projection of all players |
| `results/roc_curves_subtle.png` | ROC curves (subtle injection, test set) |
| `results/model_agreement_matrix.png` | Model agreement matrix |
| `results/learning_curves.png` | LOF data-efficiency curve |

### ✅ 4. Run the notebooks (methodology walkthrough on small dataset, ~5 minutes)

> **Note:** The notebooks run on the small Kaggle dataset (20,058 games, 299 players, 8 features) to demonstrate methodology interactively. The numbers they produce (e.g. LOF AUC ≈ 0.74) differ from the report because the report uses the full Lichess dataset. Pre-computed results from the full run are loaded automatically in Section 12 of notebook 03.

```bash
jupyter lab notebooks/01_eda.ipynb
```

### ⏱️ 5. Reproduce the full pipeline (requires Lichess dataset — hours)

The results in `results/` were produced by running the full pipeline on the Lichess July 2016 dataset (6.25M games). To reproduce exactly:

**Step 1 — Download the dataset from Kaggle:**

```bash
# Option A: Kaggle CLI (requires ~/.kaggle/kaggle.json credentials)
kaggle datasets download -d arevel/chess-games
unzip chess-games.zip -d data/raw/

# Option B: Manual — go to https://www.kaggle.com/datasets/arevel/chess-games
# Download the CSV and place it at: data/raw/lichess_jul2016.csv
```

**Step 2 — Verify the file is in the right place:**

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/lichess_jul2016.csv', nrows=5); print(df.shape[1], 'columns found')"
```

**Step 3 — Run the pipeline:**

```bash
python -m src.pipeline
```

This takes several hours on a standard laptop and writes all outputs to `results/`. All outputs are already committed to `results/` so this step is optional — the pre-computed results are ready to use immediately.

---

## Setup

### Prerequisites

- Python 3.10+ (3.12 recommended)
- **Windows only:** JupyterLab requires Long Path support. Run once in PowerShell as Administrator:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

Then restart your terminal. Alternatively, clone to a short path like `C:\ML\`. Tests and the pipeline core are not affected by this limit.

### Install dependencies

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Conda (any platform):**
```bash
conda env create -f environment.yml
conda activate chess-anomaly
```

---

## Project Layout

```
├── deliverables/
│   ├── report.pdf            # Project report
│   ├── poster.pdf            # A1 poster (landscape)
│   └── presentation.pptx     # 10-slide presentation
├── data/
│   ├── raw/games.csv         # Small Kaggle mirror (committed); replace with Lichess for full run
│   └── processed/            # Optional intermediate artifacts
├── docs/                     # Professor feedback, decisions log
├── decisions.md              # Methodological decision log (all major choices documented)
├── results/                  # ← Pre-computed pipeline outputs (charts, CSVs, metrics)
├── models/                   # Saved model files — generated on pipeline run, gitignored
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb# Feature engineering and leakage fix (Stage 2e)
│   └── 03_modeling.ipynb     # Model training, evaluation, injection recovery
├── src/
│   ├── config.py             # Feature definitions, hyperparameter search spaces
│   ├── data_loader.py        # Raw data loading and cleaning
│   ├── lichess_loader.py     # Lichess PGN parser and player aggregation
│   ├── features.py           # Feature engineering and train/val/test split
│   ├── models.py             # All six detectors + ensemble
│   ├── validation.py         # Synthetic injection, CV, metrics
│   ├── interpretation.py     # Permutation importance, UMAP, SHAP, failure analysis
│   └── pipeline.py           # End-to-end orchestration
├── tests/
│   └── test_features.py      # 8 unit tests (run with: python -m pytest)
├── scripts/
│   ├── setup.sh              # One-shot setup (macOS/Linux)
│   ├── setup.ps1             # One-shot setup (Windows)
│   └── download_kaggle_dataset.py
├── Makefile                  # make test / make pipeline / make lab
├── environment.yml           # Conda environment
├── requirements.txt          # pip dependencies
└── decisions.md              # Full methodological decision log
```
