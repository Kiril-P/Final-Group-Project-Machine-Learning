# Unmasking the Board: Behavioral Anomaly Detection in Human Chess

BCSAI - Machine Learning Foundations | Group project

Status: Approved by Prof. Matteo Turilli

Group repository: [github.com/Kiril-P/Final-Group-Project-Machine-Learning](https://github.com/Kiril-P/Final-Group-Project-Machine-Learning)

This project detects unusual player behavior patterns in online chess games using unsupervised anomaly detection. The workflow starts from game-level Lichess data, reshapes it into player-level records, engineers behavioral features, and then compares multiple detectors (Isolation Forest, One-Class SVM, Local Outlier Factor, and an autoencoder) against synthetic anomaly injections. The objective is to identify statistically unusual behavior clusters for analysis, not to assign definitive cheating labels.

## What The Pipeline Does

Running the full pipeline with `python -m src.pipeline` executes these stages:

1. Load and clean raw game data from `data/raw/games.csv`.
2. Build player-level aggregates and engineered features.
3. Split data into train/validation/test (70/15/15) and fit scaling on train only.
4. Tune model hyperparameters on injected validation anomalies.
5. Run 5-fold cross-validation on the development split.
6. Train all models and evaluate on validation and holdout test injections.
7. Export metrics, feature importance, and failure analysis outputs.

Main outputs are written to `results/` (metrics and analysis tables) and `models/` (saved fitted models).

## Prerequisites

- Python 3.10+ (3.12 recommended).

## Setup And Run

Use one of the two workflows below.

### macOS/Linux

1. Go to the project root.

~~~bash
cd "/path/to/Final-Group-Project-Machine-Learning"
~~~

2. Create the environment, install dependencies, and attempt dataset download.

~~~bash
bash scripts/setup.sh
~~~

3. Activate the environment.

~~~bash
source .venv/bin/activate
~~~

4. Run tests.

~~~bash
python -m pytest
~~~

5. Verify the dataset is readable.

~~~bash
python -c "from src.data_loader import load_raw; print(load_raw().shape)"
~~~

6. Run the full pipeline.

~~~bash
python -m src.pipeline
~~~

### Windows (PowerShell)

1. Go to the project root.

~~~powershell
cd "C:\path\to\Final-Group-Project-Machine-Learning"
~~~

2. Create the environment, install dependencies, and attempt dataset download.

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
~~~

3. Activate the environment.

~~~powershell
& .\.venv\Scripts\Activate.ps1
~~~

4. Run tests.

~~~powershell
python -m pytest
~~~

5. Verify the dataset is readable.

~~~powershell
python -c "from src.data_loader import load_raw; print(load_raw().shape)"
~~~

6. Run the full pipeline.

~~~powershell
python -m src.pipeline
~~~

## Dataset

The setup scripts above already call `scripts/download_kaggle_dataset.py`.

- If Kaggle CLI and credentials are configured, `data/raw/games.csv` is downloaded automatically.
- If Kaggle is not configured, the script prints a skip message and setup still completes.

Dataset source website: [Kaggle - Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess)

Kaggle API credentials page: [Kaggle Settings](https://www.kaggle.com/settings)

If download is skipped, manually place `games.csv` in `data/raw/`, then rerun:

~~~bash
python -c "from src.data_loader import load_raw; print(load_raw().shape)"
~~~

## Daily Use (After First Setup)

Use this when `.venv` and dependencies are already installed.

### macOS/Linux

1. Go to the project root and activate the environment.

~~~bash
cd "/path/to/Final-Group-Project-Machine-Learning"
source .venv/bin/activate
~~~

2. Run the pipeline.

~~~bash
python -m src.pipeline
~~~

3. Optional checks.

~~~bash
python -m pytest
python -m jupyter lab notebooks/01_eda.ipynb
~~~

### Windows (PowerShell)

1. Go to the project root and activate the environment.

~~~powershell
cd "C:\path\to\Final-Group-Project-Machine-Learning"
& .\.venv\Scripts\Activate.ps1
~~~

2. Run the pipeline.

~~~powershell
python -m src.pipeline
~~~

3. Optional checks.

~~~powershell
python -m pytest
python -m jupyter lab notebooks/01_eda.ipynb
~~~

## Project Layout

~~~text
├── data/
│   ├── raw/              # games.csv (not committed)
│   └── processed/        # optional intermediate artifacts
├── docs/
├── models/               # saved model files (.pkl, .pt)
├── notebooks/            # EDA and modeling notebooks
├── results/              # pipeline outputs, metrics, analysis tables
├── scripts/              # setup and dataset bootstrap scripts
├── src/                  # project code (loading, features, models, validation)
├── tests/
├── Makefile
├── environment.yml
├── requirements.txt
├── pytest.ini
└── README.md
~~~

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | Data inspection, quality checks, distributions |
| `notebooks/03_modeling.ipynb` | Model comparison and anomaly detection analysis |
