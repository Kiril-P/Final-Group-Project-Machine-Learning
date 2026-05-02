"""
Central configuration for the chess anomaly detection project.
Edit paths and hyperparameters here rather than in individual scripts.
"""

import os
import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "games.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for d in (DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Stockfish ────────────────────────────────────────────────────────────────

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "/usr/bin/stockfish"
STOCKFISH_DEPTH = 15
STOCKFISH_SAMPLE_GAMES = 500

# ── Data ─────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

RATING_BANDS = [0, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 3000]
RATING_BAND_LABELS = [
    "<800",
    "800-1000",
    "1000-1200",
    "1200-1400",
    "1400-1600",
    "1600-1800",
    "1800-2000",
    "2000-2200",
    "2200+",
]

MIN_GAMES_PER_PLAYER = 5
TIME_CONTROLS = ["blitz", "rapid", "classical", "bullet"]

# ── Feature Engineering ─────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "win_rate",
    "avg_turns",
    "opening_ply_ratio",
    "victory_efficiency",
    "rating_volatility",
    "time_control_consistency",
    "avg_acpl",
    "acpl_variance",
]

# ── Anomaly Detection ─────────────────────────────────────────────────────────

CONTAMINATION = 0.05

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": CONTAMINATION,
    "max_samples": "auto",
    "random_state": RANDOM_SEED,
}

OCSVM_PARAMS = {
    "kernel": "rbf",
    "nu": CONTAMINATION,
    "gamma": "scale",
}

AUTOENCODER_PARAMS = {
    "encoding_dim": 4,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "reconstruction_threshold_percentile": 95,
}

N_SYNTHETIC_ANOMALIES = 50
ALPHA = 0.05

# ── Hyperparameter Search Spaces ─────────────────────────────────────────────
#
# Each dict maps parameter name -> list of candidate values.
# run_hyperparameter_search() samples uniformly from these lists.
#
# Ranges are justified as follows:
#   contamination / nu : 1–15 % covers plausible cheating prevalence estimates
#                        found in prior chess integrity literature.
#   n_estimators       : diminishing returns above 300 for this dataset size.
#   max_samples        : "auto" = min(256, n) vs explicit fractions to balance
#                        diversity and bias in each tree.
#   n_neighbors (LOF)  : 10–40 spans local to semi-global density estimation.
#   kernel / gamma     : standard RBF search; "poly" included as alternative.
#   encoding_dim (AE)  : 2–8 spans lossy compression of a 6-feature input.
#   threshold_pct (AE) : 90–99 controls recall/precision trade-off on recon error.

RANDOM_SEARCH_N_ITER = 20  # random draws per model; ≈20 is cost-effective for this dataset size

ISOLATION_FOREST_SEARCH: dict = {
    "contamination": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "n_estimators":  [100, 150, 200, 300],
    "max_samples":   ["auto", 0.5, 0.8],
}

OCSVM_SEARCH: dict = {
    "nu":     [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "kernel": ["rbf", "poly"],
    "gamma":  ["scale", "auto"],
}

LOF_SEARCH: dict = {
    "contamination": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "n_neighbors":   [10, 15, 20, 30, 40],
}

# Autoencoder search is separated: architecture/threshold are searched with a
# cheap short-run (30 epochs) to find good encoding_dim and threshold percentile,
# then the best config is re-trained at full epochs.  Learning rate is kept fixed
# to cap total search cost.
AUTOENCODER_SEARCH: dict = {
    "encoding_dim":                      [2, 4, 6, 8],
    "reconstruction_threshold_percentile": [90, 93, 95, 97, 99],
}
AUTOENCODER_SEARCH_EPOCHS = 30   # epochs used during search trials (not final training)
