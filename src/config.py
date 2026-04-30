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
