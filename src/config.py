"""
Central configuration for the chess anomaly detection project.
Edit paths and hyperparameters here rather than in individual scripts.
"""

import os
import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "games.csv"       # small Kaggle dataset (~20k games)
DATA_LICHESS = ROOT_DIR / "data" / "raw" / "lichess_jul2016.csv"  # big dataset (6.25M games)
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Make sure output directories exist before anything tries to write to them
for d in (DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Stockfish ────────────────────────────────────────────────────────────────

# Check STOCKFISH_PATH env var first (CI/server), fall back to system PATH, then a default.
# Stockfish is only needed for the optional ACPL features on the small dataset.
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "/usr/bin/stockfish"
STOCKFISH_DEPTH = 15          # search depth per position — deeper = more accurate but slower
STOCKFISH_SAMPLE_GAMES = 500  # analysing every game is too slow; 500 is a practical sample

# ── Data ─────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42  # fixed seed so every run is reproducible — important for paper/report

# ── Lichess large dataset ────────────────────────────────────────────────────

# 500k games gives ~50k unique players after filtering — enough for robust anomaly detection
# without needing to process all 6.25M games on a laptop.
LICHESS_SAMPLE_N = 500_000

# We focus on rapid and classical because cheating is more plausible with longer time controls.
# Bullet/blitz players physically can't consult an engine between moves, so anomaly signals
# there are mostly noise. Restricting to slower games means our anomalies are more meaningful.
LICHESS_TIME_CONTROLS = ["rapid", "classical"]

# Only include eval-based features (blunder rate, best move rate, etc.) for players where
# at least 50% of their games have Stockfish eval annotations. Below that threshold the
# feature would just be noise from an unrepresentative sample of their games.
MIN_EVAL_COVERAGE = 0.5

# Rating bands for within-band normalisation — a 1200 with 70% win rate is more suspicious
# than a 2000 with 70% win rate, so we need to compare players to their peers, not globally.
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

# Minimum games per player before we trust their aggregated stats.
# With fewer than 5 games, win rate and rating volatility are meaningless noise.
MIN_GAMES_PER_PLAYER = 5

# Minimum games before a player is eligible to be flagged by the ensemble.
# With very few games, aggregated stats have high variance and produce false positives
# purely from statistical noise — not from genuine anomalous behaviour.
# Analysis on our dataset showed the median flagged player had only 10 games vs 29 for
# normal players; 416/915 flags had <10 games. At 15 games the major noise source is
# eliminated while retaining 71% of all players (12,738 out of 17,909).
# Players below this threshold are still scored and stored in results CSVs — only their
# ensemble_flag and ensemble_confident values are forced to False.  They still contribute
# to model training as normal examples, which is correct: a player with 8 games looks
# normal to the model (small sample), and that is fine.
MIN_GAMES_FOR_FLAG = 15
TIME_CONTROLS = ["blitz", "rapid", "classical", "bullet"]

# ── Feature Engineering ─────────────────────────────────────────────────────

# The 8 base features available on both datasets (see features.py for full definitions).
# These are what actually go into the models — kept here for reference / downstream use.
FEATURE_COLUMNS = [
    "win_rate",
    "win_rate_vs_expected",
    "avg_turns",
    "opening_ply_ratio",
    "victory_efficiency",
    "rating_volatility",
    "performance_vs_actual",
    "underdog_win_rate",
]

# ── Anomaly Detection ─────────────────────────────────────────────────────────

# 5% contamination = we expect roughly 1 in 20 players might be behaving anomalously.
# This is a reasonable prior for an online chess platform where cheating does happen.
CONTAMINATION = 0.05

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": CONTAMINATION,
    "max_samples": "auto",       # "auto" = min(256, n_samples) — sklearn's recommended default
    "random_state": RANDOM_SEED,
}

OCSVM_PARAMS = {
    "kernel": "rbf",
    "nu": CONTAMINATION,   # nu is an upper bound on the fraction of outliers — maps to contamination
    "gamma": "scale",      # scale = 1 / (n_features * X.var()) — usually better than "auto"
}

AUTOENCODER_PARAMS = {
    "encoding_dim": 4,          # compress 8 features → 4-dim bottleneck
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "reconstruction_threshold_percentile": 95,  # flag top 5% highest reconstruction errors
    # Top-k mean scoring: set to 0 to use plain mean (current behaviour).
    # k>0 was evaluated empirically (k=3) and reverted — see Decision 36 in decisions.md.
    "scoring_top_k": 0,
}

# Eval features get extra weight in the AE reconstruction loss so that suspicious
# eval signals aren't diluted by the many behavioral features sitting near zero.
# Weight 3.0 means eval errors count 3× more; normalised so mean weight = 1.0,
# which keeps loss scale and learning rate comparable to the unweighted baseline.
AUTOENCODER_EVAL_WEIGHT: float = 3.0
AUTOENCODER_EVAL_FEATURES: frozenset = frozenset({
    "avg_acpl_band_z",
    "avg_weighted_acpl_band_z",
    "avg_acpl_middlegame_band_z",
    "avg_acpl_opening_band_z",
    "avg_acpl_endgame_band_z",
    "acpl_consistency_band_z",
    "acpl_phase_gap_band_z",
    "blunder_rate",
    "best_move_rate_band_z",
    "comeback_rate",
    "performance_vs_actual",
    "underdog_win_rate",
})

# Pure chess-accuracy features used by ACPLSubAutoencoder.
# Subset of AUTOENCODER_EVAL_FEATURES: strictly move-quality signals (ACPL variants,
# blunder rate, best-move rate).  Outcome-based signals (comeback_rate,
# performance_vs_actual, underdog_win_rate) are excluded because they can reflect
# legitimate skill rather than engine assistance and would dilute the sub-model's
# focused accuracy signal.
ACPL_SUB_FEATURES: frozenset = frozenset({
    "avg_acpl_band_z",
    "avg_weighted_acpl_band_z",
    "avg_acpl_middlegame_band_z",
    "avg_acpl_opening_band_z",
    "avg_acpl_endgame_band_z",
    "acpl_consistency_band_z",
    "acpl_phase_gap_band_z",
    "blunder_rate",
    "best_move_rate_band_z",
})

N_SYNTHETIC_ANOMALIES = 50  # injected per evaluation run — enough signal without dominating the set
ALPHA = 0.05                # significance level for statistical tests

# ── Hyperparameter Search Spaces ─────────────────────────────────────────────
#
# Each dict maps parameter name -> list of candidate values.
# run_hyperparameter_search() samples uniformly from these lists (20 random draws).
#
# Why these ranges:
#   contamination / nu : 1–15% covers plausible cheating prevalence estimates
#                        in prior chess integrity literature.
#   n_estimators       : returns diminish above 300 for datasets of this size.
#   max_samples        : "auto" or explicit integers — sklearn ≥1.3 rejects
#                        floats here (0.5, 0.8 would raise a ValueError).
#                        128/256 give diversity without being too large.
#   n_neighbors (LOF)  : 10–40 spans local-to-semi-global density estimation.
#   kernel / gamma     : standard RBF search; poly is a cheap alternative.
#   encoding_dim (AE)  : 2–8 spans lossy compression of an 8-feature input.
#   threshold_pct (AE) : 90–99 controls recall/precision trade-off on recon error.

RANDOM_SEARCH_N_ITER = 20  # 20 random draws per model — cost-effective for this dataset size

ISOLATION_FOREST_SEARCH: dict = {
    "contamination": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "n_estimators":  [100, 150, 200, 300],
    # Must be "auto" or an integer. We keep only "auto" because numpy's rng.choice()
    # on a mixed list (string + int) upcasts everything to string, which causes sklearn
    # to reject the value. "auto" = min(256, n_samples) which is exactly what we want
    # anyway with 20k+ training players.
    "max_samples":   ["auto"],
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

# HDBSCAN finds clusters of arbitrary shape and marks everything outside as noise.
# No contamination parameter needed for the algorithm itself — we only use it to
# set our anomaly score threshold after fitting.
HDBSCAN_PARAMS = {
    "min_cluster_size": 15,    # minimum points to form a cluster
    "min_samples":      5,     # how many neighbors define a core point
    "contamination":    CONTAMINATION,
}

HDBSCAN_SEARCH: dict = {
    # min_cluster_size is the main knob: smaller = more sensitive (more anomalies flagged),
    # larger = more conservative. We search a wide range since optimal value depends on
    # how dense the normal player population is in feature space.
    "min_cluster_size": [10, 15, 20, 30, 50],
    # min_samples controls how strict the core-point definition is.
    # Higher = tighter clusters, more noise points = more anomalies.
    "min_samples":      [5, 10, 15],
    # contamination: only affects our score threshold, not the HDBSCAN algorithm.
    "contamination":    [0.03, 0.05, 0.08, 0.10],
}

# Autoencoder search is split: we first run cheap 30-epoch trials to find the best
# encoding_dim and threshold percentile, then re-train the winner at full epochs.
# This keeps total search cost manageable without sacrificing architecture quality.
AUTOENCODER_SEARCH: dict = {
    "encoding_dim":                      [2, 4, 6, 8],
    "reconstruction_threshold_percentile": [90, 93, 95, 97, 99],
}
AUTOENCODER_SEARCH_EPOCHS = 30   # epochs used during search trials only (not final training)
