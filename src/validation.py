"""
Validation for label-free anomaly detection: synthetic injection, ACPL correlation, stats.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import ALPHA, AUTOENCODER_SEARCH_EPOCHS, N_SYNTHETIC_ANOMALIES, RANDOM_SEED

logger = logging.getLogger(__name__)
rng = np.random.default_rng(RANDOM_SEED)


def inject_synthetic_anomalies(
    X: pd.DataFrame,
    n: int = N_SYNTHETIC_ANOMALIES,
    strategy: str = "engine_perfect",
) -> tuple[np.ndarray, np.ndarray]:
    """Append synthetic anomaly rows; return X_injected, y_true (1 = synthetic)."""
    X_arr = X.values if hasattr(X, "values") else X

    if strategy == "engine_perfect":
        # ── SANITY-CHECK benchmark, not a real-world performance metric ──────
        # Synthetic points are placed at the 99th percentile of every feature
        # in X_arr simultaneously, then jittered by 5% noise.
        #
        # WHY AUC ≈ 1.0 is EXPECTED here (and not evidence of over-fitting):
        #   LOF  → p99 points have the lowest local density by construction.
        #   IF   → extreme values are isolated in ≤ 2 splits by construction.
        #   OC-SVM → they land outside the normal-data support by construction.
        #   AE   → they produce the highest reconstruction error by construction.
        # Any functioning anomaly detector will find them. The result validates
        # that the models are numerically working, nothing more.
        #
        # The meaningful benchmark is the "subtle" strategy — those anomalies
        # stay on the data manifold and require real discriminative power.
        synthetic = np.tile(np.percentile(X_arr, 99, axis=0), (n, 1))
        synthetic += rng.normal(0, 0.05, synthetic.shape)
    elif strategy == "extreme_outlier":
        synthetic = rng.normal(3.5, 0.5, (n, X_arr.shape[1]))
    elif strategy == "subtle":
        # Smurf-like: take real player rows, perturb a couple of features by ~1.5 sd.
        # Anomalies stay close to the data manifold and are genuinely hard to separate.
        idx = rng.choice(len(X_arr), size=n, replace=True)
        synthetic = X_arr[idx].copy()
        n_features = X_arr.shape[1]
        n_perturb = max(1, n_features // 3)
        sds = X_arr.std(axis=0)
        for i in range(n):
            feats = rng.choice(n_features, size=n_perturb, replace=False)
            synthetic[i, feats] += rng.normal(0, 1.5, n_perturb) * sds[feats]

    elif strategy == "realistic_cheater":
        # ── DOMAIN-GROUNDED benchmark ─────────────────────────────────────────
        # Profile constructed from documented patterns in Lichess/Chess.com fair
        # play team reports and academic literature (Regan 2011, Guid & Bratko 2006,
        # Oxera 2023 review of computer-move detection).
        #
        # CORE INSIGHT from the literature: sophisticated engine-assisted players
        # do NOT play perfectly — they deliberately play the 2nd or 3rd engine
        # choice, occasionally allow a "blunder" to stay under detection thresholds,
        # and maintain normal behavioral patterns (game length, opening variety) to
        # blend in.  Detection therefore requires looking at the DISTRIBUTION of
        # quality across many games, not individual move perfection.
        #
        # What this means for our synthetic profile:
        #   - Push eval/performance features to p5–p10 (suspicious) or p88–p92
        #     (suspicious high) — "very good but not impossibly perfect"
        #   - Vary values across synthetic players (uniform sample in range, not
        #     a single clone) so the benchmark isn't trivially clustered
        #   - Keep all behavioral features at 0 ± small noise (population mean
        #     in scaled space) — exactly what sophisticated cheaters maintain
        #
        # Feature assignments (in StandardScaler-transformed space, 0 = mean):
        #
        #   SUSPICIOUS LOW  (very low ACPL / errors = engine-like accuracy):
        #     avg_acpl_band_z, avg_weighted_acpl_band_z  — primary ACPL signals
        #     avg_acpl_middlegame_band_z  — strongest phase: engines consulted when
        #                                   positions get complex (Guid & Bratko 2006)
        #     avg_acpl_opening_band_z, avg_acpl_endgame_band_z  — secondary phases
        #     acpl_consistency_band_z  — engines are consistently accurate; humans
        #                                fluctuate heavily game to game
        #     blunder_rate    — engines almost never drop eval > 1.5 pawns
        #     timeout_loss_rate  — engine responds instantly; never runs out of time
        #       (pushed to p8, not p1 — many legitimate players also rarely time out)
        #
        #   SUSPICIOUS HIGH (performance above peers):
        #     best_move_rate_band_z   — engine plays near-optimal move far too often
        #     win_rate                — wins more than peers at same Elo
        #     performance_vs_actual   — consistently outperforms their rating
        #     comeback_rate           — escapes losing positions abnormally often
        #     underdog_win_rate       — beats stronger players at improbable rates
        #     rating_gain_rate        — above-average climb (p75, not p92 — this
        #                               signal is noisy; new human players also improve
        #                               quickly, so we don't push it hard)
        #
        #   KEPT NORMAL (population mean ± small noise):
        #     avg_turns, turns_std, avg_opening_ply, rating_volatility,
        #     avg_opponent_rating, n_games, avg_rating, avg_rating_diff,
        #     rating_gain (raw, not rate), acpl_phase_gap_band_z
        #     — sophisticated cheaters deliberately keep these normal.

        # Start everything at population mean (0 in scaled space) + tiny jitter
        synthetic = rng.normal(0.0, 0.08, (n, X_arr.shape[1]))

        cols = list(X.columns) if hasattr(X, "columns") else []

        # (lo_pct_min, lo_pct_max) — sample uniformly in this percentile range
        suspicious_low: dict[str, tuple[float, float]] = {
            "avg_acpl_band_z":            (3,  10),
            "avg_weighted_acpl_band_z":   (3,  10),
            "avg_acpl_middlegame_band_z": (3,  10),
            "avg_acpl_opening_band_z":    (5,  12),
            "avg_acpl_endgame_band_z":    (5,  12),
            "acpl_consistency_band_z":    (5,  12),
            "blunder_rate":               (3,  10),
            "timeout_loss_rate":          (5,  12),
        }
        # (hi_pct_min, hi_pct_max)
        suspicious_high: dict[str, tuple[float, float]] = {
            "best_move_rate_band_z":  (88, 96),
            "win_rate":               (86, 94),
            "performance_vs_actual":  (88, 95),
            "comeback_rate":          (85, 93),
            "underdog_win_rate":      (82, 92),
            "rating_gain_rate":       (72, 82),   # noisy signal — mild push only
        }

        for feat, (plo, phi) in suspicious_low.items():
            if feat in cols:
                ci = cols.index(feat)
                lo = np.percentile(X_arr[:, ci], plo)
                hi = np.percentile(X_arr[:, ci], phi)
                synthetic[:, ci] = rng.uniform(lo, hi, n)

        for feat, (plo, phi) in suspicious_high.items():
            if feat in cols:
                ci = cols.index(feat)
                lo = np.percentile(X_arr[:, ci], plo)
                hi = np.percentile(X_arr[:, ci], phi)
                synthetic[:, ci] = rng.uniform(lo, hi, n)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_injected = np.vstack([X_arr, synthetic])
    y_true = np.array([0] * len(X_arr) + [1] * n)
    logger.info("Injected %s synthetic anomalies (strategy=%r)", n, strategy)
    return X_injected, y_true


def evaluate_injection_recovery(model, X_injected: np.ndarray, y_true: np.ndarray) -> Dict:
    """Precision@k, Recall@k, ROC-AUC, and Average Precision using model anomaly scores.

    k is set to n_synthetic (the number of injected anomalies), so Recall@k answers
    "of the injected anomalies, what fraction appear in the model's top-k predictions?"
    This is a proportional recall, not a fixed-cutoff recall.  Always read k alongside
    n_synthetic (stored in the returned dict) — e.g. "Recall@50" for 50 injected points.

    Average Precision (AP) is the primary ranking metric and is robust to the choice
    of k; it is reported alongside Recall@k for completeness.
    """
    scores = model.score(X_injected)
    n_synthetic = int(y_true.sum())
    k = n_synthetic  # k == n_synthetic: Recall@k is proportional, not a fixed cutoff
    top_k_idx = np.argsort(scores)[::-1][:k]
    y_pred_topk = np.zeros(len(y_true), dtype=int)
    y_pred_topk[top_k_idx] = 1

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    results = {
        "precision_at_k": float(precision_score(y_true, y_pred_topk, zero_division=0)),
        "recall_at_k": float(recall_score(y_true, y_pred_topk, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores_norm)),
        "average_precision": float(average_precision_score(y_true, scores_norm)),
        "n_synthetic": n_synthetic,
        "k_used": k,  # always equals n_synthetic; stored so CSV is self-documenting
        "n_recovered_in_top_k": int(y_pred_topk[y_true == 1].sum()),
    }
    logger.info(
        "Injection recovery (k=n_synthetic=%s) — P@k: %.3f  R@k: %.3f  ROC-AUC: %.3f  AP: %.3f",
        k,
        results["precision_at_k"],
        results["recall_at_k"],
        results["roc_auc"],
        results["average_precision"],
    )
    return results


def correlate_with_acpl(anomaly_scores: np.ndarray, acpl_values: np.ndarray) -> Dict:
    """Pearson / Spearman between anomaly scores and ACPL."""
    mask = ~(np.isnan(anomaly_scores) | np.isnan(acpl_values))
    s, a = anomaly_scores[mask], acpl_values[mask]
    pearson_r, pearson_p = stats.pearsonr(s, a)
    spearman_r, spearman_p = stats.spearmanr(s, a)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_players": int(mask.sum()),
        "significant": bool(pearson_p < ALPHA),
    }


def test_anomaly_vs_normal(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """Welch t-tests: anomaly (-1) vs normal (1) on each feature."""
    anomaly_mask = labels == -1
    normal_mask = labels == 1
    logger.info(
        "Comparing %s anomalies vs %s normal players",
        int(anomaly_mask.sum()),
        int(normal_mask.sum()),
    )
    rows = []
    for i, feat in enumerate(feature_names):
        a_vals = feature_matrix[anomaly_mask, i]
        n_vals = feature_matrix[normal_mask, i]
        t_stat, p_val = stats.ttest_ind(a_vals, n_vals, equal_var=False)
        rows.append(
            {
                "feature": feat,
                "anomaly_mean": float(a_vals.mean()),
                "normal_mean": float(n_vals.mean()),
                "diff": float(a_vals.mean() - n_vals.mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < ALPHA),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette for binary anomaly vs normal partition."""
    binary_labels = (labels == -1).astype(int)
    if len(np.unique(binary_labels)) < 2:
        logger.warning("Only one class — silhouette undefined.")
        return float("nan")
    score = silhouette_score(X, binary_labels)
    logger.info("Silhouette score: %.4f", score)
    return float(score)


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    binary_labels = (labels == -1).astype(int)
    if len(np.unique(binary_labels)) < 2:
        return float("nan")
    score = davies_bouldin_score(X, binary_labels)
    logger.info("Davies-Bouldin index: %.4f", score)
    return float(score)


def cross_validate_anomaly_models(
    X: np.ndarray,
    feature_names: list,
    best_params: dict,
    n_splits: int = 5,
    n_injected: int = 50,
    injection_strategy: str = "subtle",
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold cross-validation for all anomaly detectors.

    Splits the development set (train + val; test excluded by the caller)
    into k folds.  For each fold the StandardScaler is fit on the k-1
    training folds and applied to the held-out fold — no leakage within
    CV.  Ground truth comes from synthetic injection into each test fold.

    Args:
        X: Raw (unscaled) development-set feature matrix.
        feature_names: Column names matching X's columns.
        best_params: Tuned param dicts from run_hyperparameter_search().
        n_splits: Number of folds (default 5).
        n_injected: Synthetic anomalies injected per test fold.
        injection_strategy: Injection strategy passed to inject_synthetic_anomalies().
        random_state: Seed for KFold shuffle.

    Returns:
        raw_df: One row per (fold × model) with every metric value.
        summary_df: Mean, std, and 95 % CI across folds per (model × metric).
    """
    # Late import to avoid circular dependency (models → config, validation → config)
    from src.models import (
        AutoencoderDetector,
        HDBSCANDetector,
        IsolationForestDetector,
        LOFDetector,
        OneClassSVMDetector,
        ZScoreBaseline,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    raw_rows: list = []
    metric_cols = ["roc_auc", "average_precision", "precision_at_k", "recall_at_k"]

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
        logger.info("CV fold %d / %d...", fold_idx + 1, n_splits)

        # Scaler fit on this fold's training rows only — mirrors the main pipeline
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        # Inject synthetic anomalies into the test fold to get labelled ground truth
        X_inj, y_inj = inject_synthetic_anomalies(
            pd.DataFrame(X_te, columns=feature_names),
            n=n_injected,
            strategy=injection_strategy,
        )
        X_inj_arr = X_inj if isinstance(X_inj, np.ndarray) else X_inj.values

        # Build every model with its tuned parameters.
        # Autoencoder uses reduced epochs (AUTOENCODER_SEARCH_EPOCHS) during CV
        # to keep wall-clock time practical; final performance is still reported
        # on the held-out test set with full training.
        models = [
            ("ZScoreBaseline", ZScoreBaseline(contamination=0.05)),
            ("LOF",            LOFDetector(**best_params.get("LOF", {"contamination": 0.05}))),
            ("IsolationForest", IsolationForestDetector(**best_params.get("IsolationForest", {"contamination": 0.05}))),
            ("OneClassSVM",    OneClassSVMDetector(**best_params.get("OneClassSVM", {"nu": 0.05}))),
            ("Autoencoder",    AutoencoderDetector(
                input_dim=X_tr.shape[1],
                epochs=AUTOENCODER_SEARCH_EPOCHS,
                **best_params.get("Autoencoder", {}),
            )),
            # HDBSCAN uses reduced min_cluster_size fallback if search hasn't run yet
            ("HDBSCAN",        HDBSCANDetector(**best_params.get("HDBSCAN", {"min_cluster_size": 15}))),
        ]

        for name, m in models:
            try:
                m.fit(X_tr)
                metrics = evaluate_injection_recovery(m, X_inj_arr, y_inj)
                raw_rows.append({"fold": fold_idx + 1, "model": name, **metrics})
            except Exception as exc:
                logger.warning("CV fold %d %s failed: %s", fold_idx + 1, name, exc)

    raw_df = pd.DataFrame(raw_rows)

    # Summarise: mean, std, and 95 % CI across folds per model
    summary_rows: list = []
    for model_name, grp in raw_df.groupby("model"):
        row: dict = {"model": model_name}
        for metric in metric_cols:
            if metric not in grp.columns:
                continue
            vals = grp[metric].dropna()
            n = len(vals)
            mean, std = float(vals.mean()), float(vals.std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"]  = std
            row[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(n)) if n > 1 else float("nan")
        summary_rows.append(row)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "CV complete (%d folds).\n%s",
        n_splits,
        summary_df[["model", "roc_auc_mean", "roc_auc_std"]].to_string(index=False),
    )
    return raw_df, summary_df
