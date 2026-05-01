"""
Validation for label-free anomaly detection: synthetic injection, ACPL correlation, stats.
"""

from __future__ import annotations

import logging
from typing import Dict

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

from src.config import ALPHA, N_SYNTHETIC_ANOMALIES, RANDOM_SEED

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
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_injected = np.vstack([X_arr, synthetic])
    y_true = np.array([0] * len(X_arr) + [1] * n)
    logger.info("Injected %s synthetic anomalies (strategy=%r)", n, strategy)
    return X_injected, y_true


def evaluate_injection_recovery(model, X_injected: np.ndarray, y_true: np.ndarray) -> Dict:
    """Precision@k, recall@k, ROC-AUC, AP using model anomaly scores."""
    scores = model.score(X_injected)
    k = int(y_true.sum())
    top_k_idx = np.argsort(scores)[::-1][:k]
    y_pred_topk = np.zeros(len(y_true), dtype=int)
    y_pred_topk[top_k_idx] = 1

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    results = {
        "precision_at_k": float(precision_score(y_true, y_pred_topk, zero_division=0)),
        "recall_at_k": float(recall_score(y_true, y_pred_topk, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores_norm)),
        "average_precision": float(average_precision_score(y_true, scores_norm)),
        "n_synthetic": int(y_true.sum()),
        "n_recovered_in_top_k": int(y_pred_topk[y_true == 1].sum()),
    }
    logger.info(
        "Injection recovery — P@%s: %.3f R@%s: %.3f ROC-AUC: %.3f",
        k,
        results["precision_at_k"],
        k,
        results["recall_at_k"],
        results["roc_auc"],
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
