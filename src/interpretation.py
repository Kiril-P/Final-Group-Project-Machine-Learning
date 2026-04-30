"""
Interpretability: permutation importance, SHAP (trees), plots, failure analysis.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def permutation_feature_importance(
    model,
    X: np.ndarray,
    feature_names: list,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Mean drop in anomaly score when each feature is permuted."""
    rng = np.random.default_rng(42)
    base_scores = model.score(X)
    base_mean = float(base_scores.mean())
    importances = []
    for feat_idx, feat_name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
            perm_scores = model.score(X_perm)
            deltas.append(base_mean - float(perm_scores.mean()))
        importances.append(
            {
                "feature": feat_name,
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
            }
        )
    df = pd.DataFrame(importances).sort_values("importance_mean", ascending=False)
    logger.info("Permutation importance computed.")
    return df


def compute_shap_values(
    isolation_forest: IsolationForest,
    X: np.ndarray,
    feature_names: list,
    max_samples: int = 500,
) -> np.ndarray:
    """SHAP values for sklearn IsolationForest (TreeExplainer)."""
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    explainer = shap.TreeExplainer(isolation_forest)
    shap_values = explainer.shap_values(X_sample)
    logger.info("SHAP values computed for %s samples.", len(X_sample))
    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    save: bool = True,
) -> None:
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    if save:
        path = RESULTS_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary saved to %s", path)
    plt.show()


def per_feature_reconstruction_error(
    autoencoder,
    X: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    import torch

    autoencoder.model.eval()
    tensor = torch.FloatTensor(X).to(autoencoder.device)
    with torch.no_grad():
        recon = autoencoder.model(tensor).cpu().numpy()
    per_feat_error = (recon - X) ** 2
    df = pd.DataFrame(per_feat_error, columns=feature_names)
    df["total_error"] = df.sum(axis=1)
    return df


def analyze_false_positives(
    meta: pd.DataFrame,
    results: pd.DataFrame,
    player_df: pd.DataFrame,
) -> pd.DataFrame:
    """Heuristic explanations for ensemble-flagged players."""
    anomalies = results[results["ensemble_anomaly"] == True].copy()
    explanations = []
    for _, row in anomalies.iterrows():
        player_games = player_df[player_df["player_id"] == row["player_id"]]
        if len(player_games) == 0:
            explanations.append("unknown")
            continue
        rating_std = player_games["player_rating"].std()
        if rating_std > 200:
            explanations.append("rapid_improvement")
            continue
        avg_opp = player_games["opponent_rating"].mean()
        avg_self = player_games["player_rating"].mean()
        if avg_self - avg_opp > 200:
            explanations.append("pool_mismatch")
            continue
        explanations.append("unexplained_deviation")
    anomalies["likely_explanation"] = explanations
    logger.info(
        "Failure analysis breakdown:\n%s",
        anomalies["likely_explanation"].value_counts().to_string(),
    )
    return anomalies


def plot_anomaly_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(-1, "crimson", "Anomaly"), (1, "steelblue", "Normal")]:
        mask = labels == label
        ax.hist(scores[mask], bins=40, alpha=0.6, color=color, label=name, density=True)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Score Distribution")
    ax.legend()
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{model_name.lower()}_score_dist.png"
        plt.savefig(path, dpi=150)
        logger.info("Saved score distribution plot to %s", path)
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        importance_df["feature"],
        importance_df["importance_mean"],
        xerr=importance_df["importance_std"],
        color="steelblue",
        alpha=0.8,
    )
    ax.set_xlabel("Mean importance (drop in anomaly score)")
    ax.set_title("Permutation feature importance")
    ax.invert_yaxis()
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "feature_importance.png"
        plt.savefig(path, dpi=150)
    plt.show()
