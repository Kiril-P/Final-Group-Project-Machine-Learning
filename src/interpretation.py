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


def plot_roc_curves(
    model_scores: dict,
    y_true: np.ndarray,
    save: bool = True,
) -> pd.DataFrame:
    """ROC curves for all models on a single labelled evaluation set.

    Args:
        model_scores: {model_name: anomaly_score_array} — higher score = more anomalous.
        y_true: Binary ground-truth labels (1 = anomaly, 0 = normal).
        save: Write PNG to results/roc_curves.png.

    Returns:
        DataFrame with columns [model, fpr, tpr, threshold, roc_auc].
    """
    from sklearn.metrics import auc, roc_auc_score, roc_curve

    model_order = ["ZScoreBaseline", "LOF", "IsolationForest", "OneClassSVM", "Autoencoder"]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")

    rows = []
    for model_name, color in zip(model_order, colors):
        if model_name not in model_scores:
            continue
        scores = model_scores[model_name]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        fpr, tpr, thresholds = roc_curve(y_true, scores_norm)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
        for f, t, th in zip(fpr, tpr, thresholds):
            rows.append({"model": model_name, "fpr": f, "tpr": t, "threshold": th, "roc_auc": roc_auc})
        logger.info("ROC-AUC %s: %.4f", model_name, roc_auc)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Subtle Anomaly Detection (Test Set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "roc_curves.png"
        plt.savefig(path, dpi=150)
        logger.info("ROC curves saved to %s", path)
    plt.show()

    return pd.DataFrame(rows)


def plot_learning_curves(
    X_train: np.ndarray,
    X_val_injected: np.ndarray,
    y_val_injected: np.ndarray,
    best_params: dict,
    feature_names: list,
    train_fractions: list | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """ROC-AUC vs. training set size for each anomaly detector.

    Trains every model from scratch on increasingly large subsets of X_train
    and evaluates on the pre-injected validation set.  Shows whether models
    converge quickly or benefit from more data (bias/variance insight).

    Returns a DataFrame with columns [fraction, n_samples, model, roc_auc].
    """
    from sklearn.metrics import roc_auc_score

    from src.config import AUTOENCODER_SEARCH_EPOCHS, RANDOM_SEED
    from src.models import (
        AutoencoderDetector,
        IsolationForestDetector,
        LOFDetector,
        OneClassSVMDetector,
        ZScoreBaseline,
    )

    if train_fractions is None:
        train_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    def _norm(s: np.ndarray) -> np.ndarray:
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    for frac in train_fractions:
        n = max(10, int(len(X_train) * frac))
        idx = rng.choice(len(X_train), size=n, replace=False)
        X_sub = X_train[idx]
        logger.info("Learning curve: %.0f%% of training data (%d samples)...", frac * 100, n)

        model_configs = [
            ("ZScoreBaseline", ZScoreBaseline(contamination=0.05)),
            ("LOF", LOFDetector(**best_params.get("LOF", {"contamination": 0.05}))),
            ("IsolationForest", IsolationForestDetector(**best_params.get("IsolationForest", {"contamination": 0.05}))),
            ("OneClassSVM", OneClassSVMDetector(**best_params.get("OneClassSVM", {"nu": 0.05}))),
            ("Autoencoder", AutoencoderDetector(
                input_dim=X_sub.shape[1],
                epochs=AUTOENCODER_SEARCH_EPOCHS,
                **best_params.get("Autoencoder", {}),
            )),
        ]

        for name, m in model_configs:
            try:
                m.fit(X_sub)
                auc = float(roc_auc_score(y_val_injected, _norm(m.score(X_val_injected))))
                rows.append({"fraction": frac, "n_samples": n, "model": name, "roc_auc": auc})
                logger.info("  %s @ %.0f%%: ROC-AUC=%.4f", name, frac * 100, auc)
            except Exception as exc:
                logger.warning("Learning curve failed (%s @ %.0f%%): %s", name, frac * 100, exc)

    df = pd.DataFrame(rows)

    model_order = ["ZScoreBaseline", "LOF", "IsolationForest", "OneClassSVM", "Autoencoder"]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, color in zip(model_order, colors):
        sub = df[df["model"] == model_name]
        if sub.empty:
            continue
        ax.plot(sub["n_samples"], sub["roc_auc"], marker="o", label=model_name, color=color, linewidth=2)

    ax.set_xlabel("Training set size (players)")
    ax.set_ylabel("ROC-AUC (validation — subtle injection)")
    ax.set_title("Learning Curves — Anomaly Detection Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0.4, df["roc_auc"].min() - 0.05) if not df.empty else 0.4)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "learning_curves.png"
        plt.savefig(path, dpi=150)
        logger.info("Learning curves saved to %s", path)
    plt.show()

    return df


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
