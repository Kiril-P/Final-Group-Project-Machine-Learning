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
    # Use ensemble_flag (≥2 models agree) — replaces the old single ensemble_anomaly column.
    # ensemble_confident (≥4 models agree) is stricter; flag is the right level for FP analysis.
    anomalies = results[results["ensemble_flag"] == True].copy()
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


def generate_player_explanations(
    results: pd.DataFrame,
    agg: pd.DataFrame,
    feature_names: list,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    For every ensemble-flagged player, explain which features drove the flag
    and how confident we are in that explanation.

    We only claim to "know" why a player was flagged when a feature has a clear,
    intuitive cheating interpretation AND the player's value is extreme in the
    suspicious direction. For everything else we say it's model-detected but not
    directly interpretable — no fake certainty.

    Args:
        results:      Output of run_all_models — one row per player, includes
                      ensemble_flag, ensemble_confident, anomaly_votes.
        agg:          Full player feature DataFrame (player_features.csv).
                      Used to compute within-band z-scores for explanations.
        feature_names: Features that went into the model (in order).
        top_n:        How many top-deviating features to report per player.

    Returns:
        DataFrame with one row per flagged player, columns:
        player_id, avg_rating, rating_band, n_games, anomaly_votes,
        ensemble_confident, top_feature_{1..n}, z_score_{1..n},
        explanation_{1..n}, confident_{1..n}, summary.
    """
    # ── Which direction is suspicious for each feature ────────────────────────
    # "high" = unusually high value is the red flag
    # "low"  = unusually low value is the red flag
    # None   = either direction is flagged — explanation is model-based
    FEATURE_META = {
        "avg_acpl_band_z": {
            "direction": "low",
            "text": (
                "ACPL is unusually low for their rating band — "
                "plays with near engine-level accuracy relative to peers at the same Elo"
            ),
            "confident": True,
        },
        "avg_acpl": {
            "direction": "low",
            "text": (
                "ACPL is unusually low overall — plays with high accuracy "
                "(note: not adjusted for rating band)"
            ),
            "confident": True,
        },
        "best_move_rate_band_z": {
            "direction": "high",
            "text": (
                "Finds optimal moves at an unusually high rate for their rating band — "
                "engines consistently find the best move; humans at this Elo don't"
            ),
            "confident": True,
        },
        "best_move_rate": {
            "direction": "high",
            "text": (
                "Finds optimal moves at an unusually high rate overall "
                "(note: not adjusted for rating band)"
            ),
            "confident": True,
        },
        "win_rate_vs_expected": {
            "direction": "high",
            "text": (
                "Consistently outperforms their Elo prediction — "
                "wins more games than the rating system expects given their opponents"
            ),
            "confident": True,
        },
        "performance_vs_actual": {
            "direction": "high",
            "text": (
                "Empirical performance rating is significantly higher than their official Elo — "
                "they play measurably better than their registered rating suggests"
            ),
            "confident": True,
        },
        "underdog_win_rate": {
            "direction": "high",
            "text": (
                "Wins against much stronger opponents (100+ Elo higher) at an unusual rate — "
                "for a human this is statistically improbable over many games"
            ),
            "confident": True,
        },
        "comeback_rate": {
            "direction": "high",
            "text": (
                "Escapes clearly losing positions (eval < -1.5 pawns) at an unusual rate — "
                "this is extremely hard for humans but routine for engines"
            ),
            "confident": True,
        },
        "time_pressure_rate": {
            "direction": "low",
            "text": (
                "Rarely runs into time trouble — engine users don't spend real time thinking, "
                "so they almost never run low on the clock"
            ),
            "confident": True,
        },
        "blunder_rate": {
            "direction": "low",
            "text": (
                "Unusually few blunders — engines almost never drop pieces or miss tactics; "
                "blunder rate is roughly Elo-independent so this is meaningful at any rating"
            ),
            "confident": True,
        },
        "rating_volatility": {
            "direction": "high",
            "text": (
                "Rating changes erratically — could indicate sandbagging "
                "(deliberately losing to lower rating) or a new account climbing quickly"
            ),
            "confident": True,
        },
        # The features below are statistically anomalous but harder to interpret cleanly.
        # We report them honestly as model-detected rather than making up an explanation.
        "win_rate": {
            "direction": "high",
            "text": (
                "Win rate is unusually high — in a stable Elo system win rates "
                "converge toward ~50%, so sustained deviation is notable"
            ),
            "confident": False,
        },
        "avg_turns": {
            "direction": None,
            "text": (
                "Unusual game length distribution — "
                "flagged by the statistical model but no single clear interpretation"
            ),
            "confident": False,
        },
        "opening_ply_ratio": {
            "direction": None,
            "text": (
                "Unusual opening depth relative to rating — "
                "flagged by the model; not a strong standalone cheating signal"
            ),
            "confident": False,
        },
        "victory_efficiency": {
            "direction": None,
            "text": (
                "Unusual game outcome efficiency — "
                "flagged by the model; interpretation is not straightforward"
            ),
            "confident": False,
        },
        "move_time_cv": {
            "direction": "low",
            "text": (
                "Unusually uniform move times — "
                "engine users think for the same amount of time every move (low variation)"
            ),
            "confident": True,
        },
        # Per-phase ACPL features
        "avg_acpl_middlegame_band_z": {
            "direction": "low",
            "text": (
                "Middlegame ACPL is unusually low for their rating band — "
                "the middlegame is where engine assistance matters most; "
                "human players naturally make more mistakes in complex positions"
            ),
            "confident": True,
        },
        "avg_acpl_opening_band_z": {
            "direction": "low",
            "text": (
                "Opening ACPL is unusually low for their rating band — "
                "could reflect strong opening preparation rather than engine use; "
                "weaker signal than middlegame ACPL"
            ),
            "confident": False,  # opening accuracy is often explained by preparation
        },
        "avg_acpl_endgame_band_z": {
            "direction": "low",
            "text": (
                "Endgame ACPL is unusually low for their rating band — "
                "engines play endgames with near-perfect technique"
            ),
            "confident": True,
        },
        "acpl_phase_gap_band_z": {
            "direction": "high",
            "text": (
                "Large gap between opening and middlegame accuracy — "
                "plays at roughly human level in the opening, then dramatically "
                "better in the middlegame; consistent with turning on an engine "
                "once the position moves beyond memorised theory"
            ),
            "confident": True,
        },
        "acpl_consistency_band_z": {
            "direction": "low",
            "text": (
                "Move quality is unusually consistent across games for their rating band — "
                "humans have good days and bad days; an engine plays at the same "
                "level every game regardless of tiredness, preparation, or time pressure. "
                "Suspiciously low game-to-game ACPL variance relative to peers at the same Elo."
            ),
            "confident": True,
        },
    }

    flagged = results[results["ensemble_flag"] == True].copy()
    if flagged.empty:
        logger.warning("No ensemble-flagged players found — explanation CSV will be empty.")
        return pd.DataFrame()

    # Join with full player features to get raw feature values
    feat_cols = [c for c in feature_names if c in agg.columns]
    agg_sub = agg[["player_id", "rating_band"] + feat_cols].copy()
    flagged = flagged.merge(agg_sub, on="player_id", how="left", suffixes=("", "_feat"))

    # ── Compute within-band z-scores for all features using full population ───
    # We use the entire agg population (not just train) because we're computing
    # "how unusual is this player relative to all players at their rating?" —
    # a contextual normalisation, not a learned model parameter.
    band_means = agg.groupby("rating_band", observed=False)[feat_cols].mean()
    band_stds  = agg.groupby("rating_band", observed=False)[feat_cols].std().fillna(1.0).replace(0, 1.0)

    rows = []
    for _, player in flagged.iterrows():
        band = player.get("rating_band")
        row: dict = {
            "player_id":          player["player_id"],
            "avg_rating":         round(float(player["avg_rating"]), 0),
            "rating_band":        str(band),
            "n_games":            int(player["n_games"]),
            "anomaly_votes":      int(player["anomaly_votes"]),
            "ensemble_confident": bool(player["ensemble_confident"]),
        }

        # Compute suspiciousness score for each feature (signed z-score in suspicious direction)
        feat_scores = {}
        for feat in feat_cols:
            val = player.get(feat)
            if pd.isna(val):
                continue
            meta = FEATURE_META.get(feat, {"direction": None, "text": "Model-detected anomaly — no direct interpretation.", "confident": False})
            try:
                b_mean = float(band_means.loc[band, feat]) if band in band_means.index else float(agg[feat].mean())
                b_std  = float(band_stds.loc[band, feat])  if band in band_stds.index  else float(max(agg[feat].std(), 1e-8))
            except Exception:
                continue
            z = (float(val) - b_mean) / b_std

            # Flip sign so that the "suspicious direction" always gives a positive score
            if meta["direction"] == "high":
                score = z          # high value is suspicious → high z = high score
            elif meta["direction"] == "low":
                score = -z         # low value is suspicious → negative z flipped = high score
            else:
                score = abs(z)     # either direction — use magnitude

            feat_scores[feat] = (score, z, meta)

        # Sort by suspiciousness score, take top_n
        top = sorted(feat_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_n]

        any_confident = False
        for rank, (feat, (score, z, meta)) in enumerate(top, start=1):
            row[f"top_feature_{rank}"]   = feat
            row[f"z_score_{rank}"]       = round(z, 2)
            row[f"explanation_{rank}"]   = meta["text"]
            row[f"confident_{rank}"]     = meta["confident"]
            if meta["confident"]:
                any_confident = True

        # Summary: if we have confident signals, name them; otherwise be honest
        if not top:
            row["summary"] = "Flagged by statistical model — no individual feature stands out clearly."
        elif any_confident:
            top_readable = [
                t[0].replace("_band_z", " (band-normalised)").replace("_", " ")
                for t in top if t[1][2]["confident"]
            ]
            row["summary"] = (
                f"Flagged primarily due to: {', '.join(top_readable[:2])}. "
                f"{'All 3 voters agreed.' if player['ensemble_confident'] else '2 of 3 voters agreed.'}"
            )
        else:
            row["summary"] = (
                "Flagged by statistical model — top deviating features are not directly "
                "interpretable as cheating signals. Warrants human review."
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    logger.info(
        "Player explanations generated for %s flagged players (%s with confident signals).",
        len(out),
        out.get("confident_1", pd.Series(dtype=bool)).sum() if "confident_1" in out.columns else "?",
    )
    return out


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
