"""
Feature engineering for chess behavioral anomaly detection.

Features are computed at the PLAYER level (aggregated across games),
with controls for time control and opponent strength (see project proposal).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import RATING_BANDS, RATING_BAND_LABELS

logger = logging.getLogger(__name__)


def aggregate_player_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-game player records into one row per player."""
    agg = (
        player_df.groupby("player_id")
        .agg(
            n_games=("id", "count"),
            avg_rating=("player_rating", "mean"),
            rating_volatility=("player_rating", "std"),
            avg_opponent_rating=("opponent_rating", "mean"),
            win_rate=("won", "mean"),
            avg_turns=("turns", "mean"),
            turns_std=("turns", "std"),
            avg_opening_ply=("opening_ply", "mean"),
            avg_rating_diff=("rating_diff", "mean"),
            time_control_cat=("time_control_cat", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    agg["rating_volatility"] = agg["rating_volatility"].fillna(0)
    agg["turns_std"] = agg["turns_std"].fillna(0)
    return agg


def add_engineered_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Derived features for anomaly signals within rating context."""
    df = agg.copy()

    df["opening_ply_ratio"] = df["avg_opening_ply"] / (df["avg_rating"] + 1)
    df["victory_efficiency"] = df["avg_turns"] / (df["avg_opponent_rating"].clip(lower=1))

    df["expected_win_rate"] = 1 / (
        1 + 10 ** ((df["avg_opponent_rating"] - df["avg_rating"]) / 400)
    )
    df["win_rate_vs_expected"] = df["win_rate"] - df["expected_win_rate"]

    df["rating_band"] = pd.cut(
        df["avg_rating"],
        bins=RATING_BANDS,
        labels=RATING_BAND_LABELS,
        right=False,
    )

    logger.info("Engineered features added. Shape: %s", df.shape)
    return df


def add_acpl_features(df: pd.DataFrame, acpl_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-computed ACPL columns: player_id, avg_acpl, acpl_variance."""
    if acpl_df is None or len(acpl_df) == 0:
        logger.warning("No ACPL data provided — skipping ACPL features.")
        df = df.copy()
        df["avg_acpl"] = np.nan
        df["acpl_variance"] = np.nan
        return df

    df = df.merge(acpl_df[["player_id", "avg_acpl", "acpl_variance"]], on="player_id", how="left")
    logger.info("ACPL features merged for %s players", f"{acpl_df['player_id'].nunique():,}")
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    use_acpl: bool = False,
    time_control: Optional[str] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Return (feature_matrix, metadata, scaler) ready for anomaly detection.

    Args:
        df: Player-level dataframe with engineered features.
        use_acpl: Include ACPL columns when present.
        time_control: If set, keep players whose dominant TC matches (e.g. 'blitz').
        fit_scaler: When True (default), fits StandardScaler on the supplied data and
            returns scaled features — suitable for one-shot use (e.g. notebooks, EDA).
            Set to False when the caller will split the data first and fit the scaler
            only on the training partition; the returned scaler is unfitted and the
            feature matrix contains raw (unscaled) values.
    """
    if time_control:
        df = df[df["time_control_cat"] == time_control].copy()
        logger.info("Filtered to time control '%s': %s players", time_control, f"{len(df):,}")

    features = [
        "win_rate",
        "win_rate_vs_expected",
        "avg_turns",
        "opening_ply_ratio",
        "victory_efficiency",
        "rating_volatility",
    ]

    if use_acpl and "avg_acpl" in df.columns:
        features += ["avg_acpl", "acpl_variance"]

    df_clean = df.dropna(subset=features).copy()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        logger.warning("Dropped %s players with NaN in features", n_dropped)

    X_raw = df_clean[features].reset_index(drop=True)
    scaler = StandardScaler()

    if fit_scaler:
        X = pd.DataFrame(scaler.fit_transform(X_raw), columns=features)
    else:
        # Caller is responsible for fitting the scaler on the training split only.
        X = X_raw.copy()

    meta = df_clean[["player_id", "avg_rating", "rating_band", "n_games"]].reset_index(drop=True)

    logger.info(
        "Feature matrix: %s players × %s features (scaled=%s)",
        X.shape[0], X.shape[1], fit_scaler,
    )
    return X, meta, scaler


def normalize_within_band(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Z-score each feature within each rating band."""
    df = df.copy()
    for band in df["rating_band"].dropna().unique():
        mask = df["rating_band"] == band
        for feat in features:
            col = df.loc[mask, feat]
            mean, std = col.mean(), col.std()
            if std and std > 0:
                df.loc[mask, feat] = (col - mean) / std
            else:
                df.loc[mask, feat] = 0.0
    return df


if __name__ == "__main__":
    from src.data_loader import load_and_prepare

    _, player_df = load_and_prepare()
    agg = aggregate_player_stats(player_df)
    agg = add_engineered_features(agg)
    X, meta, scaler = get_feature_matrix(agg)
    print(X.describe())
