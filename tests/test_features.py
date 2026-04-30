"""Unit tests for feature engineering. Run: pytest from repository root."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import aggregate_player_stats, add_engineered_features, get_feature_matrix


def make_fake_player_df(n_players=20, games_per_player=10):
    rng = np.random.default_rng(0)
    records = []
    for i in range(n_players):
        for j in range(games_per_player):
            records.append(
                {
                    "id": f"game_{i}_{j}",
                    "player_id": f"player_{i}",
                    "player_rating": int(rng.integers(800, 2000)),
                    "opponent_rating": int(rng.integers(800, 2000)),
                    "won": int(rng.integers(0, 2)),
                    "turns": int(rng.integers(10, 80)),
                    "opening_ply": int(rng.integers(2, 20)),
                    "rating_diff": int(rng.integers(-400, 400)),
                    "time_control_cat": rng.choice(["blitz", "rapid"]),
                }
            )
    return pd.DataFrame(records)


def test_aggregate_player_stats():
    df = make_fake_player_df()
    agg = aggregate_player_stats(df)
    assert "player_id" in agg.columns
    assert len(agg) == 20
    assert agg["n_games"].min() >= 5


def test_add_engineered_features():
    df = make_fake_player_df()
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    assert "opening_ply_ratio" in agg.columns
    assert "victory_efficiency" in agg.columns
    assert "win_rate_vs_expected" in agg.columns
    assert "expected_win_rate" in agg.columns
    assert agg["expected_win_rate"].between(0, 1).all()


def test_get_feature_matrix():
    df = make_fake_player_df(n_players=30, games_per_player=10)
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    X, meta, scaler = get_feature_matrix(agg, use_acpl=False)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert not X.isnull().any().any()
    assert len(meta) == len(X)
    assert scaler is not None


def test_feature_matrix_no_nan():
    df = make_fake_player_df(n_players=50, games_per_player=8)
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    X, meta, _ = get_feature_matrix(agg)
    assert not X.isnull().values.any()
