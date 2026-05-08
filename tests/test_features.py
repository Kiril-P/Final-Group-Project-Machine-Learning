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
    # Sanity check: one row per player, all players have ≥ MIN_GAMES_PER_PLAYER games.
    df = make_fake_player_df()
    agg = aggregate_player_stats(df)
    assert "player_id" in agg.columns
    assert len(agg) == 20
    assert agg["n_games"].min() >= 5


def test_add_engineered_features():
    # Verify the key engineered features are produced and Elo expectancy stays in [0, 1].
    df = make_fake_player_df()
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    assert "opening_ply_ratio" in agg.columns
    assert "victory_efficiency" in agg.columns
    assert "win_rate_vs_expected" in agg.columns
    assert "expected_win_rate" in agg.columns
    # Elo expected win rate is always a probability — must be in [0, 1]
    assert agg["expected_win_rate"].between(0, 1).all()


def test_get_feature_matrix():
    # End-to-end: we should get a non-empty, NaN-free matrix with matching metadata rows.
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
    # Larger dataset — make sure NaN handling scales without issues.
    df = make_fake_player_df(n_players=50, games_per_player=8)
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    X, meta, _ = get_feature_matrix(agg)
    assert not X.isnull().values.any()


# ── New feature tests ─────────────────────────────────────────────────────────

def make_fake_player_df_with_result(n_players=20, games_per_player=10):
    """Same as make_fake_player_df but includes a 'result' column (0 / 0.5 / 1)."""
    rng = np.random.default_rng(1)
    records = []
    for i in range(n_players):
        for j in range(games_per_player):
            winner_choice = rng.choice(["white", "black", "draw"], p=[0.45, 0.45, 0.10])
            color = rng.choice(["white", "black"])
            won = 1 if winner_choice == color else 0
            result = 1.0 if winner_choice == color else (0.5 if winner_choice == "draw" else 0.0)
            records.append({
                "id": f"game_{i}_{j}",
                "player_id": f"player_{i}",
                "player_rating": int(rng.integers(800, 2000)),
                "opponent_rating": int(rng.integers(800, 2000)),
                "won": won,
                "result": result,
                "turns": int(rng.integers(10, 80)),
                "opening_ply": int(rng.integers(2, 20)),
                "rating_diff": int(rng.integers(-400, 400)),
                "time_control_cat": rng.choice(["blitz", "rapid"]),
            })
    return pd.DataFrame(records)


def test_result_column_preserved():
    """Draws must produce result=0.5, wins=1.0, losses=0.0.

    This matters for performance rating: treating a draw as a loss (result=0.0)
    would undercount the player's performance against strong opponents.
    """
    df = make_fake_player_df_with_result()
    assert set(df["result"].unique()).issubset({0.0, 0.5, 1.0})
    # Consistency check: if won==1, result must be 1.0 (not 0.5 or 0.0)
    assert (df[df["won"] == 1]["result"] == 1.0).all()


def test_performance_vs_actual():
    """performance_vs_actual should be finite and numeric for every player.

    The Elo formula (opponent_rating ± 400) can produce large but always finite
    numbers, so any NaN or inf here would indicate a bug in the aggregation.
    """
    df = make_fake_player_df_with_result(n_players=30, games_per_player=10)
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    assert "performance_vs_actual" in agg.columns
    assert agg["performance_vs_actual"].notna().all()
    assert np.isfinite(agg["performance_vs_actual"]).all()


def test_underdog_win_rate_range():
    """underdog_win_rate must be in [0, 1] and 0 for players with no underdog games.

    Players with zero underdog games must get 0.0, not NaN — NaN would cause them
    to be dropped from the feature matrix even though their data is perfectly valid.
    """
    df = make_fake_player_df_with_result(n_players=40, games_per_player=12)
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    assert "underdog_win_rate" in agg.columns
    assert (agg["underdog_win_rate"] >= 0).all()
    assert (agg["underdog_win_rate"] <= 1).all()
    # Players who never faced a stronger opponent get rate=0 (not NaN)
    no_underdog = agg[agg["underdog_games"] == 0]
    assert (no_underdog["underdog_win_rate"] == 0.0).all()


def test_new_features_in_feature_matrix():
    """performance_vs_actual and underdog_win_rate must appear in the base feature matrix.

    They're part of the 8 base features we always include — if they're missing here,
    the models would silently train without our two key new cheating signals.
    """
    df = make_fake_player_df_with_result(n_players=30, games_per_player=10)
    agg = aggregate_player_stats(df)
    agg = add_engineered_features(agg)
    X, meta, scaler = get_feature_matrix(agg, feature_set="base")
    assert "performance_vs_actual" in X.columns
    assert "underdog_win_rate" in X.columns
    assert not X.isnull().values.any()
