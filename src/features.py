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

from src.config import MIN_EVAL_COVERAGE, RATING_BANDS, RATING_BAND_LABELS

logger = logging.getLogger(__name__)


def aggregate_player_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-game player records into one row per player.

    Handles both the small Kaggle dataset (no move-time / eval columns) and
    the large Lichess dataset (has move_time_mean, move_time_std,
    time_pressure_count, avg_acpl_game, blunder_count, best_move_count,
    n_moves_with_eval, was_losing columns).  Extra columns are simply absent
    for the small dataset — no code changes needed in callers.
    """
    df = player_df.copy()

    # ── Game-level derived columns needed before groupby ─────────────────────

    # Standard Elo performance rating per game:
    #   win  → opponent_rating + 400  (performed like someone 400 pts stronger)
    #   draw → opponent_rating + 0    (matched the opponent exactly)
    #   loss → opponent_rating − 400  (performed like someone 400 pts weaker)
    # This is the industry-standard formula (used by FIDE, Lichess, Chess.com).
    # We use the float `result` column (1.0/0.5/0.0) so draws are valued correctly.
    # Fall back to binary `won` if `result` isn't in the data (treats draws as losses —
    # slightly pessimistic but avoids crashing on older/legacy data).
    if "result" in df.columns:
        df["game_perf_rating"] = df["opponent_rating"] + 400 * (2 * df["result"] - 1)
    else:
        df["game_perf_rating"] = df["opponent_rating"] + 400 * (2 * df["won"] - 1)

    # Underdog flag: opponent is 100+ Elo higher than the player.
    # At 100 Elo difference the expected win rate for the weaker side is ~36%,
    # so beating them isn't impossible — but doing it consistently is suspicious.
    # We chose 100 (not 200) to get enough underdog games per player for a meaningful rate.
    df["is_underdog"] = (df["opponent_rating"] - df["player_rating"]) > 100
    df["underdog_won"] = (df["is_underdog"] & (df["won"] == 1)).astype(int)

    # Comeback flag (big dataset only — requires Stockfish eval annotations):
    # "was_losing" means the eval dropped below -150cp at some point during the game.
    # Winning from that position is genuinely hard for humans but trivial for an engine.
    if "was_losing" in df.columns:
        df["comeback_win"] = (df["was_losing"].astype(bool) & (df["won"] == 1)).astype(int)

    # ── Base aggregation ──────────────────────────────────────────────────────
    agg_spec: dict = dict(
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
        avg_performance_rating=("game_perf_rating", "mean"),
        underdog_games=("is_underdog", "sum"),
        underdog_wins=("underdog_won", "sum"),
    )

    # ── Optional move-time columns (large Lichess dataset only) ───────────────
    for src_col, agg_name, func in [
        ("move_time_mean",      "avg_move_time_mean",  "mean"),
        ("move_time_std",       "avg_move_time_std",   "mean"),
        ("time_pressure_count", "total_time_pressure", "sum"),
    ]:
        if src_col in df.columns:
            agg_spec[agg_name] = (src_col, func)

    # ── Optional eval columns (large Lichess dataset only) ────────────────────
    for src_col, agg_name, func in [
        ("avg_acpl_game",       "avg_acpl",              "mean"),
        ("blunder_count",       "total_blunders",         "sum"),
        ("best_move_count",     "total_best_moves",       "sum"),
        ("n_moves_with_eval",   "total_moves_with_eval",  "sum"),
        ("was_losing",          "was_losing_count",       "sum"),
        # Per-phase ACPL — average across all a player's games for each phase
        ("acpl_opening_game",   "avg_acpl_opening",       "mean"),
        ("acpl_middlegame_game","avg_acpl_middlegame",     "mean"),
        ("acpl_endgame_game",   "avg_acpl_endgame",        "mean"),
        # ACPL consistency: std of avg_acpl_game across all a player's games.
        # An engine is unnaturally consistent — ACPL barely varies game to game.
        # A human has good days and bad days, so their std is naturally higher.
        # Note: needs at least ~10 games to be a reliable estimate; with only 5 games
        # (our minimum) the std is noisy. We include it anyway and let the ensemble
        # absorb the noise from low-game-count players.
        ("avg_acpl_game",       "acpl_consistency",        "std"),
    ]:
        if src_col in df.columns:
            agg_spec[agg_name] = (src_col, func)

    if "comeback_win" in df.columns:
        agg_spec["comeback_wins"] = ("comeback_win", "sum")

    agg = df.groupby("player_id").agg(**agg_spec).reset_index()

    agg["rating_volatility"] = agg["rating_volatility"].fillna(0)
    agg["turns_std"] = agg["turns_std"].fillna(0)
    return agg


def add_engineered_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Derived features for anomaly signals within rating context."""
    df = agg.copy()

    # ── Existing features ─────────────────────────────────────────────────────

    # Opening depth relative to rating — a 1200 who plays 15-move openings is more
    # suspicious than a 2000 who does; dividing by rating normalises for that.
    df["opening_ply_ratio"] = df["avg_opening_ply"] / (df["avg_rating"] + 1)

    # Game length relative to opponent strength — stronger opponents tend to drag games
    # out longer, so we normalise avg_turns by opponent rating to make this comparable.
    df["victory_efficiency"] = df["avg_turns"] / (df["avg_opponent_rating"].clip(lower=1))

    # Elo win expectancy formula: the "fair" win rate given average rating difference.
    # This is the standard formula used by all major rating systems.
    df["expected_win_rate"] = 1 / (
        1 + 10 ** ((df["avg_opponent_rating"] - df["avg_rating"]) / 400)
    )
    # How much a player over- or under-performs their expected win rate.
    # Consistently positive = winning more than their Elo predicts = suspicious.
    df["win_rate_vs_expected"] = df["win_rate"] - df["expected_win_rate"]

    # Rating band for within-band normalisation — keeps comparisons fair across skill levels.
    df["rating_band"] = pd.cut(
        df["avg_rating"],
        bins=RATING_BANDS,
        labels=RATING_BAND_LABELS,
        right=False,
    )

    # ── New features (available on both small and large datasets) ─────────────

    # Empirical performance rating vs registered Elo.
    # A player who consistently outperforms their rating by 200+ points is worth flagging —
    # it could mean rapid improvement, or it could mean engine assistance.
    df["performance_vs_actual"] = df["avg_performance_rating"] - df["avg_rating"]

    # Win rate specifically against stronger opponents (100+ Elo gap).
    # A legitimate 1400 beating 1500+ players often is unusual; for an engine it's routine.
    # Players with zero underdog games get 0.0, not NaN, so they aren't dropped from the model.
    df["underdog_win_rate"] = (
        df["underdog_wins"] / df["underdog_games"].replace(0, np.nan)
    ).fillna(0.0)

    # ── New features (large Lichess dataset only — require clock/eval annotations) ──

    # Move-time coefficient of variation = std / mean across all of a player's think times.
    # A human varies a lot: quick moves on forcing lines, long thinks on critical positions.
    # An engine tends to think for the same amount of time every move (CV near 0).
    # We add 1e-6 to avoid divide-by-zero for games with nearly zero average move time.
    if "avg_move_time_mean" in df.columns and "avg_move_time_std" in df.columns:
        df["move_time_cv"] = df["avg_move_time_std"] / (
            df["avg_move_time_mean"].replace(0, np.nan) + 1e-6
        )
    else:
        df["move_time_cv"] = np.nan  # not available on the small dataset

    # Fraction of moves played with < 10s left on the clock.
    # Engine users never get into time trouble because the engine is fast.
    # A surprisingly low time-pressure rate at lower ratings is a red flag.
    if "total_time_pressure" in df.columns:
        df["time_pressure_rate"] = df["total_time_pressure"] / df["n_games"].replace(0, np.nan)
    else:
        df["time_pressure_rate"] = np.nan

    # Blunder rate: moves where eval dropped by more than 150cp (1.5 pawns).
    # Engines essentially never blunder. A low blunder rate at low ratings is suspicious.
    if "total_blunders" in df.columns and "total_moves_with_eval" in df.columns:
        df["blunder_rate"] = (
            df["total_blunders"] / df["total_moves_with_eval"].replace(0, np.nan)
        ).fillna(np.nan)
    else:
        df["blunder_rate"] = np.nan

    # Best move rate: moves where eval didn't drop at all (≤ 10cp loss).
    # An engine plays the "best" move almost every turn. Humans don't.
    if "total_best_moves" in df.columns and "total_moves_with_eval" in df.columns:
        df["best_move_rate"] = (
            df["total_best_moves"] / df["total_moves_with_eval"].replace(0, np.nan)
        ).fillna(np.nan)
    else:
        df["best_move_rate"] = np.nan

    # Win rate in games where the player was in a clearly losing position (eval < -150cp).
    # Escaping losing positions is one of the hardest things in chess — engines do it routinely.
    # Players with no losing positions get 0.0 (not NaN) so they stay in the model.
    if "comeback_wins" in df.columns and "was_losing_count" in df.columns:
        df["comeback_rate"] = (
            df["comeback_wins"] / df["was_losing_count"].replace(0, np.nan)
        ).fillna(0.0)
    else:
        df["comeback_rate"] = np.nan

    # ── Within-rating-band normalization for eval features ────────────────────
    # avg_acpl and best_move_rate both depend heavily on Elo: a 1200-rated player
    # naturally has much higher ACPL and lower best-move rate than a 2000-rated player.
    # Without band normalization, the model would compare these across skill levels,
    # which is unfair and noisy.
    #
    # We only do this for avg_acpl and best_move_rate, NOT for blunder_rate, comeback_rate,
    # or time_pressure_rate — those are roughly Elo-independent (a blunder is a blunder
    # regardless of rating, and engine users don't run out of time at any rating).
    #
    # We keep the raw columns intact and add new *_band_z columns.
    # The raw values are still in player_features.csv for reporting and explainability.
    # Band z-scores are computed on the full population (all 28k players), which is fine
    # here — these are "contextual" statistics (what's normal for a 1400-rated player?)
    # rather than learned model parameters, so using all data doesn't cause leakage.
    for raw_feat, z_feat in [
        ("avg_acpl",            "avg_acpl_band_z"),
        ("best_move_rate",      "best_move_rate_band_z"),
        # Per-phase ACPL — each phase compared within rating band.
        # Middlegame is the key signal: engines get turned on when positions get complex.
        # Opening is less informative (cheaters often play theory normally).
        # Endgame is informative but many games end before move 30, so coverage is lower.
        ("avg_acpl_opening",    "avg_acpl_opening_band_z"),
        ("avg_acpl_middlegame", "avg_acpl_middlegame_band_z"),
        ("avg_acpl_endgame",    "avg_acpl_endgame_band_z"),
        # ACPL consistency — also band-normalized.
        # The user's insight: higher-rated players are naturally more consistent
        # (lower STDCPL) because their skill floor is higher. A 2200 player
        # fluctuates between ACPL 20-35; a 1200 might fluctuate 50-120. Without
        # band normalization we'd flag high-rated players as suspicious just for
        # being good, which is wrong. Comparing within band makes the signal fair.
        ("acpl_consistency",    "acpl_consistency_band_z"),
    ]:
        if raw_feat in df.columns and "rating_band" in df.columns:
            df[z_feat] = np.nan
            for band in df["rating_band"].dropna().unique():
                mask = df["rating_band"] == band
                col = df.loc[mask, raw_feat].dropna()
                if len(col) < 2:
                    continue
                mean, std = float(col.mean()), float(col.std())
                if std > 0:
                    df.loc[mask, z_feat] = (df.loc[mask, raw_feat] - mean) / std
                else:
                    df.loc[mask, z_feat] = 0.0
            n_computed = df[z_feat].notna().sum()
            logger.info(
                "Within-band z-score '%s' computed for %s players", z_feat, n_computed
            )

    # ── Phase gap: opening ACPL minus middlegame ACPL ─────────────────────────
    # This is the single most powerful phase-based signal. A player who plays
    # normally (high ACPL) in the opening but suspiciously well (low ACPL) in the
    # middlegame has a large positive gap. That's the classic pattern of someone
    # who knows their opening theory but turns on an engine once the position
    # gets complicated.
    #
    # A genuine strong player also has lower middlegame ACPL than opening, but
    # the gap is proportionally smaller — they're consistent throughout.
    # We band-normalize the gap because the absolute ACPL values (and thus the
    # gap magnitude) both scale with Elo — at 1200 ACPL values are larger so the
    # raw gap would be bigger even without cheating.
    if "avg_acpl_opening" in df.columns and "avg_acpl_middlegame" in df.columns:
        df["acpl_phase_gap"] = (
            df["avg_acpl_opening"] - df["avg_acpl_middlegame"]
        )
        # Band-normalize the gap
        if "rating_band" in df.columns:
            df["acpl_phase_gap_band_z"] = np.nan
            for band in df["rating_band"].dropna().unique():
                mask = df["rating_band"] == band
                col = df.loc[mask, "acpl_phase_gap"].dropna()
                if len(col) < 2:
                    continue
                mean, std = float(col.mean()), float(col.std())
                if std > 0:
                    df.loc[mask, "acpl_phase_gap_band_z"] = (
                        df.loc[mask, "acpl_phase_gap"] - mean
                    ) / std
                else:
                    df.loc[mask, "acpl_phase_gap_band_z"] = 0.0
            logger.info(
                "Phase gap feature 'acpl_phase_gap_band_z' computed for %s players",
                df["acpl_phase_gap_band_z"].notna().sum(),
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
    feature_set: str = "base",
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Return (feature_matrix, metadata, scaler) ready for anomaly detection.

    Args:
        df: Player-level dataframe with engineered features.
        use_acpl: Include ACPL columns when present (legacy flag; also activated
            automatically when feature_set='extended' and avg_acpl is available).
        time_control: If set, keep players whose dominant TC matches (e.g. 'blitz').
        fit_scaler: When True (default), fits StandardScaler on the supplied data and
            returns scaled features — suitable for one-shot use (e.g. notebooks, EDA).
            Set to False when the caller will split the data first and fit the scaler
            only on the training partition; the returned scaler is unfitted and the
            feature matrix contains raw (unscaled) values.
        feature_set: 'base' (8 features, both datasets) or 'extended' (up to 14
            features, requires large Lichess dataset with clock + eval annotations).
    """
    if time_control:
        df = df[df["time_control_cat"] == time_control].copy()
        logger.info("Filtered to time control '%s': %s players", time_control, f"{len(df):,}")

    # ── Base features (available for both small and large datasets) ───────────
    # These 8 features are computable from any dataset that has basic rating + outcome data.
    features = [
        "win_rate",
        "win_rate_vs_expected",
        "avg_turns",
        "opening_ply_ratio",
        "victory_efficiency",
        "rating_volatility",
        "performance_vs_actual",
        "underdog_win_rate",
    ]

    # ── Extended features (large Lichess dataset with clock / eval data) ──────
    # We only add an extended feature if it has coverage for at least MIN_EVAL_COVERAGE
    # (50%) of players. A feature that's NaN for most players would just force us to drop
    # those players from the model entirely, which is worse than not having the feature.
    if feature_set == "extended":
        extended_candidates = [
            "move_time_cv",                  # strongest cheating signal — uniform think times
            "time_pressure_rate",            # engine users don't run out of time
            "avg_acpl_band_z",               # overall ACPL z-score within rating band
            "blunder_rate",                  # engines almost never blunder (Elo-independent)
            "best_move_rate_band_z",         # best-move rate z-score within rating band
            "comeback_rate",                 # engines escape losing positions at superhuman rates
            # Per-phase ACPL — finer-grained than overall ACPL.
            # Most useful for detecting "mid-game engine activation" patterns.
            "avg_acpl_middlegame_band_z",    # middlegame ACPL vs band peers (strongest phase signal)
            "avg_acpl_opening_band_z",       # opening ACPL vs band peers (lower signal, theory-heavy)
            "avg_acpl_endgame_band_z",       # endgame ACPL vs band peers (lower coverage — many games end earlier)
            "acpl_phase_gap_band_z",         # opening minus middlegame ACPL, band-normalized
                                             # large positive = normal opening, engine-perfect middlegame
            "acpl_consistency_band_z",       # std of ACPL across games, band-normalized
                                             # unusually LOW = suspiciously consistent = engine-like
        ]
        for feat in extended_candidates:
            if feat in df.columns:
                coverage = df[feat].notna().mean()
                if coverage >= MIN_EVAL_COVERAGE:
                    features.append(feat)
                    logger.info(
                        "Extended feature '%s' included (coverage %.0f%%)", feat, coverage * 100
                    )
                else:
                    # Not enough data — including it would shrink the player pool too much
                    logger.warning(
                        "Extended feature '%s' skipped — only %.0f%% coverage (< %.0f%%)",
                        feat, coverage * 100, MIN_EVAL_COVERAGE * 100,
                    )

    # Legacy flag for running Stockfish ACPL on the small dataset.
    # Not used in the main pipeline (slow!) but kept for ad-hoc experiments.
    if use_acpl and "avg_acpl" in df.columns and "avg_acpl" not in features:
        features += ["avg_acpl", "acpl_variance"]

    # Drop players who are missing any of the selected features — imputing would
    # introduce bias, and the models need complete rows.
    df_clean = df.dropna(subset=features).copy()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        logger.warning("Dropped %s players with NaN in features", n_dropped)

    X_raw = df_clean[features].reset_index(drop=True)
    scaler = StandardScaler()

    if fit_scaler:
        # One-shot mode (e.g. notebooks, EDA): fit and return scaled data.
        X = pd.DataFrame(scaler.fit_transform(X_raw), columns=features)
    else:
        # Pipeline mode: the caller will split data first, then fit the scaler on training
        # rows only. Returning raw values here ensures no test-set statistics contaminate
        # the scaler's mean/std — a common and subtle form of data leakage.
        X = X_raw.copy()

    # Keep just the columns we need for downstream reporting (not used in model training).
    meta = df_clean[["player_id", "avg_rating", "rating_band", "n_games"]].reset_index(drop=True)

    logger.info(
        "Feature matrix: %s players × %s features (set=%s, scaled=%s)",
        X.shape[0], X.shape[1], feature_set, fit_scaler,
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
