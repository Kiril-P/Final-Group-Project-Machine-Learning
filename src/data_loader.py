"""
Load, validate, and perform initial cleaning of the Lichess dataset.

Kaggle dataset `datasnaek/chess` columns include:
    id, rated, created_at, last_move_at, turns, victory_status, winner,
    increment_code, white_id, white_rating, black_id, black_rating,
    moves, opening_eco, opening_name, opening_ply
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import DATA_RAW, MIN_GAMES_PER_PLAYER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """Load the raw games CSV and do minimal type fixing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Download the Kaggle chess dataset and place `games.csv` in `data/raw/` "
            "(see README)."
        )
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %s games, %s columns", f"{len(df):,}", df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if expected columns are missing."""
    required = [
        "id",
        "turns",
        "winner",
        "increment_code",
        "white_id",
        "white_rating",
        "black_id",
        "black_rating",
        "moves",
        "opening_ply",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    logger.info("Schema validation passed.")


def parse_time_control(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse increment_code (e.g. '10+0', '5+3') into base_time_sec and increment_sec,
    then classify into time control category.
    """
    df = df.copy()
    parts = df["increment_code"].astype(str).str.extract(r"(\d+)\+(\d+)")
    df["base_time_sec"] = pd.to_numeric(parts[0], errors="coerce")
    df["increment_sec"] = pd.to_numeric(parts[1], errors="coerce")

    def classify_tc(row):
        base = row["base_time_sec"]
        inc = row["increment_sec"]
        if pd.isna(base) or pd.isna(inc):
            return "unknown"
        total = base + 40 * inc
        if total < 180:
            return "bullet"
        if total < 480:
            return "blitz"
        if total < 1500:
            return "rapid"
        return "classical"

    df["time_control_cat"] = df.apply(classify_tc, axis=1)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows, fix dtypes, add basic derived columns."""
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=["white_rating", "black_rating", "moves", "turns"])
    logger.info("Dropped %s rows with missing critical fields", f"{before - len(df):,}")

    df = df[(df["white_rating"] > 0) & (df["black_rating"] > 0)]
    df = df[df["turns"] >= 5]

    df["winner"] = df["winner"].fillna("draw")
    df["rating_diff"] = df["white_rating"] - df["black_rating"]

    logger.info("After cleaning: %s games remain", f"{len(df):,}")
    return df


def to_player_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape game-level data to player-level by stacking white and black records.
    Each row is one player's perspective on one game.
    """
    white = df.copy()
    white["player_id"] = white["white_id"]
    white["player_rating"] = white["white_rating"]
    white["opponent_rating"] = white["black_rating"]
    white["won"] = (white["winner"] == "white").astype(int)
    white["color"] = "white"

    black = df.copy()
    black["player_id"] = black["black_id"]
    black["player_rating"] = black["black_rating"]
    black["opponent_rating"] = black["white_rating"]
    black["won"] = (black["winner"] == "black").astype(int)
    black["color"] = "black"

    combined = pd.concat([white, black], ignore_index=True)

    game_counts = combined["player_id"].value_counts()
    valid_players = game_counts[game_counts >= MIN_GAMES_PER_PLAYER].index
    combined = combined[combined["player_id"].isin(valid_players)]

    logger.info(
        "Player-level dataset: %s rows, %s unique players",
        f"{len(combined):,}",
        f"{combined['player_id'].nunique():,}",
    )
    return combined


def load_and_prepare(path: Path = DATA_RAW) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full data loading pipeline. Returns (game_df, player_df), both cleaned."""
    df = load_raw(path)
    validate_schema(df)
    df = parse_time_control(df)
    df = clean(df)
    player_df = to_player_level(df)
    return df, player_df


if __name__ == "__main__":
    game_df, player_df = load_and_prepare()
    print(game_df.head())
    print(player_df.head())
