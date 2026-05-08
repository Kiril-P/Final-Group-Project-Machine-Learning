"""
Loader for the large Lichess July 2016 dataset (6.25 M games, CSV format).

Dataset source: https://www.kaggle.com/datasets/arevel/chess-games
Expected file : data/raw/lichess_jul2016.csv

Column schema (raw CSV):
    Event, White, Black, Result, UTCDate, UTCTime,
    WhiteElo, BlackElo, WhiteRatingDiff, BlackRatingDiff,
    ECO, Opening, TimeControl, Termination, AN

The AN column contains moves in PGN Movetext format with optional
inline annotations:
    [%clk H:MM:SS]   — clock remaining after the move (Lichess format)
    [%eval x.xx]     — Stockfish centipawn evaluation from White's POV
    [%eval #N]       — forced mate in N (positive = White mates)

After loading this module produces the same (game_df, player_df) interface
as data_loader.py so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    LICHESS_SAMPLE_N,
    LICHESS_TIME_CONTROLS,
    MIN_GAMES_PER_PLAYER,
    RANDOM_SEED,
    RATING_BANDS,
    RATING_BAND_LABELS,
)

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Lichess embeds clock times inline in the move text, e.g.: "e4 { [%clk 0:05:00] }"
# The format is always H:MM:SS (hours can be 0 for short games).
_CLK_RE  = re.compile(r'\[%clk (\d+):(\d+):(\d+)\]')

# Eval annotations look like "[%eval 0.35]" (pawns) or "[%eval #3]" (mate in 3).
# Negative means Black is better. We convert pawns → centipawns (*100) for consistency.
_EVAL_RE = re.compile(r'\[%eval ([+-]?\d+\.?\d*|#-?\d+)\]')


# ── Low-level parsers ─────────────────────────────────────────────────────────

def parse_clock_times(an: str) -> List[float]:
    """
    Extract all [%clk H:MM:SS] annotations from a game's AN string.

    Returns a flat list of clock readings in seconds, alternating
    White/Black (index 0 = after White's 1st move, 1 = after Black's 1st, …).
    Returns an empty list if no clock annotations are present.
    """
    out: List[float] = []
    for h, m, s in _CLK_RE.findall(an):
        out.append(int(h) * 3600 + int(m) * 60 + int(s))
    return out


def parse_eval_values(an: str) -> List[float]:
    """
    Extract all [%eval ...] annotations from a game's AN string.

    Centipawn values are stored as pawns in the PGN (e.g. 0.17 = 17 cp).
    Mate scores (#N / #-N) are mapped to ±1000 cp.
    Returns centipawn integers (from White's POV); empty list if none.
    """
    out: List[float] = []
    for token in _EVAL_RE.findall(an):
        if token.startswith('#'):
            try:
                n = int(token[1:])
                out.append(1000.0 if n > 0 else -1000.0)
            except ValueError:
                continue
        else:
            try:
                out.append(float(token) * 100.0)   # pawns → centipawns
            except ValueError:
                continue
    return out


def _compute_player_move_times(
    all_clocks: List[float],
    base_time_sec: float,
    increment_sec: float,
    player_color: str,           # "white" or "black"
) -> List[float]:
    """
    Derive per-move think times (seconds) for one player from the flat
    clock sequence.

    Lichess clock format: [%clk] shows remaining time AFTER the move,
    INCLUDING the increment that was just added.

    Formula:
        move_time[0]  = base_time − clk[0]                  (no increment on first move)
        move_time[i]  = clk[i-1] − clk[i] + increment_sec  (i > 0, same colour)

    Negative values (clock drift / rounding) are clamped to 0.
    """
    # The flat clock list alternates White/Black. Slice out this player's readings:
    # White's clocks are at indices 0, 2, 4, … and Black's at 1, 3, 5, …
    start = 0 if player_color == "white" else 1
    player_clocks = all_clocks[start::2]
    if not player_clocks:
        return []  # no clock annotations in this game — skip

    times: List[float] = []
    # First move: the clock starts at base_time and then drops to clk[0] after the move.
    # The increment hasn't been awarded yet on move 1 in Lichess's convention.
    times.append(max(0.0, base_time_sec - player_clocks[0]))
    for i in range(1, len(player_clocks)):
        # Subsequent moves: time used = previous clock − current clock + increment received.
        # We clamp to 0 because rounding or server lag can produce tiny negatives.
        t = max(0.0, player_clocks[i - 1] - player_clocks[i] + increment_sec)
        times.append(t)
    return times


def _compute_player_eval_stats(
    all_evals: List[float],
    player_color: str,           # "white" or "black"
) -> dict:
    """
    Compute eval-based statistics for one player from the full eval sequence.

    Eval sequence is alternating: index 0 = after White's 1st move,
    index 1 = after Black's 1st move, etc.

    For White — centipawn LOSS on move i:
        eval_before = all_evals[2i-1]  (Black's previous response, or 0 for move 1)
        eval_after  = all_evals[2i]
        loss        = max(0, eval_before − eval_after)

    For Black — centipawn LOSS on move i:
        eval_before = all_evals[2i]    (White's just-played move eval)
        eval_after  = all_evals[2i+1]
        loss        = max(0, eval_after − eval_before)
                      (Black wants eval to go DOWN; if it goes up, Black lost ground)

    "Was losing": any point where eval favoured the opponent by > 150 cp.
    "Blunder": loss ≥ 150 cp on a single move.
    "Best move": loss ≤ 10 cp (essentially no error).
    """
    if not all_evals:
        return {
            "avg_acpl_game": np.nan, "blunder_count": 0,
            "best_move_count": 0, "n_moves_with_eval": 0, "was_losing": False,
        }

    losses: List[float] = []
    was_losing = False

    if player_color == "white":
        # Iterate over White's eval indices: 0, 2, 4, …
        for i, idx_after in enumerate(range(0, len(all_evals), 2)):
            idx_before = idx_after - 1  # Black's previous move (negative for first)
            e_after  = all_evals[idx_after]
            e_before = all_evals[idx_before] if idx_before >= 0 else 0.0
            losses.append(max(0.0, e_before - e_after))
            if e_before <= -150:          # White was losing before this move
                was_losing = True
    else:
        # Iterate over Black's eval indices: 1, 3, 5, …
        for idx_before in range(0, len(all_evals) - 1, 2):
            idx_after = idx_before + 1
            if idx_after >= len(all_evals):
                break
            e_before = all_evals[idx_before]  # after White's move (Black's turn incoming)
            e_after  = all_evals[idx_after]   # after Black's response
            losses.append(max(0.0, e_after - e_before))
            if e_before >= 150:               # Black was losing (White up 150+ cp)
                was_losing = True

    n = len(losses)
    return {
        "avg_acpl_game":    float(np.mean(losses)) if losses else np.nan,
        "blunder_count":    int(sum(1 for x in losses if x >= 150)),
        "best_move_count":  int(sum(1 for x in losses if x <= 10)),
        "n_moves_with_eval": n,
        "was_losing":       bool(was_losing),
    }


def compute_game_stats(
    an: str,
    base_time_sec: float,
    increment_sec: float,
    player_color: str,
) -> dict:
    """
    Compute all move-time and eval statistics for one player in one game.

    Returns a flat dict ready to be stored as columns on the game-level
    player DataFrame.
    """
    all_clocks = parse_clock_times(an)
    all_evals  = parse_eval_values(an)

    move_times = _compute_player_move_times(
        all_clocks, base_time_sec, increment_sec, player_color
    )
    eval_stats = _compute_player_eval_stats(all_evals, player_color)

    if move_times:
        mt_mean = float(np.mean(move_times))
        # std requires at least 2 data points; NaN for single-move games avoids misleading 0
        mt_std  = float(np.std(move_times)) if len(move_times) > 1 else np.nan
        # Time pressure = any move made with < 10s remaining on the clock.
        # Engine users almost never hit time pressure because the engine responds instantly.
        tp_count = int(sum(1 for c in (
            all_clocks[0::2] if player_color == "white" else all_clocks[1::2]
        ) if c < 10))
    else:
        mt_mean = mt_std = np.nan
        tp_count = 0  # no clock data — can't compute, but don't drop the game

    return {
        "move_time_mean":   mt_mean,
        "move_time_std":    mt_std,
        "time_pressure_count": tp_count,
        **eval_stats,
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def _parse_time_control(tc_str: str) -> Tuple[float, float, str]:
    """
    Parse a TimeControl string like '600+5' into (base_sec, increment_sec, category).

    Returns ("unknown", 0, "unknown") on parse failure.
    """
    try:
        parts = str(tc_str).split('+')
        base = float(parts[0])
        inc  = float(parts[1]) if len(parts) > 1 else 0.0
        total = base + 40 * inc           # FIDE estimated game length
        if total < 180:
            cat = "bullet"
        elif total < 480:
            cat = "blitz"
        elif total < 1500:
            cat = "rapid"
        else:
            cat = "classical"
        return base, inc, cat
    except Exception:
        return 0.0, 0.0, "unknown"


def load_lichess(
    path: Path,
    sample_n: int = LICHESS_SAMPLE_N,
    time_controls: Optional[List[str]] = None,
    rated_only: bool = True,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load a sample of games from the large Lichess CSV.

    Reads the file in chunks to avoid exhausting RAM, filters early,
    then returns at most `sample_n` rows.

    Args:
        path: Path to lichess_jul2016.csv (or equivalent).
        sample_n: Maximum number of games to return.
        time_controls: If set, keep only these categories (e.g. ['rapid', 'classical']).
        rated_only: Drop unrated games and games with missing / invalid Elo.
        random_state: Random seed for reproducible sampling.

    Returns:
        Raw DataFrame with original columns plus `base_time_sec`, `increment_sec`,
        `time_control_cat` columns added.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Lichess dataset not found at {path}.\n"
            "Download from https://www.kaggle.com/datasets/arevel/chess-games\n"
            "and save as data/raw/lichess_jul2016.csv"
        )

    logger.info("Loading Lichess dataset from %s (sample_n=%s) …", path, f"{sample_n:,}")

    chunks = []
    collected = 0
    chunksize = 50_000

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        # ── Filter: valid Elo ratings ─────────────────────────────────────────
        # Some games have "?" or missing Elo — coerce to NaN and drop them.
        # We need real ratings to compute expected win rate and rating-based features.
        for col in ("WhiteElo", "BlackElo"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        if rated_only:
            chunk = chunk.dropna(subset=["WhiteElo", "BlackElo"])
            chunk = chunk[(chunk["WhiteElo"] > 0) & (chunk["BlackElo"] > 0)]

        # ── Filter: completed games only ──────────────────────────────────────
        # Drop abandoned, forfeited, or otherwise non-standard results.
        chunk = chunk[chunk["Result"].isin(["1-0", "0-1", "1/2-1/2"])]

        # ── Parse time control ────────────────────────────────────────────────
        tc_parsed = chunk["TimeControl"].apply(_parse_time_control)
        chunk["base_time_sec"]    = tc_parsed.apply(lambda x: x[0])
        chunk["increment_sec"]    = tc_parsed.apply(lambda x: x[1])
        chunk["time_control_cat"] = tc_parsed.apply(lambda x: x[2])

        if time_controls:
            chunk = chunk[chunk["time_control_cat"].isin(time_controls)]

        chunks.append(chunk)
        collected += len(chunk)
        logger.debug("Processed chunk — collected %s games so far", f"{collected:,}")

        if collected >= sample_n * 3:
            # We have 3× what we need — stop reading. Collecting more than we'll sample
            # ensures we get a random cross-section of the filtered pool rather than
            # just the first N games (which could be biased toward early users/ratings).
            break

    if not chunks:
        raise ValueError("No games remaining after filtering. Check time_controls and rated_only settings.")

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Filtered pool: %s games", f"{len(df):,}")

    # Sample
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
        logger.info("Sampled %s games", f"{sample_n:,}")

    return df


def _to_player_level_lichess(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape game-level Lichess data to player-level with move stats computed
    per player perspective.

    This is the expensive step: parse_clock_times and parse_eval_values
    are called once per (game, colour) pair.
    """
    records = []
    total = len(game_df)
    log_every = max(1, total // 10)

    for i, row in enumerate(game_df.itertuples(index=False)):
        if i % log_every == 0:
            logger.info("  Computing move stats … %d / %d games", i, total)

        an            = str(row.AN) if hasattr(row, "AN") else ""
        base_time     = float(row.base_time_sec)
        increment     = float(row.increment_sec)
        winner_raw    = str(row.Result)

        # Map Result → winner string used by the rest of the pipeline
        winner = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}.get(winner_raw, "draw")

        # Count clock annotations to estimate total moves. Each [%clk] appears once
        # per half-move (ply), so the count equals the total number of plies played.
        # Fallback: count Black move markers (e.g. "15...") and double them — rougher.
        clk_count = len(_CLK_RE.findall(an))
        turns = clk_count if clk_count > 0 else max(
            1, len(re.findall(r'\d+\.\.\.', an)) * 2
        )

        # Opening depth heuristic — the big dataset doesn't have a real opening_ply
        # column, so we approximate it as 1/4 of total game length, capped at 12.
        # This is the weakest feature in our model anyway (permutation importance showed
        # it contributes very little), so the heuristic inaccuracy is acceptable.
        opening_ply = min(12, turns // 4)

        # Rating diff (white − black) — same sign convention as the small dataset.
        rating_diff = float(row.WhiteElo) - float(row.BlackElo)

        base_record = {
            "id":               i,
            "turns":            turns,
            "winner":           winner,
            "increment_code":   str(row.TimeControl),
            "white_id":         str(row.White),
            "white_rating":     float(row.WhiteElo),
            "black_id":         str(row.Black),
            "black_rating":     float(row.BlackElo),
            "moves":            an,
            "opening_ply":      opening_ply,
            "opening_eco":      str(row.ECO) if hasattr(row, "ECO") else "",
            "opening_name":     str(row.Opening) if hasattr(row, "Opening") else "",
            "victory_status":   str(row.Termination) if hasattr(row, "Termination") else "",
            "base_time_sec":    base_time,
            "increment_sec":    increment,
            "time_control_cat": str(row.time_control_cat),
            "rating_diff":      rating_diff,
        }

        for color, pid, p_rating, o_rating in [
            ("white", row.White, row.WhiteElo, row.BlackElo),
            ("black", row.Black, row.BlackElo, row.WhiteElo),
        ]:
            won    = 1 if winner == color else 0
            result = 1.0 if winner == color else (0.5 if winner == "draw" else 0.0)

            move_stats = compute_game_stats(an, base_time, increment, color)

            records.append({
                **base_record,
                "player_id":      str(pid),
                "player_rating":  float(p_rating),
                "opponent_rating":float(o_rating),
                "won":            won,
                "result":         result,
                "color":          color,
                **move_stats,
            })

    player_df = pd.DataFrame(records)

    # Enforce minimum games per player
    counts = player_df["player_id"].value_counts()
    valid  = counts[counts >= MIN_GAMES_PER_PLAYER].index
    player_df = player_df[player_df["player_id"].isin(valid)].reset_index(drop=True)

    logger.info(
        "Player-level dataset: %s rows, %s unique players",
        f"{len(player_df):,}",
        f"{player_df['player_id'].nunique():,}",
    )
    return player_df


def load_and_prepare_lichess(
    path: Path = None,
    sample_n: int = LICHESS_SAMPLE_N,
    time_controls: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full loading pipeline for the large Lichess dataset.

    Returns (game_df, player_df) with the same column conventions as
    data_loader.load_and_prepare(), plus move-time and eval columns
    that enable the 'extended' feature set.
    """
    from src.config import DATA_LICHESS
    path = path or DATA_LICHESS

    game_df   = load_lichess(path, sample_n=sample_n, time_controls=time_controls)
    player_df = _to_player_level_lichess(game_df)
    return game_df, player_df


if __name__ == "__main__":
    from src.config import DATA_LICHESS, LICHESS_SAMPLE_N, LICHESS_TIME_CONTROLS
    g, p = load_and_prepare_lichess(
        DATA_LICHESS, sample_n=LICHESS_SAMPLE_N, time_controls=LICHESS_TIME_CONTROLS
    )
    print("game_df :", g.shape)
    print("player_df:", p.shape)
    print(p.head())
