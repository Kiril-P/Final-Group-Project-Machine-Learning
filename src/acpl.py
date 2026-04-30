"""
Optional ACPL (average centipawn loss) via Stockfish. See `src.config` for paths and sample size.
"""

from __future__ import annotations

import logging
from typing import Optional

import chess
import chess.engine
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import STOCKFISH_DEPTH, STOCKFISH_PATH, STOCKFISH_SAMPLE_GAMES

logger = logging.getLogger(__name__)


def compute_game_acpl(moves_str: str, color: str, engine_path: str = STOCKFISH_PATH) -> Optional[float]:
    """ACPL for one color in one game; moves are space-separated UCI."""
    try:
        board = chess.Board()
        moves = moves_str.strip().split()
        centipawn_losses = []
        player_color = chess.WHITE if color == "white" else chess.BLACK

        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            for move_uci in moves:
                if board.turn != player_color:
                    try:
                        board.push_uci(move_uci)
                    except Exception:
                        break
                    continue

                info_before = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
                score_before = info_before["score"].white()

                try:
                    board.push_uci(move_uci)
                except Exception:
                    break

                info_after = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
                score_after = info_after["score"].white()

                def to_cp(score):
                    if score.is_mate():
                        return 1000 if score.score(mate_score=10000) > 0 else -1000
                    return max(-1000, min(1000, score.score()))

                cp_before = to_cp(score_before)
                cp_after = to_cp(score_after)
                if player_color == chess.WHITE:
                    loss = max(0, cp_before - cp_after)
                else:
                    loss = max(0, cp_after - cp_before)
                centipawn_losses.append(loss)

        if not centipawn_losses:
            return None
        return float(np.mean(centipawn_losses))
    except Exception as e:
        logger.debug("ACPL computation failed: %s", e)
        return None


def compute_acpl_for_dataset(
    game_df: pd.DataFrame,
    n_games: int = STOCKFISH_SAMPLE_GAMES,
    engine_path: str = STOCKFISH_PATH,
) -> pd.DataFrame:
    """Sample games, compute ACPL per side, aggregate by player."""
    sample = game_df.sample(min(n_games, len(game_df)), random_state=42)
    logger.info("Computing ACPL for %s games...", len(sample))
    records = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="ACPL"):
        for color, player_id in [("white", row["white_id"]), ("black", row["black_id"])]:
            acpl = compute_game_acpl(str(row["moves"]), color, engine_path)
            if acpl is not None:
                records.append({"player_id": player_id, "acpl": acpl})

    if not records:
        logger.warning("No ACPL values computed. Check Stockfish path.")
        return pd.DataFrame(columns=["player_id", "avg_acpl", "acpl_variance", "n_games_evaluated"])

    acpl_df = pd.DataFrame(records)
    result = acpl_df.groupby("player_id").agg(
        avg_acpl=("acpl", "mean"),
        acpl_variance=("acpl", "var"),
        n_games_evaluated=("acpl", "count"),
    ).reset_index()
    logger.info("ACPL computed for %s unique players", result["player_id"].nunique())
    return result


def check_stockfish_available(engine_path: str = STOCKFISH_PATH) -> bool:
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as _engine:
            _engine.analyse(chess.Board(), chess.engine.Limit(depth=5))
        logger.info("Stockfish is available.")
        return True
    except Exception as e:
        logger.warning("Stockfish not available at %r: %s", engine_path, e)
        return False


if __name__ == "__main__":
    print("Stockfish available:", check_stockfish_available())
