"""
End-to-end pipeline. Run from repository root:

    python -m src.pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import RESULTS_DIR
from src.data_loader import load_and_prepare
from src.features import aggregate_player_stats, add_engineered_features, get_feature_matrix
from src.interpretation import (
    analyze_false_positives,
    permutation_feature_importance,
    plot_anomaly_score_distribution,
    plot_feature_importance,
)
from src.models import IsolationForestDetector, run_all_models
from src.validation import (
    compute_davies_bouldin,
    compute_silhouette,
    evaluate_injection_recovery,
    inject_synthetic_anomalies,
    test_anomaly_vs_normal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(use_acpl: bool = False, time_control: str = "blitz"):
    logger.info("=" * 60)
    logger.info("Chess behavioral anomaly detection — full pipeline")
    logger.info("=" * 60)

    logger.info("Stage 1: Loading data...")
    game_df, player_df = load_and_prepare()

    logger.info("Stage 2: Engineering features...")
    agg = aggregate_player_stats(player_df)
    agg = add_engineered_features(agg)

    if use_acpl:
        from src.acpl import check_stockfish_available, compute_acpl_for_dataset
        from src.features import add_acpl_features

        if check_stockfish_available():
            acpl_df = compute_acpl_for_dataset(game_df)
            agg = add_acpl_features(agg, acpl_df)
        else:
            logger.warning("Stockfish unavailable — skipping ACPL features.")

    X, meta, _scaler = get_feature_matrix(agg, use_acpl=use_acpl, time_control=time_control)
    feature_names = list(X.columns)
    X_arr = X.values

    agg.to_csv(RESULTS_DIR / "player_features.csv", index=False)
    logger.info("Saved player features to %s", RESULTS_DIR / "player_features.csv")

    logger.info("Stage 3: Training anomaly detection models...")
    results = run_all_models(X_arr, meta)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    logger.info("Saved model results to %s", RESULTS_DIR / "model_results.csv")

    logger.info("Stage 4: Synthetic anomaly injection...")
    if_model = IsolationForestDetector()
    if_model.fit(X_arr)
    X_inj, y_inj = inject_synthetic_anomalies(X, n=50, strategy="engine_perfect")
    injection_results = evaluate_injection_recovery(if_model, X_inj, y_inj)
    Path(RESULTS_DIR / "injection_results.json").write_text(json.dumps(injection_results, indent=2))
    logger.info("Injection results saved.")

    logger.info("Stage 5: Statistical tests...")
    if_labels = if_model.predict(X_arr)
    divergence_df = test_anomaly_vs_normal(
        anomaly_scores=if_model.score(X_arr),
        labels=if_labels,
        feature_matrix=X_arr,
        feature_names=feature_names,
    )
    divergence_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    sil = compute_silhouette(X_arr, if_labels)
    db = compute_davies_bouldin(X_arr, if_labels)
    logger.info("Silhouette: %s, Davies-Bouldin: %s", sil, db)

    logger.info("Stage 6: Feature importance...")
    importance_df = permutation_feature_importance(if_model, X_arr, feature_names)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df)

    plot_anomaly_score_distribution(if_model.score(X_arr), if_labels, "IsolationForest")

    logger.info("Stage 7: Failure mode analysis...")
    failure_df = analyze_false_positives(meta, results, player_df)
    failure_df.to_csv(RESULTS_DIR / "failure_analysis.csv", index=False)

    logger.info("Pipeline complete. Outputs in %s", RESULTS_DIR)
    return results, importance_df, injection_results


if __name__ == "__main__":
    main(use_acpl=False, time_control="blitz")
