"""
End-to-end pipeline. Run from repository root:

    python -m src.pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, RESULTS_DIR
from src.data_loader import load_and_prepare
from src.features import aggregate_player_stats, add_engineered_features, get_feature_matrix
from src.interpretation import (
    analyze_false_positives,
    permutation_feature_importance,
    plot_anomaly_score_distribution,
    plot_feature_importance,
)
from src.models import (
    AutoencoderDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    ZScoreBaseline,
    run_all_models,
    tune_contamination,
)
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

    # fit_scaler=False: raw (unscaled) features returned so the scaler can be fit
    # exclusively on the training partition — prevents test-set statistics leaking
    # into the scaler's mean/std and contaminating evaluation metrics.
    X_raw, meta, _ = get_feature_matrix(agg, use_acpl=use_acpl, time_control=time_control, fit_scaler=False)
    feature_names = list(X_raw.columns)
    X_arr = X_raw.values

    agg.to_csv(RESULTS_DIR / "player_features.csv", index=False)
    logger.info("Saved player features to %s", RESULTS_DIR / "player_features.csv")

    # ── Stage 2a: 70 / 15 / 15 train / val / test split ────────────────────
    # Split is performed on raw (unscaled) data so the scaler never sees val/test.
    logger.info("Stage 2a: Train / val / test split (70 / 15 / 15)...")
    train_idx, temp_idx = train_test_split(
        np.arange(len(X_arr)), test_size=0.30, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=RANDOM_SEED
    )

    # ── Stage 2b: Fit scaler on training data only, then transform each split ─
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_arr[train_idx])   # learns mean/std from train
    X_val   = scaler.transform(X_arr[val_idx])          # applies same transform
    X_test  = scaler.transform(X_arr[test_idx])         # test is truly unseen

    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_val   = meta.iloc[val_idx].reset_index(drop=True)
    meta_test  = meta.iloc[test_idx].reset_index(drop=True)
    logger.info(
        "Split sizes — Train: %s | Val: %s | Test: %s",
        len(X_train), len(X_val), len(X_test),
    )

    # ── Stage 2c: Hyperparameter tuning on validation set ───────────────────
    # Inject synthetic anomalies into the val set and search for the contamination
    # rate that maximises ROC-AUC on that held-out partition.  The test set is
    # never touched during tuning.
    logger.info("Stage 2c: Hyperparameter tuning on val split (subtle injection)...")
    X_inj_tune, y_inj_tune = inject_synthetic_anomalies(
        pd.DataFrame(X_val, columns=feature_names), n=50, strategy="subtle"
    )
    X_inj_arr = X_inj_tune.values if hasattr(X_inj_tune, "values") else X_inj_tune
    tuning_df = tune_contamination(X_train, X_inj_arr, y_inj_tune)
    tuning_df.to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    best = tuning_df.loc[tuning_df.groupby("model")["roc_auc"].idxmax()]
    overrides = dict(zip(best["model"], best["contamination"]))
    logger.info("Best contamination per model: %s", overrides)

    logger.info("Stage 3: Training anomaly detection models on train split...")
    results = run_all_models(X_train, meta_train, contamination_overrides=overrides)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    logger.info("Saved model results to %s", RESULTS_DIR / "model_results.csv")

    logger.info("Stage 4: Evaluation — injection recovery...")
    if_model = IsolationForestDetector(contamination=overrides.get("IsolationForest", 0.05))
    if_model.fit(X_train)

    model_ctors = [
        (lambda: ZScoreBaseline(contamination=0.05), "ZScoreBaseline"),
        (lambda: LOFDetector(contamination=overrides.get("LOF", 0.05)), "LOF"),
        (lambda: IsolationForestDetector(contamination=overrides.get("IsolationForest", 0.05)), "IsolationForest"),
        (lambda: OneClassSVMDetector(nu=overrides.get("OneClassSVM", 0.05)), "OneClassSVM"),
        (lambda: AutoencoderDetector(input_dim=X_train.shape[1]), "Autoencoder"),
    ]

    # Stage 4a: Validation set evaluation (used for model comparison / selection)
    logger.info("Stage 4a: Validation set evaluation...")
    val_rows = []
    for strategy in ("engine_perfect", "subtle"):
        X_inj_val, y_inj_val = inject_synthetic_anomalies(
            pd.DataFrame(X_val, columns=feature_names), n=50, strategy=strategy
        )
        for ctor, name in model_ctors:
            m = ctor()
            m.fit(X_train)
            metrics = evaluate_injection_recovery(m, X_inj_val, y_inj_val)
            val_rows.append({"model": name, "strategy": strategy, "split": "val", **metrics})
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(RESULTS_DIR / "val_evaluation.csv", index=False)
    logger.info("Validation evaluation saved.\n%s", val_df.to_string(index=False))

    # Stage 4b: Test set evaluation — touched once, final reported numbers only
    logger.info("Stage 4b: Test set evaluation (final, unseen)...")
    test_rows = []
    for strategy in ("engine_perfect", "subtle"):
        X_inj_eval, y_inj_eval = inject_synthetic_anomalies(
            pd.DataFrame(X_test, columns=feature_names), n=50, strategy=strategy
        )
        for ctor, name in model_ctors:
            m = ctor()
            m.fit(X_train)
            metrics = evaluate_injection_recovery(m, X_inj_eval, y_inj_eval)
            test_rows.append({"model": name, "strategy": strategy, "split": "test", **metrics})
    holdout_df = pd.DataFrame(test_rows)
    holdout_df.to_csv(RESULTS_DIR / "holdout_evaluation.csv", index=False)
    logger.info("Test evaluation saved.\n%s", holdout_df.to_string(index=False))

    # Reference injection result (IF on test, subtle) for notebook back-compat
    X_inj, y_inj = inject_synthetic_anomalies(
        pd.DataFrame(X_test, columns=feature_names), n=50, strategy="subtle"
    )
    injection_results = evaluate_injection_recovery(if_model, X_inj, y_inj)
    Path(RESULTS_DIR / "injection_results.json").write_text(json.dumps(injection_results, indent=2))

    logger.info("Stage 5: Statistical tests (on train split)...")
    if_labels = if_model.predict(X_train)
    divergence_df = test_anomaly_vs_normal(
        anomaly_scores=if_model.score(X_train),
        labels=if_labels,
        feature_matrix=X_train,
        feature_names=feature_names,
    )
    divergence_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    sil = compute_silhouette(X_train, if_labels)
    db = compute_davies_bouldin(X_train, if_labels)
    logger.info("Silhouette: %s, Davies-Bouldin: %s", sil, db)

    logger.info("Stage 6: Feature importance...")
    importance_df = permutation_feature_importance(if_model, X_train, feature_names)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df)

    plot_anomaly_score_distribution(if_model.score(X_train), if_labels, "IsolationForest")

    logger.info("Stage 7: Failure mode analysis...")
    failure_df = analyze_false_positives(meta_train, results, player_df)
    failure_df.to_csv(RESULTS_DIR / "failure_analysis.csv", index=False)

    logger.info("Pipeline complete. Outputs in %s", RESULTS_DIR)
    return results, importance_df, injection_results


if __name__ == "__main__":
    main(use_acpl=False, time_control="blitz")
