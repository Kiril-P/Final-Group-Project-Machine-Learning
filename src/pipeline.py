"""
End-to-end pipeline. Run from repository root:

    python -m src.pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_LICHESS, LICHESS_SAMPLE_N, LICHESS_TIME_CONTROLS, RANDOM_SEED, RESULTS_DIR
from src.data_loader import load_and_prepare
from src.lichess_loader import load_and_prepare_lichess
from src.features import aggregate_player_stats, add_engineered_features, get_feature_matrix
from src.interpretation import (
    analyze_false_positives,
    permutation_feature_importance,
    plot_anomaly_score_distribution,
    plot_feature_importance,
    plot_learning_curves,
    plot_roc_curves,
)
from src.models import (
    AutoencoderDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    ZScoreBaseline,
    run_all_models,
    run_hyperparameter_search,
)
from src.validation import (
    compute_davies_bouldin,
    compute_silhouette,
    cross_validate_anomaly_models,
    evaluate_injection_recovery,
    inject_synthetic_anomalies,
    test_anomaly_vs_normal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    use_acpl: bool = False,
    time_control: Optional[str] = None,
    dataset: str = "small",
    feature_set: str = "base",
):
    """
    Run the full anomaly detection pipeline.

    Args:
        use_acpl:     Run Stockfish ACPL (small dataset only, slow).
        time_control: Filter to one time control, e.g. 'blitz'. None = all.
        dataset:      'small' (20 k-game Kaggle CSV) or 'lichess' (6.25 M-game dataset).
        feature_set:  'base' (8 features) or 'extended' (up to 14, Lichess only).
    """
    logger.info("=" * 60)
    logger.info("Chess behavioral anomaly detection — full pipeline")
    logger.info("dataset=%s  feature_set=%s  time_control=%s", dataset, feature_set, time_control)
    logger.info("=" * 60)

    logger.info("Stage 1: Loading data...")
    if dataset == "lichess":
        game_df, player_df = load_and_prepare_lichess(
            DATA_LICHESS,
            sample_n=LICHESS_SAMPLE_N,
            time_controls=LICHESS_TIME_CONTROLS,
        )
    else:
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

    # fit_scaler=False: return raw (unscaled) features here so the scaler can later be
    # fit exclusively on the training partition. If we scaled everything upfront, the
    # scaler's mean/std would include test-set information — a form of data leakage
    # that makes validation metrics look better than they'd be on truly new data.
    X_raw, meta, _ = get_feature_matrix(
        agg, use_acpl=use_acpl, time_control=time_control,
        fit_scaler=False, feature_set=feature_set,
    )
    feature_names = list(X_raw.columns)
    X_arr = X_raw.values

    agg.to_csv(RESULTS_DIR / "player_features.csv", index=False)
    logger.info("Saved player features to %s", RESULTS_DIR / "player_features.csv")

    # ── Stage 2a: 70 / 15 / 15 train / val / test split ────────────────────
    # 70% train: models learn from this.
    # 15% val: used for hyperparameter search and model selection (never in final numbers).
    # 15% test: touched exactly once at the end to report final performance — no peeking.
    # Split on raw data so the scaler we fit below never "sees" val or test statistics.
    logger.info("Stage 2a: Train / val / test split (70 / 15 / 15)...")
    train_idx, temp_idx = train_test_split(
        np.arange(len(X_arr)), test_size=0.30, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=RANDOM_SEED
    )

    # ── Stage 2b: Fit scaler on training data only, then transform each split ─
    # StandardScaler learns mean and std from X_train rows only.
    # The same learned parameters are then applied to val and test — no new fitting.
    # This is the correct way to normalise; fitting on all data would be leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_arr[train_idx])   # learns mean/std from train only
    X_val   = scaler.transform(X_arr[val_idx])          # applies same transform (no refit)
    X_test  = scaler.transform(X_arr[test_idx])         # test is truly unseen until Stage 4b

    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_val   = meta.iloc[val_idx].reset_index(drop=True)
    meta_test  = meta.iloc[test_idx].reset_index(drop=True)
    logger.info(
        "Split sizes — Train: %s | Val: %s | Test: %s",
        len(X_train), len(X_val), len(X_test),
    )

    # ── Stage 2c: Multi-parameter random search on the validation set ────────
    # We inject synthetic anomalies into the val split to get labelled data for
    # hyperparameter optimisation. Without ground-truth cheater labels, this is the
    # best way to objectively compare model configurations.
    #
    # "subtle" strategy: anomalies that look plausible but are statistically outlying —
    # chosen because obvious anomalies are easy and don't differentiate models.
    #
    # IF / OC-SVM / LOF: 20-iteration random search over contamination AND
    #   structural params (n_estimators, n_neighbors, kernel, etc.).
    # Autoencoder: exhaustive mini-grid over encoding_dim × threshold_percentile
    #   using 30-epoch trials; best config re-trained at full epochs below.
    logger.info("Stage 2c: Hyperparameter search on val split...")
    X_inj_tune, y_inj_tune = inject_synthetic_anomalies(
        pd.DataFrame(X_val, columns=feature_names), n=50, strategy="subtle"
    )
    X_inj_arr = X_inj_tune.values if hasattr(X_inj_tune, "values") else X_inj_tune

    search_results, best_params = run_hyperparameter_search(
        X_train=X_train,
        X_val_injected=X_inj_arr,
        y_val_injected=y_inj_tune,
    )
    search_results.to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    logger.info("Best params per model:\n%s", {k: v for k, v in best_params.items()})

    # ── Stage 2d: 5-fold cross-validation for variance estimation ────────────
    # Runs on the development set (train + val) only — the test set is never involved.
    # Within every fold, the scaler is re-fit on the fold's training rows, so there's
    # zero leakage even inside CV. This lets us report "ROC-AUC 0.79 ± 0.03 (95% CI)"
    # rather than a single number that might be a lucky split artifact.
    logger.info("Stage 2d: 5-fold cross-validation (development set only)...")
    dev_idx = np.concatenate([train_idx, val_idx])
    cv_raw, cv_summary = cross_validate_anomaly_models(
        X=X_arr[dev_idx],          # raw (unscaled) development data
        feature_names=feature_names,
        best_params=best_params,
    )
    cv_raw.to_csv(RESULTS_DIR / "cv_raw_results.csv", index=False)
    cv_summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    logger.info(
        "CV summary (ROC-AUC):\n%s",
        cv_summary[["model", "roc_auc_mean", "roc_auc_std", "roc_auc_ci95"]].to_string(index=False),
    )

    logger.info("Stage 3: Training anomaly detection models on train split...")
    results = run_all_models(X_train, meta_train, model_params=best_params)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    logger.info("Saved model results to %s", RESULTS_DIR / "model_results.csv")

    logger.info("Stage 4: Evaluation — injection recovery...")
    # Re-fit IF with its tuned params for use in Stages 5-7
    if_model = IsolationForestDetector(**best_params.get("IsolationForest", {"contamination": 0.05}))
    if_model.fit(X_train)

    # Capture best_params in closure default-args to avoid late-binding issues
    model_ctors = [
        (lambda: ZScoreBaseline(contamination=0.05), "ZScoreBaseline"),
        (lambda p=best_params.get("LOF", {"contamination": 0.05}):
             LOFDetector(**p), "LOF"),
        (lambda p=best_params.get("IsolationForest", {"contamination": 0.05}):
             IsolationForestDetector(**p), "IsolationForest"),
        (lambda p=best_params.get("OneClassSVM", {"nu": 0.05}):
             OneClassSVMDetector(**p), "OneClassSVM"),
        (lambda p=best_params.get("Autoencoder", {}):
             AutoencoderDetector(input_dim=X_train.shape[1], **p), "Autoencoder"),
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

    # Stage 4b: Test set evaluation — touched exactly once, these are the numbers in the report.
    # No further tuning happens after this; looking at test results and then adjusting
    # hyperparameters would invalidate the evaluation entirely.
    logger.info("Stage 4b: Test set evaluation (final, unseen)...")
    test_rows = []
    # Collect raw scores from the subtle strategy for the ROC curve plot.
    roc_scores: dict = {}
    roc_y_true: np.ndarray | None = None
    for strategy in ("engine_perfect", "subtle"):
        X_inj_eval, y_inj_eval = inject_synthetic_anomalies(
            pd.DataFrame(X_test, columns=feature_names), n=50, strategy=strategy
        )
        X_inj_eval_arr = X_inj_eval if isinstance(X_inj_eval, np.ndarray) else X_inj_eval.values
        for ctor, name in model_ctors:
            m = ctor()
            m.fit(X_train)
            metrics = evaluate_injection_recovery(m, X_inj_eval_arr, y_inj_eval)
            test_rows.append({"model": name, "strategy": strategy, "split": "test", **metrics})
            if strategy == "subtle":
                roc_scores[name] = m.score(X_inj_eval_arr)
                roc_y_true = y_inj_eval
    holdout_df = pd.DataFrame(test_rows)
    holdout_df.to_csv(RESULTS_DIR / "holdout_evaluation.csv", index=False)
    logger.info("Test evaluation saved.\n%s", holdout_df.to_string(index=False))

    logger.info("Stage 4b (cont): Plotting ROC curves (subtle, test set)...")
    if roc_y_true is not None:
        roc_df = plot_roc_curves(roc_scores, roc_y_true)
        roc_df.to_csv(RESULTS_DIR / "roc_curves.csv", index=False)

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

    logger.info("Stage 6: Feature importance and learning curves...")
    # Learning curves: inject once on val, then train each model on increasing
    # fractions of X_train to show convergence behaviour.
    X_lc_inj, y_lc_inj = inject_synthetic_anomalies(
        pd.DataFrame(X_val, columns=feature_names), n=50, strategy="subtle"
    )
    X_lc_arr = X_lc_inj if isinstance(X_lc_inj, np.ndarray) else X_lc_inj.values
    lc_df = plot_learning_curves(
        X_train=X_train,
        X_val_injected=X_lc_arr,
        y_val_injected=y_lc_inj,
        best_params=best_params,
        feature_names=feature_names,
    )
    lc_df.to_csv(RESULTS_DIR / "learning_curves.csv", index=False)

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
    # Switch to dataset="lichess", feature_set="extended" once lichess_jul2016.csv
    # is placed in data/raw/.  Until then, keep dataset="small" to verify nothing broke.
    main(dataset="lichess", feature_set="extended", time_control=None)
