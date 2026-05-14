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
from src.features import (
    aggregate_player_stats,
    add_engineered_features,
    get_feature_matrix,
    BAND_Z_PAIRS,
    compute_band_stats,
    reapply_band_zscores,
)
from src.interpretation import (
    analyze_false_positives,
    generate_player_explanations,
    permutation_feature_importance,
    plot_anomaly_score_distribution,
    plot_feature_importance,
    plot_learning_curves,
    plot_roc_curves,
    plot_umap,
)
from src.models import (
    ACPLSubAutoencoder,
    AutoencoderDetector,
    HDBSCANDetector,
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

    # ── Stage 2e: Recompute band z-scores using training players only ─────────
    # add_engineered_features() above computed band means/stds from the full
    # population (all 28k players), so test-set distribution information
    # contaminated the normalisation.  Now that we have the split, we recompute
    # the stats from training players only and propagate corrected values back
    # into X_arr before refitting the scaler.
    logger.info("Stage 2e: Recomputing band z-scores from training split only...")
    train_pids = set(meta.iloc[train_idx]["player_id"])
    band_stats = compute_band_stats(agg[agg["player_id"].isin(train_pids)])
    agg = reapply_band_zscores(agg, band_stats)

    agg_indexed = agg.set_index("player_id")
    for _raw_feat, z_feat in BAND_Z_PAIRS:
        if z_feat not in feature_names:
            continue
        col_idx = feature_names.index(z_feat)
        for row_i, pid in enumerate(meta["player_id"]):
            if pid in agg_indexed.index:
                val = agg_indexed.at[pid, z_feat]
                if pd.notna(val):
                    X_arr[row_i, col_idx] = float(val)

    # Refit scaler on the corrected X_arr (train partition only, as before).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_arr[train_idx])
    X_val   = scaler.transform(X_arr[val_idx])
    X_test  = scaler.transform(X_arr[test_idx])
    logger.info("Stage 2e complete — band z-scores now derived from training players only.")

    # Save player features after band z-scores have been corrected to train-only stats,
    # so the CSV matches what the models actually see.
    agg.to_csv(RESULTS_DIR / "player_features.csv", index=False)
    logger.info("Saved player features to %s", RESULTS_DIR / "player_features.csv")

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

    # Inject feature_names so both AE variants can identify ACPL columns.
    best_params.setdefault("Autoencoder", {})["feature_names"] = feature_names
    best_params.setdefault("ACPLSubAutoencoder", {})["feature_names"] = feature_names

    # ── Stage 2d: 5-fold cross-validation for variance estimation ────────────
    # Runs on the development set (train + val) only — the test set is never involved.
    # Within every fold, the scaler is re-fit on the fold's training rows, so there's
    # zero leakage even inside CV. This lets us report "ROC-AUC 0.79 ± 0.03 (95% CI)"
    # rather than a single number that might be a lucky split artifact.
    #
    # We run CV twice — once with the subtle benchmark (model selection / comparison)
    # and once with the realistic_cheater benchmark (honest harder estimate with CI).
    # Both use the same folds and best_params for a direct apples-to-apples comparison.
    logger.info("Stage 2d: 5-fold cross-validation (development set only)...")
    dev_idx = np.concatenate([train_idx, val_idx])
    cv_raw, cv_summary = cross_validate_anomaly_models(
        X=X_arr[dev_idx],
        feature_names=feature_names,
        best_params=best_params,
        injection_strategy="subtle",
    )
    cv_raw.to_csv(RESULTS_DIR / "cv_raw_results.csv", index=False)
    cv_summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    logger.info(
        "CV summary subtle (ROC-AUC):\n%s",
        cv_summary[["model", "roc_auc_mean", "roc_auc_std", "roc_auc_ci95"]].to_string(index=False),
    )

    logger.info("Stage 2d-ii: 5-fold CV on realistic_cheater benchmark...")
    cv_rc_raw, cv_rc_summary = cross_validate_anomaly_models(
        X=X_arr[dev_idx],
        feature_names=feature_names,
        best_params=best_params,
        injection_strategy="realistic_cheater",
        n_injected=100,     # inject more so the signal is stable across folds
    )
    cv_rc_raw.to_csv(RESULTS_DIR / "cv_raw_results_realistic.csv", index=False)
    cv_rc_summary.to_csv(RESULTS_DIR / "cv_summary_realistic.csv", index=False)
    logger.info(
        "CV summary realistic_cheater (ROC-AUC):\n%s",
        cv_rc_summary[["model", "roc_auc_mean", "roc_auc_std", "roc_auc_ci95"]].to_string(index=False),
    )

    logger.info("Stage 3: Training anomaly detection models on train split...")
    results = run_all_models(X_train, meta_train, model_params=best_params)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    logger.info("Saved model results to %s", RESULTS_DIR / "model_results.csv")

    # ── Stage 3b: Score ALL players (train + val + test) ─────────────────────
    # Models are still trained on X_train only — no leakage.
    # The scaler was already fit on X_train (Stage 2b), so X_val and X_test are
    # already in the same normalised space.  We simply apply the trained models
    # to those pre-scaled arrays.  This is identical to production deployment:
    # you train once, then score any new player that comes in.
    # Evaluation metrics in holdout_evaluation.csv are computed separately (Stage 4b)
    # and are not affected by this step.
    logger.info("Stage 3b: Scoring ALL players with train-fitted models...")
    X_all    = np.vstack([X_train, X_val, X_test])
    meta_all = pd.concat([meta_train, meta_val, meta_test]).reset_index(drop=True)
    all_results = run_all_models(
        X_train,           # fit on training data only
        meta_all,          # metadata for all players (just for labelling rows)
        model_params=best_params,
        X_score=X_all,     # score over the full scaled dataset
    )
    all_results.to_csv(RESULTS_DIR / "all_player_results.csv", index=False)
    logger.info(
        "Scored %s players total → %s",
        len(all_results),
        RESULTS_DIR / "all_player_results.csv",
    )

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
        (lambda p=best_params.get("HDBSCAN", {"min_cluster_size": 15}):
             HDBSCANDetector(**p), "HDBSCAN"),
        (lambda p=best_params.get("ACPLSubAutoencoder", {}):
             ACPLSubAutoencoder(**p), "ACPLSubAutoencoder"),
    ]

    # Stage 4a: Validation set evaluation (used for model comparison / selection)
    #
    # Two benchmark strategies:
    #   "sanity_check" (formerly "engine_perfect"):
    #       Synthetic players placed at p99 of every feature → AUC ≈ 1.0 for every
    #       working anomaly detector, by construction.  This proves the models function
    #       numerically; it is NOT evidence of real-world effectiveness or overfitting.
    #       See validation.py for the full explanation of why this result is expected.
    #   "subtle":
    #       Real player rows with ~1/3 of features perturbed by 1.5σ — stays on the
    #       data manifold.  This is the number that actually matters for model selection.
    logger.info("Stage 4a: Validation set evaluation...")
    val_rows = []
    for strategy, label, n_inj in [
        ("engine_perfect",   "sanity_check",      50),   # proves models work; AUC≈1.0 expected
        ("subtle",           "subtle",             50),   # main generalist benchmark
        ("realistic_cheater","realistic_cheater", 100),   # domain-grounded cheater profile
    ]:
        X_inj_val, y_inj_val = inject_synthetic_anomalies(
            pd.DataFrame(X_val, columns=feature_names), n=n_inj, strategy=strategy
        )
        for ctor, name in model_ctors:
            m = ctor()
            m.fit(X_train)
            metrics = evaluate_injection_recovery(m, X_inj_val, y_inj_val)
            val_rows.append({"model": name, "strategy": label, "split": "val", **metrics})
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(RESULTS_DIR / "val_evaluation.csv", index=False)
    logger.info("Validation evaluation saved.\n%s", val_df.to_string(index=False))

    # Stage 4b: Test set evaluation — touched exactly once, these are the numbers in the report.
    # No further tuning happens after this; looking at test results and then adjusting
    # hyperparameters would invalidate the evaluation entirely.
    logger.info("Stage 4b: Test set evaluation (final, unseen)...")
    test_rows = []
    # Collect raw scores for both meaningful benchmarks for the ROC curve plot.
    # sanity_check is excluded from ROC — it always gives AUC≈1.0 by construction.
    roc_scores_subtle:   dict = {}
    roc_scores_realistic: dict = {}
    roc_y_subtle:   np.ndarray | None = None
    roc_y_realistic: np.ndarray | None = None

    for strategy, label, n_inj in [
        ("engine_perfect",    "sanity_check",      50),
        ("subtle",            "subtle",             50),
        ("realistic_cheater", "realistic_cheater", 100),
    ]:
        X_inj_eval, y_inj_eval = inject_synthetic_anomalies(
            pd.DataFrame(X_test, columns=feature_names), n=n_inj, strategy=strategy
        )
        X_inj_eval_arr = X_inj_eval if isinstance(X_inj_eval, np.ndarray) else X_inj_eval.values
        for ctor, name in model_ctors:
            m = ctor()
            m.fit(X_train)
            metrics = evaluate_injection_recovery(m, X_inj_eval_arr, y_inj_eval)
            test_rows.append({"model": name, "strategy": label, "split": "test", **metrics})
            if label == "subtle":
                roc_scores_subtle[name] = m.score(X_inj_eval_arr)
                roc_y_subtle = y_inj_eval
            elif label == "realistic_cheater":
                roc_scores_realistic[name] = m.score(X_inj_eval_arr)
                roc_y_realistic = y_inj_eval

    holdout_df = pd.DataFrame(test_rows)
    holdout_df.to_csv(RESULTS_DIR / "holdout_evaluation.csv", index=False)
    logger.info("Test evaluation saved.\n%s", holdout_df.to_string(index=False))

    logger.info("Stage 4b (cont): Plotting ROC curves (subtle + realistic_cheater, test set)...")
    if roc_y_subtle is not None:
        roc_df = plot_roc_curves(roc_scores_subtle, roc_y_subtle)
        roc_df.to_csv(RESULTS_DIR / "roc_curves_subtle.csv", index=False)
    if roc_y_realistic is not None:
        roc_df_r = plot_roc_curves(roc_scores_realistic, roc_y_realistic)
        roc_df_r.to_csv(RESULTS_DIR / "roc_curves_realistic.csv", index=False)

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

    # Feature importance via permutation on LOF — our best model (AUC 0.962).
    # Originally used IsolationForest here because it's the most common choice in the
    # literature.  But IF is our weakest ensemble model (AUC 0.773, below the ZScore
    # baseline), so its importance values describe what a suboptimal model attends to.
    # LOF drives most of our detections — its importance is what belongs in the report.
    # Permutation importance is model-agnostic: shuffle one feature at a time, measure
    # mean drop in anomaly score.  Works identically for LOF as for any other model.
    lof_importance = LOFDetector(**best_params.get("LOF", {"contamination": 0.05}))
    lof_importance.fit(X_train)
    importance_df = permutation_feature_importance(lof_importance, X_train, feature_names)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df)

    plot_anomaly_score_distribution(lof_importance.score(X_train), lof_importance.predict(X_train), "LOF")

    logger.info("Stage 7: Failure mode analysis...")
    failure_df = analyze_false_positives(meta_train, results, player_df)
    failure_df.to_csv(RESULTS_DIR / "failure_analysis.csv", index=False)

    # Stage 7b: Per-player explainability for all ensemble-flagged players.
    # For each flagged player we report which features deviated most and in which
    # direction, with honest language about what we do and don't know.
    # Features with a clear cheating interpretation get a plain-English explanation;
    # features that are statistically anomalous but ambiguous are marked as
    # "model-detected" rather than inventing a reason.
    logger.info("Stage 7b: Generating per-player explanations (all players)...")
    # Use all_results so every player — not just the 70% training split — gets
    # an explanation if the ensemble flags them.
    explanations_df = generate_player_explanations(all_results, agg, feature_names)
    explanations_df.to_csv(RESULTS_DIR / "player_explanations.csv", index=False)
    logger.info(
        "Explanations saved for %s flagged players (out of %s total) → %s",
        len(explanations_df),
        len(all_results),
        RESULTS_DIR / "player_explanations.csv",
    )

    # Stage 7c: UMAP visualisation of the full player population.
    # Reduces the feature space to 2D so we can visually inspect whether the
    # flagged players cluster separately from the normal population.
    # Two panels: binary flag (left) and vote-count gradient (right).
    # This is the last stage — UMAP is slow on 17k points so we run it last.
    logger.info("Stage 7c: UMAP projection (full population)...")
    plot_umap(X_all, all_results)

    logger.info("Pipeline complete. Outputs in %s", RESULTS_DIR)
    return all_results, importance_df, injection_results


if __name__ == "__main__":
    # Switch to dataset="lichess", feature_set="extended" once lichess_jul2016.csv
    # is placed in data/raw/.  Until then, keep dataset="small" to verify nothing broke.
    main(dataset="lichess", feature_set="extended", time_control=None)
