"""
Anomaly detection models: Isolation Forest, One-Class SVM, optional Autoencoder.

Common interface:
    .fit(X) -> self
    .score(X) -> np.ndarray (higher = more anomalous)
    .predict(X) -> np.ndarray in {-1, 1}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    AUTOENCODER_EVAL_FEATURES,
    AUTOENCODER_EVAL_WEIGHT,
    AUTOENCODER_PARAMS,
    AUTOENCODER_SEARCH,
    AUTOENCODER_SEARCH_EPOCHS,
    HDBSCAN_PARAMS,
    HDBSCAN_SEARCH,
    ISOLATION_FOREST_PARAMS,
    ISOLATION_FOREST_SEARCH,
    LOF_SEARCH,
    MIN_GAMES_FOR_FLAG,
    MODELS_DIR,
    OCSVM_PARAMS,
    OCSVM_SEARCH,
    RANDOM_SEARCH_N_ITER,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class ZScoreBaseline:
    """Trivial baseline: flag rows whose max |z-score| across features exceeds a threshold.

    Threshold is set so the top `contamination` fraction of training points are flagged.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.threshold_: Optional[float] = None
        self.name = "ZScoreBaseline"

    def fit(self, X: np.ndarray) -> "ZScoreBaseline":
        scores = self.score(X)
        self.threshold_ = float(np.quantile(scores, 1 - self.contamination))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # X is already standardized upstream (StandardScaler), so values are z-scores.
        return np.max(np.abs(X), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None
        return np.where(self.score(X) > self.threshold_, -1, 1)


class LOFDetector:
    """Classical baseline: Local Outlier Factor in novelty mode."""

    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20):
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )
        self.name = "LOF"

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class IsolationForestDetector:
    """Thin wrapper around sklearn IsolationForest."""

    def __init__(self, **kwargs):
        params = {**ISOLATION_FOREST_PARAMS, **kwargs}
        self.model = IsolationForest(**params)
        self.name = "IsolationForest"

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODELS_DIR / "isolation_forest.pkl"
        joblib.dump(self.model, path)
        logger.info("Saved IsolationForest to %s", path)

    def load(self, path: Optional[Path] = None) -> "IsolationForestDetector":
        path = path or MODELS_DIR / "isolation_forest.pkl"
        self.model = joblib.load(path)
        return self


class OneClassSVMDetector:
    """Thin wrapper around sklearn OneClassSVM."""

    def __init__(self, **kwargs):
        params = {**OCSVM_PARAMS, **kwargs}
        self.model = OneClassSVM(**params)
        self.name = "OneClassSVM"

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODELS_DIR / "ocsvm.pkl"
        joblib.dump(self.model, path)
        logger.info("Saved OneClassSVM to %s", path)

    def load(self, path: Optional[Path] = None) -> "OneClassSVMDetector":
        path = path or MODELS_DIR / "ocsvm.pkl"
        self.model = joblib.load(path)
        return self


class HDBSCANDetector:
    """HDBSCAN-based anomaly detector using sklearn's built-in implementation.

    HDBSCAN finds arbitrarily-shaped clusters and labels every point that doesn't
    fit into any cluster as noise (-1). Noise points are our anomaly candidates.

    sklearn's HDBSCAN is transductive — it only labels the data it was trained on.
    For scoring NEW data (val/test sets with injected anomalies) we use a KNN
    approximation:
      1. After fitting, store each training point's cluster membership probability
         (0 = noise/anomaly, >0 = cluster member/normal)
      2. For a new point, find its k nearest training neighbors
      3. Average their membership probabilities → anomaly_score = 1 − avg_prob
         (score near 0 = surrounded by cluster members = normal)
         (score near 1 = surrounded by noise points = anomaly)
    This gives a smooth, calibrated score without needing the standalone hdbscan package.
    """

    def __init__(self, min_cluster_size: int = 15, min_samples: int = 5, contamination: float = 0.05):
        self.min_cluster_size = min_cluster_size
        self.min_samples      = min_samples
        self.contamination    = contamination
        self.name             = "HDBSCAN"

    def fit(self, X: np.ndarray) -> "HDBSCANDetector":
        from sklearn.cluster import HDBSCAN
        from sklearn.neighbors import NearestNeighbors

        # copy=True suppresses sklearn's FutureWarning about default changing in 1.10
        self._hdb = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            copy=True,
        )
        self._hdb.fit(X)

        # probabilities_: 0 for noise points, >0 for cluster members.
        # This is our proxy for "how normal is this player" — low probability = anomaly.
        self._train_probs = self._hdb.probabilities_.copy()

        # Edge case: if HDBSCAN found NO clusters at all (too few points or too sparse
        # in feature space), all probabilities are 0 and all scores become 1.0.
        # In that case, fall back to a flat score and rely on the strict > threshold
        # so we don't flag everyone. This mainly affects the small 1k-player dataset;
        # on the 28k-player Lichess data, HDBSCAN always finds meaningful clusters.
        self._all_noise = bool(self._train_probs.max() == 0.0)
        if self._all_noise:
            logger.warning(
                "HDBSCAN found no clusters (all points labeled noise). "
                "Consider reducing min_cluster_size. Scores will be uniform."
            )

        # KNN for approximating scores on new (unseen) data.
        # We use min_samples neighbors to stay consistent with HDBSCAN's own core-point definition.
        k = min(self.min_samples, len(X))
        self._knn = NearestNeighbors(n_neighbors=k).fit(X)

        # Set threshold: score must STRICTLY EXCEED this to be flagged.
        # Using strict > means the all-noise degenerate case flags 0% instead of 100%.
        train_scores = 1.0 - self._train_probs
        self._threshold = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous (range roughly 0–1)."""
        _, idx = self._knn.kneighbors(np.asarray(X))
        # Average the membership probability of k nearest training neighbors.
        neighbor_probs = self._train_probs[idx].mean(axis=1)
        return 1.0 - neighbor_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns -1 (anomaly) or 1 (normal). Uses strict > so degenerate all-noise
        case doesn't flag every single player."""
        return np.where(self.score(X) > self._threshold, -1, 1)


class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        hidden = max(input_dim, encoding_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    """Reconstruction-error anomaly score; small feedforward net."""

    def __init__(self, input_dim: Optional[int] = None, feature_names: Optional[list] = None, **kwargs):
        params = {**AUTOENCODER_PARAMS, **kwargs}
        self.encoding_dim = params["encoding_dim"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.lr = params["learning_rate"]
        self.threshold_pct = params["reconstruction_threshold_percentile"]
        self.input_dim = input_dim
        self.feature_names = feature_names            # used to build per-feature weight vector
        self._weight_vector: Optional[torch.Tensor] = None  # built in _build()
        self.model: Optional[_AutoencoderNet] = None
        self.threshold_: Optional[float] = None
        self.name = "Autoencoder"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Autoencoder will use device: %s", self.device)

    def _build(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.model = _AutoencoderNet(input_dim, self.encoding_dim).to(self.device)
        # Build per-feature weight vector if feature names are available.
        # Eval features get AUTOENCODER_EVAL_WEIGHT; all others get 1.0.
        # Normalise so mean weight = 1.0 → loss scale is unchanged vs. plain MSE.
        if self.feature_names is not None:
            weights = [
                AUTOENCODER_EVAL_WEIGHT if f in AUTOENCODER_EVAL_FEATURES else 1.0
                for f in self.feature_names
            ]
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self._weight_vector = w / w.mean()

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        torch.manual_seed(RANDOM_SEED)
        self._build(X.shape[1])
        tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()   # plain MSE for training — unweighted so the AE
        # learns a faithful reconstruction of the full data manifold.  Eval-feature
        # weighting is applied only at SCORING time (see _reconstruction_errors),
        # which amplifies their contribution to the anomaly score without biasing
        # the network weights toward over-fitting those features during training.
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %s/%s — loss: %.5f", epoch + 1, self.epochs, total_loss / len(loader))

        train_scores = self._reconstruction_errors(X)
        self.threshold_ = float(np.percentile(train_scores, self.threshold_pct))
        logger.info("Anomaly threshold (p%s): %.4f", self.threshold_pct, self.threshold_)
        return self

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            recon = self.model(tensor)
            sq_err = (recon - tensor) ** 2
            if self._weight_vector is not None:
                sq_err = sq_err * self._weight_vector
            errors = sq_err.mean(dim=1).cpu().numpy()
        return errors

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._reconstruction_errors(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        assert self.threshold_ is not None
        return np.where(scores > self.threshold_, -1, 1)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return self.model.encode(tensor).cpu().numpy()

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODELS_DIR / "autoencoder.pt"
        assert self.model is not None
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "encoding_dim": self.encoding_dim,
                "threshold": self.threshold_,
                "feature_names": self.feature_names,  # needed to rebuild weight vector on load
            },
            path,
        )
        logger.info("Saved Autoencoder to %s", path)

    def load(self, path: Optional[Path] = None) -> "AutoencoderDetector":
        path = path or MODELS_DIR / "autoencoder.pt"
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        # Restore feature_names BEFORE _build() so the weight vector is reconstructed
        self.feature_names = checkpoint.get("feature_names", None)
        self._build(checkpoint["input_dim"])
        assert self.model is not None
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.threshold_ = checkpoint["threshold"]
        return self


def tune_contamination(
    X_clean: np.ndarray,
    X_injected: np.ndarray,
    y_injected: np.ndarray,
    grid: Optional[list] = None,
) -> pd.DataFrame:
    """Lightweight contamination-only sweep (kept for notebook back-compat).

    Prefer run_hyperparameter_search() for multi-parameter tuning.
    """
    grid = grid or [0.01, 0.03, 0.05, 0.10, 0.15]
    rows = []
    for c in grid:
        for ctor, name in [
            (lambda c=c: IsolationForestDetector(contamination=c), "IsolationForest"),
            (lambda c=c: OneClassSVMDetector(nu=c), "OneClassSVM"),
            (lambda c=c: LOFDetector(contamination=c), "LOF"),
        ]:
            m = ctor()
            m.fit(X_clean)
            scores = m.score(X_injected)
            auc = roc_auc_score(y_injected, scores)
            rows.append({"model": name, "contamination": c, "roc_auc": auc})
            logger.info("Tuning %s @ contamination=%.2f -> ROC-AUC=%.4f", name, c, auc)
    return pd.DataFrame(rows)


def run_hyperparameter_search(
    X_train: np.ndarray,
    X_val_injected: np.ndarray,
    y_val_injected: np.ndarray,
    n_iter: int = RANDOM_SEARCH_N_ITER,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, dict]:
    """Random search over hyperparameters for all anomaly detectors.

    Models are trained on X_train; selection is driven by ROC-AUC on
    X_val_injected (validation data with synthetic anomalies pre-injected).
    The test set is never touched during this phase.

    For IF, OC-SVM, and LOF a full random search is run (n_iter draws each).
    For the Autoencoder, architecture and threshold are searched with short
    (30-epoch) training runs to keep compute cost practical; the winning config
    is later re-trained at full epochs in the main pipeline.

    Args:
        X_train: Scaled training data (no injected anomalies).
        X_val_injected: Validation data with synthetic anomalies already injected.
        y_val_injected: Binary ground-truth labels (1=anomaly, 0=normal).
        n_iter: Random draws per model for IF / OC-SVM / LOF.
        random_state: Seed for reproducible sampling.

    Returns:
        all_results: DataFrame of every (model, params…, roc_auc) trial.
        best_params: {model_name: {param: best_value}} for constructing final models.
    """
    rng = np.random.default_rng(random_state)
    all_rows: list = []
    best_params: dict = {}

    # ── Fast models: full random search ──────────────────────────────────────
    fast_spaces = {
        "IsolationForest": ISOLATION_FOREST_SEARCH,
        "OneClassSVM":     OCSVM_SEARCH,
        "LOF":             LOF_SEARCH,
        "HDBSCAN":         HDBSCAN_SEARCH,  # similar cost to LOF — KNN fit is fast
    }

    def _build(name: str, params: dict):
        if name == "IsolationForest":
            return IsolationForestDetector(**params)
        if name == "OneClassSVM":
            return OneClassSVMDetector(**params)
        if name == "LOF":
            return LOFDetector(**params)
        if name == "HDBSCAN":
            return HDBSCANDetector(**params)
        raise ValueError(f"Unknown model: {name}")

    for model_name, space in fast_spaces.items():
        logger.info("Random search: %s  (%d iterations)...", model_name, n_iter)
        model_rows: list = []

        for i in range(n_iter):
            # Sample one value per parameter uniformly from each candidate list
            sampled: dict = {}
            for param, candidates in space.items():
                val = rng.choice(candidates)
                sampled[param] = val.item() if isinstance(val, np.generic) else val

            try:
                m = _build(model_name, sampled)
                m.fit(X_train)
                auc = roc_auc_score(y_val_injected, m.score(X_val_injected))
                row = {"model": model_name, **sampled, "roc_auc": auc}
                model_rows.append(row)
                all_rows.append(row)
                logger.info(
                    "  [%2d/%d] %s  %s  →  AUC=%.4f",
                    i + 1, n_iter, model_name, sampled, auc,
                )
            except Exception as exc:
                logger.warning("Search trial failed (%s iter %d): %s", model_name, i + 1, exc)

        if model_rows:
            best_row = max(model_rows, key=lambda r: r["roc_auc"])
            best_params[model_name] = {
                k: v for k, v in best_row.items() if k not in ("model", "roc_auc")
            }
            logger.info(
                "Best %s → %s  (AUC=%.4f)",
                model_name, best_params[model_name], best_row["roc_auc"],
            )

    # ── Autoencoder: lightweight architecture + threshold search ─────────────
    # Full random search is impractical (100 epochs × many trials). Instead we
    # run each trial at AUTOENCODER_SEARCH_EPOCHS epochs to cheaply rank configs,
    # then select the best (encoding_dim, threshold_pct) for full re-training.
    logger.info(
        "Autoencoder search: %d configs × %d epochs each...",
        len(AUTOENCODER_SEARCH["encoding_dim"]) * len(AUTOENCODER_SEARCH["reconstruction_threshold_percentile"]),
        AUTOENCODER_SEARCH_EPOCHS,
    )
    ae_rows: list = []
    for enc_dim in AUTOENCODER_SEARCH["encoding_dim"]:
        for thr_pct in AUTOENCODER_SEARCH["reconstruction_threshold_percentile"]:
            try:
                ae = AutoencoderDetector(
                    input_dim=X_train.shape[1],
                    encoding_dim=enc_dim,
                    epochs=AUTOENCODER_SEARCH_EPOCHS,
                    reconstruction_threshold_percentile=thr_pct,
                )
                ae.fit(X_train)
                auc = roc_auc_score(y_val_injected, ae.score(X_val_injected))
                row = {
                    "model": "Autoencoder",
                    "encoding_dim": enc_dim,
                    "reconstruction_threshold_percentile": thr_pct,
                    "roc_auc": auc,
                }
                ae_rows.append(row)
                all_rows.append(row)
                logger.info(
                    "  Autoencoder  encoding_dim=%d  threshold_pct=%d  →  AUC=%.4f",
                    enc_dim, thr_pct, auc,
                )
            except Exception as exc:
                logger.warning("AE search trial failed (enc=%d thr=%d): %s", enc_dim, thr_pct, exc)

    if ae_rows:
        best_ae = max(ae_rows, key=lambda r: r["roc_auc"])
        best_params["Autoencoder"] = {
            "encoding_dim": best_ae["encoding_dim"],
            "reconstruction_threshold_percentile": best_ae["reconstruction_threshold_percentile"],
        }
        logger.info("Best Autoencoder → %s  (AUC=%.4f)", best_params["Autoencoder"], best_ae["roc_auc"])

    all_results = (
        pd.DataFrame(all_rows)
        .sort_values(["model", "roc_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return all_results, best_params


def run_all_models(
    X: np.ndarray,
    meta: pd.DataFrame,
    contamination_overrides: Optional[dict] = None,
    model_params: Optional[dict] = None,
    X_score: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Fit all detectors on X; score on X_score (or X if not provided).

    The fit/score split exists so we can train on the 70% training partition
    but produce scores for ALL players (train + val + test) without any leakage:
    models never see val/test data during training, the scaler is already fitted
    on train-only before this function is called, and evaluation metrics in
    holdout_evaluation.csv are computed separately and are not affected.

    This is equivalent to production deployment: once a model is trained and
    validated, you run it on all available data to get a complete picture.

    Args:
        X:       Scaled TRAINING feature matrix — models are fit on this only.
        meta:    Player metadata aligned with X_score rows (all players if
                 X_score is provided, else training players).
        contamination_overrides: Legacy contamination override (kept for
                 notebook back-compat).
        model_params: Full per-model param dicts from run_hyperparameter_search().
        X_score: If provided, models are fit on X but scored on X_score.
                 Pass the full dataset (train + val + test, all pre-scaled with
                 the train-fitted scaler) to get complete player coverage.
                 If None, models are scored on X (original behaviour).
    """
    overrides = contamination_overrides or {}
    params = model_params or {}

    # What we score on — either the full dataset or just train (default)
    X_eval = X_score if X_score is not None else X

    def _kwargs(name: str, default_contamination: float = 0.05) -> dict:
        """Return constructor kwargs: full search result > contamination override > default."""
        if name in params:
            return params[name]
        c = overrides.get(name, default_contamination)
        return {"contamination": c}

    results = meta.copy()
    model_list = [
        ZScoreBaseline(contamination=overrides.get("ZScoreBaseline", 0.05)),
        LOFDetector(**_kwargs("LOF")),
        IsolationForestDetector(**_kwargs("IsolationForest")),
        OneClassSVMDetector(**({k: v for k, v in _kwargs("OneClassSVM").items()})),
        AutoencoderDetector(
            input_dim=X.shape[1],
            **params.get("Autoencoder", {}),
        ),
        HDBSCANDetector(**params.get("HDBSCAN", HDBSCAN_PARAMS)),
    ]

    for m in model_list:
        logger.info("Fitting %s...", m.name)
        m.fit(X)                              # always fit on training data only
        results[f"{m.name}_score"] = m.score(X_eval)   # score on full data if provided
        results[f"{m.name}_label"] = m.predict(X_eval)

    # ── Ensemble voting — strong models only ─────────────────────────────────
    # We evaluated all 6 models and selected the 3 that contribute genuine,
    # non-redundant signal.  The other 3 are still scored and stored in every
    # results CSV so the analysis is fully transparent — they just don't vote.
    #
    # ── EXCLUDED from ensemble (comparison only) ──────────────────────────────
    #
    # HDBSCAN  ← COMPARISON ONLY — NOT USED FOR ANY OPERATIONAL DECISION
    #   We tried HDBSCAN as a density-based alternative to LOF.  It did not work
    #   on this dataset.  AUC = 0.599 on the subtle benchmark (barely above the
    #   0.5 random baseline) and AUC = 0.767 on the sanity_check benchmark.
    #   LOF already covers the density-based angle far better (AUC 0.973 subtle).
    #   All HDBSCAN scores and labels are still written to results CSVs so we can
    #   show and explain the failure — but its vote is excluded from ensemble_flag
    #   and ensemble_confident.  See Decision 25 in decisions.md.
    #
    # IsolationForest  — AUC 0.746 (subtle), below the ZScore univariate baseline
    #   (0.781).  A tree-based model should outperform a univariate z-score on
    #   multi-feature data; that it doesn't suggests the forest isn't capturing
    #   useful feature interactions here.  It also has 882 "unique" catches that
    #   neither LOF nor AE find — at 0.746 AUC those are very likely false
    #   positives, not real anomalies the better models missed.
    #
    # ZScoreBaseline  — AUC 0.781 (subtle).  Useful for understanding the lower
    #   bound; it correctly flags univariate extremes.  But a pure max-|z-score|
    #   rule completely misses players who are moderately anomalous across many
    #   features simultaneously — exactly the sophisticated cheating pattern.
    #
    # ── ENSEMBLE VOTERS ───────────────────────────────────────────────────────
    #
    #   LOF          — AUC 0.973 subtle / 1.000 sanity_check
    #                  Density-based; best single model on this dataset.
    #                  Captures players who are outliers in local neighborhoods,
    #                  which aligns well with the "unusual for their peer group"
    #                  framing of cheating detection.
    #
    #   Autoencoder  — AUC 0.937 subtle / 1.000 sanity_check
    #                  Reconstruction-based; learns the normal manifold and flags
    #                  anything that doesn't reconstruct well.  Complements LOF:
    #                  they agree on roughly 30% of flags, meaning each finds a
    #                  genuinely different subset of suspicious players.
    #
    #   OneClassSVM  — AUC 0.883 subtle / 1.000 sanity_check
    #                  Margin-based; defines a hypersphere of "normal" behavior.
    #                  Adds a third geometrically distinct perspective and catches
    #                  ~159 players that neither LOF nor AE flag.
    #
    # Flag logic: majority vote (≥2 of 3) → ensemble_flag  (recall-oriented)
    #             unanimous (= 3 of 3)     → ensemble_confident  (precision-oriented)
    ENSEMBLE_VOTERS = ["LOF", "Autoencoder", "OneClassSVM"]
    voter_label_cols = [f"{n}_label" for n in ENSEMBLE_VOTERS if f"{n}_label" in results.columns]
    results["anomaly_votes"] = (results[voter_label_cols] == -1).sum(axis=1)

    # ≥2/3 majority — high recall triage list (players worth a second look)
    results["ensemble_flag"]      = results["anomaly_votes"] >= 2
    # All 3 agree — high precision shortlist (players most likely to be anomalous)
    results["ensemble_confident"] = results["anomaly_votes"] == 3

    # ── Minimum game threshold — suppress flags for low-data players ──────────
    # Players with fewer than MIN_GAMES_FOR_FLAG games have high-variance aggregated
    # stats and produce false positives from statistical noise, not genuine anomaly.
    # Empirically: median n_games for flagged players was 10 vs 29 for normal players;
    # 416/915 flags had <10 games with the threshold disabled.
    # We keep their scores/labels in the CSV for transparency but suppress the flags
    # so they never appear in operational outputs.  They still contribute to model
    # training as normal examples — having 8 games looks normal, which is correct.
    if "n_games" in results.columns:
        low_data = results["n_games"] < MIN_GAMES_FOR_FLAG
        n_suppressed = int((low_data & results["ensemble_flag"]).sum())
        results.loc[low_data, "ensemble_flag"]      = False
        results.loc[low_data, "ensemble_confident"] = False
        if n_suppressed:
            logger.info(
                "Suppressed %d flags for players with < %d games "
                "(insufficient data for reliable scoring).",
                n_suppressed, MIN_GAMES_FOR_FLAG,
            )

    return results
