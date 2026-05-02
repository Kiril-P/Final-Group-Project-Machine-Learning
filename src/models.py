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
    AUTOENCODER_PARAMS,
    AUTOENCODER_SEARCH,
    AUTOENCODER_SEARCH_EPOCHS,
    ISOLATION_FOREST_PARAMS,
    ISOLATION_FOREST_SEARCH,
    LOF_SEARCH,
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

    def __init__(self, input_dim: Optional[int] = None, **kwargs):
        params = {**AUTOENCODER_PARAMS, **kwargs}
        self.encoding_dim = params["encoding_dim"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.lr = params["learning_rate"]
        self.threshold_pct = params["reconstruction_threshold_percentile"]
        self.input_dim = input_dim
        self.model: Optional[_AutoencoderNet] = None
        self.threshold_: Optional[float] = None
        self.name = "Autoencoder"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Autoencoder will use device: %s", self.device)

    def _build(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.model = _AutoencoderNet(input_dim, self.encoding_dim).to(self.device)

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        torch.manual_seed(RANDOM_SEED)
        self._build(X.shape[1])
        tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
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
            errors = ((recon - tensor) ** 2).mean(dim=1).cpu().numpy()
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
    }

    def _build(name: str, params: dict):
        if name == "IsolationForest":
            return IsolationForestDetector(**params)
        if name == "OneClassSVM":
            return OneClassSVMDetector(**params)
        if name == "LOF":
            return LOFDetector(**params)
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
) -> pd.DataFrame:
    """Fit all five detectors; return meta + scores, predictions, and ensemble vote.

    Args:
        X: Scaled training feature matrix.
        meta: Player metadata aligned with X rows.
        contamination_overrides: Legacy single-value override — sets contamination
            only (kept for notebook back-compat). Ignored per-model when model_params
            provides a full param dict for that model.
        model_params: Full per-model param dicts from run_hyperparameter_search(),
            e.g. {"IsolationForest": {"contamination": 0.03, "n_estimators": 200}}.
            Takes precedence over contamination_overrides for any model it covers.
    """
    overrides = contamination_overrides or {}
    params = model_params or {}

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
    ]

    for m in model_list:
        logger.info("Fitting %s...", m.name)
        m.fit(X)
        results[f"{m.name}_score"] = m.score(X)
        results[f"{m.name}_label"] = m.predict(X)

    # Ensemble vote: flag players where ≥ 2 of the three advanced models agree
    advanced = ["IsolationForest", "OneClassSVM", "Autoencoder"]
    label_cols = [f"{n}_label" for n in advanced]
    results["anomaly_votes"] = (results[label_cols] == -1).sum(axis=1)
    results["ensemble_anomaly"] = results["anomaly_votes"] >= 2
    return results
