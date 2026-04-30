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
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    AUTOENCODER_PARAMS,
    ISOLATION_FOREST_PARAMS,
    MODELS_DIR,
    OCSVM_PARAMS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


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


def run_all_models(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Fit IF, OCSVM, Autoencoder; return meta + scores and ensemble vote."""
    results = meta.copy()
    models = [
        IsolationForestDetector(),
        OneClassSVMDetector(),
        AutoencoderDetector(input_dim=X.shape[1]),
    ]
    for m in models:
        logger.info("Fitting %s...", m.name)
        m.fit(X)
        results[f"{m.name}_score"] = m.score(X)
        results[f"{m.name}_label"] = m.predict(X)

    label_cols = [f"{m.name}_label" for m in models]
    results["anomaly_votes"] = (results[label_cols] == -1).sum(axis=1)
    results["ensemble_anomaly"] = results["anomaly_votes"] >= 2
    return results
