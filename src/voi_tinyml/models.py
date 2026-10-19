"""The four candidate TinyML classifiers (Sec. 4.6)."""
from __future__ import annotations

import io
import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn

from .config import SEED

N_CLASSES = 3


def decision_tree(seed: int = SEED):
    return DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, class_weight="balanced", random_state=seed)


def random_forest(seed: int = SEED):
    return RandomForestClassifier(
        n_estimators=32, max_depth=9, min_samples_leaf=4,
        class_weight="balanced_subsample", random_state=seed, n_jobs=1,
    )


def mlp(seed: int = SEED):
    clf = MLPClassifier(
        hidden_layer_sizes=(24, 12), activation="relu", alpha=3e-4,
        early_stopping=True, max_iter=300, random_state=seed,
    )
    # normalization statistics are estimated on the training partition only
    return make_pipeline(StandardScaler(), clf)


def sklearn_size_kb(model) -> float:
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0


class TinyCNN(nn.Module):
    """Two temporal input channels, 2 conv layers x 8 filters (k=3), adaptive avg pool, 3 outputs."""

    def __init__(self, in_channels: int = 2, filters: int = 8, kernel: int = 3, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel, padding=1), nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(filters, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TinyCNNClassifier:
    """sklearn-style wrapper: Adam (lr 2e-3), <=100 epochs, early stopping on validation loss."""

    def __init__(self, seed: int = SEED, lr: float = 2e-3, max_epochs: int = 100,
                 patience: int = 10, batch_size: int = 64):  # patience/batch: assumptions
        self.seed, self.lr, self.max_epochs = seed, lr, max_epochs
        self.patience, self.batch_size = patience, batch_size

    def _norm(self, X):
        return torch.tensor((X - self.mean_) / self.std_, dtype=torch.float32)

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        Xt, yt = self._norm(X), torch.tensor(y, dtype=torch.long)
        Xv, yv = self._norm(X_val), torch.tensor(y_val, dtype=torch.long)
        counts = np.bincount(y, minlength=N_CLASSES)
        weight = torch.tensor(len(y) / (N_CLASSES * np.maximum(counts, 1)), dtype=torch.float32)
        self.model_ = TinyCNN()
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=weight)  # class balancing: assumption
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.model_(Xt[b]), yt[b]).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                v = loss_fn(self.model_(Xv), yv).item()
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(self._norm(X)), dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def size_kb(self) -> float:
        buf = io.BytesIO()
        torch.save(self.model_.state_dict(), buf)
        return len(buf.getvalue()) / 1024.0
