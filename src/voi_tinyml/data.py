"""Dataset loading, temporal feature construction and future-state labels (Sec. 4.2-4.5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATA, DataConfig


def load_co2_series() -> pd.Series:
    """Weekly Mauna Loa atmospheric CO2 (1958-03-29 .. 2001-12-29, 2,284 weeks).

    The record ships with statsmodels, so no network access is needed.
    Missing weeks are filled by linear interpolation (Sec. 4.2).
    """
    import statsmodels.api as sm

    s = sm.datasets.co2.load_pandas().data["co2"]
    s = s.interpolate(method="linear").bfill().ffill()
    s.name = "co2_ppm"
    return s


@dataclass
class Dataset:
    """Feature matrices and labels for one chronological split."""

    X_tab: np.ndarray        # (n, 51) tabular features for DT / RF / MLP
    X_seq: np.ndarray        # (n, 2, W) two temporal channels for the CNN
    y: np.ndarray            # (n,) future-state class 0/1/2
    z_next: np.ndarray       # (n,) standardized future anomaly
    t_index: np.ndarray      # (n,) position in the weekly series
    x_now: np.ndarray        # (n,) current CO2 value (ppm)
    dates: pd.DatetimeIndex


@dataclass
class Splits:
    train: Dataset
    val: Dataset
    test: Dataset
    thresholds: tuple
    series: pd.Series


def _rolling_stats(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(x)
    return s.rolling(k).mean().to_numpy(), s.rolling(k).std(ddof=1).to_numpy()


def build_splits(cfg: DataConfig = DATA) -> Splits:
    series = load_co2_series()
    x = series.to_numpy(dtype=float)
    n = len(x)
    W = cfg.window
    mu, sd = _rolling_stats(x, cfg.baseline_weeks)
    diff = np.diff(x, prepend=np.nan)
    week = series.index.isocalendar().week.to_numpy(dtype=float)

    ts = np.arange(cfg.first_index, n - 1)  # need x[t+1] for the label
    hist = np.stack([x[t - W + 1 : t + 1] for t in ts])
    dhist = np.stack([diff[t - W + 1 : t + 1] for t in ts])
    # current standardized anomaly relative to the current 52-week baseline
    anomaly = (x[ts] - mu[ts]) / sd[ts]
    sin_w = np.sin(2 * np.pi * week[ts] / 52.0)
    cos_w = np.cos(2 * np.pi * week[ts] / 52.0)
    X_tab = np.column_stack([hist, dhist, anomaly, sin_w, cos_w])  # 24+24+3 = 51

    # CNN channels: window centred on its own mean + first differences
    X_seq = np.stack([hist - hist.mean(axis=1, keepdims=True), dhist], axis=1)

    # Eq. 17: future anomaly against the baseline available at time t (no leakage)
    z_next = (x[ts + 1] - mu[ts]) / sd[ts]
    assert np.isfinite(X_tab).all() and np.isfinite(z_next).all()

    m = len(ts)
    n_tr = int(cfg.train_frac * m)
    n_va = int(round(cfg.val_frac * m))
    idx = {
        "train": slice(0, n_tr),
        "val": slice(n_tr, n_tr + n_va),
        "test": slice(n_tr + n_va, m),
    }
    # Eq. 18: thresholds from the training targets only
    q = tuple(float(v) for v in np.quantile(z_next[idx["train"]], cfg.quantiles))
    y = np.digitize(z_next, q)  # Eq. 19

    def make(sl: slice) -> Dataset:
        return Dataset(X_tab[sl], X_seq[sl], y[sl], z_next[sl], ts[sl], x[ts][sl], series.index[ts][sl])

    return Splits(make(idx["train"]), make(idx["val"]), make(idx["test"]), q, series)
