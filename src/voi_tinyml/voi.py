"""Environmental Value-of-Information score and transmit/payload policy (Sec. 3.2-3.7, 4.8)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import VOI, VoIConfig

EPS = 1e-12


def normalized_entropy(P: np.ndarray) -> np.ndarray:
    """Eq. 5: U_t = -(1/ln K) sum_k p ln(p + eps), in [0, 1]."""
    K = P.shape[1]
    return np.clip(-(P * np.log(P + EPS)).sum(axis=1) / np.log(K), 0.0, 1.0)


def severity(P: np.ndarray, weights=VOI.severity_weights) -> np.ndarray:
    """Eq. 6-7: S_t = sum_k w_k p_k with w = [0, 0.5, 1]."""
    return P @ np.asarray(weights, dtype=float)


def raw_novelty(series: np.ndarray, t_index: np.ndarray, lag: int = VOI.novelty_lag) -> np.ndarray:
    """|x_t - mean(x_{t-r..t-1})| in the sensor's own units (ppm)."""
    return np.array([abs(series[t] - series[t - lag : t].mean()) for t in t_index])


def fit_novelty_scale(series: np.ndarray, train_t_index: np.ndarray, cfg: VoIConfig = VOI) -> float:
    """tau_D: 95th percentile of the training novelty distribution (Sec. 4.8)."""
    return float(np.percentile(raw_novelty(series, train_t_index, cfg.novelty_lag), cfg.novelty_percentile))


def novelty(raw: np.ndarray, tau_d: float) -> np.ndarray:
    """Eq. 8: D_t = min(1, |x_t - xbar| / tau_D)."""
    return np.minimum(1.0, raw / tau_d)


@dataclass
class Components:
    """Ablation switches: a disabled component is removed and the remaining weights renormalized."""

    severity: bool = True
    uncertainty: bool = True
    novelty: bool = True
    age: bool = True


@dataclass
class Decisions:
    transmit: np.ndarray     # bool (n,)
    payload: np.ndarray      # int bytes (0 when suppressed)
    high_target: np.ndarray  # bool: request 0.94 delivery target
    voi: np.ndarray          # V_t actually evaluated (depends on running age)


def voi_policy(P: np.ndarray, D: np.ndarray, T1: float, T2: float,
               comp: Components = Components(), cfg: VoIConfig = VOI) -> Decisions:
    """Run the B3 controller sequentially over a test/validation stream."""
    n = len(P)
    S, U = severity(P, cfg.severity_weights), normalized_entropy(P)
    y_hat = P.argmax(axis=1)
    w = np.array([cfg.alpha_s * comp.severity, cfg.alpha_u * comp.uncertainty,
                  cfg.alpha_d * comp.novelty, cfg.alpha_a * comp.age], dtype=float)
    w = w / w.sum()  # Eq. 11: weights sum to one

    transmit = np.zeros(n, bool)
    payload = np.zeros(n, int)
    high = np.zeros(n, bool)
    V = np.zeros(n)
    t_last = -1  # assumption: the stream starts right after a transmitted update
    for t in range(n):
        A = min(1.0, (t - t_last) / cfg.t_max)                      # Eq. 9
        V[t] = w[0] * S[t] + w[1] * U[t] + w[2] * D[t] + w[3] * A    # Eq. 10
        if V[t] < T1:
            continue
        transmit[t] = True
        t_last = t
        big = V[t] >= T2 or y_hat[t] == 2 or (comp.uncertainty and U[t] >= cfg.entropy_criterion)
        payload[t] = cfg.large_payload if big else cfg.small_payload
        high[t] = V[t] >= T2 or y_hat[t] == 2
    return Decisions(transmit, payload, high, V)
