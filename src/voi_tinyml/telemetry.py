"""Communication systems B0-B3 and their evaluation on a modeled link (Sec. 4.9-4.11).

For speed, every realization precomputes, per slot, the delivery outcome and energy
for each (payload, delivery-target) combination the adaptive controller can request;
any policy is then evaluated by indexing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RF, VOI, RFConfig
from .lora import LinkRealization, success_probability, tx_energy_mj
from .voi import Components, Decisions, voi_policy

PAYLOADS = (VOI.small_payload, VOI.large_payload)


@dataclass
class Plan:
    """What a system decides to send, independent of the link."""

    name: str
    transmit: np.ndarray
    payload: np.ndarray
    high_target: np.ndarray
    adaptive: bool


def plan_b0(n: int) -> Plan:
    t = np.ones(n, bool)
    return Plan("B0 Periodic fixed RF", t, np.full(n, VOI.large_payload), np.zeros(n, bool), False)


def plan_b1(n: int) -> Plan:
    t = np.ones(n, bool)
    return Plan("B1 TinyML periodic", t, np.full(n, VOI.small_payload), np.zeros(n, bool), False)


def plan_b2(y_hat: np.ndarray) -> Plan:
    """Event-triggered TinyML: send on an elevated predicted state or a state change (assumption)."""
    changed = np.r_[True, y_hat[1:] != y_hat[:-1]]
    t = (y_hat >= 1) | changed
    payload = np.where(t, np.where(y_hat == 2, VOI.large_payload, VOI.small_payload), 0)
    return Plan("B2 Event TinyML fixed RF", t, payload, np.zeros(len(y_hat), bool), False)


def plan_b3(P, D, T1, T2, comp: Components = Components(), adaptive: bool = True,
            name: str = "B3 Proposed VoI + adaptive RF") -> Plan:
    d: Decisions = voi_policy(P, D, T1, T2, comp)
    return Plan(name, d.transmit, d.payload, d.high_target, adaptive)


class PrecomputedLink:
    """Per-slot outcomes of fixed and adaptive radio choices for one realization."""

    def __init__(self, n: int, seed: int, cfg: RFConfig = RF):
        link = LinkRealization(n, seed, cfg)
        configs = [(sf, p) for sf in cfg.spreading_factors for p in cfg.tx_powers_dbm]
        # true and estimated success probability for every config: (n_cfg, n)
        p_true = np.stack([success_probability(link.snr_db, sf, p, cfg) for sf, p in configs])
        p_est = np.stack([success_probability(link.snr_estimate_db, sf, p, cfg) for sf, p in configs])
        self.fixed = {}
        self.adaptive = {}
        k_fixed = configs.index((cfg.fixed_sf, cfg.fixed_power_dbm))
        for L in PAYLOADS:
            E = np.array([tx_energy_mj(L, sf, p, cfg) for sf, p in configs])
            self.fixed[L] = (link.uniform < p_true[k_fixed], np.full(n, E[k_fixed]))
            score = p_est - cfg.fallback_energy_penalty * np.log1p(E)[:, None]
            fallback = score.argmax(axis=0)
            for high in (False, True):
                target = cfg.target_high if high else cfg.target_low
                ok = p_est >= target
                cost = np.where(ok, E[:, None], np.inf)
                choice = np.where(ok.any(axis=0), cost.argmin(axis=0), fallback)
                cols = np.arange(n)
                self.adaptive[(L, high)] = (link.uniform < p_true[choice, cols], E[choice], choice)
        self.configs = configs


def evaluate(plan: Plan, link: PrecomputedLink, critical: np.ndarray) -> dict:
    n = len(plan.transmit)
    delivered = np.zeros(n, bool)
    energy = np.zeros(n)
    for L in PAYLOADS:
        for high in (False, True):
            m = plan.transmit & (plan.payload == L) & (plan.high_target == high)
            if not m.any():
                continue
            ok, E = (link.adaptive[(L, high)][:2] if plan.adaptive else link.fixed[L])
            delivered[m], energy[m] = ok[m], E[m]
    n_tx = int(plan.transmit.sum())
    n_crit = int(critical.sum())
    crit_del = int((delivered & critical).sum())
    return {
        "system": plan.name,
        "packets": n_tx,
        "bytes": int(plan.payload[plan.transmit].sum()),
        "energy_mj": float(energy.sum()),
        "delivered": int(delivered.sum()),
        "pdr": delivered.sum() / n_tx if n_tx else float("nan"),       # Eq. 22
        "critical_delivered": crit_del,
        "critical_total": n_crit,
        "cedr": crit_del / n_crit if n_crit else float("nan"),         # Eq. 25
    }
