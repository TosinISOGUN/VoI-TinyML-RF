"""Modeled LoRa radio: time-on-air, transmit energy, link success and adaptive configuration (Sec. 4.10).

These are simulation models, not radio measurements.
"""
from __future__ import annotations

import math

import numpy as np

from .config import RF, RFConfig


_TOA_CACHE: dict = {}


def time_on_air_s(payload_bytes: int, sf: int, cfg: RFConfig = RF) -> float:
    """Semtech SX127x time-on-air for one LoRa packet."""
    key = (payload_bytes, sf, id(cfg))
    if key not in _TOA_CACHE:
        _TOA_CACHE[key] = _toa(payload_bytes, sf, cfg)
    return _TOA_CACHE[key]


def _toa(payload_bytes: int, sf: int, cfg: RFConfig) -> float:
    t_sym = (2 ** sf) / cfg.bandwidth_hz
    de = 1 if sf in cfg.ldro_sfs else 0
    h = 0 if cfg.explicit_header else 1
    crc = 1 if cfg.crc else 0
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * h
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cfg.coding_rate + 4), 0)
    t_preamble = (cfg.preamble_symbols + 4.25) * t_sym
    return t_preamble + n_payload * t_sym


def tx_energy_mj(payload_bytes: int, sf: int, power_dbm: int, cfg: RFConfig = RF) -> float:
    """Eq. 21: E = V_s * I_TX * T_air  (V * mA * s = mJ)."""
    return cfg.supply_voltage * cfg.tx_currents_ma[power_dbm] * time_on_air_s(payload_bytes, sf, cfg)


def success_probability(snr_db, sf: int, power_dbm: int, cfg: RFConfig = RF):
    """Logistic link model on the margin SNR - SNR_req(SF) + 0.11 (P_TX - 14)."""
    margin = np.asarray(snr_db) - cfg.snr_req_db[sf] + cfg.power_gain_per_db * (power_dbm - 14)
    return 1.0 / (1.0 + np.exp(-cfg.logistic_slope * margin))


def select_config(snr_estimate_db: float, payload_bytes: int, target: float, cfg: RFConfig = RF):
    """Cheapest (SF, P_TX) meeting the target delivery probability.

    If no configuration meets the target, maximize p_success - 0.06 ln(1 + E_TX).
    """
    best, best_e = None, np.inf
    fallback, fallback_score = None, -np.inf
    for sf in cfg.spreading_factors:
        for p in cfg.tx_powers_dbm:
            prob = float(success_probability(snr_estimate_db, sf, p, cfg))
            e = tx_energy_mj(payload_bytes, sf, p, cfg)
            if prob >= target and e < best_e:
                best, best_e = (sf, p), e
            score = prob - cfg.fallback_energy_penalty * math.log1p(e)
            if score > fallback_score:
                fallback, fallback_score = (sf, p), score
    return best if best is not None else fallback


class LinkRealization:
    """One modeled RF-link realization over a sequence of transmission opportunities.

    Assumption (not specified in the paper): SNR follows a stationary AR(1)
    shadowing process around a fixed mean; the controller sees the SNR with
    Gaussian estimation error; packet reception uses a per-slot uniform draw
    shared by all systems (common random numbers), so differences between
    systems come from their decisions, not from different luck.
    """

    def __init__(self, n_slots: int, seed: int, cfg: RFConfig = RF):
        rng = np.random.default_rng(seed)
        phi, sd = cfg.snr_ar_coef, cfg.snr_std_db
        eps = rng.normal(0.0, sd * math.sqrt(1 - phi ** 2), n_slots)
        shadow = np.empty(n_slots)
        shadow[0] = rng.normal(0.0, sd)
        for i in range(1, n_slots):
            shadow[i] = phi * shadow[i - 1] + eps[i]
        self.snr_db = cfg.snr_mean_db + shadow
        self.snr_estimate_db = self.snr_db + rng.normal(0.0, cfg.snr_estimate_noise_db, n_slots)
        self.uniform = rng.random(n_slots)

    def delivered(self, slot: int, sf: int, power_dbm: int, cfg: RFConfig = RF) -> bool:
        return bool(self.uniform[slot] < success_probability(self.snr_db[slot], sf, power_dbm, cfg))
