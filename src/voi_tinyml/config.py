"""Experiment configuration.

Every constant below is taken from the paper unless marked ``# assumption``.
Section/equation numbers refer to Nnaji, Isogun, Wanisi & Dominic (2026),
ISA Journal of Engineering and Technology 3(5):50-68.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEED = 42  # Sec. 4.1, study-wide seed


@dataclass(frozen=True)
class DataConfig:
    window: int = 24              # W, Sec. 4.3
    baseline_weeks: int = 52      # rolling mean/std, Eq. 15-16
    train_frac: float = 0.70      # Table 1
    val_frac: float = 0.15
    quantiles: tuple = (0.50, 0.80)  # Eq. 18-19
    # First usable week index. 52 reproduces the paper's 2,231 sequences,
    # 1,561/335/335 split, thresholds (0.3716, 1.3221) and test class counts exactly.
    first_index: int = 52


@dataclass(frozen=True)
class VoIConfig:
    alpha_s: float = 0.40         # Eq. 12
    alpha_u: float = 0.20
    alpha_d: float = 0.20
    alpha_a: float = 0.20
    severity_weights: tuple = (0.0, 0.5, 1.0)  # Eq. 7
    novelty_lag: int = 1          # r in Eq. 8; r=1 reproduces tau_D = 0.90
    novelty_percentile: float = 95.0  # Sec. 4.8
    t_max: int = 24               # information-age horizon, Sec. 4.8
    entropy_criterion: float = 0.72   # U_t >= 0.72 -> large packet
    large_payload: int = 32       # bytes
    small_payload: int = 12
    # Validation grid search constraints (Sec. 4.8)
    min_cedr: float = 0.90
    min_pdr: float = 0.82
    grid: tuple = tuple(round(0.30 + 0.025 * i, 3) for i in range(25))  # assumption: 0.300..0.900
    calibration_realizations: int = 50  # assumption: link realizations used on validation


@dataclass(frozen=True)
class RFConfig:
    spreading_factors: tuple = (7, 8, 9, 10, 11, 12)          # Sec. 4.10
    tx_powers_dbm: tuple = (14, 17, 20, 22)
    tx_currents_ma: dict = field(default_factory=lambda: {14: 45.0, 17: 58.0, 20: 84.0, 22: 118.0})
    snr_req_db: dict = field(default_factory=lambda: {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0})
    supply_voltage: float = 3.3
    bandwidth_hz: float = 125e3
    coding_rate: int = 1          # 4/5
    preamble_symbols: int = 8
    crc: bool = True
    explicit_header: bool = True
    ldro_sfs: tuple = (11, 12)
    logistic_slope: float = 0.72
    power_gain_per_db: float = 0.11
    target_high: float = 0.94     # upper-threshold or class-2 transmissions
    target_low: float = 0.82
    fallback_energy_penalty: float = 0.06
    # Fixed radio used by B0, B1, B2 and the "without link adaptation" ablation.
    # SF9 @ 17 dBm reproduces the paper's B0/B1/B2 energies exactly.
    fixed_sf: int = 9
    fixed_power_dbm: int = 17
    # --- Link-trace model (assumption: not specified in the paper) ---
    # SNR_t = mean + AR(1) shadowing; mean calibrated so fixed-RF (SF9, 17 dBm) PDR ~= 70.8 %.
    snr_mean_db: float = -10.95
    snr_std_db: float = 2.5
    snr_ar_coef: float = 0.7
    snr_estimate_noise_db: float = 1.0  # controller sees a noisy SNR estimate


@dataclass(frozen=True)
class ExperimentConfig:
    n_realizations: int = 300     # Sec. 4.12
    mc_seed_base: int = 1000      # assumption
    ablation_link_seed: int = 202  # Sec. 4.13
    n_bootstrap: int = 2000       # Sec. 4.7


DATA = DataConfig()
VOI = VoIConfig()
RF = RFConfig()
EXP = ExperimentConfig()
