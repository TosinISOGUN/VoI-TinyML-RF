"""Checks that pin the results to values stated in the paper."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voi_tinyml.config import VOI  # noqa: E402
from voi_tinyml.data import build_splits  # noqa: E402
from voi_tinyml.lora import select_config, success_probability, tx_energy_mj  # noqa: E402
from voi_tinyml.voi import (fit_novelty_scale, normalized_entropy, severity,  # noqa: E402
                            voi_policy)


@pytest.fixture(scope="module")
def splits():
    return build_splits()


def test_dataset_matches_paper(splits):
    assert len(splits.series) == 2284
    assert (len(splits.train.y), len(splits.val.y), len(splits.test.y)) == (1561, 335, 335)
    assert splits.train.X_tab.shape[1] == 51
    assert splits.thresholds == pytest.approx((0.3716, 1.3221), abs=1e-4)
    assert np.bincount(splits.test.y).tolist() == [164, 103, 68]


def test_novelty_scale_matches_paper(splits):
    assert fit_novelty_scale(splits.series.to_numpy(), splits.train.t_index) == pytest.approx(0.90, abs=1e-6)


def test_lora_energy_reproduces_fixed_baselines():
    # 335 packets at SF9 / 17 dBm: B0 (32 B) and B1 (12 B) energies in Table 4
    assert 335 * tx_energy_mj(32, 9, 17) == pytest.approx(15823.54, abs=0.01)
    assert 335 * tx_energy_mj(12, 9, 17) == pytest.approx(9257.76, abs=0.01)


def test_link_model_monotone():
    snr = -11.0
    assert success_probability(snr, 12, 14) > success_probability(snr, 7, 14)
    assert success_probability(snr, 9, 22) > success_probability(snr, 9, 14)


def test_select_config_meets_target_cheaply():
    sf, p = select_config(-11.0, 12, 0.82)
    assert success_probability(-11.0, sf, p) >= 0.82
    assert tx_energy_mj(12, sf, p) <= tx_energy_mj(12, 12, 22)


def test_voi_components_bounds():
    P = np.array([[1.0, 0, 0], [1 / 3, 1 / 3, 1 / 3], [0, 0, 1.0]])
    U = normalized_entropy(P)
    assert U[0] == pytest.approx(0, abs=1e-6) and U[1] == pytest.approx(1, abs=1e-6)
    assert severity(P).tolist() == [0.0, 0.5, 1.0]


def test_information_age_forces_transmission():
    # confident, routine, unchanging stream: only information age can trigger updates
    n = 100
    P = np.tile([1.0, 0.0, 0.0], (n, 1))
    d = voi_policy(P, np.zeros(n), T1=0.2, T2=0.9)
    gaps = np.diff(np.flatnonzero(d.transmit))
    assert d.transmit.any() and (gaps <= VOI.t_max).all()
