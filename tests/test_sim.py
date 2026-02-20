"""Unit tests for build_sim() and extract_metrics()."""

import numpy as np
import pytest

from phasesweep.configs import CONFIGS, T_STOP, W_REF
from phasesweep.sim import build_sim, extract_metrics


CFG_NAME = "A: 4-pole SPMSM"
CFG = CONFIGS[CFG_NAME]


def test_build_sim_returns_simulation():
    sim = build_sim(CFG_NAME, CFG)
    from motulator.drive.model import Simulation
    assert isinstance(sim, Simulation)


@pytest.fixture(scope="module")
def short_result():
    sim = build_sim(CFG_NAME, CFG)
    return sim.simulate(t_stop=0.5)


@pytest.fixture(scope="module")
def full_result():
    sim = build_sim(CFG_NAME, CFG)
    return sim.simulate(t_stop=T_STOP)


@pytest.fixture(scope="module")
def full_result_heavy_load():
    sim = build_sim(CFG_NAME, CFG, load_torque=6.0)
    return sim.simulate(t_stop=T_STOP)


def test_extract_metrics_keys(short_result):
    m = extract_metrics(short_result, CFG["n_p"])
    assert set(m.keys()) == {"t_settle", "i_ss", "speed_droop", "tau_peak"}


def test_extract_metrics_i_ss_positive(full_result):
    m = extract_metrics(full_result, CFG["n_p"])
    assert m["i_ss"] > 0


def test_extract_metrics_tau_peak_positive(short_result):
    m = extract_metrics(short_result, CFG["n_p"])
    assert m["tau_peak"] > 0


def test_extract_metrics_settle_time_finite(full_result):
    m = extract_metrics(full_result, CFG["n_p"])
    assert np.isfinite(m["t_settle"])
    assert m["t_settle"] > 0


@pytest.mark.parametrize("cfg_name", list(CONFIGS.keys()))
def test_build_sim_all_configs(cfg_name):
    from motulator.drive.model import Simulation
    sim = build_sim(cfg_name, CONFIGS[cfg_name])
    assert isinstance(sim, Simulation)


def test_extract_metrics_all_positive(full_result):
    m = extract_metrics(full_result, CFG["n_p"])
    for key in ("t_settle", "i_ss", "speed_droop", "tau_peak"):
        assert np.isfinite(m[key]), f"{key} is not finite"
        assert m[key] > 0, f"{key} is not positive"


def test_speed_droop_increases_with_load(full_result, full_result_heavy_load):
    m_default = extract_metrics(full_result, CFG["n_p"])
    m_heavy = extract_metrics(full_result_heavy_load, CFG["n_p"])
    assert m_heavy["speed_droop"] > m_default["speed_droop"]


def test_i_ss_increases_with_load(full_result, full_result_heavy_load):
    m_default = extract_metrics(full_result, CFG["n_p"])
    m_heavy = extract_metrics(full_result_heavy_load, CFG["n_p"])
    assert m_heavy["i_ss"] > m_default["i_ss"]
