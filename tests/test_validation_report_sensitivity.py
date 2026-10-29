"""Sensitivity-curve consumer in scripts/validation_report.py:
percent change of a model's primary metric across perturbation deltas,
relative to the unperturbed baseline. Exercised on the geometry-less
induction tier (fast, no FEM) — the PM/geometry tracks are the same
code path with slower models.
"""
from __future__ import annotations

from itertools import pairwise

import pytest

from tests.conftest import VIZ_SKIP

pytest.importorskip("matplotlib", reason=VIZ_SKIP)

from phasesweep.machines.configs import load_motor
from scripts import validation_report as vr
from tests.conftest import REPO_ROOT

IM_NAME = "CREATOR Case IM"
IM_PARAMS = ("R_2", "L_m", "L_sigma_s", "L_sigma_r")
GEOMETRY_PARAMS = ("OD", "gap", "L_stk", "B_rem", "k_w")
TRACK = "τ_rated_im"


@pytest.fixture(scope="module")
def im_motor():
    toml_rel, _ = vr.MOTOR_DATA_MAP[IM_NAME]
    return load_motor(REPO_ROOT / toml_rel)


@pytest.fixture(scope="module")
def grid(im_motor):
    baseline = vr.run_model(im_motor, "im_torque_slip")
    assert baseline is not None and baseline.status == "OK"
    sens = vr._run_sensitivity(
        {IM_NAME: im_motor}, {IM_NAME: [baseline]},
        workers=1, cache_dir=None, timing_log=[],
    )
    return sens[IM_NAME]


def test_im_track_and_params_registered():
    assert vr._SENSITIVITY_METRIC["im_torque_slip"] == "tau_rated"
    assert vr._SENSITIVITY_LABEL["im_torque_slip"] == TRACK
    for p in IM_PARAMS:
        assert p in vr.SENSITIVITY_PARAMS
        assert 0.0 in vr.SENSITIVITY_PARAMS[p]["deltas"]


def test_metric_missing_rated_point_is_none():
    assert vr._sensitivity_metric("im_torque_slip", {"tau_rated": None}) is None


def test_baseline_slot_is_zero(grid):
    for p in IM_PARAMS:
        deltas = vr.SENSITIVITY_PARAMS[p]["deltas"]
        assert grid[p][TRACK][deltas.index(0.0)] == 0.0


def test_geometry_params_skipped_for_circuit_only_motor(grid):
    for p in GEOMETRY_PARAMS:
        curve = grid[p][TRACK]
        deltas = vr.SENSITIVITY_PARAMS[p]["deltas"]
        assert all(v is None for v, d in zip(curve, deltas) if d != 0.0)


def test_rotor_resistance_curve_is_percent_change_vs_baseline(grid, im_motor):
    """Low-slip torque falls as R_2 rises (tau ~ 1/R_2 at rated slip):
    the curve is monotonically decreasing and each point matches an
    independent perturbed run to the percent."""
    from phasesweep.machines.perturbation import perturb_motor
    deltas = vr.SENSITIVITY_PARAMS["R_2"]["deltas"]
    curve = grid["R_2"][TRACK]
    assert all(v is not None for v in curve)
    assert all(a > b for a, b in pairwise(curve))
    base = vr.run_model(im_motor, "im_torque_slip").metrics["tau_rated"]
    for d, pct in zip(deltas, curve):
        if d == 0.0:
            continue
        m = perturb_motor(im_motor, "R_2", d)
        tau = vr.run_model(m, "im_torque_slip").metrics["tau_rated"]
        assert pct == pytest.approx((tau - base) / base * 100, rel=1e-9)


def test_leakage_and_magnetizing_curves_populated(grid):
    for p in ("L_m", "L_sigma_s", "L_sigma_r"):
        curve = grid[p][TRACK]
        assert all(v is not None for v in curve)
        assert max(abs(v) for v in curve) > 0.01
