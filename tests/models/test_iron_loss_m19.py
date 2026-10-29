"""M-19 29 Ga iron-loss coefficients for the BMW i3.

Material tier only: the catalog core-loss table transcribed to
data/bmw_i3/m19_29ga_core_loss.csv must reproduce the shipped [iron]
coefficients under the same fit procedure and B cut as the CREATOR
M250-35A fit. B_core is a geometric no-load estimate with no measured
iron-loss curve behind it, so there is no machine-tier agreement test;
the iron_loss run is exercised for shape and order of magnitude only.
"""

import csv
import dataclasses
from math import pi

import numpy as np
import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.models.iron_loss import (
    bertotti_loss_density,
    fit_bertotti,
    run_iron_loss,
)
from phasesweep.sweep_types import RunConfig
from tests.conftest import M19_TABLE_SKIP, REPO_ROOT

CSV = REPO_ROOT / "data/bmw_i3/m19_29ga_core_loss.csv"
TOML = REPO_ROOT / "motors/bmw_i3.toml"
FREQS_HZ = (50, 60, 100, 150, 200, 300, 400, 600, 1000, 1500, 2000)


@pytest.fixture(scope="module")
def table():
    if not CSV.exists():
        pytest.skip(M19_TABLE_SKIP)
    with CSV.open() as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]
    assert rows[0] == ["f_Hz", "B_T", "p_W_per_kg"]
    d = np.array(rows[1:], dtype=float)
    return d[:, 0], d[:, 1], d[:, 2]


@pytest.fixture(scope="module")
def motor():
    return load_motor(TOML)


class TestM19Table:

    def test_shape(self, table):
        f, B, p = table
        assert len(f) == 112
        assert set(f) == set(FREQS_HZ)
        assert B.min() == pytest.approx(0.1) and B.max() == pytest.approx(1.7)
        assert (p > 0).all()

    def test_monotone_in_f_and_B(self, table):
        f, B, p = table
        for b in np.unique(B):
            m = B == b
            assert (np.diff(p[m][np.argsort(f[m])]) > 0).all(), f"loss vs f at {b} T"
        for fq in FREQS_HZ:
            m = f == fq
            assert (np.diff(p[m][np.argsort(B[m])]) > 0).all(), f"loss vs B at {fq} Hz"


class TestM19Fit:

    @pytest.fixture(scope="class")
    def fit(self, table):
        f, B, p = table
        m = B <= 1.5
        return fit_bertotti(f[m], B[m], p[m]), (f[m], B[m], p[m])

    def test_coefficients_are_the_shipped_i3_fit(self, fit, motor):
        (k_h, k_e, alpha), _ = fit
        assert (motor.k_h, motor.k_e, motor.alpha_fe) == (0.02127, 6.149e-5, 1.78)
        assert k_h == pytest.approx(motor.k_h, rel=1e-3)
        assert k_e == pytest.approx(motor.k_e, rel=1e-3)
        assert alpha == pytest.approx(motor.alpha_fe, abs=0.005)

    def test_quality(self, fit):
        (k_h, k_e, alpha), (f, B, p) = fit
        rel = bertotti_loss_density(f, B, k_h, k_e, alpha) / p - 1
        assert len(p) == 90
        assert np.median(np.abs(rel)) == pytest.approx(0.049, abs=0.01)
        assert np.abs(rel).max() < 0.25

    def test_spot_check_60hz_1p5t(self, fit, table):
        (k_h, k_e, alpha), _ = fit
        f, B, p = table
        i = np.flatnonzero((f == 60) & np.isclose(B, 1.5))[0]
        assert bertotti_loss_density(60.0, 1.5, k_h, k_e, alpha) == pytest.approx(p[i], rel=0.05)


class TestI3IronLossRun:

    def test_runs_at_base_speed(self, motor):
        w_ref = 4800 * 2 * pi / 60
        m = dataclasses.replace(motor, drive=dataclasses.replace(motor.drive, W_REF=w_ref))
        r = run_iron_loss(RunConfig(motor=m, model="iron_loss"))
        assert r["f_e_Hz"] == pytest.approx(480.0)
        assert r["p_fe"] == pytest.approx(r["p_fe_hysteresis"] + r["p_fe_eddy"])
        assert r["loss_density_W_per_kg"] == pytest.approx(
            bertotti_loss_density(480.0, motor.B_core, motor.k_h, motor.k_e, motor.alpha_fe))
        # Order of magnitude at the uncalibrated geometric B_core; the
        # ORNL motor map implies several times this once calibrated.
        assert 250 < r["p_fe"] < 450

    def test_motor_toml_has_no_w_ref(self, motor):
        assert motor.drive.W_REF is None
