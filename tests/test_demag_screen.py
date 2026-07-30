"""Demag screen — knee schema, model, CREATOR anchor.

The CREATOR anchor is a bound, not a calibration: the dataset's
Ferrite_BH.csv demag curve is linear to B = 0 (no knee above zero at
20 °C, B_knee = 0.0 in the TOML), and the published design runs at the
drive current limit — so the screen at MAX_I_S must report positive
margin. No measured demag data exists in-repo.
"""

import dataclasses
from pathlib import Path

import pytest

from phasesweep.configs import load_motor
from phasesweep.demag_screen import run_demag_screen
from phasesweep.geometry import default_inrunner
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solver_params import (
    j_s_from_phase_current,
    prepare_demag_screen,
)
from phasesweep.sweep_types import RunConfig, compute_run_id

MOTOR_TOML = Path(__file__).parent.parent / "motors" / "creator_case_pmsm.toml"

_COARSE = dict(maxh_fraction=0.08, n_theta=60)


@pytest.fixture(scope="module")
def creator():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def screen_at_drive_limit(creator):
    rc = RunConfig(motor=creator, model="demag_screen",
                   i_fault=creator.drive.MAX_I_S, **_COARSE)
    return run_demag_screen(rc)


def test_registry_entry(creator):
    info = MODEL_REGISTRY["demag_screen"]
    assert info.source == "computed"
    assert "margin" in info.produces
    assert "i_fault" in info.hash_fields
    info.validate(creator)


class TestPrepareValidation:

    def test_b_knee_required(self, creator):
        m = dataclasses.replace(creator, B_knee=None)
        with pytest.raises(ValueError, match="B_knee"):
            prepare_demag_screen(m)

    def test_slotted_geometry_required(self, creator):
        m = dataclasses.replace(creator, geometry=default_inrunner())
        with pytest.raises(ValueError, match="slotted"):
            prepare_demag_screen(m)

    def test_magnet_temp_needs_both_slopes(self, creator):
        m = dataclasses.replace(creator, magnet_temp=60.0, alpha_Br=-0.002)
        with pytest.raises(ValueError, match="alpha_B_knee"):
            prepare_demag_screen(m)

    def test_temperature_derating(self, creator):
        """B_rem fractional via alpha_Br, knee absolute via alpha_B_knee,
        both at magnet_temp (ferrite signs: both worsen when cold)."""
        m = dataclasses.replace(creator, magnet_temp=-20.0,
                                alpha_Br=-0.002, alpha_B_knee=-0.001)
        p = prepare_demag_screen(m)
        assert p.fem.B_rem == pytest.approx(0.415 * (1 - 0.002 * -40.0))
        assert p.B_knee == pytest.approx(0.0 - 0.001 * -40.0)


def test_i_fault_required(creator):
    rc = RunConfig(motor=creator, model="demag_screen", **_COARSE)
    with pytest.raises(ValueError, match="i_fault"):
        run_demag_screen(rc)


def test_creator_screen_at_drive_limit(creator, screen_at_drive_limit):
    """Bound anchor: the published CREATOR design operates at MAX_I_S with
    a knee-free-to-zero ferrite curve, so the screen must clear it."""
    out = screen_at_drive_limit
    assert out["margin"] > 0
    assert out["B_m_min"] > 0
    assert out["frac_below_knee"] == 0.0
    assert out["B_knee"] == 0.0
    # Every assertion above is one-sided or monotonic, so a constant
    # optimistic bias anywhere in the screen passed the whole file. Pin the
    # value two-sided at the frozen _COARSE settings — this is what catches a
    # bias in sample_magnet_Bm, which the identity below cannot see (it shifts
    # B_m_min and margin together). abs=0.03 absorbs mesh/ngsolve drift while
    # catching the +0.15 T case 5x over.
    assert out["margin"] == pytest.approx(0.2406, abs=0.03)
    # Interface pin: margin is B_m_min measured against the knee, by
    # definition. Degenerate here (B_knee = 0.0) but free.
    assert out["margin"] == pytest.approx(out["B_m_min"] - out["B_knee"])
    # No magnet operating point can exceed remanence.
    assert out["B_m_min"] < prepare_demag_screen(creator).fem.B_rem
    assert out["j_s_fault"] == pytest.approx(
        j_s_from_phase_current(creator, creator.drive.MAX_I_S))
    # worst point on the gap-facing half of the magnet (d-axis MMF
    # attacks from the bore side)
    geo = creator.geometry
    assert (geo.r_rotor + geo.r_magnet) / 2 < out["r_min"] < geo.r_magnet


def test_margin_monotonic_in_fault_current(creator, screen_at_drive_limit):
    rc2 = RunConfig(motor=creator, model="demag_screen",
                    i_fault=2 * creator.drive.MAX_I_S, **_COARSE)
    out2 = run_demag_screen(rc2)
    assert out2["margin"] < screen_at_drive_limit["margin"]


def test_run_id_hashes_i_fault(creator):
    rc_a = RunConfig(motor=creator, model="demag_screen", i_fault=0.42)
    rc_b = RunConfig(motor=creator, model="demag_screen", i_fault=0.84)
    assert compute_run_id(rc_a) != compute_run_id(rc_b)
    assert compute_run_id(rc_a) == compute_run_id(
        RunConfig(motor=creator, model="demag_screen", i_fault=0.42))
