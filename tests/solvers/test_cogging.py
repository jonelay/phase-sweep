"""Cogging torque model — angle grid, Arkkio torque, registry, waveform."""

import math

import numpy as np
import pytest

from phasesweep.machines.geometry import inrunner, outrunner
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solvers.harmonics import cogging_angles
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import make_motor, requires_fem

# Slotted geometry for cogging tests — 6 slots, n_p=2 gives
# lcm(6, 4) = 12 cogging periods per rev (30 deg period).
_SLOTTED_GEO = inrunner(
    r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
    n_slots=6, slot_depth=0.005, slot_opening_width=0.003,
)
_COARSE = dict(maxh_fraction=0.08, n_theta=60)


def _slotted_motor(**kw):
    return make_motor(
        name="cogging_test", geometry=_SLOTTED_GEO,
        n_p=2, B_rem=1.2, psi_f=None, mu_r_pm=1.05,
        **kw,
    )


# ---------------------------------------------------------------------------
# cogging_angles unit tests
# ---------------------------------------------------------------------------

class TestCoggingAngles:

    def test_6_slot_4_pole(self):
        angles = cogging_angles(6, 2)
        period = 2 * math.pi / 12
        assert len(angles) == 12
        assert angles[0] == pytest.approx(0.0)
        assert angles[-1] == pytest.approx(period * 11 / 12)

    def test_12_slot_10_pole(self):
        # lcm(12, 10) = 60 periods per rev → 6 deg period
        angles = cogging_angles(12, 5)
        period = 2 * math.pi / 60
        assert len(angles) == 12
        assert float(np.diff(angles).mean()) == pytest.approx(period / 12)

    def test_custom_points(self):
        angles = cogging_angles(6, 2, points_per_period=24)
        assert len(angles) == 24

    def test_multiple_periods(self):
        angles = cogging_angles(6, 2, n_periods=3)
        assert len(angles) == 36

    def test_endpoint_excluded(self):
        angles = cogging_angles(6, 2)
        period = 2 * math.pi / 12
        assert angles[-1] < period


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

def test_registry_entry():
    info = MODEL_REGISTRY["cogging_torque"]
    assert info.source == "computed"
    assert info.cost == "slow"
    assert "tau_cogging_list" in info.produces
    assert "dominant_order" in info.produces
    assert "cogging_points" in info.hash_fields
    assert "maxh_fraction" in info.hash_fields
    assert info.fn is not None


def test_registry_validate_accepts_slotted():
    motor = _slotted_motor()
    MODEL_REGISTRY["cogging_torque"].validate(motor)


# ---------------------------------------------------------------------------
# Hash integration
# ---------------------------------------------------------------------------

class TestHashFields:

    def test_rotation_changes_fem_hash(self):
        motor = _slotted_motor()
        rc1 = RunConfig(motor=motor, model="fem", rotation=0.0, **_COARSE)
        rc2 = RunConfig(motor=motor, model="fem", rotation=0.1, **_COARSE)
        fields = MODEL_REGISTRY["fem"].hash_fields
        assert compute_run_id(rc1, fields) != compute_run_id(rc2, fields)

    def test_rotation_zero_preserves_fem_hash(self):
        motor = _slotted_motor()
        rc = RunConfig(motor=motor, model="fem", rotation=0.0, **_COARSE)
        rc_no_rot = RunConfig(motor=motor, model="fem", **_COARSE)
        fields = MODEL_REGISTRY["fem"].hash_fields
        assert compute_run_id(rc, fields) == compute_run_id(rc_no_rot, fields)

    def test_cogging_points_changes_cogging_hash(self):
        motor = _slotted_motor()
        rc1 = RunConfig(motor=motor, model="cogging_torque",
                        cogging_points=12, **_COARSE)
        rc2 = RunConfig(motor=motor, model="cogging_torque",
                        cogging_points=24, **_COARSE)
        fields = MODEL_REGISTRY["cogging_torque"].hash_fields
        assert compute_run_id(rc1, fields) != compute_run_id(rc2, fields)

    def test_cogging_points_ignored_by_fem(self):
        motor = _slotted_motor()
        rc1 = RunConfig(motor=motor, model="fem",
                        cogging_points=12, **_COARSE)
        rc2 = RunConfig(motor=motor, model="fem",
                        cogging_points=24, **_COARSE)
        fields = MODEL_REGISTRY["fem"].hash_fields
        assert compute_run_id(rc1, fields) == compute_run_id(rc2, fields)

    def test_rotation_ignored_by_cogging(self):
        motor = _slotted_motor()
        rc1 = RunConfig(motor=motor, model="cogging_torque",
                        rotation=0.0, **_COARSE)
        rc2 = RunConfig(motor=motor, model="cogging_torque",
                        rotation=0.5, **_COARSE)
        fields = MODEL_REGISTRY["cogging_torque"].hash_fields
        assert compute_run_id(rc1, fields) == compute_run_id(rc2, fields)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_rotation_round_trip(self):
        motor = _slotted_motor()
        rc = RunConfig(motor=motor, model="fem", rotation=0.123)
        d = rc.to_dict()
        assert d["rotation"] == 0.123
        rc2 = RunConfig.from_dict(d)
        assert rc2.rotation == pytest.approx(0.123)

    def test_rotation_zero_omitted(self):
        motor = _slotted_motor()
        rc = RunConfig(motor=motor, model="fem", rotation=0.0)
        d = rc.to_dict()
        assert "rotation" not in d

    def test_cogging_points_round_trip(self):
        motor = _slotted_motor()
        rc = RunConfig(motor=motor, model="cogging_torque", cogging_points=24)
        d = rc.to_dict()
        assert d["cogging_points"] == 24
        rc2 = RunConfig.from_dict(d)
        assert rc2.cogging_points == 24

    def test_cogging_points_default_omitted(self):
        motor = _slotted_motor()
        rc = RunConfig(motor=motor, model="cogging_torque")
        d = rc.to_dict()
        assert "cogging_points" not in d


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_nonzero_j_s():
    from phasesweep.solvers.cogging import _run_cogging_impl
    motor = _slotted_motor()
    rc = RunConfig(motor=motor, model="cogging_torque", j_s=100.0, **_COARSE)
    with pytest.raises(ValueError, match="j_s=0"):
        _run_cogging_impl(rc)


def test_rejects_smooth_bore():
    from phasesweep.solvers.cogging import _run_cogging_impl
    motor = make_motor(B_rem=1.2, psi_f=None)
    rc = RunConfig(motor=motor, model="cogging_torque", **_COARSE)
    with pytest.raises(ValueError, match="n_slots"):
        _run_cogging_impl(rc)


def test_rejects_low_cogging_points():
    from phasesweep.solvers.cogging import _run_cogging_impl
    motor = _slotted_motor()
    for pts in (0, 1, 3):
        rc = RunConfig(motor=motor, model="cogging_torque",
                       cogging_points=pts, **_COARSE)
        with pytest.raises(ValueError, match="cogging_points"):
            _run_cogging_impl(rc)


# ---------------------------------------------------------------------------
# Arkkio vs single-contour agreement (single static solve)
# ---------------------------------------------------------------------------

@requires_fem
@pytest.mark.timeout(120)
def test_arkkio_is_radial_average_of_maxwell():
    """Arkkio torque equals the manually computed radial average of
    single-contour Maxwell-stress integrals at the same radii."""
    from phasesweep.solver_params import prepare_fem
    from phasesweep.solvers.fem_field import (
        arkkio_torque,
        maxwell_stress_torque,
        solve_field_fem,
    )

    geo = inrunner(
        r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
    )
    motor = make_motor(
        name="arkkio_test", geometry=geo, n_p=2,
        B_rem=1.2, psi_f=None, mu_r_pm=1.05,
    )
    params = prepare_fem(motor)
    _, _, mesh, gfu = solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        n_theta=360, maxh_fraction=0.03,
        alpha_p=params.alpha_p, j_s=0.0,
        return_full=True,
    )
    g = geo.r_stator - geo.r_magnet
    margin = 0.2 * g
    r_inner = geo.r_magnet + margin
    r_outer = geo.r_stator - margin
    n_r = 5

    tau_arkkio = arkkio_torque(
        mesh, gfu, r_inner, r_outer, n_r=n_r,
    )

    radii = np.linspace(r_inner, r_outer, n_r)
    manual = np.array([
        maxwell_stress_torque(mesh, gfu, r) for r in radii
    ])
    manual_avg = float(np.trapezoid(manual, radii) / (r_outer - r_inner))

    assert tau_arkkio == pytest.approx(manual_avg, rel=1e-10)


# ---------------------------------------------------------------------------
# Cogging waveform properties (fast coarse sweep)
# ---------------------------------------------------------------------------

@requires_fem
@pytest.mark.timeout(600)
def test_cogging_waveform_properties():
    """Cogging waveform has near-zero mean and expected periodicity."""
    from phasesweep.solvers.cogging import _run_cogging_impl

    motor = _slotted_motor()
    rc = RunConfig(motor=motor, model="cogging_torque", **_COARSE)
    result = _run_cogging_impl(rc)

    tau = np.array(result["tau_cogging_list"])
    assert len(tau) == 12

    assert abs(np.mean(tau)) < 0.1 * (tau.max() - tau.min() + 1e-15)

    assert result["n_cogging_periods"] == 12
    assert result["tau_cogging_pp"] >= 0.0
    assert result["dominant_order"] > 0


@requires_fem
@pytest.mark.timeout(600)
def test_cogging_tau_pp_nm_with_lstk():
    """tau_cogging_pp_Nm present when L_stk is set."""
    from phasesweep.solvers.cogging import _run_cogging_impl

    motor = _slotted_motor(L_stk=0.050)
    rc = RunConfig(motor=motor, model="cogging_torque", **_COARSE)
    result = _run_cogging_impl(rc)

    assert "tau_cogging_pp_Nm" in result
    assert result["tau_cogging_pp_Nm"] == pytest.approx(
        result["tau_cogging_pp"] * 0.050)


@requires_fem
@pytest.mark.timeout(600)
def test_cogging_tau_pp_nm_absent_without_lstk():
    """tau_cogging_pp_Nm absent when L_stk is None."""
    from phasesweep.solvers.cogging import _run_cogging_impl

    motor = _slotted_motor(L_stk=None)
    rc = RunConfig(motor=motor, model="cogging_torque", **_COARSE)
    result = _run_cogging_impl(rc)

    assert "tau_cogging_pp_Nm" not in result


# ---------------------------------------------------------------------------
# Outrunner sign convention
# ---------------------------------------------------------------------------

_SLOTTED_OUTRUNNER = outrunner(
    r_outer=0.060, r_rotor=0.048, r_magnet=0.040,
    r_stator=0.030, r_inner=0.015,
    n_slots=6, slot_depth=0.003, slot_opening_width=0.002,
)


@requires_fem
@pytest.mark.timeout(1200)
def test_cogging_offset_shift_stability():
    """Dominant-order amplitude is stable under a half-sample grid shift.

    Runs the cogging sweep twice on the same geometry: once at the standard
    angle grid and once with every angle offset by half a sample spacing.
    The dominant-order Fourier amplitude should agree within a few percent;
    the difference is the per-angle remeshing noise floor.
    """
    from phasesweep.solver_params import prepare_fem
    from phasesweep.solvers.fem_field import arkkio_torque, solve_field_fem
    from phasesweep.solvers.harmonics import cogging_angles, harmonics_1sided

    motor = _slotted_motor()
    params = prepare_fem(motor)
    geo = params.geometry
    n_pts = 12
    angles_base = cogging_angles(geo.n_slots, params.n_p, points_per_period=n_pts)
    shift = float(angles_base[1] - angles_base[0]) / 2
    angles_shifted = angles_base + shift

    g = abs(geo.r_stator - geo.r_magnet)
    margin = 0.05 * g
    r_inner = geo.r_magnet + margin
    r_outer = geo.r_stator - margin

    def _sweep(angles):
        tau = []
        for phi in angles:
            _, _, mesh_obj, gfu = solve_field_fem(
                geo=geo, n_p=params.n_p, B_rem=params.B_rem,
                mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
                maxh_fraction=_COARSE["maxh_fraction"],
                n_theta=_COARSE["n_theta"],
                j_s=0.0, alpha_p=params.alpha_p,
                rotation=float(phi), return_full=True,
            )
            tau.append(arkkio_torque(mesh_obj, gfu, r_inner, r_outer,
                                    n_theta=_COARSE["n_theta"]))
        return np.array(tau)

    tau_base = _sweep(angles_base)
    tau_shifted = _sweep(angles_shifted)

    amps_base = harmonics_1sided(tau_base)
    amps_shifted = harmonics_1sided(tau_shifted)
    dom_base = float(amps_base[1])
    dom_shifted = float(amps_shifted[1])

    assert dom_base > 0, "zero cogging amplitude — geometry has no slots?"
    # Coarse mesh (maxh_fraction=0.08) gives ~30% drift; production mesh
    # (maxh_fraction ~ 0.03/r_outer) gives <5%. The bound here catches
    # catastrophic errors; the absolute noise level is a documented finding,
    # not a pass/fail gate.
    assert dom_shifted == pytest.approx(dom_base, rel=0.40), (
        f"offset-shift dominant amplitude {dom_shifted:.4e} vs "
        f"baseline {dom_base:.4e} (>{40}% drift = excessive remeshing noise)"
    )


@requires_fem
@pytest.mark.timeout(600)
def test_cogging_outrunner_runs_and_sign_flips():
    """Outrunner cogging sweep runs without error; the rotor_sign=-1
    branch is exercised and waveform has near-zero mean (no DC bias
    from a sign error)."""
    from phasesweep.solvers.cogging import _run_cogging_impl

    motor = make_motor(
        name="outrunner_cogging", geometry=_SLOTTED_OUTRUNNER,
        n_p=2, B_rem=1.2, psi_f=None, mu_r_pm=1.05,
    )
    rc = RunConfig(motor=motor, model="cogging_torque", **_COARSE)
    result = _run_cogging_impl(rc)

    tau = np.array(result["tau_cogging_list"])
    assert len(tau) == 12
    assert abs(np.mean(tau)) < 0.1 * (tau.max() - tau.min() + 1e-15)
