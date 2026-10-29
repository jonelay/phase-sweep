"""Tests for fem_runner.py — the production RunConfig -> solve_field_fem
path, plus RunConfig hashing of FEM-only fields."""

from importlib.util import find_spec

import pytest

if not find_spec("ngsolve"):
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)

import numpy as np

from phasesweep.solvers import fem_field, fem_sampling
from tests.conftest import requires_fem
from tests.solvers.fem_shared import (
    _GEO,
    _H_FAST,
    _HALBACH_GEO,
    _SLOTTED_GEO_KW,
    FAST,
    _torque_geo,
)

# Module-level guard covers missing ngsolve; the marker makes the module
# selectable by lane.
pytestmark = requires_fem


def _fake_raster(mesh, gfu, n_grid, r_bound):
    grid = np.full((n_grid, n_grid), np.nan)
    return grid, grid, grid, grid


def test_fem_runner_defers_slot_params_to_geometry(monkeypatch):
    """Production FEM path passes no slot kwargs — geometry governs."""
    from phasesweep.solvers.fem_runner import run_fem
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    captured = {}

    def fake_solve(*args, **kw):
        captured.update(kw)
        th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        # runner always requests return_full=True for the raster
        return th, np.cos(2 * th), None, None

    monkeypatch.setattr(fem_field, "solve_field_fem", fake_solve)
    monkeypatch.setattr(fem_sampling, "rasterise_cross_section", _fake_raster)
    motor = make_motor(B_rem=1.2, psi_f=None)
    run_fem(RunConfig(motor=motor, model="fem"))
    assert "n_slots" not in captured
    assert "slot_width_ratio" not in captured


def test_fem_runner_j_s_requires_k_w():
    from phasesweep.solvers.fem_runner import run_fem
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    motor = make_motor(B_rem=1.2, psi_f=None, k_w=None)
    with pytest.raises(ValueError, match="k_w"):
        run_fem(RunConfig(motor=motor, model="fem", j_s=1.0))


def test_fem_runner_reports_both_slot_harmonic_sidebands(monkeypatch):
    """Slot harmonics come in pairs Q ± n_p; both sidebands are reported."""
    from phasesweep.machines.geometry import inrunner
    from phasesweep.solvers.fem_runner import run_fem
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    def fake_solve(*args, **kw):
        th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        # fundamental at n_p=2; sidebands at Q-n_p=10 and Q+n_p=14
        return th, (np.cos(2 * th) + 0.20 * np.cos(10 * th)
                    + 0.10 * np.cos(14 * th)), None, None

    monkeypatch.setattr(fem_field, "solve_field_fem", fake_solve)
    monkeypatch.setattr(fem_sampling, "rasterise_cross_section", _fake_raster)
    motor = make_motor(geometry=inrunner(**_SLOTTED_GEO_KW), B_rem=1.2, psi_f=None)
    metrics = run_fem(RunConfig(motor=motor, model="fem"))
    assert metrics["sh_pct"] == pytest.approx(20.0, rel=1e-6)
    assert metrics["sh_upper_pct"] == pytest.approx(10.0, rel=1e-6)


def test_fem_runner_zero_fundamental_raises(monkeypatch):
    """A zero fundamental is a failed solve: raise rather than report the
    slot harmonics as raw amplitudes over a stand-in 1.0."""
    from phasesweep.machines.geometry import inrunner
    from phasesweep.solvers.fem_runner import run_fem
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    def fake_solve(*args, **kw):
        th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        return th, 0.2 * np.cos(10 * th), None, None

    monkeypatch.setattr(fem_field, "solve_field_fem", fake_solve)
    monkeypatch.setattr(fem_sampling, "rasterise_cross_section", _fake_raster)
    motor = make_motor(geometry=inrunner(**_SLOTTED_GEO_KW), B_rem=1.2, psi_f=None)
    with pytest.raises(RuntimeError, match="fundamental at order n_p=2"):
        run_fem(RunConfig(motor=motor, model="fem"))


def test_fem_runner_emits_cross_section_raster():
    """Dashboard heatmap raster: |B| on a shared-axis grid,
    None outside the motor domain, additive alongside existing metrics."""
    from phasesweep.solvers.fem_runner import _RASTER_N_GRID, run_fem
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    motor = make_motor(B_rem=1.2, psi_f=None)
    metrics = run_fem(RunConfig(motor=motor, model="fem", **FAST))
    grid = metrics["B_mag_grid"]
    coords = metrics["grid_coords_list"]
    assert len(grid) == _RASTER_N_GRID
    assert all(len(row) == _RASTER_N_GRID for row in grid)
    assert len(coords) == _RASTER_N_GRID
    assert coords[-1] == pytest.approx(motor.geometry.r_outer)
    assert coords[0] == pytest.approx(-motor.geometry.r_outer)
    # corners of the square are outside the motor disk
    assert grid[0][0] is None and grid[-1][-1] is None
    vals = [v for row in grid for v in row if v is not None]
    assert vals and max(vals) > 0.1
    assert "fundamental" in metrics and "B_r_list" in metrics


def test_fem_runner_halbach_end_to_end():
    """run_fem on a Halbach motor produces different B1 than radial,
    proving the threading survives the full RunConfig → prepare_fem →
    solve_field_fem path (the 'silent radial' risk)."""
    from phasesweep.machines.motor import Motor
    from phasesweep.solvers.fem_runner import run_fem
    from phasesweep.sweep_types import RunConfig

    geo = _HALBACH_GEO
    m_rad = Motor(name="rad", geometry=geo, n_p=4, B_rem=1.2)
    m_hal = Motor(
        name="hal", geometry=geo, n_p=4, B_rem=1.2,
        magnetization="halbach", halbach_segments=4,
    )
    met_rad = run_fem(RunConfig(motor=m_rad, model="fem", **_H_FAST))
    met_hal = run_fem(RunConfig(motor=m_hal, model="fem", **_H_FAST))
    assert met_hal["fundamental"] != pytest.approx(met_rad["fundamental"], rel=0.05)
    assert met_hal["fundamental"] > met_rad["fundamental"]


def test_config_id_differs_for_nonlinear():
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig, compute_run_id
    motor = Motor(name="test", geometry=_GEO, n_p=2, R_s=3.6,
                  L_d=4e-3, L_q=4e-3, psi_f=0.1, J=1.5e-3)
    rc1 = RunConfig(motor=motor, model="fem", nonlinear=False)
    rc2 = RunConfig(motor=motor, model="fem", nonlinear=True)
    assert compute_run_id(rc1) != compute_run_id(rc2)


def test_config_roundtrip_nonlinear():
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig, compute_run_id
    motor = Motor(name="test", geometry=_GEO, n_p=2, R_s=3.6,
                  L_d=4e-3, L_q=4e-3, psi_f=0.1, J=1.5e-3)
    rc = RunConfig(motor=motor, model="fem", nonlinear=True)
    rc2 = RunConfig.from_dict(rc.to_dict())
    assert rc2.nonlinear is True
    assert compute_run_id(rc2) == compute_run_id(rc)


def test_fem_runner_reports_maxwell_torque():
    """run_fem emits tau_maxwell_per_m (and tau_maxwell with L_stk)
    when j_s != 0; no torque keys on open-circuit runs."""
    from phasesweep.machines.motor import Motor
    from phasesweep.solvers.fem_runner import run_fem
    from phasesweep.sweep_types import RunConfig
    motor = Motor(name="torque-test", geometry=_torque_geo(), n_p=2,
                  B_rem=1.0, k_w=0.966, L_stk=0.10)
    rc = RunConfig(motor=motor, model="fem", j_s=1e5,
                   n_theta=60, maxh_fraction=0.08)
    m = run_fem(rc)
    assert m["tau_maxwell"] == pytest.approx(
        0.10 * m["tau_maxwell_per_m"], rel=1e-12)
    assert m["tau_maxwell_per_m"] > 0.0  # motoring sign for the q-axis sheet
    rc0 = RunConfig(motor=motor, model="fem",
                    n_theta=60, maxh_fraction=0.08)
    m0 = run_fem(rc0)
    assert "tau_maxwell_per_m" not in m0
    assert "tau_maxwell" not in m0
