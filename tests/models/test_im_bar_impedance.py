"""im_bar_impedance: deep-bar rotor impedance from the
Bloch-periodic single-slot-pitch harmonic solve.

Algebra and rejection tests run without ngsolve; the cell gates need
the fem extra. The CREATOR comparison lives in
tests/crossval/test_creator_im_crossval.py.
"""

import dataclasses as dc
from math import nan, pi

import numpy as np
import pytest

from phasesweep.machines.geometry import default_inrunner, inrunner
from phasesweep.machines.motor import InductionParams, Motor
from phasesweep.models.im_bar_impedance import (
    ImBarParams,
    differential_leakage_coefficient,
    prepare_im_bar_impedance,
    referral_factor,
    ring_factor,
    ring_segment,
    rotor_frequency_grid,
)
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solvers.fem_materials import MU0
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import make_motor, requires_fem

_CAGE_KW = dict(r_outer=0.05, r_stator=0.032, r_magnet=0.031, r_rotor=0.031,
                r_inner=0.011, n_bars=8, bar_width=0.0034, bar_height=0.01,
                n_slots=12, slot_depth=0.008,
                end_ring_axial_length=0.008, end_ring_radial_height=0.008)
_SIG = 3e7


def make_cage_motor(geo_overrides=None, **overrides) -> Motor:
    geo = inrunner(**{**_CAGE_KW, **(geo_overrides or {})})
    ind = dict(sigma_bar=_SIG, f_rated=50.0, L_m=0.2)
    ind.update(overrides.pop("induction", {}))
    fields = dict(name="test cage", geometry=geo, n_p=2, N=10, coils_series=4,
                  k_w=0.9, L_stk=0.1, induction=InductionParams(**ind))
    fields.update(overrides)
    return Motor(**fields)


# ---------------------------------------------------------------------------
# Algebra (no FEM)
# ---------------------------------------------------------------------------

class TestAlgebra:

    @pytest.mark.parametrize("n_p, n_bars", [(2, 28), (1, 8), (3, 40)])
    def test_ring_factor_is_the_stencil_eigenvalue(self, n_p, n_bars):
        """The F2 ring stencil I_k = L·lap(u)_k/(2Z) on the progressive
        mode u_k = e^{-jαk} has eigenvalue -4 sin²(α/2), so one segment
        impedance per end appears in series with the bar as Z/(2 sin²)."""
        lap = np.zeros((n_bars, n_bars))
        for k in range(n_bars):
            lap[k, k] = -2
            lap[k, (k + 1) % n_bars] = lap[k, (k - 1) % n_bars] = 1
        v = np.exp(-2j * pi * n_p * np.arange(n_bars) / n_bars)
        lam = (lap @ v)[0] / v[0]
        assert lam.imag == pytest.approx(0, abs=1e-12)
        assert ring_factor(n_p, n_bars) == pytest.approx(2 / abs(lam.real), rel=1e-12)

    def test_differential_coefficient_is_the_harmonic_sum(self):
        n_p, n_bars = 2, 28
        ks = np.arange(1, 200001)
        brute = np.sum((n_p / (ks * n_bars + n_p)) ** 2 + (n_p / (ks * n_bars - n_p)) ** 2)
        assert differential_leakage_coefficient(n_p, n_bars) == pytest.approx(brute, rel=1e-5)  # tail 2a²/K
        assert differential_leakage_coefficient(n_p, n_bars) == pytest.approx(0.01696, rel=1e-3)

    def test_referral_factor_creator(self):
        # 4·3·(216·0.902)²/28 — loss and MMF equivalence of the cage
        assert referral_factor(216, 0.902, 28) == pytest.approx(16268.36, rel=1e-6)

    def test_ring_segment_arithmetic(self):
        geo = inrunner(**_CAGE_KW)
        R, L = ring_segment(geo, _SIG)
        D = 2 * (geo.r_rotor - 0.004)
        assert R == pytest.approx(pi * D / (8 * _SIG * 0.008 * 0.008), rel=1e-12)
        assert L > 0
        assert L / MU0 == pytest.approx(2.3 * D / 16 * np.log10(4.7 * D / 0.024), rel=1e-12)

    def test_ring_segment_needs_ring_dims(self):
        geo = inrunner(**{**_CAGE_KW, "end_ring_axial_length": None})
        with pytest.raises(ValueError, match="end_ring"):
            ring_segment(geo, _SIG)

    def test_frequency_grid_merges_tables(self):
        m = make_cage_motor(induction=dict(R_2=1.0, R_2_vs_f=((10.0, 1.1), (25.0, 1.3)),
                                           L_sigma_r=0.01,
                                           L_sigma_r_vs_f=((25.0, 0.009), (35.0, 0.008))))
        grid = rotor_frequency_grid(prepare_im_bar_impedance(m))
        assert grid[0] == 0.0
        assert grid == sorted(set(grid))
        assert {10.0, 25.0, 35.0, 50.0, 100.0} <= set(grid)
        assert grid.count(25.0) == 1


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------

class TestPrepare:

    def test_synchronous_motor_rejected(self):
        with pytest.raises(ValueError, match=r"requires an \[induction\]"):
            prepare_im_bar_impedance(make_motor())

    def test_no_geometry_rejected(self):
        m = Motor(name="im", geometry=None, n_p=2,
                  induction=InductionParams(sigma_bar=_SIG, f_rated=50.0))
        with pytest.raises(ValueError, match="needs geometry"):
            prepare_im_bar_impedance(m)

    def test_missing_ring_dims_rejected(self):
        m = make_cage_motor(geo_overrides=dict(end_ring_radial_height=None))
        with pytest.raises(ValueError, match="end_ring_radial_height"):
            prepare_im_bar_impedance(m)

    @pytest.mark.parametrize("field, msg", [
        ("sigma_bar", "sigma_bar"), ("f_rated", "f_rated"),
    ])
    def test_missing_induction_fields_rejected(self, field, msg):
        m = make_cage_motor(induction={field: None})
        with pytest.raises(ValueError, match=msg):
            prepare_im_bar_impedance(m)

    @pytest.mark.parametrize("field, msg", [
        ("k_w", "k_w"), ("L_stk", "L_stk"), ("N", "N is required"),
        ("coils_series", "coils_series"),
    ])
    def test_missing_winding_fields_rejected(self, field, msg):
        with pytest.raises(ValueError, match=msg):
            prepare_im_bar_impedance(make_cage_motor(**{field: None}))

    def test_happy_path(self):
        p = prepare_im_bar_impedance(make_cage_motor())
        assert p.N_eff == 40 and p.k_w == 0.9 and p.L_m == 0.2
        assert p.geometry.n_bars == 8 and p.f_table == ()

    def test_params_reject_pm_geometry(self):
        with pytest.raises(ValueError, match="n_bars > 0"):
            ImBarParams(geometry=default_inrunner(), n_p=2, sigma_bar=_SIG, mu_r_fe=1000.0,
                        L_stk=0.1, N_eff=40, k_w=0.9, f_rated=50.0, f_table=(), L_m=None)

    def test_registry_metadata(self):
        info = MODEL_REGISTRY["im_bar_impedance"]
        assert info.cost == "medium" and info.hash_fields == {"maxh_fraction"}
        m = make_cage_motor()
        a = compute_run_id(RunConfig(motor=m, model="im_bar_impedance"))
        b = compute_run_id(RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.02))
        c = compute_run_id(RunConfig(motor=m, model="im_bar_impedance", slip=0.5))
        assert a != b and a == c


# ---------------------------------------------------------------------------
# Cell gates (FEM)
# ---------------------------------------------------------------------------

@requires_fem
class TestCell:
    """8-bar test cage (the F2 power-balance geometry plus an end ring)."""

    @pytest.fixture(scope="class")
    def params(self):
        return prepare_im_bar_impedance(make_cage_motor())

    @pytest.fixture(scope="class")
    def cell(self, params):
        from phasesweep.solvers.fem_mesh import build_bar_cell_mesh
        return build_bar_cell_mesh(params.geometry, 0.08)

    def test_mesh_regions_and_bore(self, cell, params):
        from ngsolve import BND, CoefficientFunction, Integrate

        geo = params.geometry
        assert set(cell.GetMaterials()) == {"rotor", "bar", "airgap"}
        assert "outer" in set(cell.GetBoundaries())
        arc = Integrate(CoefficientFunction(1), cell, BND, definedon=cell.Boundaries("outer"))
        assert arc == pytest.approx(2 * pi * geo.r_stator / geo.n_bars, rel=1e-4)
        area = Integrate(CoefficientFunction(1), cell, definedon=cell.Materials("bar"))
        assert area == pytest.approx(geo.bar_area, rel=1e-3)

    def test_dc_limit_is_the_bar_resistance(self, cell, params):
        from phasesweep.models.im_bar_impedance import solve_bar_cell

        geo = params.geometry
        Z = solve_bar_cell(cell, params, 1e-3)
        assert Z.real == pytest.approx(1 / (_SIG * geo.bar_area), rel=1e-3)
        # slot leakage: the textbook h/(3w) term is a strict lower bound
        # (the bore condition adds the tooth-top flux across the gap)
        L_slot = MU0 * geo.bar_height / (3 * geo.bar_width)
        L = Z.imag / (2 * pi * 1e-3)
        assert L_slot < L < 1.3 * L_slot

    def test_converged_at_production_mesh(self, cell, params):
        """The bar and gap element sizes are fixed by the builder, so the
        global maxh barely moves the answer: 0.08 vs 0.02 within 1e-5."""
        from phasesweep.models.im_bar_impedance import solve_bar_cell
        from phasesweep.solvers.fem_mesh import build_bar_cell_mesh

        Z_coarse = solve_bar_cell(cell, params, 50.0)
        Z_fine = solve_bar_cell(build_bar_cell_mesh(params.geometry, 0.02), params, 50.0)
        assert abs(Z_fine - Z_coarse) / abs(Z_fine) < 1e-5

    def test_cell_matches_full_cage_with_flux_wall_bore(self, cell, params):
        """All eight bars of the F1 cage mesh carrying I·e^{-jαk} with the
        stator made a flux wall (ν → ∞, so A = 0 on the bore) is the same
        continuous problem as the Bloch cell: u_0/I_0 agrees to 1e-3, every
        bar sees the same impedance, and the α = 0 cell is measurably
        further away — the periodic phase is doing its job."""
        from phasesweep.models.im_bar_impedance import solve_bar_cell
        from phasesweep.solvers import fem_mesh
        from phasesweep.solvers.fem_harmonic import CageCircuit, HarmonicSpec, solve_harmonic

        geo = params.geometry
        fem_mesh.clear_mesh_cache()
        full = fem_mesh.get_or_build_cage_mesh(geo, 0.08)
        bars = tuple(f"bar{k}" for k in range(geo.n_bars))
        cur = tuple(complex(np.exp(-2j * pi * params.n_p * k / geo.n_bars))
                    for k in range(geo.n_bars))
        wall = 1e6 / MU0
        res = solve_harmonic(full, HarmonicSpec(
            omega=2 * pi * 50, nu={"rotor": 1 / (MU0 * 1000), "stator": wall, "slot.*": wall},
            sigma={b: _SIG for b in bars}, cage=CageCircuit(bars=bars, currents=cur)))
        Z_full = [res.u[b] / cur[k] for k, b in enumerate(bars)]
        assert max(abs(z - Z_full[0]) for z in Z_full) / abs(Z_full[0]) < 1e-3
        Z_cell = solve_bar_cell(cell, params, 50.0)
        err = abs(Z_cell - Z_full[0]) / abs(Z_full[0])
        assert err < 1e-3
        Z_inphase = solve_harmonic(cell, HarmonicSpec(
            omega=2 * pi * 50, nu={"rotor": 1 / (MU0 * 1000)}, sigma={"bar": _SIG},
            cage=CageCircuit(bars=("bar",), currents=(1.0,)), periodic_phase=1 + 0j)).u["bar"]
        assert abs(Z_inphase - Z_full[0]) / abs(Z_full[0]) > 3 * err

    def test_registry_run(self):
        m = make_cage_motor()
        info = MODEL_REGISTRY["im_bar_impedance"]
        out = info.fn(RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.08))
        assert set(out) == set(info.produces) | {"L_sigma_r_diff"}
        f, R, L, kR, kX, share = (np.array(out[k]) for k in (
            "f_rotor_list", "R_2_list", "L_sigma_r_list", "k_R_list",
            "k_x_list", "ring_share_list"))
        assert f[0] == 0.0 and R[0] == out["R_2_dc"] and L[0] == out["L_sigma_r_dc"]
        # Leakage permeance non-negativity (Layer 2 invariant 2b)
        assert np.all(R > 0) and np.all(L > 0)
        assert out["R_bar_dc"] > 0 and out["R_ring_seg"] > 0 and out["L_ring_seg"] > 0
        assert out["sigma_diff"] >= 0 and out["L_sigma_r_diff"] >= 0
        assert kR[0] == pytest.approx(1.0, rel=1e-3)   # meshed vs analytic bar area
        assert kX[0] == pytest.approx(1.0, rel=1e-9)
        assert np.all(np.diff(R) > 0) and np.all(np.diff(kR) >= -1e-6) and np.all(np.diff(L) < 0)
        assert np.all(np.diff(kX) <= 1e-6) and np.all(kX <= 1.0 + 1e-9)
        assert np.all((share > 0) & (share < 1)) and np.all(np.diff(share) < 0)
        R_seg, L_seg = ring_segment(m.geometry, _SIG)
        k_ref = referral_factor(40, 0.9, 8)
        assert out["R_2_dc"] == pytest.approx(
            k_ref * (out["R_bar_dc"] + R_seg * ring_factor(2, 8)), rel=1e-3)
        assert out["L_sigma_r_diff"] == pytest.approx(out["sigma_diff"] * 0.2)

    def test_no_diff_key_without_L_m(self):
        m = make_cage_motor(induction=dict(L_m=None))
        out = MODEL_REGISTRY["im_bar_impedance"].fn(
            RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.08))
        assert "L_sigma_r_diff" not in out

    @pytest.mark.parametrize("geo_kw, n_p, f_rated", [
        (dict(n_bars=8), 2, 50.0),
        (dict(n_bars=28), 2, 50.0),
        (dict(n_bars=40), 3, 50.0),
        # Deep bar: h/delta large enough to push k_R well above 1 and k_x
        # well below 1, exercising the monotonicity checks under real skin
        # depth variation rather than just changing the slot count.
        (dict(n_bars=8, bar_height=0.020, bar_width=0.003,
              r_inner=0.004), 2, 200.0),
    ], ids=["8bar-shallow", "28bar", "40bar", "8bar-deep"])
    def test_skin_effect_invariants_multiple_geometries(self, geo_kw, n_p, f_rated):
        """K_R >= 1 non-decreasing, K_x <= 1 non-increasing across cages."""
        m = make_cage_motor(geo_overrides=geo_kw, n_p=n_p,
                            induction=dict(f_rated=f_rated))
        out = MODEL_REGISTRY["im_bar_impedance"].fn(
            RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.08))
        kR = np.array(out["k_R_list"])
        kX = np.array(out["k_x_list"])
        assert np.all(kR >= 1.0 - 1e-3)
        assert np.all(kX <= 1.0 + 1e-9)
        assert kR[0] == pytest.approx(1.0, rel=1e-3)
        assert kX[0] == pytest.approx(1.0, rel=1e-9)
        assert np.all(np.diff(kR) >= -1e-6)
        assert np.all(np.diff(kX) <= 1e-6)


@requires_fem
class TestInvariantRejection:
    """Verify that the Layer 2 invariant checks fire on bad solver output."""

    def test_k_R_below_floor_rejected(self, monkeypatch):
        """k_R < 1 - 1e-3 raises."""
        from phasesweep.models import im_bar_impedance
        real_solve = im_bar_impedance.solve_bar_cell
        calls = [0]

        def fake_solve(mesh, p, f, order=3):
            calls[0] += 1
            Z = real_solve(mesh, p, f, order)
            if calls[0] == 2:
                return complex(Z.real * 0.01, Z.imag)
            return Z

        monkeypatch.setattr(im_bar_impedance, "solve_bar_cell", fake_solve)
        m = make_cage_motor()
        with pytest.raises(ValueError, match="cannot reduce bar resistance"):
            MODEL_REGISTRY["im_bar_impedance"].fn(
                RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.08))

    def test_k_x_above_ceiling_rejected(self, monkeypatch):
        """k_x > 1 + 1e-9 raises."""
        from phasesweep.models import im_bar_impedance
        real_solve = im_bar_impedance.solve_bar_cell
        calls = [0]

        def fake_solve(mesh, p, f, order=3):
            calls[0] += 1
            Z = real_solve(mesh, p, f, order)
            if calls[0] == 2:
                return complex(Z.real, Z.imag * 100)
            return Z

        monkeypatch.setattr(im_bar_impedance, "solve_bar_cell", fake_solve)
        m = make_cage_motor()
        with pytest.raises(ValueError, match="cannot increase bar"):
            MODEL_REGISTRY["im_bar_impedance"].fn(
                RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.08))

    def test_dc_leakage_zero_rejected(self, monkeypatch):
        """L_bar_dc = 0 from the FEM solve raises before division."""
        from phasesweep.models import im_bar_impedance
        real_solve = im_bar_impedance.solve_bar_cell
        calls = [0]

        def fake_solve(mesh, p, f, order=3):
            calls[0] += 1
            Z = real_solve(mesh, p, f, order)
            if calls[0] == 1:
                return complex(Z.real, 0.0)
            return Z

        monkeypatch.setattr(im_bar_impedance, "solve_bar_cell", fake_solve)
        m = make_cage_motor()
        with pytest.raises(ValueError, match="not positive and finite"):
            MODEL_REGISTRY["im_bar_impedance"].fn(
                RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.08))

    def test_negative_leakage_rejected(self, monkeypatch):
        """Negative Z_bar.imag from a sign error triggers an invariant —
        either k_x monotonicity or L_sigma_r non-negativity, depending on
        magnitude.  Any ValueError from the invariant block is correct."""
        from phasesweep.models import im_bar_impedance
        real_solve = im_bar_impedance.solve_bar_cell
        calls = [0]

        def fake_solve(mesh, p, f, order=3):
            calls[0] += 1
            Z = real_solve(mesh, p, f, order)
            if calls[0] == 2:
                return complex(Z.real, -abs(Z.imag) * 1e6)
            return Z

        monkeypatch.setattr(im_bar_impedance, "solve_bar_cell", fake_solve)
        m = make_cage_motor()
        with pytest.raises(ValueError):
            MODEL_REGISTRY["im_bar_impedance"].fn(
                RunConfig(motor=m, model="im_bar_impedance", maxh_fraction=0.08))


class TestGuards:
    """Input validation."""

    @pytest.mark.parametrize("n_p, n_bars", [(8, 8), (16, 8), (2, 2)])
    def test_degenerate_bloch_cell_rejected(self, n_p, n_bars):
        """n_p a multiple of n_bars makes the bar-to-bar phase 1, so one slot
        pitch is no longer a periodic cell — and 1/sin^2(pi n_p/n_bars) blows
        up to ~1e31 instead of failing."""
        m = make_cage_motor(geo_overrides=dict(n_bars=n_bars), n_p=n_p)
        with pytest.raises(ValueError, match="no longer a periodic cell"):
            prepare_im_bar_impedance(m)

    @pytest.mark.parametrize("n_p, n_bars", [(2, 28), (1, 8), (3, 40), (2, 8)])
    def test_ordinary_cages_still_accepted(self, n_p, n_bars):
        prepare_im_bar_impedance(
            make_cage_motor(geo_overrides=dict(n_bars=n_bars), n_p=n_p))

    @pytest.mark.parametrize("field", ["sigma_bar", "mu_r_fe", "L_stk", "k_w", "f_rated"])
    def test_nan_rejected(self, field):
        """NaN fails every comparison, so a bare `value <= 0` lets it past."""
        p = prepare_im_bar_impedance(make_cage_motor())
        with pytest.raises(ValueError, match="finite"):
            dc.replace(p, **{field: nan})

    def test_oversized_ring_section_rejected(self):
        """Alger's permeance is a log of a ratio; past its range it returns a
        negative leakage inductance rather than failing."""
        geo = inrunner(**{**_CAGE_KW, "end_ring_axial_length": 0.1,
                          "end_ring_radial_height": 0.03})
        with pytest.raises(ValueError, match="not positive"):
            ring_segment(geo, _SIG)

    def test_ring_deeper_than_the_rotor_rejected(self):
        """r_rotor 0.031 with a 0.062 m ring height drove the mean diameter to
        zero and crashed log10() inside the impedance model."""
        with pytest.raises(ValueError, match="less than r_rotor"):
            inrunner(**{**_CAGE_KW, "end_ring_radial_height": 0.062})
