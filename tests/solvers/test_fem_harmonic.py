"""Time-harmonic solver gates.

Discrete-bar gates first — TEAM-30a's rotor is a continuous conductor
and cannot exercise the per-bar constraint — then the exact benchmark.
"""

import csv
from importlib.util import find_spec
from math import pi

import numpy as np
import pytest

if not find_spec("ngsolve"):
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)

from phasesweep.machines.geometry import inrunner
from phasesweep.solvers import fem_mesh
from phasesweep.solvers import team30a as t30
from phasesweep.solvers.fem_harmonic import CageCircuit, HarmonicSpec, solve_harmonic
from phasesweep.solvers.fem_materials import MU0
from tests.conftest import REPO_ROOT, requires_fem

pytestmark = requires_fem

_TEAM30A = REPO_ROOT / "data" / "team30a"


def _table(name):
    with open(_TEAM30A / name) as f:
        return np.array([[float(v) for v in row] for row in list(csv.reader(f))[1:]])


def _rect(w, h, x0, y0, name):
    from netgen.occ import Rectangle

    f = Rectangle(w, h).Face().Move((x0, y0, 0))
    f.faces.name = name
    return f


def _mesh(faces, maxh, outer_edges):
    from netgen.occ import Glue, OCCGeometry
    from ngsolve import Mesh

    geo = Glue(faces)
    for sel in outer_edges:
        sel(geo.edges).name = "outer"
    return Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=maxh))


# ---------------------------------------------------------------------------
# (a) rectangular deep bar in a slot vs Field's skin-effect solution
# ---------------------------------------------------------------------------

_W, _H, _G, _SIG = 0.01, 0.02, 0.005, 3.5e7


@pytest.fixture(scope="module")
def slot_mesh():
    from netgen.occ import Y

    return _mesh([_rect(_W, _H, 0, 0, "bar"), _rect(_W, _G, 0, _H, "air")], 0.002,
                 [lambda e: e.Max(Y)])


def _field_factors(xi):
    kR = xi * (np.sinh(2 * xi) + np.sin(2 * xi)) / (np.cosh(2 * xi) - np.cos(2 * xi))
    kL = 3 / (2 * xi) * (np.sinh(2 * xi) - np.sin(2 * xi)) / (np.cosh(2 * xi) - np.cos(2 * xi))
    return kR, kL


class TestDeepBar:
    """Slot walls and bottom are Neumann (infinitely permeable iron), the
    slot opening carries the return flux: the 2D problem reduces to
    Field's 1D deep-bar solution, exact up to discretisation."""

    @pytest.mark.parametrize("f_hz", [1.0, 20.0, 50.0, 200.0])
    def test_ac_resistance_and_inductance(self, slot_mesh, f_hz):
        omega = 2 * pi * f_hz
        I_set = 1000.0
        spec = HarmonicSpec(omega=omega, sigma={"bar": _SIG},
                            cage=CageCircuit(bars=("bar",), currents=(I_set,)))
        res = solve_harmonic(slot_mesh, spec)
        Z = res.u["bar"] / I_set
        delta = np.sqrt(2 / (omega * MU0 * _SIG))
        kR, kL = _field_factors(_H / delta)
        R_dc, L_dc, L_air = 1 / (_SIG * _W * _H), MU0 * _H / (3 * _W), MU0 * _G / _W
        assert Z.real / R_dc == pytest.approx(kR, rel=1e-4)
        assert (Z.imag / omega - L_air) / L_dc == pytest.approx(kL, rel=1e-4)
        assert res.I_bar["bar"] == pytest.approx(I_set, abs=1e-6)
        assert res.loss("bar") == pytest.approx(0.5 * Z.real * I_set**2, rel=1e-9)

    def test_J_on_a_non_eddy_region_raises_valueerror(self, slot_mesh):
        """The friendly message sat behind an unguarded spec.sigma[region]
        lookup, so the caller saw a bare KeyError instead."""
        spec = HarmonicSpec(omega=2 * pi * 50.0, sigma={"bar": _SIG},
                            cage=CageCircuit(bars=("bar",), currents=(1.0,)))
        res = solve_harmonic(slot_mesh, spec)
        with pytest.raises(ValueError, match="is not an eddy region"):
            res.J("air")

    def test_cage_needs_conductivity(self):
        with pytest.raises(ValueError, match="no conductivity"):
            HarmonicSpec(omega=1.0, cage=CageCircuit(bars=("bar",)))

    def test_circuit_validation(self):
        with pytest.raises(ValueError, match="ideal ring"):
            CageCircuit(bars=("a", "b"), Z_ring=0)
        with pytest.raises(ValueError, match="exclusive"):
            CageCircuit(bars=("a", "b"), currents=(1, -1), Z_ring=1)
        with pytest.raises(ValueError, match="at least two"):
            CageCircuit(bars=("a",), Z_ring=1)
        with pytest.raises(ValueError, match="prescribed currents for"):
            CageCircuit(bars=("a", "b"), currents=(1,))

    @pytest.mark.parametrize("kw, msg", [
        (dict(omega=float("nan")), "finite"),
        (dict(omega=float("inf")), "finite"),
        (dict(sigma={"bar": float("nan")}), "finite"),
        (dict(sigma={"bar": float("inf")}), "finite"),
        (dict(nu={"iron": float("nan")}), "finite"),
        (dict(nu={"iron": float("inf")}), "finite"),
        (dict(slip=float("nan")), "finite"),
        (dict(omega_mech=float("nan")), "finite"),
        (dict(omega_mech=float("inf")), "finite"),
    ])
    def test_nan_inf_spec_fields_rejected(self, kw, msg):
        base = dict(omega=1.0, sigma={"bar": _SIG})
        base.update(kw)
        with pytest.raises(ValueError, match=msg):
            HarmonicSpec(**base)

    def test_nan_cage_L_stk_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            CageCircuit(bars=("a",), L_stk=float("nan"))

    def test_slip_greater_than_one_accepted(self):
        spec = HarmonicSpec(omega=1.0, sigma={"bar": _SIG}, slip=1.5)
        assert spec.slip == 1.5

    def test_negative_slip_accepted(self):
        spec = HarmonicSpec(omega=1.0, sigma={"bar": _SIG}, slip=-0.1)
        assert spec.slip == -0.1

    def test_negative_omega_mech_accepted(self):
        spec = HarmonicSpec(omega=1.0, omega_mech=-100.0)
        assert spec.omega_mech == -100.0

    @pytest.mark.parametrize("val", [float("-inf"), float("inf")])
    def test_slip_inf_rejected(self, val):
        with pytest.raises(ValueError, match="finite"):
            HarmonicSpec(omega=1.0, slip=val)


# ---------------------------------------------------------------------------
# (b) bars closed through a nonzero end-ring impedance
# ---------------------------------------------------------------------------

_BAR_S = 0.006
_BARS = ("bar0", "bar1", "bar2")
_L_STK = 0.3
_SIG_CU = 5.8e7


@pytest.fixture(scope="module")
def ring_mesh():
    from netgen.occ import X, Y

    box = _rect(0.24, 0.24, -0.12, -0.12, "air")
    coil = _rect(0.10, 0.02, -0.05, 0.05, "coil")
    bars = [_rect(_BAR_S, _BAR_S, x0 - _BAR_S / 2, -_BAR_S / 2, n)
            for x0, n in zip((-0.04, 0.0, 0.04), _BARS)]
    air = box
    for f in [coil, *bars]:
        air = air - f
    air.faces.name = "air"
    return _mesh([air, coil, *bars], 0.01,
                 [lambda e: e.Max(X), lambda e: e.Min(X), lambda e: e.Max(Y), lambda e: e.Min(Y)])


def _ring_solve(mesh, Z_ring, slip=1.0):
    spec = HarmonicSpec(omega=2 * pi * 50, sigma={b: _SIG_CU for b in _BARS},
                        J_source={"coil": 2e6}, slip=slip,
                        cage=CageCircuit(bars=_BARS, L_stk=_L_STK, Z_ring=Z_ring))
    return solve_harmonic(mesh, spec)


def _bar_avg_A(res, name):
    from ngsolve import CoefficientFunction, Integrate

    area = Integrate(CoefficientFunction(1), res.mesh, definedon=res.mesh.Materials(name))
    return complex(Integrate(res.A, res.mesh, definedon=res.mesh.Materials(name))) / area


class TestEndRing:
    """Complex bar currents satisfy the bar/ring circuit exactly:
    I_k = σS(u_k − jsω⟨A⟩_k) per bar (Ohm's law on the bar) and
    I_k = L(u_{k+1} − 2u_k + u_{k−1})/(2 Z_r) (KCL at the ring nodes
    with KVL around each adjacent-bar loop), and they move when Z_r
    moves — a ΣI = 0 check alone would pass with the coupling absent."""

    _Z = 5e-5 + 2e-5j

    def test_bar_ohm_law_and_ring_kirchhoff(self, ring_mesh):
        res = _ring_solve(ring_mesh, self._Z)
        sw = res.spec.slip * res.spec.omega
        I_bar = [res.I_bar[b] for b in _BARS]
        u = [res.u[b] for b in _BARS]
        scale = max(abs(i) for i in I_bar)
        assert scale > 1.0
        for k, b in enumerate(_BARS):
            ohm = _SIG_CU * _BAR_S**2 * (u[k] - 1j * sw * _bar_avg_A(res, b))
            assert abs(I_bar[k] - ohm) / scale < 2e-3   # area of the meshed square
            n = len(_BARS)
            ring = _L_STK * (u[(k + 1) % n] - 2 * u[k] + u[(k - 1) % n]) / (2 * self._Z)
            assert abs(I_bar[k] - ring) / scale < 1e-8
        assert abs(sum(I_bar)) / scale < 1e-8

    def test_currents_move_with_ring_impedance(self, ring_mesh):
        I_a = np.array([_ring_solve(ring_mesh, self._Z).I_bar[b] for b in _BARS])
        I_b = np.array([_ring_solve(ring_mesh, 4 * self._Z).I_bar[b] for b in _BARS])
        assert np.max(np.abs(I_a - I_b)) / np.max(np.abs(I_a)) > 0.05

    def test_ring_loss_closes_the_power_balance(self, ring_mesh):
        """No motion, no other conductors: the coil's input power is the
        bar Joule loss plus the ring I²R that loss() does not carry.
        input_power() and loss() are per metre of depth; ring_loss() is
        the whole stack, hence the L_stk."""
        res = _ring_solve(ring_mesh, self._Z)
        assert res.ring_loss() > 0.01 * res.loss(_BARS) * _L_STK
        assert res.input_power() * _L_STK == pytest.approx(
            res.loss(_BARS) * _L_STK + res.ring_loss(), rel=1e-6)
        ideal = solve_harmonic(ring_mesh, HarmonicSpec(
            omega=2 * pi * 50, sigma={b: _SIG_CU for b in _BARS}, J_source={"coil": 2e6},
            cage=CageCircuit(bars=_BARS, L_stk=_L_STK)))
        with pytest.raises(ValueError, match="Z_ring"):
            ideal.ring_loss()

    def test_ideal_ring_limit(self, ring_mesh):
        """Z_r → 0 tends to the ideal ring (common u, ΣI = 0)."""
        ideal = solve_harmonic(ring_mesh, HarmonicSpec(
            omega=2 * pi * 50, sigma={b: _SIG_CU for b in _BARS}, J_source={"coil": 2e6},
            cage=CageCircuit(bars=_BARS, L_stk=_L_STK)))
        assert len({round(abs(v), 12) for v in ideal.u.values()}) == 1
        assert abs(sum(ideal.I_bar.values())) / max(abs(v) for v in ideal.I_bar.values()) < 1e-8
        small = _ring_solve(ring_mesh, 1e-9 * self._Z)
        for b in _BARS:
            assert small.I_bar[b] == pytest.approx(ideal.I_bar[b], rel=1e-4)


# ---------------------------------------------------------------------------
# Cage mesh (F1 path): power balance per slip point
# ---------------------------------------------------------------------------

_CAGE_KW = dict(r_outer=0.05, r_stator=0.032, r_magnet=0.031, r_rotor=0.031,
                r_inner=0.011, n_bars=8, bar_width=0.0034, bar_height=0.01,
                n_slots=12, slot_depth=0.008)


class TestCagePowerBalance:
    """Slot source J = J0 e^{-j n_p θ} at supply frequency, bars at slip
    frequency through an ideal ring: bar loss = s · input power holds
    exactly for the discrete model (every space harmonic of the source
    is treated at the same slip), and the Poynting flux through the air
    gap returns the input power. The Maxwell-stress torque is NOT gated
    against n_p·P_in/ω here: the slot comb puts ~60% of the fundamental
    current amplitude into the 2 ± 12k space harmonics, and at a single
    slip their torque share moves the total by ~8% on this mesh — the
    signature of the single-slip model that F4 must document."""

    @pytest.fixture(scope="class")
    def cage(self):
        fem_mesh.clear_mesh_cache()
        geo = inrunner(**_CAGE_KW)
        return geo, fem_mesh.get_or_build_cage_mesh(geo, 0.08)

    @pytest.mark.parametrize("slip", [1.0, 0.05])
    def test_bar_loss_is_slip_times_input(self, cage, slip):
        from ngsolve import atan2, cos, sin, x, y

        geo, mesh = cage
        n_p, omega = 2, 2 * pi * 50
        theta = atan2(y, x)
        J = 3e6 * (cos(n_p * theta) - 1j * sin(n_p * theta))
        bars = tuple(f"bar{k}" for k in range(geo.n_bars))
        spec = HarmonicSpec(omega=omega, nu={"stator": 1 / (1000 * MU0), "rotor": 1 / (1000 * MU0)},
                            sigma={b: 3e7 for b in bars}, J_source={"slot.*": J}, slip=slip,
                            cage=CageCircuit(bars=bars))
        res = solve_harmonic(mesh, spec)
        P_in = res.input_power()
        P_bar = res.loss(bars)
        assert P_in > 0
        assert P_bar == pytest.approx(slip * P_in, rel=1e-6)
        assert res.gap_power(geo.r_ag, n_theta=720) == pytest.approx(P_in, rel=1e-2)
        assert abs(sum(res.I_bar.values())) / max(abs(v) for v in res.I_bar.values()) < 1e-8


# ---------------------------------------------------------------------------
# TEAM Problem 30a — exact benchmark
# ---------------------------------------------------------------------------

_KEYS = ("torque", "voltage", "rotor_loss", "steel_loss")


def _rows(table):
    return [(row[0], dict(zip(_KEYS, row[1:]))) for row in table]


class TestTeam30aAnalytic:
    """The in-repo series solution reproduces the published tables; this
    is what pins the rms-phasor and open-exterior conventions."""

    def test_field_profile_past_the_winding_is_rejected(self):
        """Beyond R4 the coefficients used are the winding layer's (mu_0, with
        a source term) while the stator steel is mu_r = 30; the clamp used to
        extrapolate silently."""
        with pytest.raises(ValueError, match="winding outer radius"):
            t30.analytic_field_on_axis(200.0, np.array([t30.R4 + 1e-3]))

    def test_fe_field_profile_past_the_winding_is_rejected(self):
        """The FE twin clamped r > R4 to R4 - eps and returned the interface
        value; it now matches the analytic guard. Checked before any mesh
        access, so no solve is needed."""
        with pytest.raises(ValueError, match="winding side of R4"):
            t30.fe_field_on_axis(None, np.array([t30.R4 + 1e-3]))

    def test_field_profile_up_to_R4_is_accepted(self):
        B_r, _ = t30.analytic_field_on_axis(200.0, np.array([t30.R3, t30.R4]))
        assert np.all(np.isfinite(B_r))

    @pytest.mark.parametrize("speed, ref", _rows(_table("table1_three_phase.csv")))
    def test_three_phase_table1(self, speed, ref):
        out = t30.analytic_team30a(True, speed)
        for k in _KEYS:
            assert out[k] == pytest.approx(ref[k], rel=5e-4)

    @pytest.mark.parametrize("speed, ref", _rows(_table("table3_single_phase.csv")))
    def test_single_phase_table3(self, speed, ref):
        out = t30.analytic_team30a(False, speed)
        for k in ("voltage", "rotor_loss", "steel_loss"):
            assert out[k] == pytest.approx(ref[k], rel=5e-4)
        # The published 39.79 rad/s torque (0.0528) sits 7% off the
        # series value (0.0492); every other row agrees to 5e-4. Pinned
        # as an absolute band there, not a loosened relative one.
        assert out["torque"] == pytest.approx(ref["torque"], rel=5e-4, abs=4e-3)

    def test_field_profile_table2(self):
        tab = _table("table2_field_profile_200rad_s.csv")
        B_r, H_t = t30.analytic_field_on_axis(200.0, tab[:, 0])
        ref_B, ref_H = tab[:, 1] + 1j * tab[:, 2], tab[:, 3] + 1j * tab[:, 4]
        assert np.max(np.abs(B_r - ref_B) / np.abs(ref_B)) < 5e-3
        assert np.max(np.abs(H_t - ref_H) / np.abs(ref_H)) < 5e-3

    def test_open_exterior_is_the_published_case(self):
        """A = 0 at the stator surface would cut torque and loss in half
        of the published values' neighbourhood; the far-field truncation
        used by the FE mesh costs < 0.05%."""
        open_ = t30.analytic_team30a(True, 0.0)
        closed = t30.analytic_team30a(True, 0.0, r_far=t30.R5)
        far = t30.analytic_team30a(True, 0.0, r_far=20 * t30.R5)
        assert closed["torque"] < 0.95 * open_["torque"]
        assert far["torque"] == pytest.approx(open_["torque"], rel=5e-4)


@pytest.fixture(scope="module")
def mesh_three():
    return t30.build_team30a_mesh(True)


@pytest.fixture(scope="module")
def mesh_single():
    return t30.build_team30a_mesh(False)


class TestTeam30aFem:
    @pytest.mark.parametrize("speed", t30.SPEEDS_THREE_PHASE)
    def test_three_phase_vs_exact_and_published(self, mesh_three, speed):
        fe = t30.solve_team30a(mesh_three, True, speed)
        exact = t30.analytic_team30a(True, speed, r_far=20 * t30.R5)
        # 5e-3: the rotor-steel loss near synchronism (400 rad/s) is
        # 100x smaller than elsewhere, so its relative discretisation
        # error is the largest of the set (0.43% at maxh 3 mm).
        for k in _KEYS:
            assert fe[k] == pytest.approx(exact[k], rel=5e-3)
        ref = dict(zip(_KEYS, _table("table1_three_phase.csv")[
            list(t30.SPEEDS_THREE_PHASE).index(speed)][1:]))
        for k in _KEYS:
            assert fe[k] == pytest.approx(ref[k], rel=5e-3)
        # energy balance: source power = air-gap Poynting flux = rotor
        # loss + mechanical power (exact for the velocity-term model)
        assert fe["input_power"] == pytest.approx(
            fe["rotor_loss"] + fe["torque"] * speed, rel=1e-4, abs=0.05)
        assert fe["result"].gap_power(t30.R_TORQUE, n_theta=1440) == pytest.approx(
            fe["input_power"], rel=1e-5)

    @pytest.mark.parametrize("speed", t30.SPEEDS_SINGLE_PHASE[::3])
    def test_single_phase_vs_exact(self, mesh_single, speed):
        fe = t30.solve_team30a(mesh_single, False, speed)
        exact = t30.analytic_team30a(False, speed, r_far=20 * t30.R5)
        for k in ("voltage", "rotor_loss", "steel_loss"):
            assert fe[k] == pytest.approx(exact[k], rel=4e-3)
        assert fe["torque"] == pytest.approx(exact["torque"], rel=4e-3, abs=2e-4)

    def test_field_profile_table2(self, mesh_three):
        tab = _table("table2_field_profile_200rad_s.csv")
        fe = t30.solve_team30a(mesh_three, True, 200.0)
        B_r, H_t = t30.fe_field_on_axis(fe["result"], tab[:, 0])
        ref_B, ref_H = tab[:, 1] + 1j * tab[:, 2], tab[:, 3] + 1j * tab[:, 4]
        assert np.max(np.abs(B_r - ref_B) / np.abs(ref_B)) < 1e-2
        # The last point sits on the winding/stator interface (a 30x
        # permeability jump); the FE derivative there carries ~2% while
        # the interior points sit within 0.1%.
        assert np.max(np.abs(H_t - ref_H) / np.abs(ref_H)) < 2.5e-2

    def test_mesh_convergence(self):
        """One convergence point: halving the mesh
        size shrinks the torque error against the exact truncated-domain
        solution by more than 2x."""
        exact = t30.analytic_team30a(True, 0.0, r_far=20 * t30.R5)["torque"]
        errs = []
        for maxh in (0.006, 0.003):
            fe = t30.solve_team30a(t30.build_team30a_mesh(True, maxh=maxh), True, 0.0)
            errs.append(abs(fe["torque"] - exact) / exact)
        assert errs[1] < errs[0] / 2
        assert errs[1] < 4e-3
