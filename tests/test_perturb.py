"""Tests for perturb_motor perturbation physics.

Coverage targets from dynamic audit:
- All 5 params × both topologies × multiple deltas
- Guard clauses (None, negative, geometry rejection, k_w bounds)
- psi_f clearing (physics: ψ_f depends on geometry through Φ₁)
- delta=0 identity (documents psi_f clearing asymmetry)
- None-inductance motors (Deylami/Belkhadir/LRK fleet pattern)
- Geometry field forwarding across both topologies
"""

import dataclasses

import pytest

from phasesweep.geometry import Geometry, default_inrunner, inrunner, outrunner
from phasesweep.motor import Motor
from phasesweep.perturbation import perturb_motor as _perturb_motor

# ---------------------------------------------------------------------------
# Motor factories
# ---------------------------------------------------------------------------

def _make_inrunner(**kw) -> Motor:
    defaults = dict(
        name="test_in", geometry=default_inrunner(),
        n_p=2, R_s=0.2, L_d=4e-3, L_q=6e-3, psi_f=None, B_rem=0.5,
        J=0.002, N=50, k_w=0.966, L_stk=0.10,
    )
    defaults.update(kw)
    return Motor(**defaults)


def _make_outrunner(**kw) -> Motor:
    geo = outrunner(
        r_outer=0.060, r_rotor=0.048, r_magnet=0.040,
        r_stator=0.030, r_inner=0.020,
    )
    defaults = dict(
        name="test_out", geometry=geo,
        n_p=4, R_s=0.3, L_d=2e-3, L_q=3e-3, psi_f=None, B_rem=1.2,
        J=0.001, N=22, k_w=0.949, L_stk=0.035,
    )
    defaults.update(kw)
    return Motor(**defaults)


def _make_slotted_inrunner() -> Motor:
    geo = inrunner(
        r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
        n_slots=12, slot_depth=0.005, slot_width_ratio=0.5,
        slot_opening_width=0.00754,  # ratio ~0.3 of slot pitch
    )
    return Motor(name="slot_in", geometry=geo, n_p=4, B_rem=1.2,
                 L_d=4e-3, L_q=6e-3, L_stk=0.05)


def _make_slotted_outrunner() -> Motor:
    geo = outrunner(
        r_outer=0.060, r_rotor=0.048, r_magnet=0.040,
        r_stator=0.030, r_inner=0.020,
        n_slots=12, slot_depth=0.005, slot_width_ratio=0.5,
        slot_opening_width=0.00471,  # ratio ~0.3 of slot pitch
    )
    return Motor(name="slot_out", geometry=geo, n_p=4, B_rem=1.2,
                 L_d=2e-3, L_q=3e-3, L_stk=0.035)


# ---------------------------------------------------------------------------
# OD perturbation
# ---------------------------------------------------------------------------

class TestOdPerturbation:
    """OD frame-size: stator+rotor shift together, gap/magnet fixed, L ∝ r."""

    @pytest.mark.parametrize("make_motor,gap_fn", [
        (_make_inrunner, lambda g: g.r_stator - g.r_magnet),
        (_make_outrunner, lambda g: g.r_magnet - g.r_stator),
    ], ids=["inrunner", "outrunner"])
    @pytest.mark.parametrize("delta", [0.05, 0.10, 0.20])
    def test_gap_preserved(self, make_motor, gap_fn, delta):
        m = make_motor()
        p = _perturb_motor(m, "OD", delta)
        assert p is not None
        assert gap_fn(p.geometry) == pytest.approx(gap_fn(m.geometry))

    def test_inrunner_magnet_thickness_preserved(self):
        """Inrunner OD: magnet thickness (r_magnet - r_rotor) stays fixed."""
        m = _make_inrunner()
        p = _perturb_motor(m, "OD", 0.10)
        mag_orig = m.geometry.r_magnet - m.geometry.r_rotor
        mag_new = p.geometry.r_magnet - p.geometry.r_rotor
        assert mag_new == pytest.approx(mag_orig)

    @pytest.mark.parametrize("make_motor", [_make_inrunner, _make_outrunner],
                             ids=["inrunner", "outrunner"])
    @pytest.mark.parametrize("delta", [0.05, 0.10])
    def test_scales_inductances(self, make_motor, delta):
        m = make_motor()
        p = _perturb_motor(m, "OD", delta)
        assert p is not None
        shift = m.geometry.r_outer * delta
        ratio = (m.geometry.r_stator + shift) / m.geometry.r_stator
        assert p.L_d == pytest.approx(m.L_d * ratio)
        assert p.L_q == pytest.approx(m.L_q * ratio)

    def test_negative_delta_rejected_if_rotor_hits_shaft(self):
        m = _make_inrunner()
        assert _perturb_motor(m, "OD", -0.90) is None

    def test_outrunner_negative_delta_rejected_if_stator_hits_inner(self):
        m = _make_outrunner()
        assert _perturb_motor(m, "OD", -0.90) is None


# ---------------------------------------------------------------------------
# Gap perturbation
# ---------------------------------------------------------------------------

class TestGapPerturbation:
    """Gap perturbation scales L ~ 1/gap."""

    @pytest.mark.parametrize("make_motor", [_make_inrunner, _make_outrunner],
                             ids=["inrunner", "outrunner"])
    @pytest.mark.parametrize("delta", [-0.10, 0.05, 0.10])
    def test_scales_inductances_inversely(self, make_motor, delta):
        m = make_motor()
        p = _perturb_motor(m, "gap", delta)
        assert p is not None
        assert p.L_d == pytest.approx(m.L_d / (1.0 + delta))
        assert p.L_q == pytest.approx(m.L_q / (1.0 + delta))

    def test_outrunner_geometry_correct(self):
        """Outrunner gap: r_magnet moves outward, r_stator and r_rotor fixed.

        Magnet thickness (r_rotor - r_magnet) changes — the magnet absorbs
        the gap change, not the yoke.
        """
        m = _make_outrunner()
        p = _perturb_motor(m, "gap", 0.10)
        assert p is not None
        assert p.geometry.r_stator == m.geometry.r_stator
        assert p.geometry.r_rotor == m.geometry.r_rotor
        assert p.geometry.r_magnet > m.geometry.r_magnet
        # Magnet thickness shrinks (r_magnet moved toward r_rotor)
        orig_mag = m.geometry.r_rotor - m.geometry.r_magnet
        new_mag = p.geometry.r_rotor - p.geometry.r_magnet
        assert new_mag < orig_mag

    def test_inrunner_rejection_stator_exceeds_outer(self):
        m = _make_inrunner()
        assert _perturb_motor(m, "gap", 100.0) is None

    def test_outrunner_rejection_magnet_exceeds_rotor(self):
        m = _make_outrunner()
        assert _perturb_motor(m, "gap", 100.0) is None


# ---------------------------------------------------------------------------
# L_stk perturbation
# ---------------------------------------------------------------------------

class TestLstkPerturbation:
    """L_stk perturbation scales L_d/L_q and R_s proportionally."""

    def test_scales_r_s(self):
        m = _make_inrunner(R_s=0.2)
        p = _perturb_motor(m, "L_stk", 0.10)
        assert p is not None
        assert p.R_s == pytest.approx(0.2 * 1.10)

    def test_negative_scales_r_s(self):
        m = _make_inrunner(R_s=0.2)
        p = _perturb_motor(m, "L_stk", -0.05)
        assert p is not None
        assert p.R_s == pytest.approx(0.2 * 0.95)

    def test_none_r_s_ok(self):
        m = _make_inrunner(R_s=None)
        p = _perturb_motor(m, "L_stk", 0.10)
        assert p is not None
        assert p.R_s is None

    def test_still_scales_inductances(self):
        m = _make_inrunner()
        p = _perturb_motor(m, "L_stk", 0.10)
        assert p is not None
        assert p.L_d == pytest.approx(m.L_d * 1.10)
        assert p.L_q == pytest.approx(m.L_q * 1.10)


# ---------------------------------------------------------------------------
# B_rem perturbation
# ---------------------------------------------------------------------------

class TestBremPerturbation:
    """B_rem perturbation must NOT change inductances."""

    def test_preserves_inductances(self):
        m = _make_inrunner()
        p = _perturb_motor(m, "B_rem", 0.10)
        assert p is not None
        assert p.L_d == m.L_d
        assert p.L_q == m.L_q


# ---------------------------------------------------------------------------
# k_w perturbation
# ---------------------------------------------------------------------------

class TestKwPerturbation:
    """k_w perturbation: psi_f cleared, inductances preserved, bounds [0.75, 1.0]."""

    @pytest.mark.parametrize("make_motor", [_make_inrunner, _make_outrunner],
                             ids=["inrunner", "outrunner"])
    def test_within_bounds(self, make_motor):
        m = make_motor()
        p = _perturb_motor(m, "k_w", -0.025)
        assert p is not None
        assert p.k_w == pytest.approx(m.k_w * 0.975)

    def test_preserves_inductances(self):
        m = _make_inrunner()
        p = _perturb_motor(m, "k_w", -0.05)
        assert p is not None
        assert p.L_d == m.L_d
        assert p.L_q == m.L_q

    def test_k_w_none_returns_none(self):
        m = _make_inrunner(k_w=None)
        assert _perturb_motor(m, "k_w", -0.05) is None

    def test_above_upper_bound(self):
        m = _make_inrunner(k_w=0.966)
        assert _perturb_motor(m, "k_w", 0.05) is None  # 1.014 > 1.0

    def test_below_lower_bound(self):
        m = _make_inrunner(k_w=0.80)
        assert _perturb_motor(m, "k_w", -0.10) is None  # 0.72 < 0.75


# ---------------------------------------------------------------------------
# psi_f clearing
# ---------------------------------------------------------------------------

class TestPsiFCleared:
    """ψ_f must be cleared on every perturbation.

    ψ_f = N·k_w·Φ₁ depends on geometry (OD/gap change bore area and
    reluctance), stack length (Φ₁ ∝ L_stk), and remanence (B₁ ∝ B_rem).
    Stale ψ_f would cause _resolve_psi_f to use the old value instead of
    re-deriving from the new geometry.
    """

    @pytest.mark.parametrize("param,delta", [
        ("OD", 0.05), ("gap", 0.05), ("L_stk", 0.05), ("B_rem", 0.05),
        ("k_w", -0.05),  # negative: k_w=0.966 has no headroom above 1.0
    ])
    def test_psi_f_cleared(self, param, delta):
        m = _make_inrunner(psi_f=0.1)
        p = _perturb_motor(m, param, delta)
        assert p is not None
        assert p.psi_f is None


# ---------------------------------------------------------------------------
# Delta=0 identity
# ---------------------------------------------------------------------------

class TestDeltaZero:
    """Zero perturbation preserves geometry and electrical params.

    Note: psi_f is still cleared — perturb_motor is not an identity at
    delta=0.  The sensitivity analysis handles delta=0 as a special case
    (validation_report.py:620-622) and never calls perturb_motor for it.

    Uses introspection so new Motor fields are covered automatically.
    """

    @pytest.mark.parametrize("param", ["OD", "gap", "L_stk", "B_rem", "k_w"])
    def test_all_fields_preserved(self, param):
        m = _make_inrunner()
        p = _perturb_motor(m, param, 0.0)
        assert p is not None
        skip = {"name", "psi_f"}
        for f in dataclasses.fields(Motor):
            if f.name in skip:
                continue
            orig = getattr(m, f.name)
            pert = getattr(p, f.name)
            assert pert == orig, f"{param}: {f.name} changed from {orig} to {pert}"
        assert p.psi_f is None  # intentional — documents the asymmetry


# ---------------------------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------------------------

class TestGuardClauses:

    def test_l_stk_none_returns_none(self):
        m = _make_inrunner(L_stk=None)
        assert _perturb_motor(m, "L_stk", 0.10) is None

    def test_l_stk_negative_result_returns_none(self):
        m = _make_inrunner(L_stk=0.01)
        assert _perturb_motor(m, "L_stk", -2.0) is None

    def test_b_rem_none_returns_none(self):
        m = _make_inrunner(B_rem=None)
        assert _perturb_motor(m, "B_rem", 0.10) is None

    def test_b_rem_negative_result_returns_none(self):
        m = _make_inrunner()
        assert _perturb_motor(m, "B_rem", -2.0) is None

    def test_gap_inrunner_stator_exceeds_outer(self):
        m = _make_inrunner()
        assert _perturb_motor(m, "gap", 100.0) is None

    def test_gap_outrunner_magnet_exceeds_rotor(self):
        m = _make_outrunner()
        assert _perturb_motor(m, "gap", 100.0) is None


# ---------------------------------------------------------------------------
# None inductances (Deylami/Belkhadir/LRK fleet pattern)
# ---------------------------------------------------------------------------

class TestNoneInductances:
    """Motors without L_d/L_q must not gain inductance values after perturbation."""

    @pytest.mark.parametrize("param,delta", [
        ("OD", 0.05), ("gap", 0.05), ("L_stk", 0.05), ("B_rem", 0.05),
        ("k_w", -0.05),
    ])
    def test_none_inductances_stay_none(self, param, delta):
        m = _make_inrunner(L_d=None, L_q=None)
        p = _perturb_motor(m, param, delta)
        if p is not None:
            assert p.L_d is None
            assert p.L_q is None


# ---------------------------------------------------------------------------
# Geometry field forwarding (regression)
# ---------------------------------------------------------------------------

class TestGeometryFieldForwarding:
    """perturb_motor must forward every Geometry field through builder calls.

    This test introspects Geometry's fields and verifies non-radii fields
    survive perturbation. If a new field is added to Geometry without being
    forwarded in perturbation.py, this test fails.
    """

    @pytest.mark.parametrize("make_motor", [
        _make_slotted_inrunner, _make_slotted_outrunner,
    ], ids=["inrunner", "outrunner"])
    @pytest.mark.parametrize("param", ["OD", "gap"])
    def test_all_fields_preserved(self, make_motor, param):
        geo_fields = {f.name for f in dataclasses.fields(Geometry)} - {"topology"}
        radii = {"r_outer", "r_stator", "r_magnet", "r_rotor", "r_ag"}
        # slot_opening_width scales with the bore so slot_opening_ratio is
        # invariant; checked separately below
        radii.add("slot_opening_width")
        m = make_motor()
        p = _perturb_motor(m, param, 0.05)
        assert p is not None
        for field in geo_fields - radii:
            orig = getattr(m.geometry, field)
            pert = getattr(p.geometry, field)
            assert pert == orig, f"{param}: {field} changed from {orig} to {pert}"
        assert p.geometry.slot_opening_ratio == pytest.approx(
            m.geometry.slot_opening_ratio, rel=1e-12)
