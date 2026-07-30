"""Tests for geometry dimension sweeps."""

import pytest

from phasesweep.geo_sweep import (
    SweepAxis,
    SweepPoint,
    SweepResult,
    generate_grid,
    run_sweep,
)
from phasesweep.geometry import default_inrunner
from phasesweep.motor import Motor

# ---------------------------------------------------------------------------
# SweepAxis
# ---------------------------------------------------------------------------

class TestSweepAxis:

    def test_values_linspace(self):
        a = SweepAxis("r_outer", 0.04, 0.06, 3)
        vals = a.values()
        assert len(vals) == 3
        assert vals[0] == pytest.approx(0.04)
        assert vals[-1] == pytest.approx(0.06)

    def test_rejects_less_than_2_steps(self):
        with pytest.raises(ValueError, match="steps"):
            SweepAxis("r_outer", 0.04, 0.06, 1)

    def test_rejects_negative_start(self):
        with pytest.raises(ValueError, match="positive"):
            SweepAxis("r_outer", -0.01, 0.06, 3)

    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError, match="unknown sweep field 'r_magnet'"):
            SweepAxis("r_magnet", 0.04, 0.06, 3)

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="unknown sweep strategy"):
            SweepAxis("r_outer", 0.04, 0.06, 3, strategy="linear")


# ---------------------------------------------------------------------------
# Proportional scaling
# ---------------------------------------------------------------------------

class TestProportionalScaling:

    def test_r_outer_sweep_scales_all_radii(self):
        base = default_inrunner()
        axis = SweepAxis("r_outer", base.r_outer * 0.5, base.r_outer * 2.0, 3)
        points = generate_grid(base, [axis])
        assert len(points) == 3

        half = points[0].geometry
        assert half.r_outer == pytest.approx(base.r_outer * 0.5)
        assert half.r_stator == pytest.approx(base.r_stator * 0.5)
        assert half.r_magnet == pytest.approx(base.r_magnet * 0.5)

    def test_proportional_preserves_ratios(self):
        base = default_inrunner()
        axis = SweepAxis("r_outer", 0.5, 1.5, 5)
        points = generate_grid(base, [axis])
        ratio = base.r_stator / base.r_outer
        for pt in points:
            assert pt.geometry.r_stator / pt.geometry.r_outer == pytest.approx(ratio)


# ---------------------------------------------------------------------------
# Fixed-gap scaling
# ---------------------------------------------------------------------------

class TestFixedGapScaling:

    def test_r_outer_fixed_gap_preserves_airgap(self):
        base = default_inrunner()
        original_gap = base.r_stator - base.r_magnet
        axis = SweepAxis("r_outer", 0.8, 1.2, 3, strategy="fixed_gap")
        points = generate_grid(base, [axis])
        for pt in points:
            gap = pt.geometry.r_stator - pt.geometry.r_magnet
            assert gap == pytest.approx(original_gap, rel=1e-10)

    def test_r_ag_sweep_varies_gap(self):
        base = default_inrunner()
        original_gap = base.r_stator - base.r_magnet
        axis = SweepAxis("r_ag", original_gap * 0.5, original_gap * 1.5, 3)
        points = generate_grid(base, [axis])
        gaps = [pt.geometry.r_stator - pt.geometry.r_magnet for pt in points]
        assert gaps[0] < gaps[-1]


# ---------------------------------------------------------------------------
# L_stk sweep
# ---------------------------------------------------------------------------

class TestLstkSweep:

    def test_l_stk_sweep_preserves_geometry(self):
        base = default_inrunner()
        axis = SweepAxis("L_stk", 0.05, 0.15, 3)
        points = generate_grid(base, [axis], base_L_stk=0.10)
        assert len(points) == 3
        for pt in points:
            assert pt.geometry == base
        assert points[0].L_stk == pytest.approx(0.05)
        assert points[-1].L_stk == pytest.approx(0.15)


class TestBackIronSweep:

    def _base(self):
        from phasesweep.geometry import outrunner
        return outrunner(r_outer=0.80, r_rotor=0.70, r_magnet=0.64,
                         r_stator=0.50, r_inner=0.10)

    def test_sweep_sets_thickness(self):
        axis = SweepAxis("back_iron_thickness", 0.02, 0.08, 4)
        points = generate_grid(self._base(), [axis])
        assert len(points) == 4
        assert points[0].geometry.back_iron_thickness == pytest.approx(0.02)
        assert points[-1].geometry.back_iron_thickness == pytest.approx(0.08)

    def test_out_of_range_skipped(self):
        # wall = 0.10; values >= wall raise in Geometry and are dropped
        axis = SweepAxis("back_iron_thickness", 0.04, 0.16, 4)
        points = generate_grid(self._base(), [axis])
        assert 0 < len(points) < 4
        assert all(p.geometry.back_iron_thickness < 0.10 for p in points)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

class TestGridGeneration:

    def test_2d_grid(self):
        base = default_inrunner()
        axes = [
            SweepAxis("r_outer", 0.8, 1.2, 3),
            SweepAxis("L_stk", 0.05, 0.15, 4),
        ]
        points = generate_grid(base, axes, base_L_stk=0.10)
        assert len(points) == 12  # 3 * 4

    def test_empty_axes_returns_base(self):
        base = default_inrunner()
        points = generate_grid(base, [])
        assert len(points) == 1
        assert points[0].geometry == base

    def test_invalid_combos_skipped(self):
        base = default_inrunner()
        gap = base.r_stator - base.r_magnet
        # Sweep gap so wide that r_stator exceeds r_outer
        axis = SweepAxis("r_ag", gap * 0.5, gap * 20.0, 5)
        points = generate_grid(base, [axis])
        assert len(points) < 5

    def test_all_points_have_valid_geometry(self):
        base = default_inrunner()
        axes = [
            SweepAxis("r_outer", 0.5, 2.0, 5),
            SweepAxis("r_ag", 0.03, 0.12, 4),
        ]
        points = generate_grid(base, axes)
        for pt in points:
            geo = pt.geometry
            assert geo.r_outer > geo.r_stator > geo.r_magnet > geo.r_rotor


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

class TestRunSweep:

    def _make_motor(self) -> Motor:
        geo = default_inrunner()
        return Motor(name="test", geometry=geo, n_p=4, B_rem=1.2)

    def test_analytical_sweep(self):
        motor = self._make_motor()
        gap = motor.geometry.r_stator - motor.geometry.r_magnet
        axis = SweepAxis("r_ag", gap * 0.5, gap * 1.5, 3)
        points = generate_grid(motor.geometry, [axis])
        results = run_sweep(motor, points, ["analytical"])
        assert len(results) == 3
        assert all(isinstance(r, SweepResult) for r in results)
        assert all(r.status == "OK" for r in results)
        fundamentals = [r.metrics["fundamental"] for r in
                        sorted(results, key=lambda r: r.point_idx)]
        # Smaller gap → higher B₁
        assert fundamentals[0] > fundamentals[-1]

    def test_multi_model_sweep(self):
        motor = self._make_motor()
        points = [SweepPoint(geometry=motor.geometry)]
        results = run_sweep(motor, points, ["analytical", "fem"])
        assert len(results) == 2
        models = {r.model for r in results}
        assert models == {"analytical", "fem"}
        assert all(r.status == "OK" for r in results)

    def test_empty_points(self):
        motor = self._make_motor()
        results = run_sweep(motor, [], ["analytical"])
        assert results == []

    def test_L_stk_carried_through(self):
        motor = self._make_motor()
        geo = motor.geometry
        pts = [SweepPoint(geometry=geo, L_stk=0.05),
               SweepPoint(geometry=geo, L_stk=0.15)]
        results = run_sweep(motor, pts, ["analytical"])
        assert len(results) == 2
        assert all(r.status == "OK" for r in results)
