"""Integration tests for fem_field.py — use coarse meshes to keep runtime low."""

import dataclasses as _dc
from importlib.util import find_spec

import pytest

if not find_spec("ngsolve"):
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)
try:
    from ngsolve import L2, GridFunction, Mesh, grad, sqrt
except ImportError:
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)

import numpy as np

from phasesweep.machines.geometry import default_inrunner, inrunner, outrunner
from phasesweep.solvers import fem_mesh
from phasesweep.solvers.analytical import (
    end_effect_factor,
    end_effect_factor_pole_pitch,
    end_winding_leakage,
)
from phasesweep.solvers.fem_field import solve_field_fem
from phasesweep.solvers.fem_mesh import DEFAULT_GEO, clear_mesh_cache, set_disk_cache_dir
from phasesweep.solvers.field_solution import derive_B_rem, zhu_howe_Br
from phasesweep.solvers.harmonics import batch_harmonics, harmonics_1sided
from tests.conftest import requires_fem
from tests.solvers.fem_shared import (
    _GEO,
    _GEO_OUT,
    _GEO_SMALL,
    _H_FAST,
    _HALBACH_GEO,
    _RADII,
    _SLOTTED_GEO_KW,
    _TORQUE_FAST,
    FAST,
    _gap_radii,
    _solve,
    _torque_geo,
)

# Module-level guard covers missing ngsolve; the marker makes the module
# selectable by lane.
pytestmark = requires_fem


def _spy_build_geometry(monkeypatch):
    """Capture kwargs passed to _build_motor_geometry, bypassing caches."""
    captured = {}
    orig = fem_mesh._build_motor_geometry

    def spy(g, **kw):
        captured.update(kw)
        return orig(g, **kw)

    monkeypatch.setattr(fem_mesh, "_build_motor_geometry", spy)
    set_disk_cache_dir(None)
    clear_mesh_cache()
    return captured


def test_solve_field_fem_slot_params_default_from_geometry(monkeypatch):
    """Omitted n_slots/slot_width_ratio fall back to the geometry's values."""
    geo = inrunner(**_SLOTTED_GEO_KW, slot_width_ratio=0.42)
    captured = _spy_build_geometry(monkeypatch)
    solve_field_fem(geo=geo, n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                    **FAST)
    assert captured["n_slots"] == 12
    assert captured["slot_width_ratio"] == 0.42


def test_solve_field_fem_explicit_zero_overrides_geometry(monkeypatch):
    """Explicit n_slots=0 forces smooth bore despite slotted geometry."""
    geo = inrunner(**_SLOTTED_GEO_KW, slot_width_ratio=0.42)
    captured = _spy_build_geometry(monkeypatch)
    solve_field_fem(geo=geo, n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                    n_slots=0, **FAST)
    assert captured["n_slots"] == 0


# ---------------------------------------------------------------------------
# Solver — smooth bore
# ---------------------------------------------------------------------------

def test_smooth_bore_returns_correct_shape():
    theta, B_r = _solve()
    assert theta.shape == (FAST["n_theta"],)
    assert B_r.shape == (FAST["n_theta"],)


def test_smooth_bore_has_dominant_harmonic():
    n_p = 2
    theta, B_r = _solve(n_p=n_p)
    amps = harmonics_1sided(B_r)
    assert np.max(amps[1:]) > 0.01


def test_smooth_bore_field_nonzero():
    theta, B_r = _solve()
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_smooth_bore_field_sign_matches_analytical():
    n_p = 2
    theta, B_fem = _solve(n_p=n_p)
    B_rem = derive_B_rem(0.1, n_p, 50, 0.966, 0.10,
                          r_stator=DEFAULT_GEO.r_stator,
                          r_magnet=DEFAULT_GEO.r_magnet,
                          r_rotor=DEFAULT_GEO.r_rotor)
    B_an = zhu_howe_Br(theta, n_p, B_rem, **_RADII)
    # FEM waveform is flat-topped (square-wave source) vs the analytical
    # pure cosine; correlation ~0.95 proves same sign/phase, not shape
    assert np.corrcoef(B_fem, B_an)[0, 1] > 0.9


def test_smooth_bore_scales_with_psi_f():
    _, B1 = _solve(psi_f=0.1)
    _, B2 = _solve(psi_f=0.2)
    ratio = np.nanmax(np.abs(B2)) / np.nanmax(np.abs(B1))
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Solver — slotted + winding
# ---------------------------------------------------------------------------

def test_slotted_returns_correct_shape():
    # Real slot faces required by the j_s guard (the default geometry
    # has slot_depth = 0, where j_s != 0 now raises instead of being
    # silently ignored).
    theta, B_r = _solve(n_slots=12, j_s=0.1, geo=inrunner(**_SLOTTED_GEO_KW))
    assert B_r.shape == (FAST["n_theta"],)


def test_armature_without_slots_raises():
    with pytest.raises(ValueError, match="n_slots > 0"):
        _solve(n_slots=0, j_s=0.1)


def test_backward_compat_n_slots_zero():
    _, B1 = _solve()
    _, B2 = _solve(n_slots=0, j_s=0.0)
    np.testing.assert_allclose(B1, B2, rtol=1e-10)


def test_slotted_solve_completes():
    n_p, Q = 2, 12
    theta, B_r = _solve(n_p=n_p, n_slots=Q, j_s=0.0)
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_winding_current_solve_completes():
    # Real slot faces required — this test used to run on the default
    # geometry (slot_depth = 0), where the j_s source integrated over an
    # empty region and the "winding current" solve was actually
    # open-circuit. Now it also asserts the current changes the
    # field.
    geo = inrunner(**_SLOTTED_GEO_KW)
    theta, B_r = _solve(n_slots=12, j_s=1e5, geo=geo)
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4
    _, B_r0 = _solve(n_slots=12, j_s=0.0, geo=geo)
    assert np.max(np.abs(B_r - B_r0)) > 1e-3


# ---------------------------------------------------------------------------
# Zhu & Howe analytical model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# End-effect factor tests
# ---------------------------------------------------------------------------

def test_end_effect_factor_long_stack():
    """k_end → 1 for stacks much longer than the air gap."""
    assert end_effect_factor(L_stk=1.0, g_eff=0.001) == pytest.approx(1.0, abs=0.001)


def test_end_effect_factor_short_stack():
    """k_end < 1 for short stacks."""
    k = end_effect_factor(L_stk=0.007, g_eff=0.00175)
    assert 0.8 < k < 0.95


def test_end_effect_factor_monotonic():
    """k_end increases monotonically with L_stk."""
    g_eff = 0.002
    prev = 0.0
    for L in [0.005, 0.01, 0.02, 0.05, 0.1]:
        k = end_effect_factor(L_stk=L, g_eff=g_eff)
        assert k > prev
        prev = k


def test_end_effect_factor_zero_limits():
    """Edge cases take the documented limits: 0 at L_stk = 0, 1 at g_eff = 0."""
    assert end_effect_factor(L_stk=0, g_eff=0.001) == 0.0
    assert end_effect_factor(L_stk=0.01, g_eff=0) == 1.0


def test_end_effect_factor_outrunner_range():
    """Outrunner-like geometry: L_stk=7mm, g_eff≈1.75mm → k_end ≈ 0.84."""
    k = end_effect_factor(L_stk=0.007, g_eff=0.00175)
    assert 0.80 < k < 0.90


def test_end_effect_factor_pole_pitch_limits():
    """Pole-pitch form: → 1 for long stacks, 0 at L_stk = 0, 1 at tau_p = 0."""
    assert end_effect_factor_pole_pitch(L_stk=1.0, tau_p=0.004) == pytest.approx(1.0, abs=0.01)
    assert end_effect_factor_pole_pitch(L_stk=0, tau_p=0.004) == 0.0
    assert end_effect_factor_pole_pitch(L_stk=0.01, tau_p=0) == 1.0


def test_end_effect_factor_pole_pitch_monotonic():
    """Pole-pitch form increases monotonically with L_stk."""
    tau_p = 0.004
    prev = 0.0
    for L in [0.005, 0.01, 0.02, 0.05, 0.1]:
        k = end_effect_factor_pole_pitch(L_stk=L, tau_p=tau_p)
        assert k > prev
        prev = k


def test_end_effect_factor_pole_pitch_outrunner_bracket():
    """Outrunner-like geometry: pole-pitch form sits below the gap-scale
    form (aggressive edge of the bracket)."""
    k_pp = end_effect_factor_pole_pitch(L_stk=0.007, tau_p=0.00378)
    k_gap = end_effect_factor(L_stk=0.007, g_eff=0.00175)
    assert 0.60 < k_pp < 0.72
    assert k_pp < k_gap


def test_end_winding_leakage_creator():
    """Boldea L_ew for CREATOR IM (semicircular) matches hand calculation."""
    L = end_winding_leakage(r_stator=0.0628, n_slots=36, n_p=2,
                            coil_pitch_slots=7, N=18, coils_series=12)
    assert L == pytest.approx(1.424e-3, rel=0.01)


def test_end_winding_leakage_increases_with_ew_ratio():
    """Longer end-turns produce more leakage."""
    kw = dict(r_stator=0.0628, n_slots=36, n_p=2,
              coil_pitch_slots=7, N=18, coils_series=12)
    L_semi = end_winding_leakage(**kw)
    L_long = end_winding_leakage(**kw, ew_ratio=2.0)
    assert L_long > L_semi > 0


def test_end_winding_leakage_raises_when_degenerate():
    """lambda_ew <= 0 raises ValueError (non-positive permeance is input error)."""
    with pytest.raises(ValueError, match="non-positive permeance"):
        end_winding_leakage(r_stator=0.01, n_slots=6, n_p=1,
                            coil_pitch_slots=1, N=10, coils_series=2,
                            ew_ratio=0.5)


def test_zhu_howe_rejects_np1():
    with pytest.raises(ValueError, match="n_p=1"):
        zhu_howe_Br(np.linspace(0, 2 * np.pi, 60), n_p=1, B_rem=1.0,
                    **_RADII)


def test_zhu_howe_is_pure_cosine():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    Br = zhu_howe_Br(theta, n_p=2, B_rem=1.0, **_RADII)
    amps = harmonics_1sided(Br)
    assert amps[2] > 0.01
    assert amps[2] > 100 * np.max(np.delete(amps[1:], 1))


@pytest.mark.parametrize("n_p,psi_f,L_d,L_q,N,k_w,L_stk", [
    (2, 0.1, 4e-3, 4e-3, 50, 0.966, 0.10),   # Config A
    (4, 0.08, 2e-3, 2e-3, 40, 0.966, 0.08),   # Config C
    (8, 0.05, 1e-3, 1e-3, 50, 0.966, 0.10),   # high pole count
    (2, 0.1144, 0.2055, 0.3320, 328, 0.866, 0.0301),  # Creator Case PMSM
])
def test_fem_matches_zhu_howe(n_p, psi_f, L_d, L_q, N, k_w, L_stk):
    """FEM smooth-bore fundamental should match Zhu & Howe within 2%."""
    B_rem = derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                          r_stator=DEFAULT_GEO.r_stator,
                          r_magnet=DEFAULT_GEO.r_magnet,
                          r_rotor=DEFAULT_GEO.r_rotor)
    n_theta = 180

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(theta, n_p, B_rem, **_RADII)
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=_GEO, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.04,
        n_slots=0, j_s=0.0,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.02)


# ---------------------------------------------------------------------------
# batch_harmonics
# ---------------------------------------------------------------------------

def test_batch_harmonics_shape():
    n_theta = 360
    B = {"A": np.sin(2 * np.linspace(0, 2 * np.pi, n_theta, endpoint=False))}
    result = batch_harmonics(B)
    assert "A" in result
    assert result["A"].shape[0] == n_theta // 2 + 1


def test_batch_harmonics_fundamental():
    n_theta = 360
    k = 3
    B_r = np.sin(k * np.linspace(0, 2 * np.pi, n_theta, endpoint=False))
    result = batch_harmonics({"test": B_r})
    amps = result["test"]
    assert amps[k] == pytest.approx(1.0, rel=0.01)
    assert amps[k] > 10 * amps[k + 1]


def test_batch_harmonics_multi():
    n_theta = 360
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    data = {"A": np.cos(2 * theta), "B": np.cos(4 * theta)}
    result = batch_harmonics(data)
    assert set(result.keys()) == {"A", "B"}
    assert result["A"][2] > result["A"][4]
    assert result["B"][4] > result["B"][2]


def test_harmonics_odd_length_last_bin_doubled():
    n = 9
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = (n - 1) // 2  # last rfft bin is a regular harmonic for odd n
    amps = harmonics_1sided(0.25 * np.cos(k * theta))
    assert amps[k] == pytest.approx(0.25)
    batch = batch_harmonics({"A": 0.25 * np.cos(k * theta)})
    assert batch["A"][k] == pytest.approx(0.25)


def test_harmonics_even_length_nyquist_not_doubled():
    n = 8
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    amps = harmonics_1sided(np.cos((n // 2) * theta))
    assert amps[n // 2] == pytest.approx(1.0)


def test_double_onesided_does_not_mutate_input():
    from phasesweep.solvers.harmonics import _double_onesided
    amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    original = amps.copy()
    _double_onesided(amps, 8)
    np.testing.assert_array_equal(amps, original)


def test_compute_thd_rejects_bad_fund_idx():
    """fund_idx out of range raises ValueError."""
    from phasesweep.solvers.harmonics import compute_thd
    amps = np.array([0.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="fund_idx.*out of range"):
        compute_thd(amps, 0)   # DC bin is not a valid fundamental
    with pytest.raises(ValueError, match="fund_idx.*out of range"):
        compute_thd(amps, 3)   # past end
    with pytest.raises(ValueError, match="fund_idx.*out of range"):
        compute_thd(amps, -1)  # negative


def test_compute_thd_rejects_nonfinite():
    """Non-finite amplitudes raise ValueError."""
    from phasesweep.solvers.harmonics import compute_thd
    with pytest.raises(ValueError, match="non-finite"):
        compute_thd(np.array([0.0, float("nan"), 0.5]), 1)
    with pytest.raises(ValueError, match="non-finite"):
        compute_thd(np.array([0.0, 1.0, float("inf")]), 1)


def test_compute_thd_nan_on_zero_fundamental():
    from phasesweep.solvers.harmonics import compute_thd
    assert np.isnan(compute_thd(np.array([0.0, 0.0, 0.5]), 1))


def test_compute_thd_pins_value_from_amplitudes():
    """Exact value from hand-built amplitudes: sqrt(0.3^2 + 0.4^2)/1.0 = 50%.

    The only numeric THD assertions elsewhere are one-sided bounds on measured
    harmonics, so a pure scale error in compute_thd was invisible suite-wide
    (a 1.10x scale passed the whole suite).
    """
    from phasesweep.solvers.harmonics import compute_thd
    # index:      0(DC)  1    2(fund)  3    4    5
    amps = np.array([0.7, 0.0, 1.0, 0.3, 0.0, 0.4])
    assert compute_thd(amps, 2) == pytest.approx(50.0)
    # DC is excluded: a large DC bin must not move the result.
    assert compute_thd(amps * np.array([100.0, 1, 1, 1, 1, 1]), 2) == pytest.approx(50.0)
    # THD is scale-invariant in the waveform amplitude.
    assert compute_thd(2.5 * amps, 2) == pytest.approx(50.0)


def test_compute_thd_square_wave_textbook_value():
    """End-to-end through the FFT: an ideal square wave has THD = 48.34%
    (sqrt(pi^2/8 - 1)); 720 samples leaves ~0.001 of series truncation."""
    from phasesweep.solvers.harmonics import compute_thd
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    amps = harmonics_1sided(np.sign(np.cos(theta)))
    assert compute_thd(amps, 1) == pytest.approx(
        np.sqrt(np.pi**2 / 8 - 1) * 100, abs=0.01
    )


# ---------------------------------------------------------------------------
# Nonlinear Picard iteration (Issue 2)
# ---------------------------------------------------------------------------

def test_nonlinear_reproduces_linear_at_low_flux():
    _, B_lin = _solve(psi_f=0.05)
    _, B_nl = _solve(psi_f=0.05, nonlinear=True)
    peak_lin = np.nanmax(np.abs(B_lin))
    peak_nl = np.nanmax(np.abs(B_nl))
    assert peak_nl == pytest.approx(peak_lin, rel=0.05)


def test_nonlinear_converges():
    theta, B_r = _solve(nonlinear=True)
    assert np.all(np.isfinite(B_r))


def test_nonlinear_changes_field_at_high_flux():
    small = inrunner(r_outer=0.05, r_stator=0.035, r_magnet=0.032, r_rotor=0.015, r_inner=0.0)
    _, B_lin = _solve(psi_f=0.3, geo=small, maxh_fraction=0.04)
    _, B_nl = _solve(psi_f=0.3, geo=small, nonlinear=True,
                     max_picard=40, picard_relax=0.15, picard_tol=0.02, maxh_fraction=0.04)
    assert not np.allclose(B_lin, B_nl, rtol=1e-3)


def test_nonlinear_raises_on_non_convergence():
    with pytest.raises(RuntimeError, match="Picard iteration did not converge"):
        _solve(nonlinear=True, nonlinear_solver="picard",
               max_picard=1, picard_tol=1e-15)


def test_picard_zero_budget_reports_inf_residual():
    """max_picard=0 skips the loop entirely, so the residual sentinel is
    what reaches the error message — it has to stay a formattable float.
    """
    with pytest.raises(RuntimeError, match=r"in 0 iterations \(residual: inf"):
        _solve(nonlinear=True, nonlinear_solver="picard",
               max_picard=0, picard_tol=1e-15)


def test_picard_damps_relax_on_divergence():
    """Pin the adaptive-relaxation trajectory.

    Fixed relaxation at >= 0.35 diverges on this saturating geometry
    (~3.5 T in iron); the damping branch is what makes it converge, and
    running without it is what silently produced non-converged nonlinear
    solves in the wedge k_end study. No other test enters the branch, so
    its arithmetic was free to be wrong.

    picard_iterations and picard_relax_final are trajectory pins — they
    move if the damping schedule changes, which is the point. Re-measure
    them deliberately on a solver change rather than loosening them.
    """
    info: dict[str, float] = {}
    _, B_r = _solve(psi_f=0.3, geo=_GEO_SMALL, nonlinear=True,
                    maxh_fraction=0.04, nonlinear_solver="picard",
                    picard_relax=0.5, picard_tol=0.02, max_picard=60,
                    info=info)
    assert np.all(np.isfinite(B_r))
    # Damping engaged: relax ended well below where it started.
    assert info["picard_relax_final"] < 0.5
    assert info["picard_relax_final"] == pytest.approx(0.08447027, rel=1e-6)
    assert info["picard_iterations"] == 26.0
    assert info["picard_residual"] < 0.02


def test_picard_fixed_point_independent_of_starting_relax():
    """Relaxation is a path parameter, not a physical one: the converged
    field must not depend on where the damping schedule starts. Robust
    where the trajectory pin above is deliberately brittle.
    """
    kw = dict(psi_f=0.3, geo=_GEO_SMALL, nonlinear=True, maxh_fraction=0.04,
              nonlinear_solver="picard", picard_tol=0.02, max_picard=60)
    _, B_a = _solve(picard_relax=0.35, **kw)
    _, B_b = _solve(picard_relax=0.8, **kw)
    assert np.nanmax(np.abs(B_a - B_b)) < 5e-3 * np.nanmax(np.abs(B_a))


def test_newton_matches_picard_at_high_flux():
    small = inrunner(r_outer=0.05, r_stator=0.035, r_magnet=0.032,
                     r_rotor=0.015, r_inner=0.0)
    info_n: dict[str, float] = {}
    # picard_tol tightened to 1e-4 so the comparison measures solver
    # agreement at the shared fixed point, not Picard's shipping-tol
    # error (dominant at this saturation level, ~3.5 T in iron).
    _, B_p = _solve(psi_f=0.3, geo=small, nonlinear=True, maxh_fraction=0.04,
                    nonlinear_solver="picard", picard_tol=1e-4)
    _, B_n = _solve(psi_f=0.3, geo=small, nonlinear=True, maxh_fraction=0.04,
                    nonlinear_solver="newton", info=info_n)
    assert "newton_fell_back" not in info_n
    assert info_n["newton_iterations"] >= 1
    # Iteration ceiling: convergence-quality regressions (bad Jacobian,
    # broken line search, indefinite spline correction) surface as
    # iteration blowups long before wrong answers — the un-normalized
    # Jacobian regression measured ~2.7x the SPD tangent count (51 vs
    # 19 on the saturating spike case). This config measures 27, so the
    # failure class lands ~70+; 40 leaves version-jitter headroom while
    # staying far under it.
    assert info_n["newton_iterations"] <= 40
    assert info_n["newton_startup_iterations"] == 2.0
    assert np.max(np.abs(B_n - B_p)) < 1e-3 * np.max(np.abs(B_p))


def test_newton_falls_back_to_picard(monkeypatch):
    import phasesweep.solvers.fem_field as ff

    def _stall(*args, **kwargs):
        raise RuntimeError("forced stall")

    monkeypatch.setattr(ff, "newton_solve", _stall)
    info: dict[str, float] = {}
    _, B_r = _solve(nonlinear=True, nonlinear_solver="newton", info=info)
    assert info.get("newton_fell_back") == 1.0
    assert "picard_iterations" in info
    assert np.all(np.isfinite(B_r))


def test_unknown_nonlinear_solver_raises():
    with pytest.raises(ValueError, match="nonlinear_solver"):
        _solve(nonlinear=True, nonlinear_solver="broyden")


def test_nonlinear_result_insensitive_to_mu_r_fe_seed():
    """Under nonlinear=True the iron follows the BH table and mu_r_fe
    merely seeds iteration 0 (docstring contract). A regression that let
    the seed leak into the converged field would silently skew every
    nonlinear run fed a datasheet mu_r_fe. Measured: a 25x seed spread
    moves the field < 5e-5 relative; 1e-3 leaves headroom."""
    fields = {}
    for mu_seed in (200.0, 5000.0):
        info: dict[str, float] = {}
        _, B_r = solve_field_fem(
            geo=_GEO_OUT, n_p=4, B_rem=1.0,
            mu_r_pm=1.05, mu_r_fe=mu_seed,
            nonlinear=True, info=info, **FAST,
        )
        assert "newton_fell_back" not in info
        fields[mu_seed] = np.asarray(B_r)
    delta = np.max(np.abs(fields[5000.0] - fields[200.0]))
    assert delta < 1e-3 * np.max(np.abs(fields[200.0]))


# ---------------------------------------------------------------------------
# Inset magnets (interpole iron + optional bridge ring)
# ---------------------------------------------------------------------------


def _solve_inset(geo, alpha_p=0.7, nonlinear=True, **kw):
    merged = {**FAST, **kw}
    return solve_field_fem(
        geo=geo, n_p=2, B_rem=0.4, mu_r_pm=1.1, mu_r_fe=1000.0,
        alpha_p=alpha_p, nonlinear=nonlinear, **merged,
    )


def test_inset_requires_nonlinear():
    geo = _dc.replace(_GEO_SMALL, magnet_mount="inset")
    with pytest.raises(ValueError, match="requires nonlinear=True"):
        _solve_inset(geo, nonlinear=False)


def test_inset_requires_arcs():
    geo = _dc.replace(_GEO_SMALL, magnet_mount="inset")
    with pytest.raises(ValueError, match="alpha_p=1.0 builds none"):
        _solve_inset(geo, alpha_p=1.0)
    # gap (1-0.99)*pi/2*0.015 = 0.24 mm < 0.5 mm OCC limit
    with pytest.raises(ValueError, match="below the 0.5 mm OCC limit"):
        _solve_inset(geo, alpha_p=0.99)


def test_inset_interpole_iron_lowers_fundamental():
    _, B_surface = _solve_inset(_GEO_SMALL)
    _, B_inset = _solve_inset(_dc.replace(_GEO_SMALL, magnet_mount="inset"))
    f_surface = harmonics_1sided(B_surface)[2]
    f_inset = harmonics_1sided(B_inset)[2]
    # interpole iron shunts magnet flux — a real reduction, but the
    # machine still works
    assert f_inset < 0.99 * f_surface
    assert f_inset > 0.5 * f_surface


def test_bridge_ring_lowers_fundamental_further():
    geo_inset = _dc.replace(_GEO_SMALL, magnet_mount="inset")
    geo_bridge = _dc.replace(_GEO_SMALL, magnet_mount="inset",
                             bridge_thickness=2e-3)
    _, B_inset = _solve_inset(geo_inset)
    _, B_bridge = _solve_inset(geo_bridge)
    f_inset = harmonics_1sided(B_inset)[2]
    f_bridge = harmonics_1sided(B_bridge)[2]
    assert f_bridge < f_inset


def test_surface_bridge_requires_nonlinear():
    geo = _dc.replace(_GEO_SMALL, bridge_thickness=2e-3)
    with pytest.raises(ValueError, match="bridge_thickness requires nonlinear"):
        solve_field_fem(
            geo=geo, n_p=2, B_rem=0.4, mu_r_pm=1.1, mu_r_fe=1000.0,
            alpha_p=1.0, nonlinear=False,
        )


# ---------------------------------------------------------------------------
# Outrunner FEM
# ---------------------------------------------------------------------------


def test_outrunner_smooth_bore_solve():
    """Outrunner FEM solve completes with nonzero field."""
    theta, B_r = solve_field_fem(
        geo=_GEO_OUT, n_p=4, B_rem=1.0,
        mu_r_pm=1.05, mu_r_fe=1000.0, **FAST,
    )
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_outrunner_back_iron_split_matches_solid_when_linear():
    """A linear back-iron ring carries the return flux, so the split (iron ring
    + non-magnetic shell) reproduces the solid yoke's airgap fundamental. The
    physical difference only appears once the thin ring saturates (nonlinear)."""
    kw = dict(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
              r_stator=0.04, r_inner=0.02)
    solid = outrunner(**kw)
    split = outrunner(**kw, back_iron_thickness=0.01)
    common = dict(n_p=4, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                  n_theta=180, maxh_fraction=0.04)
    fund_solid = harmonics_1sided(solve_field_fem(geo=solid, **common)[1])[4]
    fund_split = harmonics_1sided(solve_field_fem(geo=split, **common)[1])[4]
    assert fund_split == pytest.approx(fund_solid, rel=0.02)


def test_outrunner_fem_matches_analytical():
    """Outrunner FEM fundamental should match Zhu & Howe within 2%."""
    n_p = 4
    B_rem = 1.0
    n_theta = 180

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(
        theta, n_p, B_rem, r_eval=_GEO_OUT.r_ag,
        r_stator=_GEO_OUT.r_stator, r_magnet=_GEO_OUT.r_magnet,
        r_rotor=_GEO_OUT.r_rotor, mu_r_pm=1.05,
    )
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=_GEO_OUT, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.04,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.02)


def test_outrunner_slotted_solve():
    """Outrunner slotted FEM solve completes."""
    geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                    r_stator=0.04, r_inner=0.02, n_slots=6, slot_depth=0.005)
    theta, B_r = solve_field_fem(
        geo=geo, n_p=4, B_rem=1.0,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_slots=6, **FAST,
    )
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_outrunner_nonlinear_solve():
    """Outrunner nonlinear solve at the BH knee — third iteration-ceiling
    geometry: _GEO_SMALL guards deep saturation (~3.5 T iron, measures
    27 Newton iterations), CREATOR the inset-bridge regime (measures 2);
    this config sits on the knee itself, where the differential
    permeability swings fastest per step."""
    info: dict[str, float] = {}
    theta, B_r = solve_field_fem(
        geo=_GEO_OUT, n_p=4, B_rem=1.4,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        nonlinear=True, info=info, **FAST,
    )
    assert np.all(np.isfinite(B_r))
    assert "newton_fell_back" not in info
    # Regime check: iron peak stays in the knee band — if geometry or
    # B_rem edits drift it out, this stops guarding the knee regime.
    assert 1.3 < info["b_iron_max"] < 1.7
    # Ceiling: this config measures 7; the broken-Jacobian failure class
    # runs ~2.7x that (~19). 12 leaves version-jitter headroom under it.
    assert info["newton_iterations"] <= 12


# ---------------------------------------------------------------------------
# Paper validation: FEM vs analytical cross-validation
# ---------------------------------------------------------------------------

def test_paper_inrunner_fem_vs_analytical(paper_geometry_8pole):
    """Paper inrunner (2p=8, Zhu & Howe dims): FEM vs analytical < 1% on fundamental."""
    n_p = 4
    B_rem = 1.2
    mu_r_pm = 1.05
    n_theta = 360
    geo = paper_geometry_8pole

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(
        theta, n_p, B_rem, r_eval=geo.r_ag,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet,
        r_rotor=geo.r_rotor, mu_r_pm=mu_r_pm,
    )
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=mu_r_pm, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.03,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.01), (
        f"FEM fundamental {fem_fund:.6f} vs analytical {an_fund:.6f}"
    )


def test_paper_outrunner_fem_vs_analytical(paper_geometry_8pole_outrunner):
    """Paper outrunner (rearranged dims): FEM vs analytical < 1% on fundamental."""
    n_p = 4
    B_rem = 1.2
    mu_r_pm = 1.05
    n_theta = 360
    geo = paper_geometry_8pole_outrunner

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(
        theta, n_p, B_rem, r_eval=geo.r_ag,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet,
        r_rotor=geo.r_rotor, mu_r_pm=mu_r_pm,
    )
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=mu_r_pm, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.03,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.01), (
        f"FEM fundamental {fem_fund:.6f} vs analytical {an_fund:.6f}"
    )


# ---------------------------------------------------------------------------
# P0: derive_B_rem roundtrip (Issue audit)
# ---------------------------------------------------------------------------

def test_derive_B_rem_scaled_radii():
    psi_f, n_p, N, k_w, L_stk = 0.1, 2, 50, 0.966, 0.10
    B_rem_default = derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                                  r_stator=DEFAULT_GEO.r_stator,
                                  r_magnet=DEFAULT_GEO.r_magnet,
                                  r_rotor=DEFAULT_GEO.r_rotor)
    B_rem_scaled = derive_B_rem(
        psi_f, n_p, N, k_w, L_stk,
        r_stator=0.35, r_magnet=0.32, r_rotor=0.15,
    )
    assert B_rem_scaled != pytest.approx(B_rem_default, rel=0.01)
    B_peak = zhu_howe_Br(
        np.array([0.0]), n_p, B_rem_scaled, r_eval=0.35,
        r_stator=0.35, r_magnet=0.32, r_rotor=0.15,
    )[0]
    psi_f_recovered = B_peak * 2 * N * k_w * 0.35 * L_stk / n_p
    assert psi_f_recovered == pytest.approx(psi_f, rel=1e-10)


def test_derive_B_rem_roundtrip():
    psi_f, n_p, N, k_w, L_stk = 0.1, 2, 50, 0.966, 0.10
    B_rem = derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                          r_stator=DEFAULT_GEO.r_stator,
                          r_magnet=DEFAULT_GEO.r_magnet,
                          r_rotor=DEFAULT_GEO.r_rotor)
    assert 0.01 < B_rem < 1.0, f"B_rem={B_rem} outside physically reasonable range"
    r_si = DEFAULT_GEO.r_stator
    B_peak = zhu_howe_Br(np.array([0.0]), n_p, B_rem, r_eval=r_si,
                         **_RADII)[0]
    psi_f_recovered = B_peak * 2 * N * k_w * r_si * L_stk / n_p
    assert psi_f_recovered == pytest.approx(psi_f, rel=1e-10)


# ---------------------------------------------------------------------------
# P1: _demoivre direct evaluation (Issue audit)
# ---------------------------------------------------------------------------

def test_demoivre_matches_trig():
    from math import cos, pi, sin

    from netgen.occ import OCCGeometry, WorkPlane

    from phasesweep.solvers.fem_field import _demoivre

    face = WorkPlane().Circle(1.0).Face()
    geo = OCCGeometry(face, dim=2)
    ngmesh = Mesh(geo.GenerateMesh(maxh=0.5))

    for n_p in (2, 8):
        cos_np, sin_np = _demoivre(n_p)
        for angle in (0.0, pi / (2 * n_p)):
            r = 0.5
            mp = ngmesh(r * cos(angle), r * sin(angle))
            assert cos_np(mp) == pytest.approx(cos(n_p * angle), abs=1e-12)
            assert sin_np(mp) == pytest.approx(sin(n_p * angle), abs=1e-12)


# ---------------------------------------------------------------------------
# P1: Nonlinear saturation validation (Issue audit)
# ---------------------------------------------------------------------------

def test_nonlinear_scaling_sublinear_at_high_flux():
    kw_base = dict(n_theta=60, maxh_fraction=0.04,
                   max_picard=40, picard_relax=0.15, picard_tol=0.02)
    _, B_lo = _solve(psi_f=0.05, nonlinear=True, **kw_base)
    _, B_hi = _solve(psi_f=0.10, nonlinear=True, **kw_base)
    ratio_low = np.nanmax(np.abs(B_hi)) / np.nanmax(np.abs(B_lo))

    # Square-wave source has a lower crest than cosine at
    # equal fundamental, so saturation onset moved up: psi_f=1→2 stays
    # linear; 2→4 is in the transition regime (ratio ≈ 1.94)
    _, B_lo2 = _solve(psi_f=2.0, nonlinear=True, **kw_base)
    _, B_hi2 = _solve(psi_f=4.0, nonlinear=True, **kw_base)
    ratio_high = np.nanmax(np.abs(B_hi2)) / np.nanmax(np.abs(B_lo2))

    assert ratio_low == pytest.approx(2.0, abs=0.01), (
        f"Low-flux ratio should be ~2.0 (got {ratio_low:.4f})"
    )
    assert ratio_high < ratio_low, (
        f"High-flux ratio ({ratio_high:.4f}) should be < low-flux ({ratio_low:.4f}) "
        f"due to iron saturation"
    )


# ---------------------------------------------------------------------------
# Physics validation
# ---------------------------------------------------------------------------

def _peak_B_iron(mesh_obj, gfu):
    """Extract peak |B| in iron regions from a FEM solution."""
    B_mag = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    fes_L2 = L2(mesh_obj, order=0)
    gfu_B = GridFunction(fes_L2)
    gfu_B.Set(B_mag)

    iron_names = {"stator", "shaft", "yoke"}
    iron_cf = {
        name: (1.0 if name in iron_names else 0.0)
        for name in mesh_obj.GetMaterials()
    }
    iron_ind = GridFunction(fes_L2)
    iron_ind.Set(mesh_obj.MaterialCF(iron_cf))

    B_vals = gfu_B.vec.FV().NumPy()
    mask = iron_ind.vec.FV().NumPy() > 0.5
    if not np.any(mask):
        return 0.0
    return float(np.max(B_vals[mask]))


def test_mesh_convergence():
    """FEM fundamental converges as mesh is refined; FAST mesh is adequate.

    Solves at 4 mesh densities and checks that the coarse test mesh
    (maxh_fraction=0.08) is within 2% of the finest mesh, proving the
    FAST mesh is in the asymptotic convergence regime.
    """
    maxh_fracs = [0.10, 0.06, 0.04, 0.02]
    funds = []
    for mf in maxh_fracs:
        _, B_r = solve_field_fem(
            geo=_GEO, n_p=2, B_rem=1.0,
            mu_r_pm=1.05, mu_r_fe=1000.0,
            n_theta=360, maxh_fraction=mf,
        )
        funds.append(harmonics_1sided(B_r)[2])

    # Successive differences should decrease (convergence)
    diffs = [abs(funds[i+1] - funds[i]) for i in range(len(funds) - 1)]
    for i in range(len(diffs) - 1):
        assert diffs[i+1] < diffs[i], (
            f"Convergence stalled: delta[{i}]={diffs[i]:.6e}, "
            f"delta[{i+1}]={diffs[i+1]:.6e}"
        )

    # FAST mesh (0.08, between first two points) within 2% of finest
    _, B_r_fast = solve_field_fem(
        geo=_GEO, n_p=2, B_rem=1.0,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=360, maxh_fraction=0.08,
    )
    fund_fast = harmonics_1sided(B_r_fast)[2]
    fund_finest = funds[-1]
    assert fund_fast == pytest.approx(fund_finest, rel=0.02), (
        f"FAST mesh fundamental {fund_fast:.6f} vs finest {fund_finest:.6f}"
    )


def test_fem_analytical_gap_decreases_with_mu_r_fe():
    """FEM-analytical error decreases as mu_r_fe increases.

    The analytical model assumes infinite iron permeability. FEM uses
    finite mu_r_fe. The gap between them should shrink as mu_r_fe → ∞,
    confirming the 2% tolerance is physically motivated.
    Uses fine mesh (maxh_fraction=0.02) to isolate the mu_r_fe effect.
    """
    n_p = 2
    B_rem = 1.0
    n_theta = 360
    maxh = 0.02

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(theta, n_p, B_rem, **_RADII)
    an_fund = harmonics_1sided(Br_an)[n_p]

    mu_r_fe_values = [100, 1000, 10000]
    errors = []
    for mu_r_fe in mu_r_fe_values:
        _, Br_fem = solve_field_fem(
            geo=_GEO, n_p=n_p, B_rem=B_rem,
            mu_r_pm=1.05, mu_r_fe=mu_r_fe,
            n_theta=n_theta, maxh_fraction=maxh,
        )
        fem_fund = harmonics_1sided(Br_fem)[n_p]
        errors.append(abs(fem_fund - an_fund) / an_fund)

    # Errors must decrease monotonically
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i+1], (
            f"Error at mu_r_fe={mu_r_fe_values[i]} ({errors[i]:.6f}) "
            f"should exceed error at mu_r_fe={mu_r_fe_values[i+1]} "
            f"({errors[i+1]:.6f})"
        )

    # At mu_r_fe=10000, error should be well under the 2% tolerance
    assert errors[-1] < 0.005, (
        f"Error at mu_r_fe=10000 = {errors[-1]:.6f}, expected < 0.005"
    )


def test_outrunner_inner_bc_effect():
    """Outrunner FEM matches analytical within 2% for both small and large bore.

    The FEM applies Neumann BC at the inner boundary (hollow center),
    while the analytical model assumes Dirichlet (infinite iron). For
    practical geometries the effect should be small.
    """
    n_p = 4
    B_rem = 1.2
    mu_r_pm = 1.05
    n_theta = 360

    geo_small = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                          r_stator=0.04, r_inner=0.005)
    geo_large = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                          r_stator=0.04, r_inner=0.025)

    errors = []
    for geo in [geo_small, geo_large]:
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        Br_an = zhu_howe_Br(
            theta, n_p, B_rem, r_eval=geo.r_ag,
            r_stator=geo.r_stator, r_magnet=geo.r_magnet,
            r_rotor=geo.r_rotor, mu_r_pm=mu_r_pm,
        )
        an_fund = harmonics_1sided(Br_an)[n_p]

        _, Br_fem = solve_field_fem(
            geo=geo, n_p=n_p, B_rem=B_rem,
            mu_r_pm=mu_r_pm, mu_r_fe=1000.0,
            n_theta=n_theta, maxh_fraction=0.03,
        )
        fem_fund = harmonics_1sided(Br_fem)[n_p]
        errors.append(abs(fem_fund - an_fund) / an_fund)

    # Both within 2%
    for i, (geo, err) in enumerate(zip([geo_small, geo_large], errors)):
        assert err < 0.02, (
            f"{'Small' if i == 0 else 'Large'} bore: FEM-analytical "
            f"error {err:.4f} exceeds 2%"
        )


def test_picard_relax_recovery_and_raw_criterion():
    """Tier-1 Picard guards: relax recovers after divergence damping, and
    the stop criterion is honored on the raw (unrelaxed) step.

    Saturating thin-yoke geometry with deliberately high picard_relax=0.5
    (diverges without damping). Pre-Tier-1, damping ratcheted relax to its
    floor (0.02-0.025) with no recovery, and the relax-scaled criterion
    accepted raw residuals ~0.4.
    """
    geo = inrunner(r_outer=0.038, r_stator=0.035, r_magnet=0.030,
                   r_rotor=0.020)
    info: dict[str, float] = {}
    solve_field_fem(
        geo=geo, n_p=2, B_rem=1.5, mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=180, maxh_fraction=0.03,
        nonlinear=True, nonlinear_solver="picard",
        max_picard=60, picard_tol=0.005, picard_relax=0.5,
        info=info,
    )
    assert info["picard_residual"] < 0.005, (
        f"Stop criterion not honored on raw step: "
        f"residual {info['picard_residual']:.4e}"
    )
    # Damping re-engages repeatedly on this deeply saturated case, so full
    # recovery to 0.5 is not expected — observed ~0.075. Pre-Tier-1 pinning
    # left relax at 0.02-0.025 with no growth path.
    assert info["picard_relax_final"] >= 0.05, (
        f"relax pinned near floor ({info['picard_relax_final']:.4f}); "
        f"recovery after divergence damping is broken"
    )


@pytest.mark.slow
def test_picard_relaxation_independence():
    """Converged nonlinear solution is independent of relaxation parameter.

    Uses a purpose-built thin-yoke geometry (3mm stator yoke, n_p=2) to
    guarantee iron saturation at B_rem=1.5T. Asserts peak |B_iron| > 1.2T
    as a precondition so the test fails loudly if saturation isn't reached.
    """
    geo = inrunner(r_outer=0.038, r_stator=0.035, r_magnet=0.030,
                   r_rotor=0.020)
    n_p = 2
    B_rem = 1.5
    common = dict(
        geo=geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=180, maxh_fraction=0.03,
        nonlinear=True, nonlinear_solver="picard",
        max_picard=60, picard_tol=0.005,
        return_full=True,
    )

    theta_a, B_r_a, mesh_a, gfu_a = solve_field_fem(**common, picard_relax=0.10)
    theta_b, B_r_b, mesh_b, gfu_b = solve_field_fem(**common, picard_relax=0.25)
    # relax=0.35 and 0.50 previously diverged; adaptive backtracking fixes this
    theta_c, B_r_c, mesh_c, gfu_c = solve_field_fem(**common, picard_relax=0.35)
    theta_d, B_r_d, mesh_d, gfu_d = solve_field_fem(**common, picard_relax=0.50)

    # Precondition: iron must be saturated for this test to be meaningful
    B_iron_a = _peak_B_iron(mesh_a, gfu_a)
    assert B_iron_a > 1.2, (
        f"Test requires saturated iron but peak |B| = {B_iron_a:.2f} T; "
        f"adjust geometry or B_rem"
    )

    fund_a = harmonics_1sided(B_r_a)[n_p]
    fund_b = harmonics_1sided(B_r_b)[n_p]
    fund_c = harmonics_1sided(B_r_c)[n_p]
    fund_d = harmonics_1sided(B_r_d)[n_p]
    for label, fund in [("0.25", fund_b), ("0.35", fund_c), ("0.50", fund_d)]:
        assert fund_a == pytest.approx(fund, rel=0.01), (
            f"Relaxation 0.10 fundamental {fund_a:.6f} vs "
            f"relaxation {label} fundamental {fund:.6f}"
        )


# ---------------------------------------------------------------------------
# Discrete magnet arcs tests
# ---------------------------------------------------------------------------

class TestDiscreteArcs:
    """Tests for FEM with discrete magnet arcs."""

    @pytest.mark.timeout(60)
    def test_alpha_p_1_invariance(self):
        """alpha_p=1.0 should give same B_r as no alpha_p."""
        geo = default_inrunner()
        _, B_r_base = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            **FAST)
        _, B_r_full = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=1.0, **FAST)
        np.testing.assert_allclose(B_r_base, B_r_full, atol=1e-12)

    @pytest.mark.timeout(60)
    def test_alpha_p_half_reduces_fundamental(self):
        """alpha_p=0.5 should significantly reduce B₁."""
        geo = default_inrunner()
        _, B_r_full = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            **FAST)
        _, B_r_half = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.5, **FAST)
        B1_full = harmonics_1sided(B_r_full)[4]
        B1_half = harmonics_1sided(B_r_half)[4]
        assert B1_half < B1_full
        assert B1_half > 0.5 * B1_full  # not more than 50% drop

    @pytest.mark.timeout(60)
    def test_outrunner_arcs_meshable(self):
        """Outrunner with discrete arcs should solve without error."""
        geo = outrunner(r_outer=0.050, r_rotor=0.045, r_magnet=0.040,
                        r_stator=0.035, r_inner=0.015)
        theta, B_r = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.8, **FAST)
        assert len(B_r) == FAST["n_theta"]
        assert np.max(np.abs(B_r)) > 0

    @pytest.mark.timeout(120)
    def test_near_full_pitch(self):
        """alpha_p=0.95 should solve and give result close to full-pitch."""
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015)
        _, B_r_full = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000, **FAST)
        _, B_r_95 = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.95, **FAST)
        B1_full = harmonics_1sided(B_r_full)[4]
        B1_95 = harmonics_1sided(B_r_95)[4]
        assert B1_95 > 0
        assert B1_95 / B1_full > 0.95  # near full pitch → small reduction

    @pytest.mark.timeout(120)
    def test_thin_arcs(self):
        """alpha_p=0.3 (thin arcs) should solve."""
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015)
        theta, B_r = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.3, **FAST)
        B1 = harmonics_1sided(B_r)[4]
        assert B1 > 0

    @pytest.mark.timeout(120)
    def test_slotted_plus_arcs(self):
        """Both slots and discrete arcs together."""
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015, n_slots=6, slot_depth=0.005,
                       slot_width_ratio=0.5)
        theta, B_r = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            n_slots=6, slot_width_ratio=0.5, alpha_p=0.75,
            **FAST)
        assert np.max(np.abs(B_r)) > 0

    @pytest.mark.timeout(120)
    def test_sub_threshold_gap_raises(self):
        """Interpole gap < 0.5 mm: OCC cannot cut the arcs, and solving the
        collapsed full ring would answer alpha_p=1.0 to an alpha_p=0.93
        request, so the solve refuses."""
        # gap at inner radius (fem v7) = (1-0.93)·π/7·0.010 = 0.314 mm
        # < 0.5 mm threshold (was 0.471 mm at the outer radius r_magnet)
        geo = inrunner(r_outer=0.030, r_stator=0.017, r_magnet=0.015,
                       r_rotor=0.010)
        with pytest.raises(ValueError, match="0.314 mm is below the 0.5 mm OCC limit"):
            solve_field_fem(
                geo, n_p=7, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
                alpha_p=0.93, **FAST)

    @pytest.mark.timeout(120)
    def test_supra_threshold_gap_solves(self):
        """A comfortably wide gap builds the arcs and solves."""
        geo = default_inrunner()
        _, B_r = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.75, **FAST)
        assert np.max(np.abs(B_r)) > 0

    @pytest.mark.timeout(120)
    def test_gap_band_honors_per_face_refinement(self):
        """0.5–1.0 mm gaps honor requested maxh ≤ gap/2 via per-face maxh.

        The global maxh floor (0.5 mm) used to mesh
        these gaps at the same requested resolution as the whole domain,
        violating the maxh ≤ gap × 0.5 rule. Pins the mean
        element density in the pm_gap regions against the equilateral-
        triangle area at the requested maxh (netgen honors maxh only
        approximately, so worst-edge metrics are too noisy to pin).
        """
        from ngsolve import VOL

        # gap at inner radius (fem v7) = (1-0.953)·π/4·0.015 = 0.554 mm —
        # inside the floored band [0.5, 1.0] mm
        alpha_p = 0.953
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015)
        gap = (1 - alpha_p) * np.pi / 4 * geo.r_rotor
        _, _, mesh, _ = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=alpha_p, return_full=True, **FAST)
        n_gap_els = sum(1 for el in mesh.Elements(VOL)
                        if el.mat == "pm_gap")
        assert n_gap_els > 0
        gap_area = (1 - alpha_p) * np.pi * (geo.r_magnet**2 - geo.r_rotor**2)
        eq_area = np.sqrt(3) / 4 * (gap * 0.5) ** 2
        # measured ~1.4 with the per-face maxh, ~3.4 under the old
        # global floor
        assert gap_area / n_gap_els <= 2.5 * eq_area


def test_j_s_without_slot_faces_raises():
    """n_slots > 0 with slot_depth = 0 builds no slot faces; j_s must not
    be silently ignored (this passed quietly before)."""
    with pytest.raises(ValueError, match="no slot regions"):
        solve_field_fem(geo=_GEO, n_slots=12, j_s=1e5, **_TORQUE_FAST)


# ---------------------------------------------------------------------------
# Magnet-pattern rotation — cogging-sweep support
# ---------------------------------------------------------------------------

class TestMagnetRotation:
    """rotation rotates arcs + magnetization sign; the stator stays fixed."""

    @pytest.mark.timeout(60)
    def test_rotation_zero_identity(self):
        """Explicit rotation=0.0 is bit-identical to the default."""
        geo = default_inrunner()
        _, B_r_base = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.5, **FAST)
        _, B_r_rot0 = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.5, rotation=0.0, **FAST)
        np.testing.assert_allclose(B_r_base, B_r_rot0, atol=1e-12)

    @pytest.mark.timeout(120)
    def test_rotation_shifts_pattern_preserves_amplitude(self):
        """Rotating the magnet pattern by phi shifts every magnet space
        harmonic's phase by -nu*phi and leaves amplitudes unchanged (up to
        mesh scatter — the rotated arcs build a different mesh)."""
        geo = default_inrunner()
        phi = np.deg2rad(10.0)
        kw = dict(n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
                  alpha_p=0.5, n_theta=360, maxh_fraction=0.08)
        _, Br0 = solve_field_fem(geo, **kw)
        _, Br1 = solve_field_fem(geo, rotation=float(phi), **kw)
        c0 = np.fft.rfft(Br0)
        c1 = np.fft.rfft(Br1)
        for nu in (4, 12):  # fundamental + 3rd magnet harmonic
            assert abs(c1[nu]) == pytest.approx(abs(c0[nu]), rel=0.02)
            phase_err = np.angle(
                (c1[nu] / c0[nu]) * np.exp(1j * nu * phi))
            assert abs(phase_err) < 0.02, (
                f"order {nu}: phase shift off by {phase_err:.4f} rad"
            )


# ---------------------------------------------------------------------------
# Current-sheet phase — d-axis MMF for the demag screen
# ---------------------------------------------------------------------------
# J_z = -j_s*k_w*cos(n_p*theta - sheet_phase): beta = 0 is the q-axis sheet
# (the original behavior), beta = +/-pi/2 a pure d-axis MMF.

class TestSheetPhase:

    def test_sheet_phase_zero_identity(self):
        """Explicit sheet_phase=0.0 is bit-identical to the default."""
        geo = _torque_geo()
        _, base, _, _ = solve_field_fem(geo=geo, j_s=1e5, **_TORQUE_FAST)
        _, zero, _, _ = solve_field_fem(geo=geo, j_s=1e5, sheet_phase=0.0,
                                        **_TORQUE_FAST)
        np.testing.assert_allclose(zero, base, atol=1e-12)

    def test_d_axis_sheet_no_interaction_torque(self):
        """At sheet_phase = +/-pi/2 the order-n_p interaction torque with
        the magnet field vanishes (measured ~1e-4 of the q-axis torque —
        pure d-axis MMF; the sign/axis invariant behind the demag screen)."""
        from phasesweep.solvers.fem_sampling import maxwell_interaction_torque_order
        geo = _torque_geo()
        kw = dict(geo=geo, n_p=2, mu_r_pm=1.05, mu_r_fe=1000.0, k_w=0.966,
                  n_theta=60, maxh_fraction=0.08, return_full=True)
        _, _, mesh, gfu_mag = solve_field_fem(j_s=0.0, B_rem=1.0, **kw)
        _, _, _, gfu_q = solve_field_fem(j_s=1e5, B_rem=1e-30, **kw)
        r = _gap_radii(geo)[1]
        tau_q = maxwell_interaction_torque_order(mesh, gfu_mag, gfu_q, r,
                                                 order=2)
        for beta in (np.pi / 2, -np.pi / 2):
            _, _, _, gfu_d = solve_field_fem(j_s=1e5, B_rem=1e-30,
                                             sheet_phase=beta, **kw)
            tau_d = maxwell_interaction_torque_order(mesh, gfu_mag, gfu_d,
                                                     r, order=2)
            assert abs(tau_d) < 0.01 * abs(tau_q), (
                f"beta={beta:+.3f}: tau_d {tau_d:.2f} vs tau_q {tau_q:.1f}"
            )

    def test_d_axis_sign_and_superposition(self):
        """sheet_phase = +pi/2 demagnetizes (gap B1 down), -pi/2 aids;
        the two shifts are symmetric (exact superposition in the linear
        solve). Pins the demag sign convention documented in the solver."""
        geo = _torque_geo()
        kw = dict(geo=geo, **_TORQUE_FAST)
        b1 = {}
        for beta in (None, np.pi / 2, -np.pi / 2):
            j = 0.0 if beta is None else 1e5
            _, Br, _, _ = solve_field_fem(
                j_s=j, sheet_phase=(beta or 0.0), **kw)
            b1[beta] = float(harmonics_1sided(Br)[2])
        drop = b1[None] - b1[np.pi / 2]
        gain = b1[-np.pi / 2] - b1[None]
        assert drop > 1e-3 * b1[None]  # demag shift well above noise
        assert gain == pytest.approx(drop, rel=1e-3)


# ---------------------------------------------------------------------------
# Halbach Phase 3 — source CF: per-segment magnetization direction
# ---------------------------------------------------------------------------


class TestHalbachSourceCF:

    def test_radial_bit_identical(self):
        """Default call vs explicit magnetization='radial' is bit-identical."""
        t1, Br1 = solve_field_fem(
            _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            **_H_FAST,
        )
        t2, Br2 = solve_field_fem(
            _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="radial", **_H_FAST,
        )
        np.testing.assert_allclose(Br1, Br2, atol=1e-12)

    def test_halbach_phase_aligned_with_radial(self):
        """Halbach fundamental phase matches radial convention (d-axis at
        theta=0). A consistent placement offset would corrupt sheet_phase
        d-axis conventions (demag screen, armature torque)."""
        n_p = 4
        _, Br_rad = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            **_H_FAST,
        )
        _, Br_h = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=4, **_H_FAST,
        )
        phase_rad = np.angle(np.fft.rfft(Br_rad)[n_p])
        phase_h = np.angle(np.fft.rfft(Br_h)[n_p])
        delta = phase_h - phase_rad
        while delta > np.pi:
            delta -= 2 * np.pi
        while delta < -np.pi:
            delta += 2 * np.pi
        assert abs(delta) < 0.05, (
            f"Halbach fundamental phase offset {np.degrees(delta):.1f} deg "
            f"from radial — d-axis alignment broken"
        )

    def test_explicit_angles_match_ideal_rule(self):
        """Passing halbach_angles_rad equal to the ideal-rule values
        produces bit-identical Br — pins indexing, mirror convention,
        and entry-0-is-d-axis."""
        from math import pi as _pi
        n_p = 4
        kappa_m = 3
        seg_pitch = _pi / (n_p * kappa_m)
        sign = -1.0  # inrunner
        ideal_angles = tuple(
            sign * n_p * j * seg_pitch for j in range(kappa_m)
        )
        _, Br_ideal = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=kappa_m, **_H_FAST,
        )
        _, Br_explicit = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=kappa_m,
            halbach_angles_rad=ideal_angles, **_H_FAST,
        )
        np.testing.assert_allclose(Br_explicit, Br_ideal, atol=1e-12)
        # Guard the vacuous twin: if halbach_angles_rad were silently
        # dropped, explicit == ideal would hold trivially. A perturbed
        # middle segment must move the field (measures 23% here).
        perturbed = tuple(a + (0.5 if j == 1 else 0.0)
                          for j, a in enumerate(ideal_angles))
        _, Br_pert = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=kappa_m,
            halbach_angles_rad=perturbed, **_H_FAST,
        )
        Br_pert, Br_ideal = np.asarray(Br_pert), np.asarray(Br_ideal)
        assert (np.max(np.abs(Br_pert - Br_ideal))
                > 0.05 * np.max(np.abs(Br_ideal)))

    def test_kappa1_near_radial(self):
        """kappa_m=1 ideal Halbach is close to radial but not identical:
        each segment carries a constant M at its center angle, while
        radial has M = B_rem * rhat(theta) varying continuously. The
        tangential component within each segment shifts the fundamental
        by ~10-15%."""
        t_rad, Br_rad = solve_field_fem(
            _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            **_H_FAST,
        )
        t_h, Br_h = solve_field_fem(
            _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=1, **_H_FAST,
        )
        h_rad = harmonics_1sided(Br_rad)
        h_h = harmonics_1sided(Br_h)
        assert h_h[4] == pytest.approx(h_rad[4], rel=0.15)

    def test_flux_concentration_inrunner(self):
        """Halbach kappa_m=4: airgap B1 higher AND rotor-side B1 lower
        than radial — pins the (1-p) concentration direction."""
        from phasesweep.solvers.fem_sampling import sample_Br
        _, _, m_rad, g_rad = solve_field_fem(
            _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            return_full=True, **_H_FAST,
        )
        _, _, m_h, g_h = solve_field_fem(
            _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=4,
            return_full=True, **_H_FAST,
        )
        r_ag = _HALBACH_GEO.r_ag
        r_back = _HALBACH_GEO.r_rotor + 0.001
        _, Br_ag_rad = sample_Br(m_rad, g_rad, r_ag, 120)
        _, Br_ag_h = sample_Br(m_h, g_h, r_ag, 120)
        _, Br_back_rad = sample_Br(m_rad, g_rad, r_back, 120)
        _, Br_back_h = sample_Br(m_h, g_h, r_back, 120)
        h_ag_rad = harmonics_1sided(Br_ag_rad)
        h_ag_h = harmonics_1sided(Br_ag_h)
        h_back_rad = harmonics_1sided(Br_back_rad)
        h_back_h = harmonics_1sided(Br_back_h)
        assert h_ag_h[4] > h_ag_rad[4] * 1.05
        assert h_back_h[4] < h_back_rad[4] * 0.95

    def test_flux_concentration_outrunner(self):
        """Outrunner Halbach concentrates field inward: airgap B1 at
        least as high as radial, AND outer-shell field reduced."""
        from phasesweep.solvers.fem_sampling import sample_Br
        geo = outrunner(
            r_outer=0.060, r_rotor=0.048, r_magnet=0.040,
            r_stator=0.030, r_inner=0.020,
        )
        _, _, m_rad, g_rad = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            return_full=True, **_H_FAST,
        )
        _, _, m_h, g_h = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=4,
            return_full=True, **_H_FAST,
        )
        r_ag = geo.r_ag
        r_back = geo.r_rotor - 0.001
        _, Br_ag_rad = sample_Br(m_rad, g_rad, r_ag, 120)
        _, Br_ag_h = sample_Br(m_h, g_h, r_ag, 120)
        _, Br_back_rad = sample_Br(m_rad, g_rad, r_back, 120)
        _, Br_back_h = sample_Br(m_h, g_h, r_back, 120)
        h_ag_rad = harmonics_1sided(Br_ag_rad)
        h_ag_h = harmonics_1sided(Br_ag_h)
        h_back_rad = harmonics_1sided(Br_back_rad)
        h_back_h = harmonics_1sided(Br_back_h)
        assert h_ag_h[4] >= h_ag_rad[4] * 0.99
        assert h_back_h[4] < h_back_rad[4]

    def test_rotation_preserves_amplitude_and_shifts_phase(self):
        """Halbach rotated by phi: amplitude preserved, fundamental
        phase shifted by -n_p*phi (mechanical-to-electrical)."""
        n_p = 4
        phi = 0.15
        t0, Br0 = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=3, **_H_FAST,
        )
        t1, Br1 = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=3,
            rotation=phi, **_H_FAST,
        )
        h0 = harmonics_1sided(Br0)
        h1 = harmonics_1sided(Br1)
        assert h1[n_p] == pytest.approx(h0[n_p], rel=0.02)
        fft0 = np.fft.rfft(Br0)
        fft1 = np.fft.rfft(Br1)
        phase0 = np.angle(fft0[n_p])
        phase1 = np.angle(fft1[n_p])
        expected_shift = -n_p * phi
        actual_shift = phase1 - phase0
        while actual_shift > np.pi:
            actual_shift -= 2 * np.pi
        while actual_shift < -np.pi:
            actual_shift += 2 * np.pi
        assert actual_shift == pytest.approx(expected_shift, abs=0.05)

    def test_halbach_b1_ratio_regression(self):
        """Pin the Halbach/radial B1 ratio as a regression literal.
        kappa_m=4 ideal Halbach concentrates flux ~19% above radial
        for this geometry (the curl-of-M source involves both M_r and
        M_theta; a simple M_r Fourier ratio doesn't predict B1 — the
        full BVP is needed, validated in Phase 5 against Zhang 2021)."""
        n_p = 4
        kappa_m = 4
        t_rad, Br_rad = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            **_H_FAST,
        )
        t_h, Br_h = solve_field_fem(
            _HALBACH_GEO, n_p=n_p, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=kappa_m, **_H_FAST,
        )
        h_rad = harmonics_1sided(Br_rad)
        h_h = harmonics_1sided(Br_h)
        ratio = h_h[n_p] / h_rad[n_p]
        assert ratio == pytest.approx(1.19, rel=0.03)

    def test_bad_magnetization_string_raises(self):
        with pytest.raises(ValueError, match="magnetization"):
            solve_field_fem(
                _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="Halbach", **_H_FAST,
            )

    def test_halbach_without_segments_raises(self):
        with pytest.raises(ValueError, match="halbach_segments"):
            solve_field_fem(
                _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="halbach", **_H_FAST,
            )

    def test_halbach_alpha_p_not_one_raises(self):
        with pytest.raises(ValueError, match="alpha_p"):
            solve_field_fem(
                _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="halbach", halbach_segments=3, alpha_p=0.8,
                **_H_FAST,
            )

    def test_halbach_angles_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            solve_field_fem(
                _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="halbach", halbach_segments=3,
                halbach_angles_rad=(0.0, -0.7), **_H_FAST,
            )

    def test_nonlinear_halbach_converges(self):
        """Nonlinear Picard solve completes on a Halbach geometry."""
        _, Br = solve_field_fem(
            _HALBACH_GEO, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=3,
            nonlinear=True, **_H_FAST,
        )
        h = harmonics_1sided(Br)
        assert h[4] > 0


# ---------------------------------------------------------------------------
# Model-free field invariants — these fail no matter which
# solver path silently ran, so they backstop the whole silent-fallback class.
# ---------------------------------------------------------------------------

class TestAirgapFieldInvariants:

    # divisible by 2*n_p (half-wave shift) and by the 12-slot/4-pole-pair
    # common period
    N_T = 80
    N_P = 4

    @pytest.fixture(scope="class")
    def arc_solution(self):
        geo = default_inrunner()
        theta, B_r, mesh, gfu = solve_field_fem(
            geo=geo, n_p=self.N_P, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
            alpha_p=0.75, return_full=True, n_theta=self.N_T,
            maxh_fraction=0.08,
        )
        return geo, np.asarray(B_r), mesh, gfu

    def _gap_radii(self, geo):
        g = geo.r_stator - geo.r_magnet
        return [geo.r_magnet + f * g for f in (0.15, 0.5, 0.85)]

    def test_net_radial_flux_zero_across_gap(self, arc_solution):
        """No net flux crosses any closed circle in the airgap (div B = 0,
        no monopoles): the mean of B_r on a uniform grid vanishes relative
        to the waveform scale. Measured <= 9.2e-5 at this config."""
        from phasesweep.solvers.fem_sampling import sample_Br

        geo, _, mesh, gfu = arc_solution
        for r in self._gap_radii(geo):
            _, br = sample_Br(mesh, gfu, r, n_theta=self.N_T)
            assert abs(np.mean(br)) < 1e-3 * np.max(np.abs(br))

    def test_fundamental_obeys_laplace_radial_decay(self, arc_solution):
        """In the current-free airgap each harmonic of B_r follows
        a*r^(n-1) + b*r^(-n-1) (Laplace) regardless of the boundary
        conditions that produced it. Fit (a, b) on the inner and outer
        sampling circles and predict the middle one — a wrong-radius
        sampling regression (the historical midgap/bore class) breaks
        this while leaving each single-radius waveform plausible.
        Measured residual 4.1e-3 (FEM discretization); 2e-2 bound."""
        from phasesweep.solvers.fem_sampling import sample_Br

        geo, _, mesh, gfu = arc_solution
        n = self.N_P
        r1, r2, r3 = self._gap_radii(geo)
        c = []
        for r in (r1, r2, r3):
            _, br = sample_Br(mesh, gfu, r, n_theta=self.N_T)
            c.append(np.fft.rfft(br)[n] / self.N_T)
        A = np.array([[r1 ** (n - 1), r1 ** (-n - 1)],
                      [r3 ** (n - 1), r3 ** (-n - 1)]])
        scale = max(abs(x) for x in c)
        for part in (np.real, np.imag):
            a, b = np.linalg.solve(A, np.array([part(c[0]), part(c[2])]))
            pred = a * r2 ** (n - 1) + b * r2 ** (-n - 1)
            assert abs(pred - part(c[1])) < 2e-2 * scale

    def test_pole_pair_periodicity_and_antisymmetry(self, arc_solution):
        """The smooth-bore waveform repeats every 2*pi/n_p and flips sign
        every pi/n_p (alternating pole signs). The unstructured mesh is
        not n_p-fold symmetric, so the residual is discretization-level:
        measured 1.3e-3 / 1.6e-3; 1e-2 bound."""
        _, B_r, _, _ = arc_solution
        peak = np.max(np.abs(B_r))
        s_pp = self.N_T // self.N_P
        s_hp = self.N_T // (2 * self.N_P)
        assert np.max(np.abs(np.roll(B_r, s_pp) - B_r)) < 1e-2 * peak
        assert np.max(np.abs(np.roll(B_r, s_hp) + B_r)) < 1e-2 * peak

    def test_slotted_pole_pair_periodicity(self):
        """Periodicity survives slotting when the slot count shares the
        pole-pair period (12 slots / 4 pole pairs = 3 slots per period).
        Measured 5.8e-3 (slot-edge sampling noise); 3e-2 bound."""
        geo = inrunner(**_SLOTTED_GEO_KW)
        _, B_r = solve_field_fem(
            geo=geo, n_p=self.N_P, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
            alpha_p=0.75, n_theta=self.N_T, maxh_fraction=0.08,
        )
        B_r = np.asarray(B_r)
        s_pp = self.N_T // self.N_P
        assert (np.max(np.abs(np.roll(B_r, s_pp) - B_r))
                < 3e-2 * np.max(np.abs(B_r)))

    def test_rotation_by_pole_pair_is_identity(self, arc_solution):
        """Rotating the magnet pattern by a full pole-pair pitch maps the
        arc set and sign pattern onto themselves. The rotated build meshes
        the geometrically identical arc set, so the solve reproduces the
        rotation=0 field to near round-off (measured 6.2e-7; 1e-4 leaves
        meshing-nondeterminism headroom)."""
        geo, B_0, _, _ = arc_solution
        _, B_rot = solve_field_fem(
            geo=geo, n_p=self.N_P, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
            alpha_p=0.75, rotation=2 * np.pi / self.N_P,
            n_theta=self.N_T, maxh_fraction=0.08,
        )
        assert (np.max(np.abs(np.asarray(B_rot) - B_0))
                < 1e-4 * np.max(np.abs(B_0)))
