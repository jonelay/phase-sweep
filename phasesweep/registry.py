"""Model registry mapping model keys to metadata and solver functions.

Registry is a plain dict. No Protocol/ABC/plugin system.

Validation uses the prepare_* factories (each living next to its model's
run function) as the single source of truth: they validate Motor
completeness and derive missing fields (B_rem <-> psi_f) via the shared
solver_params utilities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig


@dataclass(frozen=True)
class ModelInfo:
    """Metadata and execution hooks for one registered model.

    ``needs`` lists motor fields required for applicability validation.
    ``needs_runtime`` lists additional ``RunConfig`` fields that have no
    usable default and whose absence makes the model runner raise.
    """

    name: str
    source: Literal["computed", "measured"]
    cost: Literal["fast", "medium", "slow", "none"]
    produces: frozenset[str]
    needs: frozenset[str]
    hash_fields: frozenset[str]
    needs_runtime: frozenset[str] = frozenset()
    # Model-code version: bump when the model's OUTPUTS change with no
    # input change (physics fixes), so stale results cannot serve from
    # the result store. v1 is omitted from run-ID hashing for backward
    # compatibility. v2: square-wave magnetization convention +
    # Carter-consistent B_rem inversion (affects derived psi_f / B_rem,
    # hence all computed models). v3: Carter-consistent psi_f resolution
    # (drive_sim/rated_torque/stall_torque), BH vacuum-slope extrapolation
    # past 2.2 T (fem nonlinear), analytical waveform evaluated at
    # geo.r_ag (matches FEM sampling radius). fem v4: raw-residual Picard
    # stop criterion + relax recovery — all nonlinear FEM outputs shift.
    # fem v5: per-face airgap mesh refinement (maxh = gap/3) +
    # sample_Br direct evaluation (no L2 projection) — all FEM outputs
    # shift slightly (~0.1% B₁ on thin-gap motors, ~1e-5 T elsewhere).
    # analytical v4: odd-harmonic series — waveform/THD/peak change,
    # fundamental unchanged. fem v6: per-face pm_gap mesh refinement
    # (maxh = gap/2) — outputs shift only for
    # interpole gaps in the 0.5–1.0 mm band (previously floored at
    # 0.5 mm global maxh); wider-gap meshes are unchanged.
    # fem v7: _interpole_gap_width measured at the inner PM-annulus radius
    # (the wedge's narrowest edge) instead of the outer radius — the
    # collapse threshold and pm_gap/global-maxh clamp both used an
    # overestimated gap. Finer gap mesh shifts every alpha_p<1 FEM output
    # (mm-scale arc motors most, where the clamp binds the global maxh);
    # near-threshold runs may now collapse to the full ring where they
    # previously built arcs. mesh cache prefix -> mesh_v6.
    # fem v8 (+ fem_axial/cogging_torque/demag_screen v2): BH table
    # interpolated with a monotone PCHIP spline instead of piecewise
    # linear (nu shifts up to ~6% around the knee, 20% below 0.1 T),
    # analytic d nu/dB / dB/dH from the spline, and Newton (line-search
    # damped, relaxed-Picard start-up, Picard fallback) becomes the
    # default nonlinear solver — every nonlinear FEM output shifts;
    # linear solves unchanged. Mesh cache prefix unchanged (mesh_v6).
    # (no bump): Halbach magnetization — input-schema addition with
    # identity-preserving default (magnetization="radial"). Radial
    # motors produce byte-identical mesh, source CF, and outputs.
    # torque_speed v2: envelope optimizer vectorized over the speed axis
    # (perf audit F1, ~5.7x). Same 601-point i_d grid and 80 golden-section
    # contractions, but the refinement now reuses the surviving probe
    # (one kernel evaluation per contraction, the textbook form) and the
    # kernel's speed-only terms are hoisted, so the arithmetic reassociates.
    # tau_curve_* shift by <= 4e-14 relative, always toward the better
    # optimum; base/max_speed and the MTPA branch are untouched.
    # torque_speed v3: bracketing i_d grid thinned 601 -> 151 points. The
    # grid only brackets; the 80 golden-section contractions resolve the
    # optimum from there, so this costs no accuracy — against a 2001-point
    # reference, every grid from 601 down to 31 agrees to <= 2e-11 of curve
    # scale across 10195 swept synthetic designs (saliency 0.01-100,
    # near-resistive-limit and unbounded-CPSR included). On the four bundled
    # model, tau_curve_* shift by <= 7.8e-14 relative (creator_case_pmsm;
    # <= 8.2e-16 on the rest) — the refinement converges from a different
    # bracket, so only the last bits move. On rare designs it can reach
    # ~2e-9 relative (either direction) where a different local bracket is
    # selected near the current-circle/voltage-limit kink.
    # base/max_speed and the MTPA branch are untouched. ~1.6x alone, and
    # it unblocks batching: with the grid scan no longer dominant, a
    # 200-design batch runs 6.1x faster per design than v2 serial.
    version: int = 1
    # True when waveform outputs contain only the fundamental, so
    # waveform-shape extracts (max/min) are not meaningful comparands.
    # No registry model sets it since analytical v4 landed the
    # odd-harmonic series; the crossval skip machinery it drives is kept
    # for future fundamental-only entries.
    fundamental_only: bool = False
    validate: Callable[[Motor], None] | None = None
    fn: Callable[[RunConfig], dict[str, Any]] | None = None
    # Execution-strategy hint only: does not enter hash_fields or
    # compute_run_id. Outputs must be byte-identical to fn per config.
    batch_fn: Callable[[Sequence[RunConfig]], list[dict[str, Any] | Exception]] | None = None


# ---------------------------------------------------------------------------
# Validation via solver_params factories (construct and discard)
# ---------------------------------------------------------------------------

def _validate_analytical(motor: Motor) -> None:
    from phasesweep.solvers.analytical import prepare_analytical
    prepare_analytical(motor)


def _validate_analytical_axial(motor: Motor) -> None:
    from phasesweep.solvers.analytical_axial import prepare_analytical_axial
    prepare_analytical_axial(motor)


def _validate_fem(motor: Motor) -> None:
    from phasesweep.solvers.fem_runner import prepare_fem
    prepare_fem(motor)


def _validate_fem_axial(motor: Motor) -> None:
    from phasesweep.solvers.fem_axial import prepare_fem_axial
    prepare_fem_axial(motor)


def _validate_drive_sim(motor: Motor) -> None:
    from phasesweep.simulation.params import prepare_drive_sim
    prepare_drive_sim(motor)


def _validate_rated_torque(motor: Motor) -> None:
    from phasesweep.models.rated_torque import prepare_rated_torque
    prepare_rated_torque(motor)


def _validate_stall_torque(motor: Motor) -> None:
    from phasesweep.models.rated_torque import prepare_stall_torque
    prepare_stall_torque(motor)


def _validate_thermal_duty(motor: Motor) -> None:
    from phasesweep.models.thermal_duty import _check_thermal_duty_motor
    _check_thermal_duty_motor(motor)


def _validate_torque_speed(motor: Motor) -> None:
    from phasesweep.models.torque_speed import prepare_torque_speed
    prepare_torque_speed(motor)


def _validate_iron_loss(motor: Motor) -> None:
    from phasesweep.models.iron_loss import prepare_iron_loss
    prepare_iron_loss(motor)


def _validate_demag_screen(motor: Motor) -> None:
    from phasesweep.models.demag_screen import prepare_demag_screen
    prepare_demag_screen(motor)


# ---------------------------------------------------------------------------
# Analytical runner
# ---------------------------------------------------------------------------

def _run_analytical_impl(config: RunConfig) -> dict[str, Any]:
    from phasesweep.solver_params import psi_f_carter
    from phasesweep.solvers.analytical import (
        carter_adjusted_radii,
        end_effect_factor,
        prepare_analytical,
        zhu_howe_Br_series,
    )
    from phasesweep.solvers.harmonics import compute_thd, harmonics_1sided

    params = prepare_analytical(config.motor)
    geo = params.geometry
    n_p = params.n_p

    # Carter factor correction for slotted stators (Zhu & Howe 1993 III)
    r_stator, r_magnet, _k_c = carter_adjusted_radii(geo, params.mu_r_pm)

    if config.n_theta <= 2 * n_p + 1:
        # before the solve — a bad config must not cost the series eval.
        # Strictly stricter than the FFT-bin bound (n_theta > 2*n_p): at
        # n_theta == 2*n_p + 1 the series' max_order = n_theta // 2 == n_p
        # cutoff drops the k=1 term and the waveform is silently all zeros.
        raise ValueError(
            f"n_theta={config.n_theta} cannot resolve the fundamental at "
            f"n_p={n_p}; need n_theta >= 2*n_p + 2"
        )
    theta = np.linspace(0, 2 * np.pi, config.n_theta, endpoint=False)
    # r_eval = geo.r_ag matches the FEM sampling radius (always interior to
    # the Carter-widened annulus: Carter moves the bore away from the magnets)
    # Odd-harmonic series of the square-wave magnetization; the
    # fundamental bin is identical to the old single-term zhu_howe_Br.
    # max_order keeps sampled orders below the FFT Nyquist bin.
    B_r = zhu_howe_Br_series(
        theta, n_p, params.B_rem, r_eval=geo.r_ag,
        r_stator=r_stator, r_magnet=r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=params.mu_r_pm, alpha_p=params.alpha_p,
        max_order=config.n_theta // 2,
    )

    amps = harmonics_1sided(B_r)
    fundamental = float(amps[n_p])
    thd_pct = compute_thd(amps, n_p)

    m = config.motor
    # Carter radii for the field, original bore for the evaluation point
    # and winding formula (mirrors _derive_B_rem_from_psi_f; the midgap `fundamental`
    # above must NOT be used here — it is off by the B1 midgap/bore ratio,
    # high for inrunners, low for outrunners)
    flux_linkage_peak = psi_f_carter(m, r_stator, r_magnet, params.B_rem)
    if flux_linkage_peak is not None:
        flux_linkage_peak = float(flux_linkage_peak)

    # Computed per-motor end-effect factor (Motor.k_end, from the offline
    # 3D wedge solve) corrects the B_rem-derived flux linkage. Gated on
    # B_rem being a true input: when B_rem was itself derived from a
    # measured psi_f, that psi_f already includes end effects and the
    # correction would double-count them.
    if (m.k_end is not None and m.B_rem is not None
            and flux_linkage_peak is not None):
        k_end_applied = True
        flux_linkage_peak = float(flux_linkage_peak * m.k_end)
    else:
        k_end_applied = False

    # Explicit psi_f wins; otherwise derive Carter-consistently (Carter
    # radii for the field, original bore for the winding) — which is
    # exactly flux_linkage_peak (k_end-corrected above when available)
    backemf_fundamental = None
    if m.drive.W_REF is not None:
        w_e = m.drive.W_REF * m.n_p
        if m.psi_f is not None:
            backemf_fundamental = float(w_e * m.psi_f)
        elif flux_linkage_peak is not None:
            backemf_fundamental = float(w_e * flux_linkage_peak)

    # The reported k_end: the motor's computed value when present;
    # otherwise the uncalibrated analytical heuristic, informational only
    # and NOT applied (it disagrees with the 3D wedge FEM by ~8 points /
    # 9% relative on the one motor where both exist). k_end_applied says
    # whether flux_linkage_peak above actually carries the correction —
    # a "computed" source with k_end_applied False means the psi_f-derived
    # double-count guard suppressed it.
    if m.k_end is not None:
        k_end = m.k_end
        k_end_source = "computed"
    elif m.L_stk is not None:
        g = abs(geo.r_stator - geo.r_magnet)
        h_m = abs(geo.r_magnet - geo.r_rotor)
        k_end = float(end_effect_factor(m.L_stk, g + h_m / params.mu_r_pm))
        k_end_source = "heuristic"
    else:
        k_end = None
        k_end_source = None

    return {
        "theta_list": theta.tolist(),
        "B_r_list": B_r.tolist(),
        "fundamental": fundamental,
        "peak_Br": float(np.max(np.abs(B_r))),
        "thd_pct": thd_pct,
        "backemf_fundamental": backemf_fundamental,
        "flux_linkage_peak": flux_linkage_peak,
        "k_end": k_end,
        "k_end_source": k_end_source,
        "k_end_applied": k_end_applied,
    }


# ---------------------------------------------------------------------------
# Axial analytical runner
# ---------------------------------------------------------------------------

def _run_analytical_axial_impl(config: RunConfig) -> dict[str, Any]:
    from math import pi

    from phasesweep.solver_params import n_eff
    from phasesweep.solvers.analytical_axial import (
        GAP_LINKAGE,
        axial_layer_Bz_series,
        carter_airgap_axial,
        pm_leakage_factor,
        prepare_analytical_axial,
        winding_transfer_axial,
    )
    from phasesweep.solvers.harmonics import compute_thd, harmonics_1sided

    params = prepare_analytical_axial(config.motor)
    geo = params.geometry
    n_p = params.n_p

    if config.n_theta <= 2 * n_p + 1:
        # same bound as the radial runner: at n_theta == 2*n_p + 1 the
        # series' max_order cutoff drops the k=1 term silently.
        raise ValueError(
            f"n_theta={config.n_theta} cannot resolve the fundamental at "
            f"n_p={n_p}; need n_theta >= 2*n_p + 2"
        )

    tau_p = pi * geo.r_mean / n_p
    g_prime, k_c = carter_airgap_axial(geo, params.mu_r_pm)

    theta = np.linspace(0, 2 * np.pi, config.n_theta, endpoint=False)
    B_z = axial_layer_Bz_series(
        theta, n_p, params.B_rem,
        tau_p=tau_p, g_prime=g_prime, l_pm=geo.magnet_thickness,
        mu_r_pm=params.mu_r_pm, alpha_p=params.alpha_p,
        max_order=config.n_theta // 2,
    )

    amps = harmonics_1sided(B_z)
    fundamental = float(amps[n_p])
    thd_pct = compute_thd(amps, n_p)

    k_sigma_pm = float(pm_leakage_factor(
        geo, params.mu_r_pm, params.alpha_p, g_prime, tau_p,
    ))

    # Flux linkage: leakage-corrected fundamental through the axial
    # winding transfer, times the gaps the winding circuit links.
    m = config.motor
    flux_linkage_peak = None
    if m.N is not None and m.k_w is not None and m.coils_series is not None:
        flux_linkage_peak = float(
            GAP_LINKAGE[geo.variant]
            * k_sigma_pm * fundamental
            * winding_transfer_axial(geo, n_eff(m), m.k_w, n_p)
        )

    # Explicit psi_f wins; otherwise the model-derived flux linkage
    backemf_fundamental = None
    if m.drive.W_REF is not None:
        w_e = m.drive.W_REF * m.n_p
        if m.psi_f is not None:
            backemf_fundamental = float(w_e * m.psi_f)
        elif flux_linkage_peak is not None:
            backemf_fundamental = float(w_e * flux_linkage_peak)

    return {
        "theta_list": theta.tolist(),
        "B_r_list": B_z.tolist(),
        "fundamental": fundamental,
        "peak_Br": float(np.max(np.abs(B_z))),
        "thd_pct": thd_pct,
        "backemf_fundamental": backemf_fundamental,
        "flux_linkage_peak": flux_linkage_peak,
        "k_c": float(k_c),
        "k_sigma_pm": k_sigma_pm,
    }


def _run_fem_axial_impl(config: RunConfig) -> dict[str, Any]:
    """Multi-plane quasi-3D FEM: K unrolled Cartesian planes.

    The solver returns the geometric flux sum S = Σ_k B1(r_k)·r_k·Δr
    and the radial edge-leakage factor; the winding transfer
    (GAP_LINKAGE, N_eff, k_w) is applied here so the axial models share
    one flux-linkage convention.
    """
    from phasesweep.solver_params import n_eff
    from phasesweep.solvers.analytical_axial import GAP_LINKAGE
    from phasesweep.solvers.fem_axial import prepare_fem_axial, solve_axial_multiplane
    from phasesweep.solvers.harmonics import compute_thd, harmonics_1sided

    params = prepare_fem_axial(config.motor, n_planes=config.n_planes,
                               j_s=config.j_s)
    geo = params.geometry
    n_p = params.n_p

    if config.n_theta <= 2 * n_p + 1:
        # same bound as the other axial/radial runners
        raise ValueError(
            f"n_theta={config.n_theta} cannot resolve the fundamental at "
            f"n_p={n_p}; need n_theta >= 2*n_p + 2"
        )

    out = solve_axial_multiplane(
        geo, n_p, params.B_rem, params.mu_r_pm, params.mu_r_fe,
        params.alpha_p,
        n_planes=params.n_planes,
        maxh_fraction=config.maxh_fraction,
        n_theta=config.n_theta,
        nonlinear=config.nonlinear,
        nonlinear_solver=config.nonlinear_solver,
    )

    theta = out["theta_list"]
    B_z = out["B_z_list"]
    amps = harmonics_1sided(B_z)
    thd_pct = compute_thd(amps, n_p)

    m = config.motor
    flux_linkage_peak = None
    if m.N is not None and m.k_w is not None and m.coils_series is not None:
        # 2·N_eff·k_w·S/n_p is winding_transfer_axial with the
        # per-plane sum standing in for B1·r_mean·active_length.
        flux_linkage_peak = float(
            GAP_LINKAGE[geo.variant]
            * out["k_sigma_edge"]
            * 2.0 * n_eff(m) * m.k_w * out["S"] / n_p
        )

    backemf_fundamental = None
    if m.drive.W_REF is not None:
        w_e = m.drive.W_REF * m.n_p
        if m.psi_f is not None:
            backemf_fundamental = float(w_e * m.psi_f)
        elif flux_linkage_peak is not None:
            backemf_fundamental = float(w_e * flux_linkage_peak)

    return {
        "theta_list": theta.tolist(),
        "B_r_list": B_z.tolist(),
        "fundamental": float(out["fundamental"]),
        "peak_Br": float(out["peak"]),
        "thd_pct": thd_pct,
        "backemf_fundamental": backemf_fundamental,
        "flux_linkage_peak": flux_linkage_peak,
        "k_sigma_edge": float(out["k_sigma_edge"]),
        "plane_r_list": out["plane_r_list"],
        "plane_B1_list": out["plane_B1_list"],
    }


# ---------------------------------------------------------------------------
# Rated torque runner (delegated to rated_torque module)
# ---------------------------------------------------------------------------

def _lazy_rated_torque(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.rated_torque import run_rated_torque
    return run_rated_torque(config)


def _lazy_stall_torque(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.rated_torque import run_stall_torque
    return run_stall_torque(config)


def _lazy_thermal_duty(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.thermal_duty import run_thermal_duty
    return run_thermal_duty(config)


def _lazy_torque_speed(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.torque_speed import run_torque_speed
    return run_torque_speed(config)


def _lazy_torque_speed_batch(
        configs: Sequence[RunConfig]) -> list[dict[str, Any] | Exception]:
    from phasesweep.models.torque_speed import run_torque_speed_batch
    return run_torque_speed_batch(configs)


def _lazy_iron_loss(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.iron_loss import run_iron_loss
    return run_iron_loss(config)


def _lazy_demag_screen(config: RunConfig) -> dict[str, Any]:
    _check_fem("demag_screen")
    from phasesweep.models.demag_screen import run_demag_screen
    return run_demag_screen(config)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelInfo] = {
    "analytical": ModelInfo(
        name="analytical",
        source="computed",
        # v4: odd-harmonic square-wave series — B_r_list becomes the
        # full waveform (was pure fundamental), thd_pct becomes non-zero,
        # peak_Br added; the fundamental bin is unchanged.
        # (v4, no bump): optional Motor.k_end (computed 3D wedge factor,
        # input-schema change — hashed into Motor.config_id) now corrects
        # the B_rem-derived flux_linkage_peak / backemf_fundamental;
        # outputs unchanged for motors without k_end. Additive
        # k_end_source ("computed" | "heuristic" | None) and
        # k_end_applied (bool: whether flux_linkage_peak carries the
        # correction) outputs.
        version=4,
        cost="fast",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "peak_Br", "thd_pct",
            "backemf_fundamental", "flux_linkage_peak", "k_end",
            "k_end_source", "k_end_applied",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p", "mu_r_pm"}),
        hash_fields=frozenset({"n_theta", "W_REF"}),
        validate=_validate_analytical,
        fn=_run_analytical_impl,
    ),
    "analytical_axial": ModelInfo(
        name="analytical_axial",
        source="computed",
        # v1: quasi-3D Cartesian layer model at r_mean (Parviainen 2005
        # Ch. 2): odd-harmonic series, Carter on the physical gap
        # (Eq. 2.11), Qu PM-leakage factor into flux_linkage_peak.
        # B_r_list holds B_z at the stator surface of one gap.
        version=1,
        cost="fast",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "peak_Br", "thd_pct",
            "backemf_fundamental", "flux_linkage_peak", "k_c", "k_sigma_pm",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p", "mu_r_pm"}),
        hash_fields=frozenset({"n_theta", "W_REF"}),
        validate=_validate_analytical_axial,
        fn=_run_analytical_axial_impl,
    ),
    "fem_axial": ModelInfo(
        name="fem_axial",
        source="computed",
        # v1: multi-plane quasi-3D 2D Cartesian FEM.
        # K unrolled mirror planes at midpoint radii, A_z with Dirichlet
        # on the magnet-center planes; explicit slots (no Carter);
        # flux linkage from the per-plane sum with a radial-edge
        # leakage factor. B_r_list holds B_z of the middle plane.
        # (v2): PCHIP BH spline + Newton-default nonlinear solve
        # (shared fem_field/fem_materials change, see ModelInfo
        # comment); nonlinear outputs only.
        version=2,
        cost="slow",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "peak_Br", "thd_pct",
            "backemf_fundamental", "flux_linkage_peak", "k_sigma_edge",
            "plane_r_list", "plane_B1_list",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p",
                         "mu_r_pm", "mu_r_fe"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear",
                               "nonlinear_solver", "n_planes", "W_REF"}),
        validate=_validate_fem_axial,
        fn=lambda config: _lazy_fem_axial(config),
    ),
    "fem": ModelInfo(
        name="fem",
        source="computed",
        # (v5, no bump): additive Maxwell-stress torque outputs
        # tau_maxwell_per_m / tau_maxwell on armature (j_s != 0) runs;
        # existing outputs unchanged. Also: j_s != 0 with no slot
        # faces in the mesh (slot_depth = 0) is now a loud ValueError —
        # it was silently ignored before.
        # (v5, no bump): additive cross-section raster
        # (B_mag_grid / grid_coords_list) for the dashboard heatmap;
        # existing outputs unchanged.
        # (v6): per-face pm_gap refinement, see ModelInfo comment.
        # Additive info flag arcs_collapsed on sub-0.5 mm interpole gaps.
        # (v7): interpole gap width measured at the inner radius,
        # see ModelInfo comment; mesh prefix -> mesh_v6.
        # (v8): PCHIP BH spline + Newton-default nonlinear solve,
        # see ModelInfo comment; nonlinear outputs only.
        version=8,
        cost="slow",
        produces=frozenset({
            "theta_list", "B_r_list", "peak_Br", "fundamental", "thd_pct",
            "sh_pct", "sh_upper_pct", "b_iron_max",
            "tau_maxwell_per_m", "tau_maxwell",
            "B_mag_grid", "grid_coords_list",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear",
                               "nonlinear_solver", "j_s", "rotation"}),
        validate=_validate_fem,
        fn=lambda config: _lazy_fem(config),
    ),
    "drive_sim": ModelInfo(
        name="drive_sim",
        source="computed",
        # (v3, no bump): additive downsampled time-domain traces
        # (t/w_M/tau_M/|i_s|) for the dashboard sim-waveforms panel;
        # scalar outputs unchanged.
        version=3,
        cost="medium",
        produces=frozenset({
            "t_settle", "i_ss", "speed_droop", "tau_peak",
            "t_list", "w_M_list", "tau_M_list", "i_s_abs_list",
        }),
        needs=frozenset({"R_s", "L_d", "L_q", "psi_f", "J", "n_p"}),
        hash_fields=frozenset({
            "sim_plan",
            "U_DC", "MAX_I_S", "W_REF", "I_LIMIT",
        }),
        needs_runtime=frozenset({"sim_plan"}),
        validate=_validate_drive_sim,
        fn=lambda config: _lazy_sim(config),
    ),
    "drive_sim_two_mass": ModelInfo(
        name="drive_sim_two_mass",
        source="computed",
        cost="medium",
        produces=frozenset({
            "t_settle", "i_ss", "speed_droop", "tau_peak", "tau_S_peak",
            "t_list", "w_M_list", "tau_M_list", "i_s_abs_list",
            "w_L_list", "tau_S_list",
        }),
        needs=frozenset({"R_s", "L_d", "L_q", "psi_f", "J", "n_p"}),
        hash_fields=frozenset({
            "sim_plan", "load_mech",
            "U_DC", "MAX_I_S", "W_REF", "I_LIMIT",
        }),
        needs_runtime=frozenset({"sim_plan", "load_mech"}),
        validate=_validate_drive_sim,
        fn=lambda config: _lazy_sim(config),
    ),
    "rated_torque": ModelInfo(
        name="rated_torque",
        source="computed",
        # v4: reverse-salient (L_d > L_q) machines take the MTPA branch
        # (gamma < 0, magnetizing i_d) instead of degrading to k_T·I_rated;
        # adds k_T_effective output.
        version=4,
        cost="fast",
        produces=frozenset({
            "tau_mtpa", "k_T", "k_T_rms", "k_T_effective", "gamma_opt_deg",
            "I_curve", "gamma_curve_deg", "angle_d_curve_deg", "tau_curve",
        }),
        needs=frozenset({"n_p", "psi_f", "I_rated"}),
        hash_fields=frozenset(),
        validate=_validate_rated_torque,
        fn=_lazy_rated_torque,
    ),
    "stall_torque": ModelInfo(
        name="stall_torque",
        source="computed",
        # v4: reverse-salient MTPA branch + k_T_effective (see rated_torque).
        version=4,
        cost="fast",
        produces=frozenset({
            "tau_stall", "I_stall", "k_T", "k_T_effective", "gamma_opt_deg",
            "tau_stall_electromagnetic", "I_stall_electromagnetic",
            "saturation_ratio", "saturation_warning",
            "I_curve", "gamma_curve_deg", "angle_d_curve_deg", "tau_curve",
        }),
        needs=frozenset({"n_p", "psi_f", "R_s"}),
        hash_fields=frozenset({"U_DC", "MAX_I_S", "I_LIMIT"}),
        validate=_validate_stall_torque,
        fn=_lazy_stall_torque,
    ),
    "thermal_duty": ModelInfo(
        name="thermal_duty",
        source="computed",
        # v2: R_s derated to the winding operating-temperature limit
        # for copper loss — raises consumed loss (and the over_budget_ratio
        # verdict) on the thermal_resistance budget path; cancels on the
        # rated_current path.
        # v3: reverse-salient torque→current inverse uses the MTPA locus
        # (was non-salient k_T mapping) — lowers thermal current for
        # L_d > L_q machines.
        # (v3, no bump): optional [iron] p_fe joins consumption +
        # rated_current budget — outputs change only when the new inputs
        # are set (input-schema rule); [iron] hashes via config_id, W_REF
        # added to hash_fields. Also: optional
        # thermal_time_constant enables the first-order transient march
        # (transient_peak_ratio/within_s1_transient/winding_temp_peak) —
        # also input-schema, hashes via config_id.
        version=3,
        cost="fast",
        produces=frozenset({
            "p_cu_avg", "p_cu_peak", "p_fe", "p_total_avg", "p_s1_budget",
            "budget_source", "over_budget_ratio", "within_s1",
            "sustainable_duty_fraction", "total_cycle_time",
            "transient_peak_ratio", "within_s1_transient",
            "winding_temp_peak",
        }),
        needs=frozenset({"n_p", "psi_f", "R_s"}),
        hash_fields=frozenset({"duty_profile", "W_REF"}),
        needs_runtime=frozenset({"duty_profile"}),
        validate=_validate_thermal_duty,
        fn=_lazy_thermal_duty,
    ),
    "torque_speed": ModelInfo(
        name="torque_speed",
        source="computed",
        cost="fast",
        version=3,
        produces=frozenset({
            "speed_curve", "tau_curve_peak", "tau_curve_cont",
            "base_speed_peak", "base_speed_cont",
            "max_speed_peak", "max_speed_cont",
            "p_max_peak", "p_max_cont", "u_max",
            "I_env_peak", "I_env_cont",
        }),
        needs=frozenset({"n_p", "psi_f", "R_s", "L_d", "L_q"}),
        hash_fields=frozenset({"U_DC", "MAX_I_S", "I_LIMIT"}),
        validate=_validate_torque_speed,
        fn=_lazy_torque_speed,
        batch_fn=_lazy_torque_speed_batch,
    ),
    "iron_loss": ModelInfo(
        name="iron_loss",
        source="computed",
        cost="fast",
        produces=frozenset({
            "p_fe", "p_fe_hysteresis", "p_fe_eddy",
            "loss_density_W_per_kg", "f_e_Hz",
        }),
        needs=frozenset({"n_p", "k_h", "k_e", "alpha_fe", "m_core", "B_core"}),
        hash_fields=frozenset({"W_REF"}),
        validate=_validate_iron_loss,
        fn=_lazy_iron_loss,
    ),
    "demag_screen": ModelInfo(
        name="demag_screen",
        source="computed",
        # v1: FEM demag screen — magnet field + pure
        # demagnetizing d-axis sheet (sheet_phase = +π/2) at i_fault,
        # worst-point B_m margin against B_knee(T).
        # (v2): PCHIP BH spline + Newton-default nonlinear solve
        # (shared fem change); nonlinear outputs only.
        version=2,
        cost="slow",
        produces=frozenset({
            "B_m_min", "B_knee", "margin", "frac_below_knee",
            "r_min", "theta_min", "i_fault", "j_s_fault", "b_iron_max",
        }),
        needs=frozenset({
            "B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p",
            "B_knee", "N", "coils_series", "k_w",
        }),
        hash_fields=frozenset({"maxh_fraction", "nonlinear",
                               "nonlinear_solver", "i_fault"}),
        needs_runtime=frozenset({"i_fault"}),
        validate=_validate_demag_screen,
        fn=_lazy_demag_screen,
    ),
    "cogging_torque": ModelInfo(
        name="cogging_torque",
        source="computed",
        # v1: rotor sweep at j_s=0 with Arkkio radial-averaging torque.
        # RunConfig.rotation is an input-schema addition with an
        # identity-preserving default (no fem version bump).
        # (v2): PCHIP BH spline + Newton-default nonlinear solve
        # (shared fem change); nonlinear outputs only.
        version=2,
        cost="slow",
        produces=frozenset({
            "rotation_list", "tau_cogging_list", "tau_cogging_pp",
            "tau_cogging_pp_Nm", "dominant_order", "n_cogging_periods",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear",
                               "nonlinear_solver", "cogging_points"}),
        validate=_validate_fem,
        fn=lambda config: _lazy_cogging(config),
    ),
    "backemf_capture": ModelInfo(
        name="backemf_capture",
        source="measured",
        cost="none",
        produces=frozenset({
            "backemf_fundamental",
            "harmonics", "fundamental", "thd_pct",
        }),
        needs=frozenset(),
        hash_fields=frozenset(),
    ),
    "inductance_test": ModelInfo(
        name="inductance_test",
        source="measured",
        cost="none",
        produces=frozenset({"L_d", "L_q"}),
        needs=frozenset(),
        hash_fields=frozenset(),
    ),
    "resistance_test": ModelInfo(
        name="resistance_test",
        source="measured",
        cost="none",
        produces=frozenset({"R_s"}),
        needs=frozenset(),
        hash_fields=frozenset(),
    ),
    "torque_test": ModelInfo(
        name="torque_test",
        source="measured",
        cost="none",
        produces=frozenset({"tau_mtpa", "gamma_opt_deg"}),
        needs=frozenset(),
        hash_fields=frozenset(),
    ),
    "iron_loss_test": ModelInfo(
        name="iron_loss_test",
        source="measured",
        cost="none",
        produces=frozenset({"p_fe"}),
        needs=frozenset(),
        hash_fields=frozenset(),
    ),
    "airgap_flux_test": ModelInfo(
        name="airgap_flux_test",
        source="measured",
        cost="none",
        produces=frozenset({"B_ag_fundamental", "B_ag_peak", "backemf_peak", "flux_linkage_peak"}),
        needs=frozenset(),
        hash_fields=frozenset(),
    ),
}


def _ensure_ngsolve_libs() -> None:
    """Preload libmkl_rt from the venv's lib/ so ngsolve's compiled
    extensions can find it without LD_LIBRARY_PATH."""
    import ctypes
    import os
    import sys
    venv_lib = os.path.join(sys.prefix, "lib")
    mkl = os.path.join(venv_lib, "libmkl_rt.so.2")
    if os.path.exists(mkl):
        ctypes.CDLL(mkl, mode=ctypes.RTLD_GLOBAL)


_fem_checked = False


def _check_fem(model_name: str) -> None:
    # find_spec + CDLL cost real time per call; one success is good for
    # the process lifetime (a failure keeps re-checking, so the error
    # message stays per-model).
    global _fem_checked
    if _fem_checked:
        return
    from importlib.util import find_spec
    if find_spec("ngsolve") is None:
        raise ImportError(
            f"model '{model_name}' requires: pip install phasesweep[fem]"
        )
    _ensure_ngsolve_libs()
    try:
        import ngsolve  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"model '{model_name}' requires: pip install phasesweep[fem] "
            f"(ngsolve found but failed to load: {e})"
        ) from e
    _fem_checked = True


def _lazy_fem(config: RunConfig) -> dict[str, Any]:
    _check_fem("fem")
    from phasesweep.solvers.fem_runner import _run_fem_impl
    return _run_fem_impl(config)


def _lazy_fem_axial(config: RunConfig) -> dict[str, Any]:
    _check_fem("fem_axial")
    return _run_fem_axial_impl(config)


def _lazy_cogging(config: RunConfig) -> dict[str, Any]:
    _check_fem("cogging_torque")
    from phasesweep.solvers.cogging import _run_cogging_impl
    return _run_cogging_impl(config)


def _lazy_sim(config: RunConfig) -> dict[str, Any]:
    from importlib.util import find_spec
    if find_spec("motulator") is None:
        raise ImportError(
            "model 'drive_sim' requires: pip install phasesweep[sim]"
        )
    from phasesweep.simulation.sim_runner import _run_sim_impl
    return _run_sim_impl(config)
