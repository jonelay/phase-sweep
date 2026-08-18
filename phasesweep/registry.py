"""Model registry mapping model keys to metadata and solver functions.

Registry is a plain dict. No Protocol/ABC/plugin system.

Validation uses solver_params factories as the single source of truth:
prepare_analytical, prepare_fem, prepare_drive_sim, prepare_rated_torque,
prepare_stall_torque validate Motor completeness and derive missing fields
(B_rem <-> psi_f).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig


@dataclass(frozen=True)
class ModelInfo:
    name: str
    source: Literal["computed", "measured"]
    cost: Literal["fast", "medium", "slow", "none"]
    produces: frozenset[str]
    needs: frozenset[str]
    hash_fields: frozenset[str]
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
    # (maxh = gap/2 rule) — outputs shift only for
    # interpole gaps in the 0.5–1.0 mm band (previously floored at
    # 0.5 mm global maxh); wider-gap meshes are unchanged.
    # fem v7: _interpole_gap_width measured at the inner PM-annulus radius
    # (the wedge's narrowest edge) instead of the outer radius — the
    # collapse threshold and pm_gap/global-maxh clamp both used an
    # overestimated gap. Finer gap mesh shifts every alpha_p<1 FEM output
    # (mm-scale arc motors most, where the clamp binds the global maxh);
    # near-threshold runs may now collapse to the full ring where they
    # previously built arcs. mesh cache prefix -> mesh_v6.
    version: int = 1
    # True when waveform outputs contain only the fundamental, so
    # waveform-shape extracts (max/min) are not meaningful comparands.
    # No registry model sets it since analytical v4 landed the
    # odd-harmonic series; the crossval skip machinery it drives is kept
    # for future fundamental-only entries.
    fundamental_only: bool = False
    validate: Callable[[Motor], None] | None = None
    fn: Callable[[RunConfig], dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Validation via solver_params factories (construct and discard)
# ---------------------------------------------------------------------------

def _validate_analytical(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_analytical
    prepare_analytical(motor)


def _validate_fem(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_fem
    prepare_fem(motor)


def _validate_drive_sim(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_drive_sim
    prepare_drive_sim(motor)


def _validate_rated_torque(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_rated_torque
    prepare_rated_torque(motor)


def _validate_stall_torque(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_stall_torque
    prepare_stall_torque(motor)


def _validate_thermal_duty(motor: Motor) -> None:
    from phasesweep.solver_params import _check_thermal_duty_motor
    _check_thermal_duty_motor(motor)


def _validate_torque_speed(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_torque_speed
    prepare_torque_speed(motor)


def _validate_iron_loss(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_iron_loss
    prepare_iron_loss(motor)


def _validate_demag_screen(motor: Motor) -> None:
    from phasesweep.solver_params import prepare_demag_screen
    prepare_demag_screen(motor)


# ---------------------------------------------------------------------------
# Analytical runner
# ---------------------------------------------------------------------------

def _run_analytical_impl(config: RunConfig) -> dict[str, Any]:
    from phasesweep.solver_params import prepare_analytical, psi_f_carter
    from phasesweep.solvers.analytical import (
        carter_adjusted_radii,
        end_effect_factor,
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

    # Explicit psi_f wins; otherwise derive Carter-consistently (Carter
    # radii for the field, original bore for the winding) — which is
    # exactly flux_linkage_peak
    backemf_fundamental = None
    if m.drive.W_REF is not None:
        w_e = m.drive.W_REF * m.n_p
        if m.psi_f is not None:
            backemf_fundamental = float(w_e * m.psi_f)
        elif flux_linkage_peak is not None:
            backemf_fundamental = float(w_e * flux_linkage_peak)

    # Informational only — flux_linkage_peak and backemf_fundamental above
    # are end-effect-UNcorrected (k_end is uncalibrated until Tier-2 3D
    # validation; do not apply it to psi_f here)
    k_end = None
    if m.L_stk is not None:
        g = abs(geo.r_stator - geo.r_magnet)
        h_m = abs(geo.r_magnet - geo.r_rotor)
        k_end = float(end_effect_factor(m.L_stk, g + h_m / params.mu_r_pm))

    return {
        "theta_list": theta.tolist(),
        "B_r_list": B_r.tolist(),
        "fundamental": fundamental,
        "peak_Br": float(np.max(np.abs(B_r))),
        "thd_pct": thd_pct,
        "backemf_fundamental": backemf_fundamental,
        "flux_linkage_peak": flux_linkage_peak,
        "k_end": k_end,
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
        version=4,
        cost="fast",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "peak_Br", "thd_pct",
            "backemf_fundamental", "flux_linkage_peak", "k_end",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p", "mu_r_pm"}),
        hash_fields=frozenset({"n_theta", "W_REF"}),
        validate=_validate_analytical,
        fn=_run_analytical_impl,
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
        version=7,
        cost="slow",
        produces=frozenset({
            "theta_list", "B_r_list", "peak_Br", "fundamental", "thd_pct",
            "sh_pct", "sh_upper_pct", "b_iron_max",
            "tau_maxwell_per_m", "tau_maxwell",
            "B_mag_grid", "grid_coords_list",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear", "j_s", "rotation"}),
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
        validate=_validate_thermal_duty,
        fn=_lazy_thermal_duty,
    ),
    "torque_speed": ModelInfo(
        name="torque_speed",
        source="computed",
        cost="fast",
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
        cost="slow",
        produces=frozenset({
            "B_m_min", "B_knee", "margin", "frac_below_knee",
            "r_min", "theta_min", "i_fault", "j_s_fault", "b_iron_max",
        }),
        needs=frozenset({
            "B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p",
            "B_knee", "N", "coils_series", "k_w",
        }),
        hash_fields=frozenset({"maxh_fraction", "nonlinear", "i_fault"}),
        validate=_validate_demag_screen,
        fn=_lazy_demag_screen,
    ),
    "cogging_torque": ModelInfo(
        name="cogging_torque",
        source="computed",
        # v1: rotor sweep at j_s=0 with Arkkio radial-averaging torque.
        # RunConfig.rotation is an input-schema addition with an
        # identity-preserving default (no fem version bump).
        cost="slow",
        produces=frozenset({
            "rotation_list", "tau_cogging_list", "tau_cogging_pp",
            "tau_cogging_pp_Nm", "dominant_order", "n_cogging_periods",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear", "cogging_points"}),
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


def _check_fem(model_name: str) -> None:
    from importlib.util import find_spec
    if find_spec("ngsolve") is None:
        raise ImportError(
            f"model '{model_name}' requires: pip install phasesweep[fem]"
        )


def _lazy_fem(config: RunConfig) -> dict[str, Any]:
    _check_fem("fem")
    from phasesweep.solvers.fem_runner import _run_fem_impl
    return _run_fem_impl(config)


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
