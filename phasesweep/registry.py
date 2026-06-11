"""Model registry mapping model keys to metadata and solver functions.

Five computational model entries (plus five measured-source keys; ten total):
- analytical: Zhu & Howe closed-form air-gap field
- fem: NGSolve 2D magnetostatic FEM (linear/nonlinear; smooth/slotted from
  Geometry.n_slots)
- drive_sim: motulator PMSM drive simulation
- rated_torque, stall_torque: MTPA circuit models

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
    from phasesweep.motor import Motor as MotorDC
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
    # Carter-consistent B_rem inversion (S110; affects derived psi_f /
    # B_rem, hence all computed models).
    version: int = 1
    validate: Callable[[MotorDC], None] | None = None
    fn: Callable[[RunConfig], dict[str, Any]] | None = None
    import_fn: Callable | None = None


# ---------------------------------------------------------------------------
# Validation via solver_params factories (construct and discard)
# ---------------------------------------------------------------------------

def _validate_analytical(motor: MotorDC) -> None:
    from phasesweep.solver_params import prepare_analytical
    prepare_analytical(motor)


def _validate_fem(motor: MotorDC) -> None:
    from phasesweep.solver_params import prepare_fem
    prepare_fem(motor)


def _validate_drive_sim(motor: MotorDC) -> None:
    from phasesweep.solver_params import prepare_drive_sim
    prepare_drive_sim(motor)


def _validate_rated_torque(motor: MotorDC) -> None:
    from phasesweep.solver_params import prepare_rated_torque
    prepare_rated_torque(motor)


def _validate_stall_torque(motor: MotorDC) -> None:
    from phasesweep.solver_params import prepare_stall_torque
    prepare_stall_torque(motor)


# ---------------------------------------------------------------------------
# Analytical runner
# ---------------------------------------------------------------------------

def _run_analytical_impl(config: RunConfig) -> dict[str, Any]:
    from phasesweep.fem_field import (
        carter_adjusted_radii,
        compute_thd,
        end_effect_factor,
        harmonics_1sided,
        zhu_howe_Br,
    )
    from phasesweep.solver_params import prepare_analytical, psi_f_carter

    params = prepare_analytical(config.motor)
    geo = params.geometry
    n_p = params.n_p

    # Carter factor correction for slotted stators (Zhu & Howe 1993 III)
    r_stator, r_magnet, _k_c = carter_adjusted_radii(geo, params.mu_r_pm)

    theta = np.linspace(0, 2 * np.pi, config.n_theta, endpoint=False)
    B_r = zhu_howe_Br(
        theta, n_p, params.B_rem,
        r_stator=r_stator, r_magnet=r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=params.mu_r_pm, alpha_p=params.alpha_p,
    )

    amps = harmonics_1sided(B_r)
    fund_idx = min(n_p, len(amps) - 1)
    fundamental = float(amps[fund_idx])
    thd_pct = compute_thd(amps, fund_idx)

    m = config.motor
    # Carter radii for the field, original bore for the evaluation point
    # and winding formula (mirrors _derive_b_rem; the midgap `fundamental`
    # above must NOT be used here — it is high by ~g_eff/(2·r_bore))
    flux_linkage_peak = psi_f_carter(m, r_stator, r_magnet, params.B_rem)
    if flux_linkage_peak is not None:
        flux_linkage_peak = float(flux_linkage_peak)

    # Explicit psi_f wins; otherwise derive Carter-consistently (Carter
    # radii for the field, original bore for the winding) — which is
    # exactly flux_linkage_peak
    backemf_fundamental = None
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
        "thd_pct": thd_pct,
        "backemf_fundamental": backemf_fundamental,
        "flux_linkage_peak": flux_linkage_peak,
        "k_end": k_end,
    }


# ---------------------------------------------------------------------------
# Rated torque runner (delegated to rated_torque module)
# ---------------------------------------------------------------------------

def _lazy_rated_torque(config: RunConfig) -> dict[str, Any]:
    from phasesweep.rated_torque import run_rated_torque
    return run_rated_torque(config)


def _lazy_stall_torque(config: RunConfig) -> dict[str, Any]:
    from phasesweep.rated_torque import run_stall_torque
    return run_stall_torque(config)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelInfo] = {
    "analytical": ModelInfo(
        name="analytical",
        source="computed",
        version=2,
        cost="fast",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "thd_pct",
            "backemf_fundamental", "flux_linkage_peak", "k_end",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p"}),
        hash_fields=frozenset({"n_theta", "W_REF"}),
        validate=_validate_analytical,
        fn=_run_analytical_impl,
    ),
    "fem": ModelInfo(
        name="fem",
        source="computed",
        version=2,
        cost="slow",
        produces=frozenset({
            "theta_list", "B_r_list", "peak_Br", "fundamental", "thd_pct",
            "sh_pct", "b_iron_max",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear", "j_s"}),
        validate=_validate_fem,
        fn=lambda config: _lazy_fem(config),
    ),
    "drive_sim": ModelInfo(
        name="drive_sim",
        source="computed",
        version=2,
        cost="medium",
        produces=frozenset({
            "t_settle", "i_ss", "speed_droop", "tau_peak",
        }),
        needs=frozenset({"R_s", "L_d", "L_q", "psi_f", "J", "n_p"}),
        hash_fields=frozenset({
            "sim_plan",
            "U_DC", "MAX_I_S", "W_REF", "I_LIMIT",
        }),
        validate=_validate_drive_sim,
        fn=lambda config: _lazy_sim(config),
    ),
    "rated_torque": ModelInfo(
        name="rated_torque",
        source="computed",
        version=2,
        cost="fast",
        produces=frozenset({
            "tau_mtpa", "k_T", "k_T_rms", "gamma_opt_deg",
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
        version=2,
        cost="fast",
        produces=frozenset({
            "tau_stall", "I_stall", "k_T", "gamma_opt_deg",
            "tau_stall_electromagnetic", "I_stall_electromagnetic",
            "saturation_ratio", "saturation_warning",
            "I_curve", "gamma_curve_deg", "angle_d_curve_deg", "tau_curve",
        }),
        needs=frozenset({"n_p", "psi_f", "R_s"}),
        hash_fields=frozenset({"U_DC", "MAX_I_S", "I_LIMIT"}),
        validate=_validate_stall_torque,
        fn=_lazy_stall_torque,
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
    "airgap_flux_test": ModelInfo(
        name="airgap_flux_test",
        source="measured",
        cost="none",
        produces=frozenset({"B_ag_fundamental", "B_ag_peak", "backemf_peak", "flux_linkage_peak"}),
        needs=frozenset(),
        hash_fields=frozenset(),
    ),
}


def _lazy_fem(config: RunConfig) -> dict[str, Any]:
    from phasesweep.fem_runner import _run_fem_impl
    return _run_fem_impl(config)


def _lazy_sim(config: RunConfig) -> dict[str, Any]:
    from phasesweep.sim_runner import _run_sim_impl
    return _run_sim_impl(config)
