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

from phasesweep._extras import install_hint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "MODEL_REGISTRY",
    "ModelInfo",
]


@dataclass(frozen=True)
class ModelInfo:
    """Metadata and execution hooks for one registered model.

    ``produces`` is every key the model may emit, including keys that
    appear only for some inputs (named in the entry's comment); calibration
    and measured import select models by it, so an undeclared key is
    invisible to them.
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
    # compatibility. Per-version history lives in CHANGELOG.md: each
    # release names the model and the v_n -> v_n+1 step (0.3.0 carries a
    # retroactive "Model versions" note for fem v3-v5).
    version: int = 1
    validate: Callable[[Motor], None] | None = None
    fn: Callable[[RunConfig], dict[str, Any]] | None = None
    # Execution-strategy hint only: does not enter hash_fields or
    # compute_run_id. Outputs must be byte-identical to fn per config.
    batch_fn: Callable[[Sequence[RunConfig]], list[dict[str, Any] | Exception]] | None = None


# ---------------------------------------------------------------------------
# Validation via solver_params factories (construct and discard)
# ---------------------------------------------------------------------------

def _lazy_validate(
    module_path: str, func_name: str, *extra_args: object,
) -> Callable[[Motor], None]:
    """Build a validate callback that lazy-imports a prepare function."""
    def _validate(motor: Motor) -> None:
        from importlib import import_module
        getattr(import_module(module_path), func_name)(motor, *extra_args)
    return _validate


_validate_analytical = _lazy_validate(
    "phasesweep.solvers.analytical", "prepare_analytical")
_validate_analytical_axial = _lazy_validate(
    "phasesweep.solvers.analytical_axial", "prepare_analytical_axial")
_validate_fem = _lazy_validate(
    "phasesweep.solvers.fem_runner", "prepare_fem")
_validate_fem_axial = _lazy_validate(
    "phasesweep.solvers.fem_axial", "prepare_fem_axial")
_validate_drive_sim = _lazy_validate(
    "phasesweep.simulation.params", "prepare_drive_sim")
_validate_rated_torque = _lazy_validate(
    "phasesweep.models.rated_torque", "prepare_rated_torque")
_validate_stall_torque = _lazy_validate(
    "phasesweep.models.rated_torque", "prepare_stall_torque")
_validate_thermal_duty = _lazy_validate(
    "phasesweep.models.thermal_duty", "_check_thermal_duty_motor")
_validate_torque_speed = _lazy_validate(
    "phasesweep.models.torque_speed", "prepare_torque_speed")
_validate_operating_point = _lazy_validate(
    "phasesweep.models.operating_point", "prepare_operating_point")
_validate_iron_loss = _lazy_validate(
    "phasesweep.models.iron_loss", "prepare_iron_loss")
_validate_motor_thermal = _lazy_validate(
    "phasesweep.models.motor_thermal", "prepare_motor_thermal")
_validate_demag_screen = _lazy_validate(
    "phasesweep.models.demag_screen", "prepare_demag_screen")
_validate_im_circuit = _lazy_validate(
    "phasesweep.models.im_circuit", "prepare_im_circuit")
_validate_im_bar_impedance = _lazy_validate(
    "phasesweep.models.im_bar_impedance", "prepare_im_bar_impedance")
_validate_im_fem_harmonic = _lazy_validate(
    "phasesweep.models.im_fem_harmonic", "prepare_im_fem_harmonic")
_validate_im_fem_locked_rotor = _lazy_validate(
    "phasesweep.models.im_fem_harmonic", "prepare_im_fem_harmonic",
    "im_fem_locked_rotor")


def _lazy_analytical(config: RunConfig) -> dict[str, Any]:
    from phasesweep.solvers.analytical import run_analytical
    return run_analytical(config)


def _lazy_analytical_axial(config: RunConfig) -> dict[str, Any]:
    from phasesweep.solvers.analytical_axial import run_analytical_axial
    return run_analytical_axial(config)


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


def _lazy_operating_point(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.operating_point import run_operating_point
    return run_operating_point(config)


def _lazy_iron_loss(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.iron_loss import run_iron_loss
    return run_iron_loss(config)


def _lazy_motor_thermal(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.motor_thermal import run_motor_thermal
    return run_motor_thermal(config)


def _lazy_demag_screen(config: RunConfig) -> dict[str, Any]:
    _check_fem("demag_screen")
    from phasesweep.models.demag_screen import run_demag_screen
    return run_demag_screen(config)


def _lazy_im_equivalent_circuit(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.im_circuit import run_im_equivalent_circuit
    return run_im_equivalent_circuit(config)


def _lazy_im_torque_slip(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.im_circuit import run_im_torque_slip
    return run_im_torque_slip(config)


def _lazy_im_bar_impedance(config: RunConfig) -> dict[str, Any]:
    from phasesweep.models.im_bar_impedance import run_im_bar_impedance
    return run_im_bar_impedance(config)


def _lazy_im_fem_harmonic(config: RunConfig) -> dict[str, Any]:
    _check_fem("im_fem_harmonic")
    from phasesweep.models.im_fem_harmonic import run_im_fem_harmonic
    return run_im_fem_harmonic(config)


def _lazy_im_fem_locked_rotor(config: RunConfig) -> dict[str, Any]:
    _check_fem("im_fem_locked_rotor")
    from phasesweep.models.im_fem_harmonic import run_im_fem_locked_rotor
    return run_im_fem_locked_rotor(config)


# ---------------------------------------------------------------------------
# Optional-dependency gates
# ---------------------------------------------------------------------------

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
    global _fem_checked
    if _fem_checked:
        return
    from importlib.util import find_spec
    if find_spec("ngsolve") is None:
        raise ImportError(f"model '{model_name}' needs ngsolve; {install_hint('fem')}")
    _ensure_ngsolve_libs()
    try:
        import ngsolve  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"model '{model_name}': ngsolve found but failed to load ({e}); {install_hint('fem')}"
        ) from e
    _fem_checked = True


def _lazy_fem(config: RunConfig) -> dict[str, Any]:
    _check_fem("fem")
    from phasesweep.solvers.fem_runner import run_fem
    return run_fem(config)


def _lazy_fem_axial(config: RunConfig) -> dict[str, Any]:
    _check_fem("fem_axial")
    from phasesweep.solvers.fem_axial import run_fem_axial
    return run_fem_axial(config)


def _lazy_cogging(config: RunConfig) -> dict[str, Any]:
    _check_fem("cogging_torque")
    from phasesweep.solvers.cogging import run_cogging
    return run_cogging(config)


def _lazy_sim(config: RunConfig) -> dict[str, Any]:
    from importlib.util import find_spec
    if find_spec("motulator") is None:
        raise ImportError(f"model 'drive_sim' needs motulator; {install_hint('sim')}")
    from phasesweep.simulation.sim_runner import run_drive_sim
    return run_drive_sim(config)


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
        # (v4, no bump): additive backemf_source ("psi_f" | "computed" |
        # None) output; backemf_fundamental unchanged.
        version=4,
        cost="fast",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "peak_Br", "thd_pct",
            "backemf_fundamental", "backemf_source", "flux_linkage_peak",
            "k_end", "k_end_source", "k_end_applied",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p", "mu_r_pm"}),
        hash_fields=frozenset({"n_theta", "W_REF"}),
        validate=_validate_analytical,
        fn=_lazy_analytical,
    ),
    "analytical_axial": ModelInfo(
        name="analytical_axial",
        source="computed",
        # v1: quasi-3D Cartesian layer model at r_mean (Parviainen 2005
        # Ch. 2): odd-harmonic series, Carter on the physical gap
        # (Eq. 2.11), Qu PM-leakage factor into flux_linkage_peak.
        # B_r_list holds B_z at the stator surface of one gap.
        # (v1, no bump): additive backemf_source output.
        version=1,
        cost="fast",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "peak_Br", "thd_pct",
            "backemf_fundamental", "backemf_source", "flux_linkage_peak",
            "k_c", "k_sigma_pm",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p", "mu_r_pm"}),
        hash_fields=frozenset({"n_theta", "W_REF"}),
        validate=_validate_analytical_axial,
        fn=_lazy_analytical_axial,
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
        # (v2, no bump): additive backemf_source output.
        version=2,
        cost="slow",
        produces=frozenset({
            "theta_list", "B_r_list", "fundamental", "peak_Br", "thd_pct",
            "backemf_fundamental", "backemf_source", "flux_linkage_peak", "k_sigma_edge",
            "plane_r_list", "plane_B1_list",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "alpha_p",
                         "mu_r_pm", "mu_r_fe"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear",
                               "nonlinear_solver", "n_planes", "W_REF"}),
        validate=_validate_fem_axial,
        fn=_lazy_fem_axial,
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
        # (v8, no bump): sub-0.5 mm interpole gaps now raise; the
        # arcs_collapsed flag is gone.
        version=8,
        cost="slow",
        produces=frozenset({
            "theta_list", "B_r_list", "peak_Br", "fundamental", "thd_pct",
            "sh_pct", "sh_upper_pct", "b_iron_max",
            "tau_maxwell_per_m", "tau_maxwell",
            "B_mag_grid", "grid_coords_list",
            # nonlinear solve diagnostics (fem_field solve info), per solver
            "picard_iterations", "picard_relax_final", "picard_residual",
            "newton_iterations", "newton_startup_iterations",
            "newton_backtracks", "newton_rel_residual", "newton_fell_back",
        }),
        needs=frozenset({"B_rem", "n_p", "geometry", "mu_r_pm", "mu_r_fe", "alpha_p"}),
        hash_fields=frozenset({"maxh_fraction", "n_theta", "nonlinear",
                               "nonlinear_solver", "j_s", "rotation"}),
        validate=_validate_fem,
        fn=_lazy_fem,
    ),
    "drive_sim": ModelInfo(
        name="drive_sim",
        source="computed",
        # (v3, no bump): additive downsampled time-domain traces
        # (t/w_M/tau_M/|i_s|) for the dashboard sim-waveforms panel;
        # scalar outputs unchanged. FluxMap machine model (machine_model
        # = "flux_map") is a new input path — the map hashes into
        # Motor.config_id, so constant-L runs keep their IDs and outputs.
        # v4: R_s at winding_temp_limit, psi_f and the flux map at
        # magnet_temp (shared solver_params.derate) — moves only motors
        # that set those fields (hot R_s ≈ +43% at 130 °C, psi_f −10% at
        # 100 °C NdFeB); cold motors are output-identical.
        version=4,
        cost="medium",
        produces=frozenset({
            "t_settle", "i_ss", "speed_droop", "tau_peak",
            "t_list", "w_M_list", "tau_M_list", "i_s_abs_list",
            "machine_model",
        }),
        needs=frozenset({"R_s", "L_d", "L_q", "psi_f", "J", "n_p"}),
        hash_fields=frozenset({
            "sim_plan",
            "U_DC", "MAX_I_S", "W_REF", "I_LIMIT",
        }),
        needs_runtime=frozenset({"sim_plan"}),
        validate=_validate_drive_sim,
        fn=_lazy_sim,
    ),
    "drive_sim_two_mass": ModelInfo(
        name="drive_sim_two_mass",
        source="computed",
        # v2: shares the prepare_drive_sim derate of drive_sim v4 — R_s at
        # winding_temp_limit, psi_f / flux map at magnet_temp. Missed when
        # drive_sim went to v4; cold motors output-identical.
        version=2,
        cost="medium",
        produces=frozenset({
            "t_settle", "i_ss", "speed_droop", "tau_peak", "tau_S_peak",
            "t_list", "w_M_list", "tau_M_list", "i_s_abs_list",
            "w_L_list", "tau_S_list", "machine_model",
        }),
        needs=frozenset({"R_s", "L_d", "L_q", "psi_f", "J", "n_p"}),
        hash_fields=frozenset({
            "sim_plan", "load_mech",
            "U_DC", "MAX_I_S", "W_REF", "I_LIMIT",
        }),
        needs_runtime=frozenset({"sim_plan", "load_mech"}),
        validate=_validate_drive_sim,
        fn=_lazy_sim,
    ),
    "rated_torque": ModelInfo(
        name="rated_torque",
        source="computed",
        # v4: reverse-salient (L_d > L_q) machines take the MTPA branch
        # (gamma < 0, magnetizing i_d) instead of degrading to k_T·I_rated;
        # adds k_T_effective output.
        # v5: psi_f and the flux map at magnet_temp (shared
        # solver_params.derate); tau_mtpa −10% at 100 °C NdFeB, cold
        # motors output-identical.
        # v6: a map-only psi_f is referred to 20 °C from the map's
        # magnet_temp_ref_C before derating (was taken as 20 °C: −7% psi_f
        # for a map extracted at 80 °C); the map MTPA runs only when the
        # grid covers the current circle, else analytical (was pinned to
        # the grid edge, labeled flux_map). Shipped motors output-identical.
        # The four *_curve keys are analytical-path only (absent when the
        # flux map supplies the MTPA point).
        version=6,
        cost="fast",
        produces=frozenset({
            "tau_mtpa", "k_T", "k_T_rms", "k_T_effective", "gamma_opt_deg",
            "mtpa_source",
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
        # v5: psi_f / map at magnet_temp and I_stall_electromagnetic at
        # the winding_temp_limit R_s (−30% at 130 °C); cold motors
        # output-identical.
        # v6: map psi_f reference temperature and MTPA grid-coverage rule
        # (see rated_torque v6); adds mtpa_source_electromagnetic.
        # Curve keys analytical-path only, as rated_torque.
        version=6,
        cost="fast",
        produces=frozenset({
            "tau_stall", "I_stall", "k_T", "k_T_effective", "gamma_opt_deg",
            "mtpa_source",
            "tau_stall_electromagnetic", "I_stall_electromagnetic",
            "mtpa_source_electromagnetic", "saturation_ratio", "saturation_warning",
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
        # (v3, no bump): i_d/i_q_curve_* are additive outputs — the
        # envelope currents were computed and dropped before; same-version
        # cached rows lack the keys, so consumers treat them as optional.
        # v4: R_s at winding_temp_limit, psi_f and the flux map at
        # magnet_temp (shared solver_params.derate); the hot envelope
        # loses the low-speed torque with psi_f and moves the voltage
        # corner with R_s. Cold motors output-identical.
        # v5: a map-only psi_f is referred to 20 °C from the map's
        # magnet_temp_ref_C before derating (rated_torque v6). Shipped
        # motors output-identical.
        version=5,
        produces=frozenset({
            "speed_curve", "tau_curve_peak", "tau_curve_cont",
            "i_d_curve_peak", "i_q_curve_peak",
            "i_d_curve_cont", "i_q_curve_cont",
            "base_speed_peak", "base_speed_cont",
            "max_speed_peak", "max_speed_cont",
            "p_max_peak", "p_max_cont", "u_max",
            "I_env_peak", "I_env_cont", "envelope_source",
        }),  # *_cont keys only when I_rated is set
        needs=frozenset({"n_p", "psi_f", "R_s", "L_d", "L_q"}),
        hash_fields=frozenset({"U_DC", "MAX_I_S", "I_LIMIT"}),
        validate=_validate_torque_speed,
        fn=_lazy_torque_speed,
        batch_fn=_lazy_torque_speed_batch,
    ),
    "operating_point": ModelInfo(
        name="operating_point",
        source="computed",
        # v1: minimum-current (i_d, i_q) for a (w_mech, tau_ref)
        # demand — constant-L MTPA + three-stage constant-torque field
        # weakening, or a bracketed contour search on the flux map;
        # temperature-continuous on both paths (R_s(T), psi_f(T) / map
        # psi_d shift). The demand is an operating point, so it lives on
        # RunConfig and in hash_fields, not on the Motor.
        # (v1, no bump): psi_d / psi_q are additive outputs — the dq flux
        # linkages the voltages were built from; same-version rows written
        # before them lack the keys.
        # v2: map-only psi_f referred to 20 °C from magnet_temp_ref_C
        # (torque_speed v5) on the off-map path. Shipped motors
        # output-identical.
        cost="fast",
        version=2,
        produces=frozenset({
            "i_d", "i_q", "I_s_peak", "I_s_rms", "u_d", "u_q", "u_mag",
            "u_max", "m_index", "pf", "gamma_deg", "regime", "tau",
            "w_mech", "f_e_Hz", "p_cu", "p_shaft", "p_elec", "R_s_op",
            "psi_f_op", "op_source", "psi_d", "psi_q",
        }),
        needs=frozenset({"n_p", "psi_f", "R_s", "L_d", "L_q"}),
        hash_fields=frozenset({"U_DC", "MAX_I_S", "I_LIMIT", "w_mech", "tau_ref"}),
        needs_runtime=frozenset({"w_mech", "tau_ref"}),
        validate=_validate_operating_point,
        fn=_lazy_operating_point,
    ),
    "iron_loss": ModelInfo(
        name="iron_loss",
        source="computed",
        # v1: lumped Bertotti stator-core loss. Synchronous
        # machines at f_e = n_p·W_REF/2π; induction machines at the rated
        # supply f_rated (induction path added without a bump: previously
        # rejected, so no cached output changes). The rotor fundamental
        # s_rated·f_rated is the f_rotor_Hz key, emitted only on induction
        # machines with s_rated set.
        cost="fast",
        produces=frozenset({
            "p_fe", "p_fe_hysteresis", "p_fe_eddy",
            "loss_density_W_per_kg", "f_e_Hz", "f_rotor_Hz",
        }),
        needs=frozenset({"n_p", "k_h", "k_e", "alpha_fe", "m_core", "B_core"}),
        hash_fields=frozenset({"W_REF"}),
        validate=_validate_iron_loss,
        fn=_lazy_iron_loss,
    ),
    "motor_thermal": ModelInfo(
        name="motor_thermal",
        source="computed",
        # v1: coupled steady solve of the three-node W/Fe/M
        # lumped network with the operating point — copper loss at the
        # winding node temperature, PM flux at the magnet node, iron loss
        # scaled by |psi_s|/psi_ref at f_e = n_p·w_mech/2π. The demand and
        # the coolant temperature are operating conditions on RunConfig;
        # the network is a machine property ([thermal_network], hashed
        # via config_id). No W_REF.
        # (v1, additive): headroom_winding_K / headroom_magnet_K /
        # binding_limit / within_s1 against winding_temp_limit and
        # magnet_temp_max (None when unset); temperatures unchanged.
        # v2: map-only psi_f referred to 20 °C from magnet_temp_ref_C on
        # the off-map path (operating_point v2), and the map-path psi_ref
        # likewise (was the map at its own reference temperature). Shipped
        # motors output-identical.
        cost="fast",
        version=2,
        produces=frozenset({
            "T_winding", "T_iron", "T_magnet", "T_coolant",
            "p_cu", "p_fe", "p_shaft", "p_elec", "psi_s", "psi_ratio",
            "i_d", "i_q", "I_s_peak", "u_mag", "m_index", "regime",
            "op_source", "R_s_op", "psi_f_op", "f_e_Hz",
            "iterations", "residual_K",
            "headroom_winding_K", "headroom_magnet_K", "binding_limit",
            "within_s1",
        }),
        needs=frozenset({
            "n_p", "psi_f", "R_s", "L_d", "L_q", "alpha_Br",
            "k_h", "k_e", "alpha_fe", "m_core", "B_core", "thermal_network",
        }),
        hash_fields=frozenset({
            "U_DC", "MAX_I_S", "I_LIMIT", "w_mech", "tau_ref", "T_coolant",
        }),
        needs_runtime=frozenset({"w_mech", "tau_ref", "T_coolant"}),
        validate=_validate_motor_thermal,
        fn=_lazy_motor_thermal,
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
    "im_equivalent_circuit": ModelInfo(
        name="im_equivalent_circuit",
        source="computed",
        # v1: steady-state per-phase IEEE equivalent circuit
        # solved at RunConfig.slip, rated supply (V_LL_rated / f_rated
        # from [induction]). Deep-bar R_2(f_r) / leakage tables and
        # L_m(I_m) saturation applied when supplied. Per-phase rms
        # convention (outputs suffixed _rms); eta_circuit includes copper
        # and (when R_fe is set) core loss; friction+windage excluded.
        cost="fast",
        produces=frozenset({
            "slip", "f_supply_Hz", "f_rotor_Hz", "V_phase_rms",
            "I_1_rms", "I_2_rms", "I_m_rms", "pf", "tau",
            "p_in", "p_ag", "p_cu_s", "p_cu_r", "p_fe",
            "p_mech", "eta_circuit",
            "R_2_used", "R_fe_used", "r_fe_over_x_m",
            "L_m_used", "L_sigma_s_used", "L_sigma_r_used",
        }),
        needs=frozenset({"induction", "R_s", "n_p"}),
        hash_fields=frozenset({"slip"}),
        needs_runtime=frozenset({"slip"}),
        validate=_validate_im_circuit,
        fn=_lazy_im_equivalent_circuit,
    ),
    "im_torque_slip": ModelInfo(
        name="im_torque_slip",
        source="computed",
        # v1: torque-slip sweep of the same circuit on a
        # fixed 400-point geometric slip grid (1e-3..1) at rated supply;
        # breakdown/starting/rated-point extracts plus the
        # constant-parameter Thévenin closed form for reference. The
        # grid is not a RunConfig knob — changing it is a
        # version-bumping model change.
        cost="fast",
        produces=frozenset({
            "slip_list", "tau_slip_list", "I_1_rms_list", "pf_list",
            "eta_circuit_list", "p_fe_list",
            "tau_breakdown", "slip_breakdown",
            "tau_starting", "I_locked_rotor_rms", "R_fe_used",
            "slip_breakdown_thevenin", "tau_breakdown_thevenin",
            "tau_rated", "I_1_rated_rms", "slip_rated",
        }),
        needs=frozenset({"induction", "R_s", "n_p"}),
        hash_fields=frozenset(),
        validate=_validate_im_circuit,
        fn=_lazy_im_torque_slip,
    ),
    "im_bar_impedance": ModelInfo(
        name="im_bar_impedance",
        source="computed",
        # v1: predicted deep-bar R_2(f_r) / L_sigma_r(f_r)
        # tables from a Bloch-periodic single-slot-pitch harmonic solve
        # (A_z = 0 on the bore, prescribed bar current) plus the lumped
        # end-ring segment and the cage turns ratio. Slot + tooth-top +
        # ring leakage only; differential leakage σ_d·L_m is a separate
        # optional key (L_sigma_r_diff, when [induction] L_m is set), skew
        # is not modeled. Grid = fixed fractions of f_rated merged with
        # the motor's measured table frequencies.
        cost="medium",
        produces=frozenset({
            "f_rotor_list", "R_2_list", "L_sigma_r_list", "k_R_list",
            "k_x_list", "ring_share_list", "R_2_dc", "L_sigma_r_dc", "R_bar_dc",
            "R_ring_seg", "L_ring_seg", "referral_factor", "sigma_diff",
        }),
        needs=frozenset({"induction", "geometry", "n_p", "N", "k_w",
                         "coils_series", "L_stk", "mu_r_fe"}),
        hash_fields=frozenset({"maxh_fraction"}),
        validate=_validate_im_bar_impedance,
        fn=_lazy_im_bar_impedance,
    ),
    "im_fem_harmonic": ModelInfo(
        name="im_fem_harmonic",
        source="computed",
        # v1: full-machine 2D time-harmonic solve on the cage
        # mesh with a per-slot double-layer winding at RunConfig.slip and
        # the rated supply (V_LL_rated / f_rated). Linear iron, R_1 at
        # its reference temperature. Per-phase rms outputs like the
        # circuit models. Slip sweeps go through the sweep engine (slip
        # is the RunConfig axis), not a built-in grid.
        # RunConfig.nonlinear (F5) replaces the linear iron by the
        # energy-equivalent saturation fixed point on the motor's
        # [materials] bh_curve (or the generic curve): iron_model,
        # bh_iterations and B_peak_iron_max report which and how far.
        # (v1, no bump): Geometry.skew_angle turns the solve into
        # the decoupled multi-slice average (skew enters Motor.config_id,
        # so skewed runs hash apart); additive skew_model /
        # n_skew_slices outputs, unskewed outputs unchanged. Skew with
        # nonlinear=True is rejected.
        # (v2): Boldea stator end-winding leakage L_ew added to Z_phase
        # (π/2 semicircular default). All I²-scaled outputs shift: I_rms
        # decreases ~1–2%, torque/losses follow. Additive L_ew_H output.
        # (v3): one stator-circuit owner. Operating points carry the 2D
        # Z_int; operating_point_at_voltage alone adds R_1 + jωL_ew (and
        # R_fe), so the saturation fixed point now targets the FEM-branch
        # current I_int with L_ew and R_fe in (it saturated at V/|R_1+Z_int|,
        # CREATOR ~+0.7 % current). Z_phase_re/_im are terminal, R_fe
        # included; r_fe_over_x_m from Z_int. Linear outputs unchanged.
        version=3,
        cost="slow",
        produces=frozenset({
            "slip", "f_supply_Hz", "f_rotor_Hz", "V_phase_rms",
            "I_1_rms", "pf", "p_in", "p_cu_s", "p_ag", "p_cu_r", "p_ring",
            "p_fe", "p_mech", "eta_circuit", "tau", "tau_power", "I_bar_rms_max",
            "Z_phase_re", "Z_phase_im", "phase_asymmetry", "p_gap_poynting",
            "k_w_layout", "ring_model", "iron_model", "skew_model",
            "n_skew_slices", "L_ew_H", "bh_iterations",
            "B_peak_iron_max", "B_mag_grid", "grid_coords_list",
            "I_int_rms", "R_fe_used", "r_fe_over_x_m",
        }),
        needs=frozenset({"induction", "geometry", "n_p", "N", "k_w",
                         "coils_series", "L_stk", "R_s", "mu_r_fe"}),
        hash_fields=frozenset({"maxh_fraction", "slip", "nonlinear"}),
        needs_runtime=frozenset({"slip"}),
        validate=_validate_im_fem_harmonic,
        fn=_lazy_im_fem_harmonic,
    ),
    "im_fem_locked_rotor": ModelInfo(
        name="im_fem_locked_rotor",
        source="computed",
        # v1: virtual blocked-rotor solves (s = 1) at the
        # motor's table frequencies ∪ f_rated, identified through the
        # IEEE-112 locked-rotor step (im_circuit.identify_locked_rotor,
        # leakage split from [induction] design_letter or 0.5) into
        # R_2 / L_sigma_s / L_sigma_r tables in the [induction] shape,
        # plus tau_per_A2 and Z per frequency so measured test points can
        # be predicted, and the LINEAR-IRON no-load current as a labeled
        # diagnostic (not an L_m identification). Reported beside the
        # motor's measured tables, never substituted. The no-load sweep
        # (*_no_load_list, L_m_list, I_m_rms_list over fractions of the
        # rated voltage through identify_from_tests) is the L_m(I_m)
        # identification when RunConfig.nonlinear runs the saturation
        # fixed point (F5) and the flat linear value otherwise; the
        # locked-rotor lists stay linear in both cases.
        # (v1, no bump): multi-slice skew as on im_fem_harmonic;
        # additive skew_model / n_skew_slices outputs.
        # (v2): Boldea stator end-winding leakage L_ew added to the
        # identified L_sigma_s (after the IEEE-112 split, so it stays on
        # the stator side). L_sigma_s_list increases by L_ew at every
        # frequency; R_2, L_sigma_r, Z, X_lr unchanged. Additive L_ew_H.
        # (v3): the no-load quantities include L_ew as terminal values
        # (I_no_load_rms_*, L_no_load_*; they were 2D-only while L_sigma_s
        # carried L_ew, CREATOR ~−0.6 % current); identification drops the
        # L_ew voltage so L_m is unchanged under linear iron; the
        # saturated sweep targets I_int (as im_fem_harmonic v3).
        # Locked-rotor lists unchanged.
        version=3,
        cost="slow",
        produces=frozenset({
            "f_locked_rotor_list", "R_2_list", "L_sigma_s_list",
            "L_sigma_r_list", "R_lr_list", "X_lr_list", "tau_per_A2_list",
            "Z_re_list", "Z_im_list", "phase_asymmetry_max",
            "leakage_split_used", "R_1", "k_w_layout", "ring_model",
            "skew_model", "n_skew_slices", "L_ew_H",
            "I_no_load_rms_linear_iron", "L_no_load_linear_iron",
            "f_no_load_Hz", "V_phase_rms", "iron_model",
            "V_no_load_list", "I_no_load_rms_list", "L_no_load_list",
            "L_m_list", "I_m_rms_list", "B_peak_iron_no_load_list",
            "bh_iterations_list",
        }),
        needs=frozenset({"induction", "geometry", "n_p", "N", "k_w",
                         "coils_series", "L_stk", "R_s", "mu_r_fe"}),
        hash_fields=frozenset({"maxh_fraction", "nonlinear"}),
        validate=_validate_im_fem_locked_rotor,
        fn=_lazy_im_fem_locked_rotor,
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
        fn=_lazy_cogging,
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
