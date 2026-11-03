"""Model registry mapping model keys to metadata and solver functions.

Registry is a plain dict. No Protocol/ABC/plugin system.

Validation uses the prepare_* factories (each living next to its model's
run function) as the single source of truth: they validate Motor
completeness and derive missing fields (B_rem <-> psi_f) via the shared
solver_params utilities.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from phasesweep._extras import MissingExtra, require_extra

if TYPE_CHECKING:
    from collections.abc import Sequence

    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "MODEL_REGISTRY",
    "ModelInfo",
    "headline_key",
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

    ``headline`` names the 2–5 outputs that summarize one result: a key
    from ``produces``, or ``max:<key>`` for the maximum of a list output.
    Keys present only for some inputs may be absent from a given result.
    ``units`` gives the native unit of each headline key as stored ("1"
    for dimensionless, "" for strings and booleans). ``limits`` lists
    model-inherent omissions; ``phasesweep.trust`` reports them on every
    OK result alongside caveats derived from the result's provenance keys.
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
    headline: tuple[str, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    limits: tuple[str, ...] = ()


def headline_key(entry: str) -> str:
    """The ``produces`` key behind a headline entry (``max:k`` -> ``k``)."""
    return entry.removeprefix("max:")


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
        raise MissingExtra(f"model '{model_name}' needs ngsolve", "fem")
    _ensure_ngsolve_libs()
    try:
        import ngsolve  # noqa: F401
    except ImportError as e:
        raise MissingExtra(
            f"model '{model_name}': ngsolve found but failed to load ({e})", "fem"
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
    require_extra("sim", _SIM_MODULES, f"model {config.model!r}")
    from phasesweep.simulation.sim_runner import run_drive_sim
    return run_drive_sim(config)


_SIM_MODULES = ("motulator",)


def require_model_extra(key: str) -> None:
    """Raise MissingExtra if model *key* runs through an extra that is not
    installed. Covers the drive-sim models; the FEM models check ngsolve
    only when they run."""
    info = MODEL_REGISTRY.get(key)
    if info is not None and info.fn is _lazy_sim:
        require_extra("sim", _SIM_MODULES, f"model {key!r}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelInfo] = {
    "analytical": ModelInfo(
        name="analytical",
        source="computed",
        # Odd-harmonic square-wave series: B_r_list is the full waveform,
        # peak_Br and thd_pct follow from it. Optional Motor.k_end
        # (computed 3D wedge factor, hashed into Motor.config_id)
        # corrects the B_rem-derived flux_linkage_peak /
        # backemf_fundamental; k_end_source is "computed" | "heuristic" |
        # None and k_end_applied says whether flux_linkage_peak carries
        # the correction. backemf_source is "psi_f" | "computed" | None.
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
        # Quasi-3D Cartesian layer model at r_mean (Parviainen 2005
        # Ch. 2): odd-harmonic series, Carter on the physical gap
        # (Eq. 2.11), Qu PM-leakage factor into flux_linkage_peak.
        # B_r_list holds B_z at the stator surface of one gap.
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
        # Multi-plane quasi-3D 2D Cartesian FEM.
        # K unrolled mirror planes at midpoint radii, A_z with Dirichlet
        # on the magnet-center planes; explicit slots (no Carter);
        # flux linkage from the per-plane sum with a radial-edge
        # leakage factor. B_r_list holds B_z of the middle plane.
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
        # Maxwell-stress torque (tau_maxwell_per_m / tau_maxwell) only on
        # armature (j_s != 0) runs; B_mag_grid / grid_coords_list are the
        # cross-section raster for the dashboard heatmap. Guards: j_s != 0
        # with no slot faces in the mesh (slot_depth = 0) raises
        # ValueError; interpole gaps under 0.5 mm (measured at the inner
        # radius) raise.
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
        # t/w_M/tau_M/i_s_abs lists are downsampled time-domain traces
        # for the dashboard sim-waveforms panel. machine_model =
        # "flux_map" selects the FluxMap machine (the map hashes into
        # Motor.config_id). R_s is taken at winding_temp_limit, psi_f and
        # the flux map at magnet_temp (solver_params.derate); motors
        # without those fields run cold.
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
        # Same prepare_drive_sim derate as drive_sim: R_s at
        # winding_temp_limit, psi_f / flux map at magnet_temp.
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
        # Reverse-salient (L_d > L_q) machines take the MTPA branch
        # (gamma < 0, magnetizing i_d); k_T_effective reports the result.
        # psi_f and the flux map are derated to magnet_temp
        # (solver_params.derate); a map-only psi_f is first referred to
        # 20 °C from the map's magnet_temp_ref_C. The map MTPA runs only
        # when the grid covers the current circle, else analytical
        # (mtpa_source). The four *_curve keys are analytical-path only
        # (absent when the flux map supplies the MTPA point).
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
        # MTPA branch, derate and map rules as rated_torque; additionally
        # I_stall_electromagnetic at the winding_temp_limit R_s, with
        # mtpa_source_electromagnetic. Curve keys analytical-path only,
        # as rated_torque.
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
        # Copper loss uses R_s derated to winding_temp_limit — raises
        # consumed loss (and the over_budget_ratio verdict) on the
        # thermal_resistance budget path, cancels on the rated_current
        # path. The torque→current inverse uses the MTPA locus. Optional
        # [iron] p_fe joins consumption and the rated_current budget
        # ([iron] hashes via config_id; W_REF is in hash_fields for it).
        # Optional thermal_time_constant enables the first-order transient
        # march (transient_peak_ratio / within_s1_transient /
        # winding_temp_peak), also hashed via config_id.
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
        # i_d/i_q_curve_* may be absent on older cached rows; consumers
        # treat them as optional. R_s at winding_temp_limit, psi_f and
        # the flux map at magnet_temp (solver_params.derate); a map-only
        # psi_f is first referred to 20 °C from the map's
        # magnet_temp_ref_C.
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
        # Minimum-current (i_d, i_q) for a (w_mech, tau_ref)
        # demand — constant-L MTPA + three-stage constant-torque field
        # weakening, or a bracketed contour search on the flux map;
        # temperature-continuous on both paths (R_s(T), psi_f(T) / map
        # psi_d shift). The demand is an operating point, so it lives on
        # RunConfig and in hash_fields, not on the Motor. A map-only
        # psi_f is referred to 20 °C from magnet_temp_ref_C on the
        # off-map path. psi_d / psi_q (the dq flux linkages the voltages
        # were built from) may be absent on older cached rows.
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
        # Lumped Bertotti stator-core loss. Synchronous
        # machines at f_e = n_p·W_REF/2π; induction machines at the rated
        # supply f_rated. The rotor fundamental s_rated·f_rated is the
        # f_rotor_Hz key, emitted only on induction machines with
        # s_rated set.
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
        # Coupled steady solve of the three-node W/Fe/M
        # lumped network with the operating point — copper loss at the
        # winding node temperature, PM flux at the magnet node, iron loss
        # scaled by |psi_s|/psi_ref at f_e = n_p·w_mech/2π. The demand and
        # the coolant temperature are operating conditions on RunConfig;
        # the network is a machine property ([thermal_network], hashed
        # via config_id). No W_REF. headroom_winding_K / headroom_magnet_K
        # / binding_limit / within_s1 are measured against
        # winding_temp_limit and magnet_temp_max (None when unset). A
        # map-only psi_f is referred to 20 °C from magnet_temp_ref_C on
        # the off-map path, and the map-path psi_ref likewise.
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
        # FEM demag screen — magnet field + pure
        # demagnetizing d-axis sheet (sheet_phase = +π/2) at i_fault,
        # worst-point B_m margin against B_knee(T).
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
        # Steady-state per-phase IEEE equivalent circuit
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
        # Torque-slip sweep of the same circuit on a
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
        # Predicted deep-bar R_2(f_r) / L_sigma_r(f_r)
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
        # Full-machine 2D time-harmonic solve on the cage
        # mesh with a per-slot double-layer winding at RunConfig.slip and
        # the rated supply (V_LL_rated / f_rated). R_1 at its reference
        # temperature. Per-phase rms outputs like the circuit models.
        # Slip sweeps go through the sweep engine (slip is the RunConfig
        # axis), not a built-in grid. Linear iron by default;
        # RunConfig.nonlinear replaces it by the energy-equivalent
        # saturation fixed point on the motor's [materials] bh_curve (or
        # the generic curve): iron_model, bh_iterations and
        # B_peak_iron_max report which and how far. Geometry.skew_angle
        # turns the solve into the decoupled multi-slice average (skew
        # enters Motor.config_id); skew_model / n_skew_slices report it.
        # Skew with nonlinear=True is rejected. Boldea stator end-winding
        # leakage L_ew (π/2 semicircular default) is part of Z_phase;
        # L_ew_H reports it. One stator-circuit owner: operating points
        # carry the 2D Z_int, operating_point_at_voltage alone adds
        # R_1 + jωL_ew (and R_fe), and the saturation fixed point targets
        # the FEM-branch current I_int. Z_phase_re/_im are terminal, R_fe
        # included; r_fe_over_x_m from Z_int.
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
        # Virtual blocked-rotor solves (s = 1) at the
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
        # fixed point and the flat linear value otherwise; the
        # locked-rotor lists stay linear in both cases. Multi-slice skew
        # as on im_fem_harmonic; skew_model / n_skew_slices report it.
        # Boldea stator end-winding leakage L_ew is added to the
        # identified L_sigma_s after the IEEE-112 split (stator side;
        # L_ew_H reports it); R_2, L_sigma_r, Z and X_lr do not carry it.
        # The no-load quantities (I_no_load_rms_*, L_no_load_*) are
        # terminal values including L_ew; identification drops the L_ew
        # voltage so L_m is unaffected under linear iron; the saturated
        # sweep targets I_int (as im_fem_harmonic).
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
        # Rotor sweep at j_s=0 with Arkkio radial-averaging torque.
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


# ---------------------------------------------------------------------------
# Trust metadata: headline (key -> native unit, in display order) and limits
# ---------------------------------------------------------------------------
# Units verified against the producing code. Conventions the unit strings
# do not repeat: PM-machine currents, voltages and flux linkages are peak
# per phase unless the key ends in _rms; induction-machine currents and
# voltages are rms per phase, power and torque machine totals; speeds are
# mechanical rad/s. "1" = dimensionless (ratios
# and fractions, not percent); "" = string or boolean.
# Limits restate README "Scope and limitations", docs/validation-report.md,
# or the entry's own comment above — no new claims.

_LIM_2D = "two-dimensional field solution; 3D end effects are not included"
_LIM_QUASI_3D = ("quasi-3D: several 2D planes; 3D effects are estimated "
                 "only by separate offline studies")
_LIM_PM_SAT = ("saturation enters only through a user-supplied [flux_map]; "
               "no saturated inductance is predicted from geometry")
_LIM_IM_SAT = ("saturation enters only through a user-supplied [induction] "
               "L_m_vs_i table")
_LIM_LOSSES = "AC winding losses and magnet eddy-current losses are not implemented"

_TRUST: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "analytical": (
        {"fundamental": "T", "thd_pct": "%", "backemf_fundamental": "V_peak",
         "flux_linkage_peak": "Wb"},
        ("two-dimensional field model; 3D end effects enter only through k_end",),
    ),
    "analytical_axial": (
        {"fundamental": "T", "thd_pct": "%", "backemf_fundamental": "V_peak",
         "flux_linkage_peak": "Wb"},
        (_LIM_QUASI_3D,),
    ),
    "fem": (
        {"fundamental": "T", "thd_pct": "%", "b_iron_max": "T", "tau_maxwell": "N·m"},
        (_LIM_2D,),
    ),
    "fem_axial": (
        {"fundamental": "T", "thd_pct": "%", "backemf_fundamental": "V_peak"},
        (_LIM_QUASI_3D,),
    ),
    "cogging_torque": (
        {"tau_cogging_pp_Nm": "N·m", "dominant_order": "1/rev"},
        (_LIM_2D + "; skew is not modelled",),
    ),
    "demag_screen": (
        {"margin": "T", "B_m_min": "T", "frac_below_knee": "1"},
        (_LIM_2D,),
    ),
    "rated_torque": (
        {"tau_mtpa": "N·m", "k_T": "N·m/A_peak", "gamma_opt_deg": "deg"},
        (_LIM_PM_SAT,),
    ),
    "stall_torque": (
        {"tau_stall": "N·m", "I_stall": "A_peak", "saturation_ratio": "1"},
        (_LIM_PM_SAT,),
    ),
    "torque_speed": (
        {"max:tau_curve_peak": "N·m", "p_max_peak": "W",
         "base_speed_peak": "rad/s", "max_speed_peak": "rad/s"},
        (_LIM_PM_SAT,),
    ),
    "operating_point": (
        {"I_s_rms": "A_rms", "pf": "1", "m_index": "1", "regime": ""},
        (_LIM_PM_SAT,),
    ),
    "thermal_duty": (
        {"within_s1": "", "winding_temp_peak": "°C", "over_budget_ratio": "1"},
        (_LIM_LOSSES,),
    ),
    "motor_thermal": (
        {"T_winding": "°C", "T_magnet": "°C", "binding_limit": "", "within_s1": ""},
        (_LIM_PM_SAT, _LIM_LOSSES),
    ),
    "iron_loss": (
        {"p_fe": "W", "loss_density_W_per_kg": "W/kg"},
        ("stator-core loss only; on the CREATOR IM it under-predicts measured "
         "no-load loss about 3.9×",),
    ),
    "drive_sim": (
        {"t_settle": "s", "speed_droop": "1", "tau_peak": "N·m"},
        (_LIM_PM_SAT,),
    ),
    "drive_sim_two_mass": (
        {"t_settle": "s", "speed_droop": "1", "tau_peak": "N·m", "tau_S_peak": "N·m"},
        (_LIM_PM_SAT,),
    ),
    "im_equivalent_circuit": (
        {"tau": "N·m", "I_1_rms": "A_rms", "pf": "1", "eta_circuit": "1"},
        (_LIM_IM_SAT, "eta_circuit excludes friction and windage"),
    ),
    "im_fem_harmonic": (
        {"tau": "N·m", "I_1_rms": "A_rms", "pf": "1", "eta_circuit": "1"},
        (_LIM_2D,),
    ),
    "im_torque_slip": (
        {"tau_breakdown": "N·m", "tau_starting": "N·m", "slip_breakdown": "1",
         "tau_rated": "N·m"},
        (_LIM_IM_SAT, "breakdown point is sampled on a fixed 400-point slip grid"),
    ),
    "im_bar_impedance": (
        {"R_2_dc": "Ω", "L_sigma_r_dc": "H"},
        (_LIM_2D, "skew is not modelled; differential leakage is reported "
                  "separately (L_sigma_r_diff)"),
    ),
    "im_fem_locked_rotor": (
        {"I_no_load_rms_linear_iron": "A_rms", "L_no_load_linear_iron": "H",
         "phase_asymmetry_max": "1"},
        (_LIM_2D, "the locked-rotor tables and the *_linear_iron no-load values "
                  "always use linear iron; the latter are a diagnostic, not an "
                  "L_m identification"),
    ),
}

for _key, (_units, _limits) in _TRUST.items():
    MODEL_REGISTRY[_key] = replace(
        MODEL_REGISTRY[_key], headline=tuple(_units), units=_units, limits=_limits)
del _key, _units, _limits
