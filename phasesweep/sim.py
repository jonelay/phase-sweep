from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phasesweep.solver_params import DriveSimParams

import numpy as np
from motulator.drive.control.sm import (
    CurrentVectorController,
    CurrentVectorControllerCfg,
    SpeedController,
    VectorControlSystem,
)
from motulator.drive.model import (
    Drive,
    MechanicalSystem,
    Simulation,
    SynchronousMachine,
    SynchronousMachinePars,
    VoltageSourceConverter,
)
from motulator.drive.utils import Step

from phasesweep.rated_torque import magnet_torque_constant


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


@dataclass(frozen=True)
class SimPlan:
    """Self-contained execution plan for a drive simulation.

    Carries load parameters AND metric extraction timing windows.
    Built by plan_sim(); consumed by build_sim() and extract_metrics().
    """

    # Execution
    load_torque: float      # Nm
    load_time: float        # s — when load step is applied
    t_stop: float           # s — simulation end
    speed_step_time: float  # s — when speed ref steps from 0

    # Metric extraction windows
    settle_threshold: float            # fraction of w_ref (e.g. 0.05)
    ss_window: float                   # s — averaging window before t_stop
    droop_window: float                # s — post-load window for speed dip
    accel_window: tuple[float, float]  # (start, end) for peak torque

    # Controller tuning
    alpha_s: float          # speed-controller bandwidth (rad/s)
    alpha_c: float          # current-controller bandwidth (rad/s)
    T_s: float              # control sampling period (s)

    # Diagnostic
    tau_m: float            # mechanical time constant (s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_torque": self.load_torque,
            "load_time": self.load_time,
            "t_stop": self.t_stop,
            "speed_step_time": self.speed_step_time,
            "settle_threshold": self.settle_threshold,
            "ss_window": self.ss_window,
            "droop_window": self.droop_window,
            "accel_window": list(self.accel_window),
            "alpha_s": self.alpha_s,
            "alpha_c": self.alpha_c,
            "T_s": self.T_s,
            "tau_m": self.tau_m,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimPlan:
        aw = d["accel_window"]
        return cls(
            load_torque=d["load_torque"],
            load_time=d["load_time"],
            t_stop=d["t_stop"],
            speed_step_time=d["speed_step_time"],
            settle_threshold=d["settle_threshold"],
            ss_window=d["ss_window"],
            droop_window=d["droop_window"],
            accel_window=(aw[0], aw[1]),
            alpha_s=d.get("alpha_s", 2 * np.pi * 4),
            alpha_c=d.get("alpha_c", 2 * np.pi * 200),
            T_s=d.get("T_s", 125e-6),
            tau_m=d["tau_m"],
        )


def _current_loop_timing(params: DriveSimParams) -> tuple[float, float, float]:
    """Electrical time constant, controller sample time, and current-loop
    bandwidth shared by the speed- and torque-mode planners.
    """
    tau_e = params.L_d / params.R_s if params.R_s > 0 else 1e-3
    T_s = _clamp(tau_e / 5, 10e-6, 125e-6)
    # high-pole/high-speed machines: keep >= 20 samples per electrical period
    if params.drive.W_REF is not None and params.drive.W_REF > 0:
        T_elec = 2 * np.pi / (params.n_p * params.drive.W_REF)
        T_s = min(T_s, T_elec / 20)
    alpha_c = _clamp(0.2 / T_s, 2 * np.pi * 200, 2 * np.pi * 5000)
    return tau_e, T_s, alpha_c


def plan_sim(params: DriveSimParams, load_fraction: float = 0.5) -> SimPlan:
    """Derive a SimPlan for speed-control mode from motor physics.

    k_t approximation uses non-salient formula (1.5 * n_p * psi_f).
    This is conservative for salient motors since reluctance torque
    adds capability, making load_fraction more conservative.

    tau_peak is sized from I_LIMIT (thermal/application limit) while
    build_sim caps the controller at MAX_I_S — when I_LIMIT < MAX_I_S the
    load steps reflect I_LIMIT but transients may draw up to MAX_I_S.
    """
    k_t = magnet_torque_constant(params.n_p, params.psi_f)
    drive = params.drive
    _require_sim_drive_fields(params)
    # mypy narrowing only — the guard above already raised ValueError
    assert drive.U_DC is not None and drive.MAX_I_S is not None and drive.W_REF is not None
    I_limit = drive.I_LIMIT if drive.I_LIMIT is not None else drive.MAX_I_S
    tau_peak = k_t * I_limit

    u_emf = params.n_p * drive.W_REF * params.psi_f
    u_avail = drive.U_DC / np.sqrt(3)
    if u_emf > 0.9 * u_avail:
        warnings.warn(
            f"W_REF={drive.W_REF:.1f} rad/s needs back-EMF {u_emf:.1f} V "
            f"vs {u_avail:.1f} V available (U_DC/sqrt(3)) — setpoint enters "
            "field weakening; t_settle may never be reached (NaN)",
            stacklevel=2,
        )

    load_torque = load_fraction * tau_peak

    tau_m = params.J * drive.W_REF / tau_peak
    alpha_s = _clamp(3.0 / tau_m, 2 * np.pi * 2, 2 * np.pi * 50)

    _tau_e, T_s, alpha_c = _current_loop_timing(params)

    speed_step_time = _clamp(0.5 * tau_m, 0.01, 0.5)
    settle_duration = _clamp(8 * tau_m, 0.1, 3.0)
    load_time = speed_step_time + settle_duration
    post_load = _clamp(8 * tau_m, 0.1, 3.0)
    t_stop = load_time + post_load

    ss_window = _clamp(2 * tau_m, 0.01, 0.5)
    droop_window = _clamp(3 * tau_m, 0.02, 0.5)
    accel_end = speed_step_time + _clamp(4 * tau_m, 0.05, 1.0)
    accel_window = (speed_step_time, accel_end)

    return SimPlan(
        load_torque=load_torque,
        load_time=load_time,
        t_stop=t_stop,
        speed_step_time=speed_step_time,
        settle_threshold=0.05,
        ss_window=ss_window,
        droop_window=droop_window,
        accel_window=accel_window,
        alpha_s=alpha_s,
        alpha_c=alpha_c,
        T_s=T_s,
        tau_m=tau_m,
    )


def plan_torque_sim(
    params: DriveSimParams,
    tau_ref: float,
    *,
    step_time: float = 0.05,
    ss_window: float = 0.05,
    load_torque: float | None = None,
) -> SimPlan:
    """Derive a SimPlan for torque-control mode.

    Defaults `load_torque = tau_ref` so the mechanical system absorbs the
    commanded torque and the rotor holds at w_M = 0 — back-EMF stays zero
    and the steady-state current matches the closed-form MTPA anchor
    |i_s| = current_for_torque(n_p, psi_f, L_d, L_q, tau_ref) without race
    against voltage saturation. For a salient machine the controller exploits
    reluctance, so |i_s| is below the magnet-only tau_ref / (1.5 * n_p * psi_f)
    by the reluctance saving (which that anchor only equals when L_d == L_q);
    verified to ~1e-4 against motulator for saliency up to 2.06. Pass
    `load_torque=0` for free acceleration, or any explicit value for other
    load profiles.

    SimPlan fields unused in torque mode (alpha_s, settle_threshold,
    droop_window) are set to neutral defaults.
    """
    if tau_ref <= 0:
        raise ValueError(f"tau_ref must be > 0, got {tau_ref}")

    tau_e, T_s, alpha_c = _current_loop_timing(params)

    settle = max(5 * tau_e, 5 / alpha_c)
    t_stop = step_time + settle + ss_window

    if load_torque is None:
        load_torque = tau_ref

    return SimPlan(
        load_torque=load_torque,
        load_time=step_time,
        t_stop=t_stop,
        speed_step_time=step_time,
        settle_threshold=0.05,
        ss_window=ss_window,
        droop_window=0.0,
        accel_window=(step_time, step_time + settle),
        alpha_s=2 * np.pi * 4,
        alpha_c=alpha_c,
        T_s=T_s,
        tau_m=0.0,
    )


def _fix_mtpv_nan(ref_gen: Any) -> None:
    """Patch motulator's MTPV lookup for SPMSM with small parameters.

    motulator 0.7.3 computes _psi_s_mtpv inside compute_flux_and_torque_refs.
    When the MTPV locus returns NaN (common for SPMSM), downstream
    compute_current_ref uses phase(NaN) and produces NaN current references.
    Fix: after each call, replace NaN with j*psi_f (phase = pi/2, i.e. no
    MTPV constraint — correct for non-salient machines).

    Salient IPMs (Lq > Ld) compute a finite MTPV locus, so this patch never
    fires for them and its non-salient j*psi_f value is not exercised —
    verified across both in-repo IPM anchors (saliency up to 2.06) over
    0.1-1.5x rated speed. The patch matters only for the near-SPMSM
    case it targets.
    """
    _orig = ref_gen.compute_flux_and_torque_refs

    def _safe(tau_M_ref: float, w_m: float, u_dc: float,
              _rg: Any = ref_gen, _fn: Any = _orig) -> tuple[float, float]:
        result = _fn(tau_M_ref, w_m, u_dc)
        if np.isnan(_rg._psi_s_mtpv):
            _rg._psi_s_mtpv = 1j * _rg.par.psi_f
        return result

    ref_gen.compute_flux_and_torque_refs = _safe


def _require_sim_drive_fields(params: DriveSimParams) -> None:
    """DriveSimParams validates the motor constants but not the drive
    fields (prepare_drive_sim does) — a directly constructed instance can
    carry None U_DC/MAX_I_S/W_REF into motulator, which fails obscurely
    (or not at all). Same message shape as solver_params._require_drive_fields."""
    missing = [f for f in ("U_DC", "MAX_I_S", "W_REF")
               if getattr(params.drive, f) is None]
    if missing:
        raise ValueError(
            f"drive_sim needs [drive] {', '.join(missing)}"
        )


def build_sim(
    params: DriveSimParams,
    plan: SimPlan | None = None,
    *,
    torque_ref: Step | None = None,
) -> Simulation:
    """Build a motulator Simulation from validated DriveSimParams.

    Default is speed-control mode (current behavior, unchanged). Pass
    `torque_ref` to use torque-control mode: SpeedController is skipped
    and the vector controller is driven directly by the torque step.

    Reverse-salient machines (L_d > L_q) are rejected: motulator 0.7.3's
    MTPA locus is silently wrong for them — below psi_f/(L_d - L_q) the
    angle search hits a zero-torque sentinel, above it brentq converges on
    the demagnetizing stationary point (negative torque). Unfixed upstream;
    remove this guard when motulator ships a fix.
    """
    _require_sim_drive_fields(params)
    if params.L_d > params.L_q:
        raise ValueError(
            f"reverse-salient machine (L_d={params.L_d} > L_q={params.L_q}): "
            "motulator's MTPA current reference is wrong for L_d > L_q "
            "(zero- or negative-torque locus); drive_sim is blocked for "
            "these machines until fixed upstream"
        )
    is_torque_mode = torque_ref is not None

    lt = plan.load_torque if plan else 3.0
    lt_time = plan.load_time if plan else 1.2
    sst = plan.speed_step_time if plan else 0.2

    par = SynchronousMachinePars(
        n_p=params.n_p,
        R_s=params.R_s,
        L_d=params.L_d,
        L_q=params.L_q,
        psi_f=params.psi_f,
    )

    drive = params.drive
    machine = SynchronousMachine(par)
    converter = VoltageSourceConverter(u_dc=drive.U_DC)
    mechanics = MechanicalSystem(J=params.J)
    mechanics.set_external_load_torque(
        lambda t, tau=lt, t0=lt_time: tau * (t > t0))
    mdl = Drive(converter=converter, machine=machine, mechanics=mechanics)

    _alpha_s = plan.alpha_s if plan else 2 * np.pi * 4
    _alpha_c = plan.alpha_c if plan else 2 * np.pi * 200
    _T_s = plan.T_s if plan else 125e-6

    cfg_ctrl = CurrentVectorControllerCfg(
        i_s_max=drive.MAX_I_S, J=params.J, alpha_c=_alpha_c)
    vector_ctrl = CurrentVectorController(
        par=par, cfg=cfg_ctrl, sensorless=False, T_s=_T_s)

    if is_torque_mode:
        ctrl_sys = VectorControlSystem(vector_ctrl=vector_ctrl, speed_ctrl=None)
        ctrl_sys.set_torque_ref(torque_ref)
    else:
        speed_ctrl = SpeedController(J=params.J, alpha_s=_alpha_s)
        ctrl_sys = VectorControlSystem(vector_ctrl=vector_ctrl, speed_ctrl=speed_ctrl)
        ctrl_sys.set_speed_ref(
            Step(step_time=sst, step_value=drive.W_REF, initial_value=0))

    _fix_mtpv_nan(vector_ctrl.reference_gen)

    return Simulation(mdl=mdl, ctrl=ctrl_sys, show_progress=False)


def extract_waveforms(res: Any, max_points: int = 1200) -> dict[str, list[float]]:
    """Downsampled time-domain traces for the dashboard sim-waveforms panel.
    Additive alongside extract_metrics — scalar metrics are
    unchanged."""
    t = np.asarray(res.mdl.t)
    stride = max(1, -(-len(t) // max_points))
    return {
        "t_list": np.round(t[::stride], 6).tolist(),
        "w_M_list": np.round(np.asarray(res.mdl.mechanics.w_M)[::stride], 4).tolist(),
        "tau_M_list": np.round(np.asarray(res.mdl.machine.tau_M)[::stride], 4).tolist(),
        "i_s_abs_list": np.round(
            np.abs(np.asarray(res.mdl.machine.i_s_ab))[::stride], 4).tolist(),
    }


def extract_metrics(
    res: Any,
    *,
    plan: SimPlan | None = None,
    w_ref: float,
) -> dict[str, float]:
    """Extract scalar performance metrics from a motulator simulation result.

    t_settle is the time after the speed step at which the normalised speed
    permanently enters the ±settle_threshold band.
    """
    _t_stop = plan.t_stop if plan else 1.8
    _load_time = plan.load_time if plan else 1.2
    _sst = plan.speed_step_time if plan else 0.2
    _settle_thr = plan.settle_threshold if plan else 0.05
    _ss_window = plan.ss_window if plan else 0.2
    _droop_window = plan.droop_window if plan else 0.3
    _accel = plan.accel_window if plan else (0.2, 0.8)

    t = res.mdl.t
    w_M = res.mdl.mechanics.w_M
    i_s = np.abs(res.mdl.machine.i_s_ab)
    tau_M = res.mdl.machine.tau_M

    post_step = t >= _sst
    err = np.abs(w_M / w_ref - 1.0)
    outside = post_step & (err >= _settle_thr)
    if outside.any():
        idx = np.flatnonzero(outside)[-1]
        if idx + 1 < len(t) and np.all(err[idx + 1:] < _settle_thr):
            t_settle = t[idx + 1] - _sst
        else:
            t_settle = np.nan
    elif post_step.any():
        t_settle = 0.0
    else:
        t_settle = np.nan

    ss_mask = t >= (_t_stop - _ss_window)
    i_ss = float(np.mean(i_s[ss_mask])) if ss_mask.any() else np.nan

    pre_load = (t >= (_load_time - _ss_window)) & (t < _load_time)
    post_load = (t >= _load_time) & (t < _load_time + _droop_window)
    if pre_load.any() and post_load.any():
        w_pre = np.mean(w_M[pre_load])
        w_dip = w_pre - np.min(w_M[post_load])
        speed_droop = w_dip / w_ref
    else:
        speed_droop = np.nan

    accel = (t >= _accel[0]) & (t < _accel[1])
    tau_peak = np.max(tau_M[accel]) if accel.any() else np.nan

    return dict(t_settle=t_settle, i_ss=i_ss,
                speed_droop=speed_droop, tau_peak=tau_peak)

