from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phasesweep.result_store import ResultStore, SlimResult
    from phasesweep.sweep_types import MotorSweepConfig, SweepResult

import numpy as np
from numpy.typing import NDArray

from motulator.drive.model import (
    Drive, SynchronousMachine, MechanicalSystem,
    VoltageSourceConverter, SynchronousMachinePars, Simulation)
from motulator.drive.control.sm import (
    CurrentVectorController, CurrentVectorControllerCfg,
    VectorControlSystem, SpeedController)
from motulator.drive.utils import Step

from phasesweep.configs import (
    CONFIGS, MotorConfig, U_DC, MAX_I_S, T_STOP, W_REF,
    SWEEP_BASE, SWEEP_LOAD, LOAD_T, T_STOP_SWEEP,
    PSI_F_VALS, RATIO_VALS,
)


def build_sim(
    cfg_name: str,
    cfg: MotorConfig,
    load_torque: float = 3.0,
    load_time: float = 1.2,
) -> Simulation:
    par = SynchronousMachinePars(
        n_p=cfg["n_p"],
        R_s=cfg["R_s"],
        L_d=cfg["L_d"],
        L_q=cfg["L_q"],
        psi_f=cfg["psi_f"],
    )

    machine = SynchronousMachine(par)
    converter = VoltageSourceConverter(u_dc=U_DC)
    mechanics = MechanicalSystem(J=cfg["J"])
    mechanics.set_external_load_torque(
        lambda t, tau=load_torque, t0=load_time: tau * (t > t0))
    mdl = Drive(converter=converter, machine=machine, mechanics=mechanics)

    cfg_ctrl = CurrentVectorControllerCfg(i_s_max=MAX_I_S, J=cfg["J"])
    vector_ctrl = CurrentVectorController(par=par, cfg=cfg_ctrl, sensorless=False)
    speed_ctrl = SpeedController(J=cfg["J"], alpha_s=2 * np.pi * 4)
    ctrl_sys = VectorControlSystem(vector_ctrl=vector_ctrl, speed_ctrl=speed_ctrl)

    w_ref_mech = W_REF / cfg["n_p"]
    ctrl_sys.set_speed_ref(Step(step_time=0.2, step_value=w_ref_mech, initial_value=0))

    return Simulation(mdl=mdl, ctrl=ctrl_sys, show_progress=False)


def run_all() -> dict[str, Any]:
    results = {}
    for name, cfg in CONFIGS.items():
        print(f"Simulating {name}...")
        sim = build_sim(name, cfg)
        res = sim.simulate(t_stop=T_STOP)
        results[name] = res
        print(f"  done ({T_STOP:.2f}s simulated)")
    return results


def extract_metrics(res: Any, n_p: int) -> dict[str, float]:
    t = res.mdl.t
    w_m = res.mdl.mechanics.w_M
    w_e = w_m * n_p
    i_s = np.abs(res.mdl.machine.i_s_ab)
    tau_M = res.mdl.machine.tau_M

    w_norm = w_e / W_REF

    post_step = t >= 0.2
    settled = post_step & (np.abs(w_norm - 1.0) < 0.05)
    t_settle = t[settled][0] - 0.2 if settled.any() else np.nan

    ss_mask = t >= (T_STOP_SWEEP - 0.2)
    i_ss = np.mean(i_s[ss_mask])

    pre_load = (t >= 1.0) & (t < LOAD_T)
    post_load = (t >= LOAD_T) & (t < LOAD_T + 0.3)
    w_pre = np.mean(w_e[pre_load])
    w_dip = w_pre - np.min(w_e[post_load]) if post_load.any() else np.nan
    speed_droop = w_dip / W_REF

    accel = (t >= 0.2) & (t < 0.8)
    tau_peak = np.max(tau_M[accel]) if accel.any() else np.nan

    return dict(t_settle=t_settle, i_ss=i_ss,
                speed_droop=speed_droop, tau_peak=tau_peak)


def _collect_results(
    store: ResultStore,
    in_memory: dict[str, SweepResult],
    completed: set[str],
) -> dict[str, SlimResult]:
    from phasesweep.result_store import SlimResult
    merged: dict[str, SlimResult] = {}
    if completed:
        merged.update(store.load_slim())
    for cid, r in in_memory.items():
        merged[cid] = SlimResult(
            config_id=cid, status=r.status, metrics=r.metrics,
        )
    return merged


def run_sweep(
    output_dir: str = "results/psi_lq_sweep",
    workers: int = 1,
    timeout_s: int = 60,
    resume: bool = True,
) -> dict[str, NDArray[np.floating]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    from phasesweep.result_store import ResultStore
    from phasesweep.sim_runner import run_sim_safe
    from phasesweep.sweep_types import MotorSweepConfig

    n_psi = len(PSI_F_VALS)
    n_ratio = len(RATIO_VALS)
    _METRIC_KEYS = ("t_settle", "i_ss", "speed_droop", "tau_peak")

    # Build full grid of configs with positional lookup
    grid_configs: dict[str, tuple[int, int]] = {}  # config_id -> (i, j)
    all_configs: list[MotorSweepConfig] = []
    L_d = float(SWEEP_BASE["L_d"])
    for i, psi_f in enumerate(PSI_F_VALS):
        for j, ratio in enumerate(RATIO_VALS):
            config = MotorSweepConfig(
                n_p=int(SWEEP_BASE["n_p"]),
                R_s=float(SWEEP_BASE["R_s"]),
                L_d=L_d,
                L_q=float(L_d * ratio),
                psi_f=float(psi_f),
                J=float(SWEEP_BASE["J"]),
                load_torque=SWEEP_LOAD,
                load_time=LOAD_T,
                t_stop=T_STOP_SWEEP,
            )
            grid_configs[config.config_id] = (i, j)
            all_configs.append(config)

    store = ResultStore(Path(output_dir))
    completed = store.get_known_ids() if resume else set()
    to_run = [config for config in all_configs if config.config_id not in completed]

    total = len(all_configs)
    skipped = total - len(to_run)
    if skipped:
        print(f"  Resuming: {skipped}/{total} already complete")

    done = skipped
    in_memory: dict[str, SweepResult] = {}

    def _save_and_report(result: SweepResult) -> None:
        nonlocal done
        store.save(result)
        in_memory[result.config.config_id] = result
        done += 1
        config = result.config
        print(f"  [{done}/{total}] psi_f={config.psi_f:.3f} "
              f"Lq/Ld={config.L_q/config.L_d:.2f} → {result.status}")

    if workers <= 1:
        for config in to_run:
            _save_and_report(run_sim_safe(config, timeout_s))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_sim_safe, config, timeout_s): config for config in to_run}
            for future in as_completed(futures):
                _save_and_report(future.result())

    # Reconstruct grid from in-memory + disk results
    grid: dict[str, NDArray[np.floating]] = {
        k: np.full((n_psi, n_ratio), np.nan) for k in _METRIC_KEYS
    }
    for cid, slim in _collect_results(store, in_memory, completed).items():
        if cid not in grid_configs or slim.status != "OK" or not slim.metrics:
            continue
        i, j = grid_configs[cid]
        for k in _METRIC_KEYS:
            if k in slim.metrics:
                grid[k][i, j] = slim.metrics[k]

    return grid


def _build_fem_config(
    cfg: MotorConfig, n_slots: int = 0, j_s: float = 0.0, nonlinear: bool = False,
) -> MotorSweepConfig:
    from phasesweep.sweep_types import MotorSweepConfig
    return MotorSweepConfig(
        n_p=int(cfg["n_p"]),
        R_s=float(cfg["R_s"]),
        L_d=float(cfg["L_d"]),
        L_q=float(cfg["L_q"]),
        psi_f=float(cfg["psi_f"]),
        J=float(cfg["J"]),
        n_slots=int(n_slots),
        j_s=float(j_s),
        nonlinear=nonlinear,
        N=int(cfg.get("N", 50)),
        k_w=float(cfg.get("k_w", 0.966)),
        L_stk=float(cfg.get("L_stk", 0.10)),
    )


def _run_fem_batch(
    configs_and_names: list[tuple[str, MotorSweepConfig]],
    output_dir: str,
    resume: bool,
    timeout_s: int,
    label: str,
) -> tuple[dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]], dict[str, NDArray[np.floating]]]:
    from pathlib import Path
    from phasesweep.fem_field import batch_harmonics
    from phasesweep.fem_runner import run_fem_safe
    from phasesweep.result_store import ResultStore

    id_to_name: dict[str, str] = {config.config_id: name for name, config in configs_and_names}

    store = ResultStore(Path(output_dir))
    completed = store.get_known_ids() if resume else set()
    to_run = [(name, config) for name, config in configs_and_names
              if config.config_id not in completed]

    total = len(configs_and_names)
    skipped = total - len(to_run)
    if skipped:
        print(f"  Resuming: {skipped}/{total} already complete")

    done = skipped
    in_memory: dict[str, SweepResult] = {}
    for name, config in to_run:
        print(f"  {label} {name}...")
        result = run_fem_safe(config, timeout_s)
        store.save(result)
        in_memory[result.config.config_id] = result
        done += 1
        print(f"  [{done}/{total}] {name} → {result.status}"
              f" ({result.elapsed_s:.1f}s)")

    fem_results: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]] = {}
    for cid, slim in _collect_results(store, in_memory, completed).items():
        if cid not in id_to_name or slim.status != "OK" or not slim.metrics:
            continue
        name = id_to_name[cid]
        theta = np.array(slim.metrics["theta_list"])
        B_r = np.array(slim.metrics["B_r_list"])
        fem_results[name] = (theta, B_r)

    harmonics = batch_harmonics({name: B for name, (_, B) in fem_results.items()})
    return fem_results, harmonics


def run_field_fem(
    output_dir: str = "results/fem_field",
    resume: bool = True,
    timeout_s: int = 300,
    nonlinear: bool = False,
) -> tuple[dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]], dict[str, NDArray[np.floating]]]:
    configs = [
        (name, _build_fem_config(cfg, nonlinear=nonlinear))
        for name, cfg in CONFIGS.items()
    ]
    return _run_fem_batch(configs, output_dir, resume, timeout_s, "FEM")


def run_field_fem_slotted(
    output_dir: str = "results/fem_field_slotted",
    resume: bool = True,
    timeout_s: int = 300,
    nonlinear: bool = False,
) -> tuple[dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]], dict[str, NDArray[np.floating]]]:
    configs = [
        (name, _build_fem_config(cfg, n_slots=cfg["n_slots"], j_s=cfg["j_s"],
                                  nonlinear=nonlinear))
        for name, cfg in CONFIGS.items()
    ]
    return _run_fem_batch(configs, output_dir, resume, timeout_s, "Slotted FEM")


def run_armature_decomposition(
    cfg_name: str = "A: 4-pole SPMSM",
    nonlinear: bool = False,
) -> tuple[NDArray[np.floating], dict[str, NDArray[np.floating]], dict[str, NDArray[np.floating]], str, int, int]:
    from phasesweep.fem_field import solve_field_fem, batch_harmonics
    cfg = CONFIGS[cfg_name]
    n_p, psi_f = cfg["n_p"], cfg["psi_f"]
    L_d, L_q = cfg["L_d"], cfg["L_q"]
    Q, j_s = cfg["n_slots"], cfg["j_s"]

    print("  PM only (j_s=0) ...")
    theta, B_pm = solve_field_fem(n_p=n_p, psi_f=psi_f, L_d=L_d, L_q=L_q,
                                   n_slots=Q, j_s=0.0, nonlinear=nonlinear)
    print("  Winding only (psi_f~0) ...")
    _, B_arm = solve_field_fem(n_p=n_p, psi_f=1e-4, L_d=L_d, L_q=L_q,
                                n_slots=Q, j_s=j_s, nonlinear=nonlinear)
    print("  Combined ...")
    _, B_comb = solve_field_fem(n_p=n_p, psi_f=psi_f, L_d=L_d, L_q=L_q,
                                 n_slots=Q, j_s=j_s, nonlinear=nonlinear)

    components = {"PM only": B_pm, "Winding only": B_arm, "Combined": B_comb}
    harmonics = batch_harmonics(components)
    return theta, components, harmonics, cfg_name, Q, n_p


def run_slot_sweep(
    slot_counts: tuple[int, ...] = (6, 9, 12, 18, 24, 36),
    cfg_name: str = "A: 4-pole SPMSM",
    output_dir: str = "results/slot_sweep",
    workers: int = 1,
    resume: bool = True,
    timeout_s: int = 300,
    nonlinear: bool = False,
) -> tuple[dict[int, NDArray[np.floating]], int, str]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path
    from phasesweep.fem_field import harmonics_1sided
    from phasesweep.fem_runner import run_fem_safe
    from phasesweep.result_store import ResultStore

    cfg = CONFIGS[cfg_name]

    all_configs: list[tuple[int, MotorSweepConfig]] = []
    slot_map: dict[str, int] = {}
    for Q in slot_counts:
        sweep_config = _build_fem_config(cfg, n_slots=Q, j_s=cfg["j_s"], nonlinear=nonlinear)
        slot_map[sweep_config.config_id] = Q
        all_configs.append((Q, sweep_config))

    store = ResultStore(Path(output_dir))
    completed = store.get_known_ids() if resume else set()
    to_run = [(Q, config) for Q, config in all_configs if config.config_id not in completed]

    total = len(all_configs)
    skipped = total - len(to_run)
    if skipped:
        print(f"  Resuming: {skipped}/{total} already complete")

    done = skipped
    in_memory: dict[str, SweepResult] = {}

    def _save_and_report(result: SweepResult) -> None:
        nonlocal done
        store.save(result)
        in_memory[result.config.config_id] = result
        done += 1
        Q = slot_map[result.config.config_id]
        print(f"  [{done}/{total}] Q={Q} → {result.status}"
              f" ({result.elapsed_s:.1f}s)")

    if workers <= 1:
        for Q, config in to_run:
            print(f"  Slot sweep Q={Q} ...")
            _save_and_report(run_fem_safe(config, timeout_s))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_fem_safe, config, timeout_s): config
                       for _, config in to_run}
            for future in as_completed(futures):
                _save_and_report(future.result())

    results: dict[int, NDArray[np.floating]] = {}
    for cid, slim in _collect_results(store, in_memory, completed).items():
        if cid not in slot_map or slim.status != "OK" or not slim.metrics:
            continue
        Q = slot_map[cid]
        B_r = np.array(slim.metrics["B_r_list"])
        results[Q] = harmonics_1sided(B_r)

    return results, cfg["n_p"], cfg_name
