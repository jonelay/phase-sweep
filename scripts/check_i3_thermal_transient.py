"""March the BMW i3 [thermal_network] through the Burress 2017 slide 9 trace.

The steady fit (scripts/fit_i3_thermal_network.py) sees only the three
plateaus; the capacitances (handbook specific heats x ORNL masses) and
the modes of C^-1 G are unchecked by it. This script drives the network
with the digitized shaft-power trace at 7000 rpm, sample by sample
(piecewise-constant power, losses from the operating point at the
marching winding and magnet temperatures, the same loss path as the
coupled solve), from the model's own no-load equilibrium, and compares
the rise between plateaus with the digitized winding trace: the level
before and after each step and the time to 63.2 % and 90 % of the rise.

The comparison is at 7000 rpm only. Steady temperatures are independent
of G_FeM (the rotor node carries no source); the winding transient sees
every mode, the slow one weakly, so the trace constrains G_FeM only
entangled with the capacitances.

Usage:
    uv run python scripts/check_i3_thermal_transient.py [--coolant 60] [--out trace.csv]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_i3_thermal_network import (
    COOLANT_C,
    CSV,
    POWER_STEPS_KW,
    STEP_TOLERANCE_KW,
    TOML,
    W_MECH,
    plateau_windows,
)

from phasesweep import load_motor
from phasesweep.models.iron_loss import iron_loss_at
from phasesweep.models.motor_thermal import (
    _modes,
    assemble_network,
    prepare_motor_thermal,
    segment_map,
    steady_temps,
)
from phasesweep.models.operating_point import operating_point

# Re-evaluate the operating point when the power or a node temperature
# has moved this much since the last evaluation (R_s and the PM flux
# follow the nodes; 1 K is 0.4 % of copper loss).
REEVAL_KW = 0.25
REEVAL_K = 1.0
BEFORE_WINDOW_S = 60.0
FRACTIONS = (0.632, 0.9)
EQUILIBRIUM_TOL_K = 0.01
EQUILIBRIUM_MAX_ITER = 50


def trace() -> list[tuple[float, float, float]]:
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            p = float(r["P_shaft_kW"])
            if math.isnan(p):
                continue
            rows.append((float(r["t_s"]), float(r["T_winding_C"]), p))
    return rows


class Losses:
    """(p_cu, p_fe) at shaft power P and node temperatures, memoized on
    the re-evaluation thresholds."""

    def __init__(self, params) -> None:
        self.p = params
        self.last: tuple[float, float, float] | None = None
        self.value = (0.0, 0.0)
        self.evaluations = 0

    def __call__(self, P_kW: float, T: np.ndarray) -> tuple[float, float]:
        key = (P_kW, float(T[0]), float(T[2]))
        if self.last is not None and (
                abs(key[0] - self.last[0]) < REEVAL_KW
                and abs(key[1] - self.last[1]) < REEVAL_K
                and abs(key[2] - self.last[2]) < REEVAL_K):
            return self.value
        return self.evaluate(P_kW, T)

    def evaluate(self, P_kW: float, T: np.ndarray) -> tuple[float, float]:
        """Unmemoized evaluation; updates the memo."""
        key = (P_kW, float(T[0]), float(T[2]))
        op = self.p.op
        r = operating_point(op.n_p, op.psi_f, op.R_s, op.L_d, op.L_q, op.I_max,
                            op.U_max, op.U_DC, W_MECH, P_kW * 1e3 / W_MECH,
                            flux_map=op.flux_map, winding_temp=key[1],
                            magnet_temp=key[2], alpha_Br=op.alpha_Br)
        psi_ratio = math.hypot(r.psi_d, r.psi_q) / self.p.psi_ref
        p_fe = iron_loss_at(op.n_p, self.p.k_h, self.p.k_e, self.p.alpha_fe,
                            self.p.m_core, self.p.B_core, W_MECH, psi_ratio)
        self.last, self.value = key, (r.p_cu, p_fe)
        self.evaluations += 1
        return self.value


def march(T_coolant: float, G_FeS: float | None = None,
          ) -> tuple[list[tuple[float, float, float]], np.ndarray, dict]:
    """(trace rows, model node temperatures per row, info). G_FeS
    overrides the TOML block's value (sensitivity; the steady fit moves
    it with the assumed coolant)."""
    rows = trace()
    motor = load_motor(TOML)
    if G_FeS is not None:
        motor = dataclasses.replace(
            motor, thermal_network=dataclasses.replace(motor.thermal_network, G_FeS=G_FeS))
    p = prepare_motor_thermal(motor)
    G, g_s, C = assemble_network(p.network)
    losses = Losses(p)
    # Start at the model's own equilibrium for the opening (no-load) power:
    # the fixed point of losses(T) -> steady network, iterated on the
    # unmemoized losses to EQUILIBRIUM_TOL_K (the coupled solve's loop
    # without its relaxation; at the block's values it closes in a few
    # iterations, an override can make it wander, hence the check).
    T = np.full(3, T_coolant + 20.0)
    for _ in range(EQUILIBRIUM_MAX_ITER):
        p_cu, p_fe = losses.evaluate(rows[0][2], T)
        T_new = steady_temps(G, g_s, np.array([p_cu, p_fe, 0.0]), T_coolant)
        resid = float(np.max(np.abs(T_new - T)))
        T = T_new
        if resid <= EQUILIBRIUM_TOL_K:
            break
    else:
        raise SystemExit(
            f"opening equilibrium did not converge in {EQUILIBRIUM_MAX_ITER} "
            f"iterations (residual {resid:.3g} K at T_winding {T[0]:.1f} degC)")
    out = np.empty((len(rows), 3))
    out[0] = T
    for i in range(1, len(rows)):
        dt = rows[i][0] - rows[i - 1][0]
        p_cu, p_fe = losses(rows[i - 1][2], T)
        M, v = segment_map(G, g_s, C, np.array([p_cu, p_fe, 0.0]), T_coolant, dt)
        T = M @ T + v
        out[i] = T
    lam, _, _ = _modes(G, C)
    info = {"evaluations": losses.evaluations,
            "tau_s": sorted(1.0 / lam),
            "T0": out[0].tolist()}
    return rows, out, info


def rise_metrics(t: np.ndarray, T: np.ndarray, P: np.ndarray) -> list[dict]:
    """Per step: onset, level before (last BEFORE_WINDOW_S before onset),
    level after (the fit's plateau window) and the time from onset to
    each FRACTIONS of the rise. Onset is the first sample within
    STEP_TOLERANCE_KW of the step after the previous plateau (17 kW for
    the 25 kW step, after the ramp's first ~6 K), so these are
    remaining-rise times from that sample, threshold-dependent by
    construction, applied identically to both traces."""
    out = []
    t_prev_end = t[0]
    for step, (_, _, t0, t1, _) in zip(POWER_STEPS_KW, plateau_windows()):
        member = np.flatnonzero((np.abs(P - step) <= STEP_TOLERANCE_KW) & (t > t_prev_end))
        onset = t[member[0]]
        before = float(np.mean(T[(t >= onset - BEFORE_WINDOW_S) & (t < onset)]))
        after = float(np.mean(T[(t >= t0) & (t <= t1)]))
        times = {}
        for f in FRACTIONS:
            level = before + f * (after - before)
            hit = np.flatnonzero((t >= onset) & (T >= level))
            times[f] = float(t[hit[0]] - onset) if len(hit) else math.nan
        out.append({"step": step, "onset": float(onset), "before": before,
                    "after": after, "times": times})
        t_prev_end = t1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coolant", type=float, default=COOLANT_C)
    ap.add_argument("--G-FeS", type=float, dest="G_FeS",
                    help="override the block's G_FeS (W/K) for a sensitivity run")
    ap.add_argument("--out", type=Path, help="write t, measured, model nodes as CSV")
    a = ap.parse_args()

    rows, model, info = march(a.coolant, a.G_FeS)
    t = np.array([r[0] for r in rows])
    T_meas = np.array([r[1] for r in rows])
    P = np.array([r[2] for r in rows])
    T_model = model[:, 0]
    print(f"coolant {a.coolant:.0f} degC{f', G_FeS {a.G_FeS:.1f} W/K' if a.G_FeS else ''}; network modes "
          f"{', '.join(f'{x:.0f}' for x in info['tau_s'])} s; "
          f"{info['evaluations']} operating-point evaluations over {len(rows)} samples")
    print(f"opening equilibrium: model winding {info['T0'][0]:.1f} degC, "
          f"measured {T_meas[0]:.1f} degC at {P[0]:.1f} kW")
    meas, mod = rise_metrics(t, T_meas, P), rise_metrics(t, T_model, P)
    for m, k in zip(meas, mod):
        print(f"step {m['step']:.0f} kW at t = {m['onset']:.0f} s: rise "
              f"{m['before']:.1f} -> {m['after']:.1f} degC measured, "
              f"{k['before']:.1f} -> {k['after']:.1f} degC model")
        for f in FRACTIONS:
            print(f"  t{f * 100:.0f}: measured {m['times'][f]:.0f} s, "
                  f"model {k['times'][f]:.0f} s")
    resid = T_model - T_meas
    print(f"winding residual over the trace (model - measured): mean {resid.mean():+.1f} K, "
          f"rms {math.sqrt(np.mean(resid ** 2)):.1f} K, "
          f"range {resid.min():+.1f}..{resid.max():+.1f} K")
    if a.out:
        with a.out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "P_shaft_kW", "T_winding_meas_C", "T_winding_C",
                        "T_iron_C", "T_magnet_C"])
            for i in range(len(rows)):
                w.writerow([f"{t[i]:.1f}", f"{P[i]:.1f}", f"{T_meas[i]:.1f}",
                            *(f"{x:.2f}" for x in model[i])])
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
