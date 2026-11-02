"""Fit the BMW i3 [thermal_network] G_FeS to the Burress 2017 continuous test.

One-parameter least squares in G_FeS through the coupled motor_thermal
solve: the predicted winding temperature at 7000 rpm and each of the
three shaft-power steps (25, 50, 75 kW) against the digitized winding
plateaus (last 5 min of each step). R_WFe comes from Tikadar et al. 2020 (Motor-CAD
hot spot, two-point back-solve), R_FeM from the Becker-Kaye air-gap
correlation; both are invisible to the fit at steady state or held fixed.
Capacitances are handbook specific heats times the ORNL masses.

The fit absorbs every loss the model does not carry (AC copper, PWM,
mechanical, the uncalibrated [iron] B_core), so the block it emits is a
pin of the fit, not a validation. Reported: the fitted G_FeS at the
chosen coolant temperature and at both bracket ends, the residual per
point, the model's prediction for the opening no-load segment (not
fitted), and dG_FeS/dR_WFe from re-fits at R_WFe +/- 20 %.

The coolant temperature is unpublished and the trace does not identify
it. The opening segment (7000 rpm, under 1 kW shaft power) holds the
winding at 63.2 degC, drifting 0.1 K over its first 135 s, which bounds
the coolant from above if the motor is at equilibrium there (positive
losses put the winding above the sink; the drift bounds a residual rise
on the 1015 s mode to ~1 K; a warming coolant loop is not excluded).
It is not the coolant: at 7000 rpm the model carries about 540 W of
no-load iron loss, which at the fitted G_FeS sits the winding about
14 K above the sink. Taking the reading as the sink would put the
coolant at 63 degC; taking the model's no-load loss at face value would
put it near 49 degC; the uncalibrated B_core is the largest suspect,
with the conductances and the paths the network omits (housing) behind
it. The fit stays at 60 degC with an assumed sensitivity bracket topped
at the bound, and the block discloses the no-load residual.

Usage:
    uv run python scripts/fit_i3_thermal_network.py [--coolant 60] [--workers 3]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scipy.optimize import minimize_scalar

from phasesweep import load_motor
from phasesweep.machines.motor import ThermalNetwork
from phasesweep.models.motor_thermal import coupled_steady, prepare_motor_thermal

ROOT = Path(__file__).resolve().parent.parent
TOML = ROOT / "motors" / "bmw_i3.toml"
CSV = ROOT / "data" / "bmw_i3" / "burress_2017_slide9_continuous_test_7000rpm.csv"

RPM = 7000.0
W_MECH = RPM * 2.0 * math.pi / 60.0
POWER_STEPS_KW = (25.0, 50.0, 75.0)
PLATEAU_WINDOW_S = 300.0
STEP_TOLERANCE_KW = 8.0

COOLANT_C = 60.0
# Assumed sensitivity bracket; the top is the opening no-load winding
# reading (63.2 degC), an upper bound if the motor is at equilibrium there.
COOLANT_BRACKET_C = (55.0, 63.0)
NOLOAD_MAX_KW = 1.0

# Tikadar et al. 2020 Fig 4/5 two-point back-solve at 4500 rpm, h = 5000 W/m2K
# (Motor-CAD hot spot, not a measurement): R_WFe = 0.0235 K/W.
R_WFE_K_PER_W = 0.0235
# Air gap, Becker & Kaye (1962) Taylor-Couette correlation at 7000 rpm
# (data/bmw_i3/README.md): gap d = 0.7 mm, r = 89.3 mm, L = 132.3 mm, air
# at ~80 degC (nu = 2.1e-5 m2/s, k = 0.030 W/mK). Surface speed
# w r = 65.5 m/s, Re = w r d / nu = 2.2e3, Ta = Re^2 d / r = 3.7e4,
# Nu = 0.409 Ta^0.241 = 5.2 with Nu = h (2 d) / k -> h = 110 W/m2K over
# 2 pi r L = 0.0742 m2 -> G_FeM = 8.2 W/K. Invisible at steady state; a
# correlation estimate, not fitted.
G_FEM_W_PER_K = 8.2
# Handbook specific heats x ORNL masses (EETT Table C-3, Burress 2016
# slide 23). The rotor lump includes the 2.0 kg of magnets.
C_W_J_PER_K = 385.0 * 7.1
C_FE_J_PER_K = 460.0 * 13.7
C_M_J_PER_K = 460.0 * 14.2

G_FES_BOUNDS = (2.0, 2000.0)


def plateaus() -> list[tuple[float, float]]:
    """(shaft power kW, winding plateau degC) per step: the mean over the
    samples whose power sits within STEP_TOLERANCE_KW of the step and
    whose time lies in the last PLATEAU_WINDOW_S seconds of that set
    (`plateau_windows` reports the endpoints and sample counts). This is
    the authoritative plateau definition; the data README quotes it."""
    return [(P, T) for P, T, _, _, _ in plateau_windows()]


def plateau_windows() -> list[tuple[float, float, float, float, int]]:
    """(step kW, mean degC, t_start s, t_end s, n samples) per step."""
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            p = float(r["P_shaft_kW"])
            if math.isnan(p):
                continue
            rows.append((float(r["t_s"]), float(r["T_winding_C"]), p))
    out = []
    for step in POWER_STEPS_KW:
        member = [(t, T) for t, T, p in rows if abs(p - step) <= STEP_TOLERANCE_KW]
        t_end = member[-1][0]
        window = [(t, T) for t, T in member if t >= t_end - PLATEAU_WINDOW_S]
        mean = sum(T for _, T in window) / len(window)
        out.append((step, mean, window[0][0], t_end, len(window)))
    return out


def noload_segment() -> tuple[float, float, float, float, int]:
    """(mean shaft power kW, mean winding degC, t_start s, t_end s, n
    samples) of the opening segment: every sample before the first
    power step whose shaft power is at most NOLOAD_MAX_KW. Not fitted;
    the block reports the model's prediction there."""
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            p = float(r["P_shaft_kW"])
            if math.isnan(p):
                continue
            if p > NOLOAD_MAX_KW:
                break
            rows.append((float(r["t_s"]), float(r["T_winding_C"]), p))
    n = len(rows)
    return (sum(p for _, _, p in rows) / n, sum(T for _, T, _ in rows) / n,
            rows[0][0], rows[-1][0], n)


def _network(G_FeS: float, G_WFe: float) -> ThermalNetwork:
    return ThermalNetwork(G_WFe=G_WFe, G_FeS=G_FeS, G_FeM=G_FEM_W_PER_K,
                          C_W=C_W_J_PER_K, C_Fe=C_FE_J_PER_K, C_M=C_M_J_PER_K)


def _predict(args: tuple[float, float, float, float]) -> float:
    G_FeS, G_WFe, T_coolant, P_kW = args
    motor = dataclasses.replace(load_motor(TOML), thermal_network=_network(G_FeS, G_WFe))
    p = prepare_motor_thermal(motor)
    return coupled_steady(p, W_MECH, P_kW * 1e3 / W_MECH, T_coolant)["T_winding"]


@dataclasses.dataclass
class Fit:
    G_FeS: float
    T_pred: list[float]
    residual_K: list[float]

    @property
    def rms_K(self) -> float:
        return math.sqrt(sum(r * r for r in self.residual_K) / len(self.residual_K))


def fit(data: list[tuple[float, float]], G_WFe: float, T_coolant: float,
        pool: ProcessPoolExecutor) -> Fit:
    def predict(G_FeS: float) -> list[float]:
        return list(pool.map(_predict, [(G_FeS, G_WFe, T_coolant, P) for P, _ in data]))

    def sse(log_G: float) -> float:
        return sum((Tp - Tm) ** 2 for Tp, (_, Tm) in zip(predict(math.exp(log_G)), data))

    lo, hi = (math.log(b) for b in G_FES_BOUNDS)
    res = minimize_scalar(sse, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    if not res.success:
        raise SystemExit(f"minimizer failed: {res.message}")
    # A bounded minimizer stops a little inside its bounds; a monotone
    # objective would otherwise pass a strict inequality at the edge.
    if min(res.x - lo, hi - res.x) < 1e-2:
        raise SystemExit(
            f"fit drove G_FeS to the bound {math.exp(res.x):.4g} W/K "
            f"(bounds {G_FES_BOUNDS})")
    G_FeS = math.exp(res.x)
    T_pred = predict(G_FeS)
    return Fit(G_FeS, T_pred, [Tp - Tm for Tp, (_, Tm) in zip(T_pred, data)])


def block(G_FeS: float, G_WFe: float, T_coolant: float, f: Fit,
          bracket: dict[float, float], dG_dR: float,
          noload: tuple[float, float, float]) -> str:
    """noload — (shaft power kW, measured degC, predicted degC) of the
    opening segment."""
    resid = ", ".join(f"{P:.0f} kW {r:+.1f} K" for (P, _), r in zip(plateaus(), f.residual_K))
    lo, hi = (bracket[T] for T in COOLANT_BRACKET_C)
    P0, T0, Tp0 = noload
    return f"""# Three-node lumped network (winding W, iron Fe, magnet/rotor M) for the
# motor_thermal model; sink = RunConfig.T_coolant. G_FeS FITTED to the
# Burress 2017 slide 9 continuous test at 7000 rpm (three winding
# plateaus, data/bmw_i3/burress_2017_slide9_continuous_test_7000rpm.csv)
# through the coupled solve at an assumed coolant of {T_coolant:.0f} degC
# (unpublished; the opening no-load segment, {T0:.1f} degC at {P0:.1f} kW, bounds
# it from above if the motor is at equilibrium there; assumed bracket
# {COOLANT_BRACKET_C[0]:.0f}-{COOLANT_BRACKET_C[1]:.0f} degC gives G_FeS {lo:.1f}-{hi:.1f} W/K).
# Residuals (predicted - measured): {resid}
# (rms {f.rms_K:.1f} K); at the no-load segment, not fitted, the model
# predicts {Tp0:.1f} degC ({Tp0 - T0:+.1f} K): its no-load iron loss at 7000 rpm
# holds the winding above the sink where the trace sits near it. The
# fit absorbs every loss the model does not carry (AC copper, PWM,
# mechanical, the uncalibrated [iron] B_core): a pin, not a validation.
# scripts/fit_i3_thermal_network.py regenerates this block;
# scripts/check_i3_thermal_transient.py marches it through the trace.
[thermal_network]
G_WFe = {G_WFe:.2f}   # W/K, 1/R_WFe, R_WFe = {1 / G_WFe:.4f} K/W from Tikadar et al. 2020 (Motor-CAD hot spot
                # two-point back-solve at 4500 rpm, h = 5000 W/m2K); dG_FeS/dR_WFe =
                # {dG_dR:.0f} W/K per K/W from re-fits at R_WFe +/- 20 %
G_FeS = {G_FeS:.2f}   # W/K, fitted (R_FeS = {1 / G_FeS:.4f} K/W; the Tikadar et al. model implies 0.0029)
G_FeM = {G_FEM_W_PER_K:.1f}     # W/K, air gap, Becker-Kaye correlation at 7000 rpm; invisible at steady state
C_W = {C_W_J_PER_K:.1f}    # J/K, 385 J/kgK x 7.1 kg copper (EETT Table C-3)
C_Fe = {C_FE_J_PER_K:.1f}   # J/K, 460 J/kgK x 13.7 kg stator core (EETT Table C-3)
C_M = {C_M_J_PER_K:.1f}    # J/K, 460 J/kgK x 14.2 kg rotor incl. 2.0 kg magnets (Burress 2016 slide 23)
winding_basis = "ETD"   # assumed: coil-mounted stator sensors (BMW i3 manual, Burress 2016
                        # slide 21); nothing shows the Burress 2017 trace came from them
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coolant", type=float, default=COOLANT_C)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    data = plateaus()
    for P, T, t0, t1, n in plateau_windows():
        print(f"plateau {P:.0f} kW: {T:.2f} degC, {n} samples, t = {t0:.1f}..{t1:.1f} s")
    P0, T0, t0, t1, n = noload_segment()
    print(f"no-load segment: {T0:.2f} degC at {P0:.2f} kW, {n} samples, t = {t0:.1f}..{t1:.1f} s")
    G_WFe = 1.0 / R_WFE_K_PER_W
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        main_fit = fit(data, G_WFe, a.coolant, pool)
        print(f"coolant {a.coolant:.0f} degC: G_FeS = {main_fit.G_FeS:.2f} W/K "
              f"(R_FeS = {1 / main_fit.G_FeS:.4f} K/W), rms {main_fit.rms_K:.2f} K")
        for (P, Tm), Tp, r in zip(data, main_fit.T_pred, main_fit.residual_K):
            print(f"  {P:.0f} kW: predicted {Tp:.1f}, measured {Tm:.1f}, residual {r:+.1f} K")
        Tp0 = _predict((main_fit.G_FeS, G_WFe, a.coolant, P0))
        print(f"  no-load {P0:.1f} kW (not fitted): predicted {Tp0:.1f}, measured {T0:.1f}, "
              f"residual {Tp0 - T0:+.1f} K")

        bracket = {}
        for T_c in COOLANT_BRACKET_C:
            bracket[T_c] = fit(data, G_WFe, T_c, pool).G_FeS
            print(f"coolant {T_c:.0f} degC: G_FeS = {bracket[T_c]:.2f} W/K")

        sens = {}
        for scale in (0.8, 1.2):
            R = R_WFE_K_PER_W * scale
            sens[R] = fit(data, 1.0 / R, a.coolant, pool).G_FeS
            print(f"R_WFe {R:.4f} K/W: G_FeS = {sens[R]:.2f} W/K")
    (R_lo, R_hi) = sorted(sens)
    dG_dR = (sens[R_hi] - sens[R_lo]) / (R_hi - R_lo)
    print(f"dG_FeS/dR_WFe = {dG_dR:.1f} W/K per K/W")
    print()
    print(block(main_fit.G_FeS, G_WFe, a.coolant, main_fit, bracket, dG_dR, (P0, T0, Tp0)))


if __name__ == "__main__":
    main()
