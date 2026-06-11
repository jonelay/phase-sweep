#!/usr/bin/env python3
"""Back-EMF validation: analytical vs FEM vs measured for the actuator steel rotor.

Produces output/back_emf_validation/report.md with comparison tables,
sensitivity sweeps, and B_r waveform plots.
"""

from __future__ import annotations

import json
import sys
from math import pi
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phasesweep.configs import load_motor
from phasesweep.fem_field import (
    _derive_B_rem,
    carter_adjusted_radii,
    compute_thd,
    end_effect_factor,
    harmonics_1sided,
    solve_field_fem,
    zhu_howe_Br,
)
from phasesweep.motor import Motor
from phasesweep.solver_params import (
    _derive_psi_f,
    n_eff,
    prepare_analytical,
    prepare_drive_sim,
    prepare_fem,
    psi_f_carter,
    winding_transfer,
)

# ── Constants ──────────────────────────────────────────────────────────
TOML = ROOT / "motors" / "actuator_steel_rotor.toml"
OUT = ROOT / "output" / "back_emf_validation"
CAPTURES_DIR = ROOT / "data" / "actuator_steel_rotor" / "captures"
SWEEP_JSON = ROOT / "data" / "actuator_steel_rotor" / "backemf_speed_sweep.json"

# Test conditions (80 rps reference point)
W_MECH = 501.8      # rad/s mechanical (79.86 rps actual)
N_P = 6
W_ELEC = W_MECH * N_P  # 3010.8 rad/s
SPEED_RPM = 4792

# Measured values (3-channel speed sweep 2026-03-18, linear fit R²=0.99997)
# Supersedes single-channel 80 rps estimate from 2026-03-16.
PSI_F_MEAS = 0.000268   # Wb (from Ke / n_p)
V_LN_PEAK_MEAS = 0.8087 # V (3-channel avg at 79.86 rps)
KE_MEAS = 0.001607       # V/(rad/s) mechanical (combined linear fit)
KT_MEAS = 0.002412       # Nm/A (1.5 * n_p * psi_f)
THD_MEAS_BOUND = 2.5     # % upper bound (5th harmonic ~1.6% dominates)

# ── Paul Tol vibrant palette (colorblind-safe) ───────────────────────
# https://personal.sron.nl/~pault/data/colourschemes.pdf
TOL_BLUE = "#0077BB"
TOL_CYAN = "#33BBEE"
TOL_TEAL = "#009988"
TOL_ORANGE = "#EE7733"
TOL_RED = "#CC3311"
TOL_MAGENTA = "#EE3377"
TOL_GREY = "#BBBBBB"

# Semantic mapping for consistent use across all plots
CLR_ANALYTICAL = TOL_BLUE
CLR_FEM = TOL_ORANGE
CLR_MEASURED = TOL_RED
CLR_CH1 = TOL_RED       # red wire
CLR_CH2 = TOL_TEAL      # green wire (Tol teal for deuteranopia)
CLR_CH4 = TOL_BLUE      # blue wire
CLR_SMOOTH = TOL_CYAN      # smooth-bore analytical (no Carter)
CLR_NEUTRAL = TOL_GREY
CLR_ACCENT = TOL_MAGENTA
CLR_FEM_KEND = TOL_TEAL    # FEM + end-effect correction


class FieldResult(TypedDict):
    """Air-gap field solution and derived scalars (run_analytical / run_fem)."""
    theta: np.ndarray
    B_r: np.ndarray
    B_g1: float
    thd_pct: float
    psi_f: float
    backemf: float | None
    amps: np.ndarray
    psi_f_smooth: NotRequired[float | None]  # analytical only
    k_c: NotRequired[float]                  # analytical only


class EndEffectResult(TypedDict):
    """FEM scalars with Russell-Norsworthy k_end applied (apply_end_effect)."""
    B_g1: float
    psi_f: float
    backemf: float
    k_end: float


class EffectiveBremResult(TypedDict):
    """Back-calculated effective remanence with per-source perturbations.

    uncertainty: "psi_f"/"k_w"/"alpha_p" -> (lo, hi, half_width); "N=k" -> point estimate.
    """
    B_rem_eff: float
    grade: str
    grade_lo: str
    grade_hi: str
    uncertainty: dict[str, tuple[float, float, float] | float]
    rss: float


def compute_end_effect(motor: Motor) -> tuple[float, float]:
    """Compute Russell-Norsworthy end-effect factor k_end for a motor.

    Returns (k_end, g_eff) tuple.  k_end in (0, 1] — multiply B_g1 by k_end
    to account for axial flux leakage at stack ends.
    """
    geo = motor.geometry
    g = abs(geo.r_stator - geo.r_magnet)
    h_m = abs(geo.r_magnet - geo.r_rotor)
    g_eff = g + h_m / motor.mu_r_pm
    k_end = end_effect_factor(motor.L_stk, g_eff)
    return k_end, g_eff


def apply_end_effect(fem_result: FieldResult, k_end: float, motor: Motor) -> EndEffectResult:
    """Create a corrected copy of FEM results with end-effect factor applied.

    B_g1 is scaled by k_end; psi_f and backemf follow linearly.
    """
    B_g1_corr = fem_result["B_g1"] * k_end
    psi_f_corr = B_g1_corr * winding_transfer(motor)
    backemf_corr = W_ELEC * psi_f_corr
    return {
        "B_g1": B_g1_corr,
        "psi_f": psi_f_corr,
        "backemf": backemf_corr,
        "k_end": k_end,
    }


def _psi_f_carter(motor, r_s_c, r_m_c):
    """Derive psi_f using Carter-adjusted geometry for the field, original bore for winding."""
    if motor.B_rem is None:
        return None
    return psi_f_carter(motor, r_s_c, r_m_c, motor.B_rem)


def _fem_psi_f_scaled(fem, motor, B_rem=None, k_w=None):
    """Scale FEM psi_f linearly for B_rem or k_w changes (no geometry change).

    Valid because B_g1 is linear in B_rem (no saturation) and k_w is a
    post-solve multiplier in the winding formula.
    """
    scale = 1.0
    if B_rem is not None:
        scale *= B_rem / motor.B_rem
    if k_w is not None:
        scale *= k_w / motor.k_w
    return fem["psi_f"] * scale


def run_analytical(motor: Motor) -> FieldResult:
    params = prepare_analytical(motor)
    geo = params.geometry

    # Carter factor (same logic as registry runner)
    r_stator, r_magnet, k_c = carter_adjusted_radii(geo, params.mu_r_pm)
    r_rotor = geo.r_rotor

    n_theta = 720
    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    B_r = zhu_howe_Br(
        theta, params.n_p, params.B_rem,
        r_stator=r_stator, r_magnet=r_magnet, r_rotor=r_rotor,
        mu_r_pm=params.mu_r_pm, alpha_p=params.alpha_p,
    )
    amps = harmonics_1sided(B_r)
    fund_idx = min(params.n_p, len(amps) - 1)
    B_g1 = float(amps[fund_idx])
    thd = compute_thd(amps, fund_idx)

    # psi_f from Carter-corrected B_g1 (consistent with B_g1 in the table)
    psi_f = B_g1 * winding_transfer(motor)

    # Also keep smooth-bore baseline for reference
    psi_f_smooth = _derive_psi_f(motor)

    # Back-EMF at test speed
    backemf = W_ELEC * psi_f if psi_f else None

    return {
        "theta": theta, "B_r": B_r,
        "B_g1": B_g1, "thd_pct": thd,
        "psi_f": psi_f, "backemf": backemf,
        "psi_f_smooth": psi_f_smooth,
        "k_c": k_c,
        "amps": amps,
    }


def run_fem(motor: Motor, nonlinear: bool = False) -> FieldResult:
    params = prepare_fem(motor)
    geo = params.geometry

    theta, B_r = solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        maxh_fraction=0.03, n_theta=720,
        nonlinear=nonlinear,
        j_s=0.0,
        alpha_p=params.alpha_p,
    )
    amps = harmonics_1sided(B_r)
    fund_idx = min(params.n_p, len(amps) - 1)
    B_g1 = float(amps[fund_idx])
    thd = compute_thd(amps, fund_idx)

    # FEM-derived psi_f: use B_g1 from field solution with winding formula
    psi_f_fem = B_g1 * winding_transfer(motor)

    backemf = W_ELEC * psi_f_fem

    return {
        "theta": theta, "B_r": B_r,
        "B_g1": B_g1, "thd_pct": thd,
        "psi_f": psi_f_fem, "backemf": backemf,
        "amps": amps,
    }


def sweep_parameter(motor, param_name, values, carter_geo=None):
    """Sweep a single parameter and return psi_f for each value."""
    from dataclasses import replace
    results = []
    for val in values:
        if param_name == "B_rem":
            m = replace(motor, B_rem=val)
        elif param_name == "alpha_p":
            m = replace(motor, alpha_p=val)
        elif param_name == "k_w":
            m = replace(motor, k_w=val)
        else:
            raise ValueError(f"Unknown param: {param_name}")
        if carter_geo:
            psi_f = _psi_f_carter(m, *carter_geo)
        else:
            psi_f = _derive_psi_f(m)
        results.append(psi_f)
    return np.array(results)


def find_matching_value(motor, param_name, values, target_psi_f, carter_geo=None):
    """Find parameter value that gives target psi_f by interpolation."""
    psi_f_arr = sweep_parameter(motor, param_name, values, carter_geo=carter_geo)
    # Linear interpolation to find crossing
    for i in range(len(psi_f_arr) - 1):
        if (psi_f_arr[i] - target_psi_f) * (psi_f_arr[i+1] - target_psi_f) <= 0:
            frac = (target_psi_f - psi_f_arr[i]) / (psi_f_arr[i+1] - psi_f_arr[i])
            return values[i] + frac * (values[i+1] - values[i])
    return None


def _measured_equiv_B_g1(motor):
    """Back-derive B_g1 from measured psi_f using winding formula (inverse)."""
    return PSI_F_MEAS / winding_transfer(motor)


def _extract_cycle(signal, dt, f_e):
    """Find a rising zero-crossing and extract one electrical cycle."""
    T_e = 1.0 / f_e
    samples_per_cycle = round(T_e / dt)
    skip = len(signal) // 10
    for i in range(skip, len(signal) - samples_per_cycle - 1):
        if signal[i] <= 0 < signal[i + 1]:
            seg = slice(i, i + samples_per_cycle)
            theta = np.arange(samples_per_cycle) * dt / T_e * 360.0
            return theta, signal[seg]
    return None, None


def compute_measurement_uncertainty():
    """Quantify psi_f/Ke measurement uncertainty from per-channel spread."""
    if not SWEEP_JSON.exists():
        return None

    with open(SWEEP_JSON) as f:
        sweep = json.load(f)

    per_ch = sweep["per_channel_Ke_mV_per_rads"]
    ke_values = np.array(list(per_ch.values())) * 1e-3  # V/(rad/s)
    ke_mean = np.mean(ke_values)
    ke_std = np.std(ke_values, ddof=1)
    ke_spread = np.max(ke_values) - np.min(ke_values)

    psi_f_values = ke_values / N_P
    psi_f_mean = np.mean(psi_f_values)
    psi_f_std = np.std(psi_f_values, ddof=1)

    # Per-speed Ke variation
    points = sweep["speed_sweep"]
    per_speed_ke = []
    for pt in points:
        omega = pt["actual_rps"] * 2 * pi
        v_avg = np.mean(pt["V_pk_mV"]) * 1e-3
        per_speed_ke.append(v_avg / omega)
    ke_speed_std = np.std(per_speed_ke, ddof=1)

    return {
        "ke_mean": ke_mean,
        "ke_std": ke_std,
        "ke_spread": ke_spread,
        "ke_rel_pct": ke_std / ke_mean * 100,
        "ke_speed_std": ke_speed_std,
        "psi_f_mean": psi_f_mean,
        "psi_f_std": psi_f_std,
        "psi_f_rel_pct": psi_f_std / psi_f_mean * 100,
        "per_channel": per_ch,
    }


def _ndfeb_grade(B_rem):
    """Map B_rem to approximate NdFeB grade (room temperature)."""
    grades = [
        (1.00, "N30"), (1.08, "N33"), (1.13, "N35"), (1.17, "N38"),
        (1.21, "N40"), (1.25, "N42"), (1.28, "N45"), (1.32, "N48"),
        (1.37, "N50"), (1.42, "N52"), (1.45, "N55"),
    ]
    grade = "below N30"
    for threshold, g in grades:
        if B_rem >= threshold:
            grade = g
    return grade


def _b_rem_from_params(psi_f, n_p, N_eff, k_w, L_stk, geo, mu_r_pm, alpha_p,
                       carter_geo=None):
    """Invert psi_f to B_rem. With Carter, uses adjusted radii for field but original for winding."""
    r_s_c, r_m_c = carter_geo if carter_geo else (None, None)
    return _derive_B_rem(
        psi_f, n_p, N_eff, k_w, L_stk,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=mu_r_pm, alpha_p=alpha_p,
        r_stator_c=r_s_c, r_magnet_c=r_m_c,
    )


def compute_effective_B_rem(
    motor: Motor,
    psi_f_measured: float,
    meas_unc: dict[str, Any] | None = None,
    carter_geo: tuple[float, float] | None = None,
) -> EffectiveBremResult:
    """Back-calculate effective B_rem from measured psi_f with uncertainty.

    This is a *combined* effective B_rem that absorbs errors in k_w, alpha_p,
    and leakage — it's what the magnet would need to be if everything else
    were nominal.
    """
    geo = motor.geometry
    N_eff = n_eff(motor)
    cg = carter_geo
    B_rem_eff = _b_rem_from_params(
        psi_f_measured, motor.n_p, N_eff, motor.k_w, motor.L_stk,
        geo, motor.mu_r_pm, motor.alpha_p, carter_geo=cg,
    )

    unc = {}

    if meas_unc:
        psi_f_lo = psi_f_measured - meas_unc["psi_f_std"]
        psi_f_hi = psi_f_measured + meas_unc["psi_f_std"]
        b_lo = _b_rem_from_params(psi_f_lo, motor.n_p, N_eff, motor.k_w, motor.L_stk,
                                   geo, motor.mu_r_pm, motor.alpha_p, carter_geo=cg)
        b_hi = _b_rem_from_params(psi_f_hi, motor.n_p, N_eff, motor.k_w, motor.L_stk,
                                   geo, motor.mu_r_pm, motor.alpha_p, carter_geo=cg)
        unc["psi_f"] = (b_lo, b_hi, abs(b_hi - b_lo) / 2)

    kw_lo, kw_hi = motor.k_w * 0.95, motor.k_w * 1.05
    b_kw_lo = _b_rem_from_params(psi_f_measured, motor.n_p, N_eff, kw_lo, motor.L_stk,
                                  geo, motor.mu_r_pm, motor.alpha_p, carter_geo=cg)
    b_kw_hi = _b_rem_from_params(psi_f_measured, motor.n_p, N_eff, kw_hi, motor.L_stk,
                                  geo, motor.mu_r_pm, motor.alpha_p, carter_geo=cg)
    unc["k_w"] = (b_kw_hi, b_kw_lo, abs(b_kw_lo - b_kw_hi) / 2)

    ap_lo, ap_hi = motor.alpha_p * 0.95, motor.alpha_p * 1.05
    b_ap_lo = _b_rem_from_params(psi_f_measured, motor.n_p, N_eff, motor.k_w, motor.L_stk,
                                  geo, motor.mu_r_pm, ap_lo, carter_geo=cg)
    b_ap_hi = _b_rem_from_params(psi_f_measured, motor.n_p, N_eff, motor.k_w, motor.L_stk,
                                  geo, motor.mu_r_pm, ap_hi, carter_geo=cg)
    unc["alpha_p"] = (b_ap_hi, b_ap_lo, abs(b_ap_lo - b_ap_hi) / 2)

    for N_alt, label in [(motor.N - 1, "N-1"), (motor.N - 2, "N-2")]:
        if N_alt < 1:
            continue
        N_eff_alt = N_alt * motor.coils_series
        b_alt = _b_rem_from_params(psi_f_measured, motor.n_p, N_eff_alt, motor.k_w, motor.L_stk,
                                    geo, motor.mu_r_pm, motor.alpha_p, carter_geo=cg)
        unc[f"N={N_alt}"] = b_alt

    # RSS of independent uncertainties (psi_f + k_w)
    components = []
    if "psi_f" in unc:
        components.append(unc["psi_f"][2])
    components.append(unc["k_w"][2])
    rss = np.sqrt(sum(c**2 for c in components))

    return {
        "B_rem_eff": B_rem_eff,
        "grade": _ndfeb_grade(B_rem_eff),
        "grade_lo": _ndfeb_grade(B_rem_eff - rss),
        "grade_hi": _ndfeb_grade(B_rem_eff + rss),
        "uncertainty": unc,
        "rss": rss,
    }


def extract_measured_harmonics(rps_cmd, n_p, actual_rps, n_harmonics=7):
    """Hann-windowed FFT harmonic extraction from raw capture.

    Returns harmonics as % of fundamental + THD, averaged across 3 channels.
    Returns None if capture file missing.
    """
    csv_path = CAPTURES_DIR / f"backemf_{rps_cmd:03d}rps.csv"
    if not csv_path.exists():
        return None

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["time_s"]
    dt = t[1] - t[0]
    f_e = actual_rps * n_p
    N_pts = len(t)

    target_ks = [1, 2, 3, 5, 7]
    target_ks = [k for k in target_ks if k <= n_harmonics]

    channels = [data["CH1_V"], data["CH2_V"], data["CH4_V"]]
    all_harmonics = []  # per-channel dicts

    for sig in channels:
        window = np.hanning(N_pts)
        windowed = sig * window
        spectrum = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(N_pts, dt)
        magnitudes = np.abs(spectrum) * 2.0 / np.sum(window)
        df = freqs[1]

        harmonics = {}
        for k in target_ks:
            f_target = k * f_e
            bin_center = f_target / df
            # Search +/- 2 bins around expected location for peak
            lo = max(1, int(bin_center) - 2)
            hi = min(len(magnitudes) - 2, int(bin_center) + 3)
            peak_bin = lo + np.argmax(magnitudes[lo:hi])
            # Parabolic interpolation around peak
            alpha = magnitudes[peak_bin - 1]
            beta = magnitudes[peak_bin]
            gamma = magnitudes[peak_bin + 1]
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-30:
                p = 0.5 * (alpha - gamma) / denom
                interp_mag = beta - 0.25 * (alpha - gamma) * p
            else:
                interp_mag = beta
            harmonics[k] = interp_mag

        all_harmonics.append(harmonics)

    # Average across channels
    avg = {}
    fund_avg = np.mean([h.get(1, 0) for h in all_harmonics])
    if fund_avg == 0:
        return None

    for k in target_ks:
        vals = [h.get(k, 0) for h in all_harmonics]
        avg[k] = np.mean(vals) / fund_avg * 100  # % of fundamental

    # THD = sqrt(sum of squares of harmonics 2+) / fundamental
    thd_sum = 0
    for k in target_ks:
        if k > 1:
            thd_sum += (avg[k] / 100) ** 2
    thd = np.sqrt(thd_sum) * 100

    return {"harmonics_pct": avg, "thd_pct": thd, "fundamental_V": fund_avg, "rps": actual_rps, "n_pts": N_pts}


def analyze_phase_balance(rps_cmd, n_p, actual_rps):
    """Find rising zero-crossings per channel and compute pairwise phase angles.

    Returns None if capture file missing.
    """
    csv_path = CAPTURES_DIR / f"backemf_{rps_cmd:03d}rps.csv"
    if not csv_path.exists():
        return None

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["time_s"]
    dt = t[1] - t[0]
    f_e = actual_rps * n_p
    T_e = 1.0 / f_e
    min_sep = 0.5 * T_e

    channel_names = ["CH1_V", "CH2_V", "CH4_V"]
    channel_labels = ["CH1 (red)", "CH2 (green)", "CH4 (blue)"]
    crossings = {}

    for name, label in zip(channel_names, channel_labels):
        sig = data[name]
        times = []
        last_t = -1e9
        for i in range(len(sig) - 1):
            if sig[i] <= 0 < sig[i + 1]:
                # Sub-sample interpolation
                frac = -sig[i] / (sig[i + 1] - sig[i])
                t_cross = t[i] + frac * dt
                if t_cross - last_t >= min_sep:
                    times.append(t_cross)
                    last_t = t_cross
        crossings[label] = np.array(times)

    # Compute pairwise phase angles using matching crossings
    # Find nearest crossing (positive or negative time) and wrap to [0, 360)
    labels = channel_labels
    pairs = [(0, 1), (1, 2), (2, 0)]
    pair_names = [f"{labels[a]}->{labels[b]}" for a, b in pairs]
    angles = {}

    for (a, b), name in zip(pairs, pair_names):
        t_a = crossings[labels[a]]
        t_b = crossings[labels[b]]
        n_use = min(len(t_a), len(t_b), 50)
        if n_use == 0:
            continue
        deltas = []
        for i in range(n_use):
            diffs = t_b - t_a[i]
            # Find smallest absolute diff (nearest crossing)
            if len(diffs) == 0:
                continue
            idx = np.argmin(np.abs(diffs))
            dt = diffs[idx]
            # Convert to angle, wrap to [0, 360)
            angle = (dt / T_e * 360.0) % 360.0
            deltas.append(angle)
        if deltas:
            # Circular mean to handle wrap-around
            rads = np.radians(deltas)
            mean_angle = np.degrees(np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))) % 360.0
            angles[name] = mean_angle

    return {"crossings": {k: len(v) for k, v in crossings.items()},
            "angles": angles, "rps": actual_rps}


def plot_phase_phasors(balance_results, out_dir):
    """Polar phasor diagram: measured vs ideal 120 degree positions."""
    if not balance_results:
        return False

    angles = balance_results["angles"]
    if len(angles) < 3:
        return False

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    # Ideal phasors at 0, 120, 240 degrees
    ideal = [0, 120, 240]
    ideal_rad = [np.radians(a) for a in ideal]
    ax.plot(ideal_rad, [1, 1, 1], 'o', color=CLR_NEUTRAL, markersize=10, alpha=0.5,
            label="Ideal 120 deg")
    for a in ideal_rad:
        ax.annotate("", xy=(a, 1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=CLR_NEUTRAL, alpha=0.3, lw=1.5))

    # Measured phasors (cumulative angles from CH1)
    pair_keys = list(angles.keys())
    meas_angles = [0.0]  # CH1 at 0
    cumulative = 0.0
    for key in pair_keys[:2]:
        cumulative += angles[key]
        meas_angles.append(cumulative)

    colors = [CLR_CH1, CLR_CH2, CLR_CH4]
    labels = ["CH1 (red)", "CH2 (green)", "CH4 (blue)"]
    for a, c, lbl in zip(meas_angles, colors, labels):
        a_rad = np.radians(a)
        ax.plot(a_rad, 1, 'D', color=c, markersize=10, zorder=5,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate("", xy=(a_rad, 1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=c, lw=2))

    # Annotate pairwise angles
    for key, val in angles.items():
        ax.annotate(f"{val:.1f} deg", xy=(0.05, 0.95 - 0.05 * list(angles.keys()).index(key)),
                     xycoords="axes fraction", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=CLR_NEUTRAL, alpha=0.8))

    ax.set_ylim(0, 1.3)
    ax.set_title(f"Phase Balance at {balance_results['rps']:.0f} rps", pad=20)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "phase_phasors.png", dpi=150)
    plt.close(fig)
    return True


def run_calibrated_drive_sim(motor, psi_f_calibrated):
    """Run drive sim with calibrated psi_f vs uncalibrated. Gracefully handles crash."""
    from dataclasses import replace

    from phasesweep.sim import build_sim, extract_metrics, plan_sim

    results = {}

    # Uncalibrated
    try:
        params_uncal = prepare_drive_sim(motor)
        plan_uncal = plan_sim(params_uncal)
        sim_uncal = build_sim(params_uncal, plan=plan_uncal)
        res_uncal = sim_uncal.simulate(t_stop=plan_uncal.t_stop)
        metrics_uncal = extract_metrics(
            res_uncal, plan=plan_uncal, w_ref=params_uncal.drive.W_REF)
        results["uncalibrated"] = {"psi_f": params_uncal.psi_f, "metrics": metrics_uncal}
    except Exception as e:
        results["uncalibrated"] = {"error": str(e)}

    # Calibrated
    try:
        motor_cal = replace(motor, psi_f=psi_f_calibrated)
        params_cal = prepare_drive_sim(motor_cal)
        plan_cal = plan_sim(params_cal)
        sim_cal = build_sim(params_cal, plan=plan_cal)
        res_cal = sim_cal.simulate(t_stop=plan_cal.t_stop)
        metrics_cal = extract_metrics(
            res_cal, plan=plan_cal, w_ref=params_cal.drive.W_REF)
        results["calibrated"] = {"psi_f": params_cal.psi_f, "metrics": metrics_cal}
    except Exception as e:
        results["calibrated"] = {"error": str(e)}

    return results


def load_capture_aligned(rps_cmd: int, n_p: int, actual_rps: float):
    """Load a scope CSV and extract one cycle per channel, each aligned to its own zero-crossing.

    Returns list of (theta_deg, voltage_V, label, color) tuples, or None if file missing.
    """
    csv_path = CAPTURES_DIR / f"backemf_{rps_cmd:03d}rps.csv"
    if not csv_path.exists():
        return None

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["time_s"]
    dt = t[1] - t[0]
    f_e = actual_rps * n_p

    channels = [
        (data["CH1_V"], "CH1 (red)", CLR_CH1),
        (data["CH2_V"], "CH2 (green)", CLR_CH2),
        (data["CH4_V"], "CH4 (blue)", CLR_CH4),
    ]

    results = []
    for sig, label, color in channels:
        theta, cycle = _extract_cycle(sig, dt, f_e)
        if theta is not None:
            results.append((theta, cycle, label, color))

    return results if results else None


def plot_measured_overlay(ana, fem, motor, out_dir, fem_kend=None):
    """Back-EMF waveform at 80 rps: model predictions + 3 measured channels.

    Each measured channel is aligned to its own rising zero-crossing so all three
    collapse onto the same waveform shape (they are 120 electrical degrees apart
    in the raw capture).
    """
    channels = load_capture_aligned(80, motor.n_p, actual_rps=79.86)
    if channels is None:
        print("  [skip] 80 rps capture not found — run after CSV publish")
        return False

    w_e_80 = 79.86 * 2 * pi * motor.n_p
    k_end = fem_kend["k_end"] if fem_kend else 1.0
    fem_label = "FEM+k_end" if fem_kend else "FEM"

    # Terminal voltage = fundamental only (winding filters slot harmonics)
    v_ana_pk = ana["psi_f"] * w_e_80 * 1e3 if ana["psi_f"] else 0
    v_fem_pk = fem["psi_f"] * k_end * w_e_80 * 1e3

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Model fundamental sinusoids (what appears at terminals)
    theta_plot = np.linspace(0, 360, 720)
    theta_rad = np.radians(theta_plot)

    ax.plot(theta_plot, v_ana_pk * np.sin(theta_rad),
            label=f'Analytical ({v_ana_pk:.0f} mV pk)',
            linewidth=1.5, color=CLR_ANALYTICAL)
    ax.plot(theta_plot, v_fem_pk * np.sin(theta_rad),
            label=f'{fem_label} ({v_fem_pk:.0f} mV pk)',
            linewidth=1.5, color=CLR_FEM_KEND if fem_kend else CLR_FEM, linestyle="--")

    # Measured channels — each aligned to its own zero-crossing
    for theta, voltage, label, color in channels:
        v_pk = float(np.max(np.abs(voltage))) * 1e3
        ax.plot(theta, voltage * 1e3, linewidth=0.8, color=color, alpha=0.6,
                label=f'{label} ({v_pk:.0f} mV pk)')

    ax.set_xlabel("Electrical angle (degrees)")
    ax.set_ylabel("Phase back-EMF (mV)")
    ax.set_title("Back-EMF Waveform at 80 rps — Model vs Measured (3 channels aligned)")
    ax.set_xlim(0, 360)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "measured_overlay_80rps.png", dpi=150)
    plt.close(fig)
    return True


def plot_vpeak_vs_speed(motor, out_dir, carter_geo=None, fem=None, fem_kend=None):
    """V_peak vs mechanical speed: model prediction line + measured data points."""
    if not SWEEP_JSON.exists():
        print("  [skip] speed sweep JSON not found")
        return False

    with open(SWEEP_JSON) as f:
        sweep = json.load(f)

    points = sweep["speed_sweep"]
    omegas = np.array([pt["actual_rps"] * 2 * pi for pt in points])
    v_avg = np.array([np.mean(pt["V_pk_mV"]) for pt in points])  # mV
    v_ch1 = np.array([pt["V_pk_mV"][0] for pt in points])
    v_ch2 = np.array([pt["V_pk_mV"][1] for pt in points])
    v_ch4 = np.array([pt["V_pk_mV"][2] for pt in points])

    # Model prediction lines
    psi_f = _psi_f_carter(motor, *carter_geo) if carter_geo else _derive_psi_f(motor)
    psi_f_smooth = _derive_psi_f(motor)
    Ke_model = psi_f * motor.n_p  # V/(rad/s)
    Ke_smooth = psi_f_smooth * motor.n_p

    # Measured linear fit (forced through origin)
    Ke_meas = np.sum(omegas * v_avg * 1e-3) / np.sum(omegas**2)  # V/(rad/s)

    omega_line = np.linspace(0, max(omegas) * 1.05, 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-channel points
    ax.scatter(omegas, v_ch1, marker='o', s=20, color=CLR_CH1, alpha=0.4, zorder=3)
    ax.scatter(omegas, v_ch2, marker='o', s=20, color=CLR_CH2, alpha=0.4, zorder=3)
    ax.scatter(omegas, v_ch4, marker='o', s=20, color=CLR_CH4, alpha=0.4, zorder=3)

    # 3-channel average
    ax.scatter(omegas, v_avg, marker='D', s=40, color=CLR_MEASURED, edgecolors="black",
               linewidths=0.5, zorder=4, label=f"Measured avg (Ke={Ke_meas*1e3:.3f} mV/(rad/s))")

    # Measured fit line
    ax.plot(omega_line, Ke_meas * omega_line * 1e3,
            color=CLR_MEASURED, linewidth=1.5, linestyle="--",
            label="Measured fit (R²=0.99997)")

    # Smooth-bore analytical
    ax.plot(omega_line, Ke_smooth * omega_line * 1e3,
            color=CLR_SMOOTH, linewidth=1.2, linestyle="--", alpha=0.8,
            label=f"Smooth-bore (Ke={Ke_smooth*1e3:.3f}, +{(Ke_smooth/Ke_meas - 1)*100:.1f}%)")

    # Carter-corrected analytical
    ax.plot(omega_line, Ke_model * omega_line * 1e3,
            color=CLR_ANALYTICAL, linewidth=1.5,
            label=f"Carter-corr. (Ke={Ke_model*1e3:.3f}, +{(Ke_model/Ke_meas - 1)*100:.1f}%)")

    # FEM
    if fem:
        Ke_fem = fem["psi_f"] * motor.n_p
        ax.plot(omega_line, Ke_fem * omega_line * 1e3,
                color=CLR_FEM, linewidth=1.5, linestyle="-.",
                label=f"FEM (Ke={Ke_fem*1e3:.3f}, +{(Ke_fem/Ke_meas - 1)*100:.1f}%)")

    # FEM + end-effect correction
    if fem_kend:
        Ke_kend = fem_kend["psi_f"] * motor.n_p
        ax.plot(omega_line, Ke_kend * omega_line * 1e3,
                color=CLR_FEM_KEND, linewidth=1.5, linestyle=":",
                label=f"FEM+k_end (Ke={Ke_kend*1e3:.3f}, {(Ke_kend/Ke_meas - 1)*100:+.1f}%)")

    # Shade the gap between FEM (or Carter) and measured
    Ke_upper = fem["psi_f"] * motor.n_p if fem else Ke_model
    ax.fill_between(omega_line,
                     Ke_meas * omega_line * 1e3,
                     Ke_upper * omega_line * 1e3,
                     alpha=0.08, color=CLR_NEUTRAL, label="Model range")

    ax.set_xlabel("Mechanical speed ω (rad/s)")
    ax.set_ylabel("Phase back-EMF V$_{peak}$ (mV)")
    ax.set_title("Back-EMF Amplitude vs Speed — Model vs Measured")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, omega_line[-1])
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(out_dir / "vpeak_vs_speed.png", dpi=150)
    plt.close(fig)
    return True


def plot_br_waveforms(ana, fem, motor, out_dir, fem_kend=None):
    B_g1_meas = _measured_equiv_B_g1(motor)
    k_end = fem_kend["k_end"] if fem_kend else 1.0
    B_g1_fem_eff = fem["B_g1"] * k_end
    backemf_fem_eff = fem["backemf"] * k_end
    fem_label = "FEM+k_end" if fem_kend else "FEM"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── Top panel: air-gap B_r ──
    ax = axes[0]
    theta_deg_ana = np.degrees(ana["theta"])
    theta_deg_fem = np.degrees(fem["theta"])
    ax.plot(theta_deg_ana, ana["B_r"],
            label=f'Analytical (B$_1$={ana["B_g1"]:.4f} T)', linewidth=1.5, color=CLR_ANALYTICAL)
    ax.plot(theta_deg_fem, fem["B_r"] * k_end,
            label=f'{fem_label} (B$_1$={B_g1_fem_eff:.4f} T)', linewidth=1.5,
            color=CLR_FEM_KEND if fem_kend else CLR_FEM, linestyle="--")
    # Measured-equivalent fundamental as cosine envelope
    theta_rad = ana["theta"]
    B_meas_cos = B_g1_meas * np.cos(motor.n_p * theta_rad)
    ax.plot(theta_deg_ana, B_meas_cos,
            label=f'Measured equiv. (B$_1$={B_g1_meas:.4f} T)', linewidth=2,
            color=CLR_MEASURED, linestyle="-.", alpha=0.85)
    ax.set_ylabel("B$_r$ (T)")
    ax.set_title("Air-Gap Radial Flux Density")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Bottom panel: back-EMF voltage ──
    ax2 = axes[1]
    v_scale = W_ELEC * winding_transfer(motor)

    ax2.plot(theta_deg_ana, ana["B_r"] * v_scale,
             label=f'Analytical ({ana["backemf"]:.4f} V pk)', linewidth=1.5, color=CLR_ANALYTICAL)
    ax2.plot(theta_deg_fem, fem["B_r"] * k_end * v_scale,
             label=f'{fem_label} ({backemf_fem_eff:.4f} V pk)', linewidth=1.5,
             color=CLR_FEM_KEND if fem_kend else CLR_FEM, linestyle="--")
    # Measured back-EMF fundamental
    v_meas_cos = V_LN_PEAK_MEAS * np.cos(motor.n_p * theta_rad)
    ax2.plot(theta_deg_ana, v_meas_cos,
             label=f'Measured ({V_LN_PEAK_MEAS:.4f} V pk)', linewidth=2,
             color=CLR_MEASURED, linestyle="-.", alpha=0.85)
    ax2.axhline(V_LN_PEAK_MEAS, color=CLR_MEASURED, linewidth=0.8, alpha=0.4)
    ax2.axhline(-V_LN_PEAK_MEAS, color=CLR_MEASURED, linewidth=0.8, alpha=0.4)
    ax2.set_xlabel("Electrical angle (degrees)")
    ax2.set_ylabel("Phase back-EMF (V)")
    ax2.set_title("Back-EMF Voltage at 4820 RPM (phase-to-neutral)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)

    fig.tight_layout()
    fig.savefig(out_dir / "br_waveforms.png", dpi=150)
    plt.close(fig)


def plot_harmonics(ana, fem, n_p, out_dir, measured_harmonics=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    max_order = min(60, len(ana["amps"]) - 1, len(fem["amps"]) - 1)
    orders = np.arange(1, max_order + 1)
    ana_fund = ana["amps"][n_p] or 1
    fem_fund = fem["amps"][n_p] or 1
    ana_pct = ana["amps"][1:max_order+1] / ana_fund * 100
    fem_pct = fem["amps"][1:max_order+1] / fem_fund * 100
    width = 0.35
    ax.bar(orders - width/2, ana_pct, width, label="Analytical", alpha=0.7, color=CLR_ANALYTICAL)
    ax.bar(orders + width/2, fem_pct, width, label="FEM", alpha=0.7, color=CLR_FEM)

    # Measured harmonic data — use extracted values if available, else fallback
    if measured_harmonics is not None:
        meas_orders = {}
        for k, pct in measured_harmonics["harmonics_pct"].items():
            k = int(k)
            if k > 1:
                spatial_order = k * n_p
                meas_orders[spatial_order] = pct
        thd_label = f"Measured THD = {measured_harmonics['thd_pct']:.2f}%"
    else:
        meas_orders = {3 * n_p: 0.47, 5 * n_p: 0.94}
        thd_label = f"Measured THD < {THD_MEAS_BOUND}%"

    for order, pct in meas_orders.items():
        if order <= max_order:
            ax.plot(order, pct, 'D', color=CLR_MEASURED, markersize=8, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.5)
    ax.plot([], [], 'D', color=CLR_MEASURED, markersize=8, markeredgecolor="black",
            markeredgewidth=0.5, label="Measured (voltage harmonics)")

    ax.annotate(thd_label,
                xy=(max_order * 0.65, 6), fontsize=9, color=CLR_MEASURED,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CLR_MEASURED, alpha=0.8))

    ax.set_xlabel("Spatial Harmonic Order")
    ax.set_ylabel("Amplitude (% of fundamental)")
    ax.set_title("Harmonic Spectrum — B$_r$ (model) vs Voltage (measured)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, max_order + 1)
    ax.set_ylim(0, max(15, ana_pct.max() * 1.1, fem_pct.max() * 1.1))
    ax.axvline(n_p, color=CLR_NEUTRAL, linestyle=":", alpha=0.5)
    ax.text(n_p + 0.5, ax.get_ylim()[1] * 0.92, f"n$_p$={n_p}", fontsize=8, color=CLR_NEUTRAL)
    fig.tight_layout()
    fig.savefig(out_dir / "harmonics.png", dpi=150)
    plt.close(fig)


def plot_sensitivity(
    motor: Motor,
    out_dir: Path,
    carter_geo: tuple[float, float] | None = None,
    fem: FieldResult | None = None,
    fem_alpha_p: list[tuple[float, float]] | None = None,
    fem_kend: EndEffectResult | None = None,
) -> tuple[float | None, float | None, float | None]:
    cg = carter_geo
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # B_rem sweep — range must span measured psi_f in both directions
    b_vals = np.linspace(0.90, 1.60, 50)
    psi_b = sweep_parameter(motor, "B_rem", b_vals, carter_geo=cg) * 1e3
    psi_b_smooth = sweep_parameter(motor, "B_rem", b_vals) * 1e3
    axes[0].plot(b_vals, psi_b_smooth, color=CLR_SMOOTH, linewidth=1.2, linestyle="--",
                 label="Smooth-bore", alpha=0.8)
    axes[0].plot(b_vals, psi_b, color=CLR_ANALYTICAL, linewidth=2, label="Carter-corrected")
    if fem:
        psi_b_fem = np.array([_fem_psi_f_scaled(fem, motor, B_rem=b) for b in b_vals]) * 1e3
        axes[0].plot(b_vals, psi_b_fem, color=CLR_FEM, linewidth=1.5, linestyle="-.",
                     label="FEM (linear)", alpha=0.9)
    if fem_kend:
        psi_b_kend = np.array([_fem_psi_f_scaled(fem, motor, B_rem=b) * fem_kend["k_end"]
                                for b in b_vals]) * 1e3
        axes[0].plot(b_vals, psi_b_kend, color=CLR_FEM_KEND, linewidth=1.5, linestyle=":",
                     label="FEM+k_end", alpha=0.9)
    axes[0].axhline(PSI_F_MEAS * 1e3, color=CLR_MEASURED, linestyle="--", linewidth=1,
                     label=f"Measured ({PSI_F_MEAS*1e3:.3f} mWb)")
    b_match = find_matching_value(motor, "B_rem", b_vals, PSI_F_MEAS, carter_geo=cg)
    if b_match:
        axes[0].axvline(b_match, color=CLR_NEUTRAL, linestyle=":", alpha=0.7)
        axes[0].annotate(f"B_rem = {b_match:.3f} T", xy=(b_match, PSI_F_MEAS*1e3),
                        xytext=(b_match - 0.1, PSI_F_MEAS*1e3 + 0.03),
                        arrowprops=dict(arrowstyle="->", color=CLR_NEUTRAL), fontsize=9)
    axes[0].set_xlabel("B_rem (T)")
    axes[0].set_ylabel("ψ_f (mWb)")
    axes[0].set_title("B_rem Sensitivity")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # alpha_p sweep
    a_vals = np.linspace(0.35, 0.75, 50)
    psi_a = sweep_parameter(motor, "alpha_p", a_vals, carter_geo=cg) * 1e3
    psi_a_smooth = sweep_parameter(motor, "alpha_p", a_vals) * 1e3
    axes[1].plot(a_vals, psi_a_smooth, color=CLR_SMOOTH, linewidth=1.2, linestyle="--",
                 label="Smooth-bore", alpha=0.8)
    axes[1].plot(a_vals, psi_a, color=CLR_ANALYTICAL, linewidth=2, label="Carter-corrected")
    if fem_alpha_p:
        ap_pts, psi_pts = zip(*fem_alpha_p)
        axes[1].plot(ap_pts, np.array(psi_pts) * 1e3, 'o-', color=CLR_FEM, linewidth=1.5,
                     markersize=5, label="FEM (linear)", alpha=0.9)
        if fem_kend:
            axes[1].plot(ap_pts, np.array(psi_pts) * fem_kend["k_end"] * 1e3,
                         's:', color=CLR_FEM_KEND, linewidth=1.5,
                         markersize=4, label="FEM+k_end", alpha=0.9)
    axes[1].axhline(PSI_F_MEAS * 1e3, color=CLR_MEASURED, linestyle="--", linewidth=1,
                     label=f"Measured ({PSI_F_MEAS*1e3:.3f} mWb)")
    a_match = find_matching_value(motor, "alpha_p", a_vals, PSI_F_MEAS, carter_geo=cg)
    if a_match:
        axes[1].axvline(a_match, color=CLR_NEUTRAL, linestyle=":", alpha=0.7)
        axes[1].annotate(f"α_p = {a_match:.3f}", xy=(a_match, PSI_F_MEAS*1e3),
                        xytext=(a_match + 0.03, PSI_F_MEAS*1e3 + 0.03),
                        arrowprops=dict(arrowstyle="->", color=CLR_NEUTRAL), fontsize=9)
    axes[1].set_xlabel("α_p (pole-arc ratio)")
    axes[1].set_ylabel("ψ_f (mWb)")
    axes[1].set_title("α_p Sensitivity")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # k_w sweep
    k_vals = np.linspace(0.60, 1.0, 50)
    psi_k = sweep_parameter(motor, "k_w", k_vals, carter_geo=cg) * 1e3
    psi_k_smooth = sweep_parameter(motor, "k_w", k_vals) * 1e3
    axes[2].plot(k_vals, psi_k_smooth, color=CLR_SMOOTH, linewidth=1.2, linestyle="--",
                 label="Smooth-bore", alpha=0.8)
    axes[2].plot(k_vals, psi_k, color=CLR_ANALYTICAL, linewidth=2, label="Carter-corrected")
    if fem:
        psi_k_fem = np.array([_fem_psi_f_scaled(fem, motor, k_w=k) for k in k_vals]) * 1e3
        axes[2].plot(k_vals, psi_k_fem, color=CLR_FEM, linewidth=1.5, linestyle="-.",
                     label="FEM (linear)", alpha=0.9)
    if fem_kend:
        psi_k_kend = np.array([_fem_psi_f_scaled(fem, motor, k_w=k) * fem_kend["k_end"]
                                for k in k_vals]) * 1e3
        axes[2].plot(k_vals, psi_k_kend, color=CLR_FEM_KEND, linewidth=1.5, linestyle=":",
                     label="FEM+k_end", alpha=0.9)
    axes[2].axhline(PSI_F_MEAS * 1e3, color=CLR_MEASURED, linestyle="--", linewidth=1,
                     label=f"Measured ({PSI_F_MEAS*1e3:.3f} mWb)")
    k_match = find_matching_value(motor, "k_w", k_vals, PSI_F_MEAS, carter_geo=cg)
    if k_match:
        axes[2].axvline(k_match, color=CLR_NEUTRAL, linestyle=":", alpha=0.7)
        axes[2].annotate(f"k_w = {k_match:.3f}", xy=(k_match, PSI_F_MEAS*1e3),
                        xytext=(k_match + 0.03, PSI_F_MEAS*1e3 + 0.03),
                        arrowprops=dict(arrowstyle="->", color=CLR_NEUTRAL), fontsize=9)
    axes[2].set_xlabel("k_w (winding factor)")
    axes[2].set_ylabel("ψ_f (mWb)")
    axes[2].set_title("k_w Sensitivity")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Sensitivity: Which Parameter Matches Measured ψ_f?", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return b_match, a_match, k_match


def plot_brem_kw_contour(
    motor: Motor,
    out_dir: Path,
    carter_geo: tuple[float, float] | None = None,
    fem: FieldResult | None = None,
    fem_kend: EndEffectResult | None = None,
) -> bool:
    """2D contour: (B_rem, k_w) combinations that produce measured psi_f."""
    from dataclasses import replace

    b_vals = np.linspace(0.95, 1.60, 60)
    k_vals = np.linspace(0.65, 1.00, 60)
    B, K = np.meshgrid(b_vals, k_vals)
    PSI = np.zeros_like(B)

    for i in range(len(k_vals)):
        for j in range(len(b_vals)):
            m = replace(motor, B_rem=b_vals[j], k_w=k_vals[i])
            psi = _psi_f_carter(m, *carter_geo) if carter_geo else _derive_psi_f(m)
            PSI[i, j] = psi * 1e3 if psi else 0

    # Smooth-bore grid
    PSI_SMOOTH = np.zeros_like(B)
    for i in range(len(k_vals)):
        for j in range(len(b_vals)):
            m = replace(motor, B_rem=b_vals[j], k_w=k_vals[i])
            psi = _derive_psi_f(m)
            PSI_SMOOTH[i, j] = psi * 1e3 if psi else 0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Filled contours — sequential blue-to-orange (Tol sunset-inspired)
    from matplotlib.colors import LinearSegmentedColormap
    tol_cmap = LinearSegmentedColormap.from_list(
        "tol_seq", ["#332288", TOL_BLUE, TOL_TEAL, "#DDCC77", TOL_ORANGE])

    levels = np.linspace(PSI.min(), PSI.max(), 20)
    cf = ax.contourf(B, K, PSI, levels=levels, cmap=tol_cmap, alpha=0.7)
    plt.colorbar(cf, ax=ax, label="ψ_f (mWb)")

    # Smooth-bore target contour
    cs_sm = ax.contour(B, K, PSI_SMOOTH, levels=[PSI_F_MEAS * 1e3],
                        colors=[CLR_SMOOTH], linewidths=1.5, linestyles="--")
    ax.clabel(cs_sm, fmt="smooth-bore", fontsize=7)

    # Carter-corrected target contour (measured psi_f)
    cs = ax.contour(B, K, PSI, levels=[PSI_F_MEAS * 1e3],
                     colors=[TOL_RED], linewidths=2.5)
    ax.clabel(cs, fmt=f"{PSI_F_MEAS*1e3:.3f} mWb", fontsize=9)

    # FEM target contour — psi_f_fem ∝ B_rem × k_w (scaled from nominal)
    if fem:
        PSI_FEM = np.zeros_like(B)
        for i in range(len(k_vals)):
            for j in range(len(b_vals)):
                PSI_FEM[i, j] = _fem_psi_f_scaled(fem, motor, B_rem=b_vals[j], k_w=k_vals[i]) * 1e3
        cs_fem = ax.contour(B, K, PSI_FEM, levels=[PSI_F_MEAS * 1e3],
                             colors=[CLR_FEM], linewidths=2, linestyles="-.")
        ax.clabel(cs_fem, fmt="FEM", fontsize=7)

    # FEM+k_end target contour
    if fem_kend and fem:
        PSI_KEND = PSI_FEM * fem_kend["k_end"]
        cs_kend = ax.contour(B, K, PSI_KEND, levels=[PSI_F_MEAS * 1e3],
                              colors=[CLR_FEM_KEND], linewidths=2, linestyles=":")
        ax.clabel(cs_kend, fmt="FEM+k_end", fontsize=7)

    # Nominal point
    ax.plot(motor.B_rem, motor.k_w, 'D', color="white", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=5)
    ax.annotate(f"Motor definition\n({motor.B_rem}, {motor.k_w})",
                xy=(motor.B_rem, motor.k_w),
                xytext=(motor.B_rem - 0.15, motor.k_w - 0.04),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9, ha="center")

    # NdFeB grade markers — find intersection with target contour
    grade_marks = [(1.28, "N45"), (1.37, "N50"), (1.45, "N52")]
    target = PSI_F_MEAS * 1e3
    for gi, (b_grade, g_label) in enumerate(grade_marks):
        ax.axvline(b_grade, color=TOL_GREY, linestyle=":", alpha=0.5, linewidth=0.8)

        # Interpolate k_w at this B_rem on the target contour
        j_idx = np.searchsorted(b_vals, b_grade)
        if j_idx < 1 or j_idx >= len(b_vals):
            continue
        # Linear interpolation between adjacent B_rem columns
        frac = (b_grade - b_vals[j_idx - 1]) / (b_vals[j_idx] - b_vals[j_idx - 1])
        psi_col = PSI[:, j_idx - 1] * (1 - frac) + PSI[:, j_idx] * frac
        # Find where this column crosses the target
        for ii in range(len(psi_col) - 1):
            if (psi_col[ii] - target) * (psi_col[ii + 1] - target) <= 0:
                f2 = (target - psi_col[ii]) / (psi_col[ii + 1] - psi_col[ii])
                kw_hit = k_vals[ii] + f2 * (k_vals[ii + 1] - k_vals[ii])
                # Marker on the contour
                ax.plot(b_grade, kw_hit, 's', color="white", markersize=7,
                        markeredgecolor=TOL_RED, markeredgewidth=1.5, zorder=6)
                # Label: alternate above/below to avoid overlap
                y_off = -22 if gi % 2 == 0 else 14
                ax.annotate(f"{g_label}\nk_w={kw_hit:.2f}",
                            xy=(b_grade, kw_hit),
                            xytext=(0, y_off), textcoords="offset points",
                            fontsize=8, ha="center", color=TOL_RED, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec=TOL_RED, alpha=0.85))
                break

    ax.set_xlabel("B_rem (T)")
    ax.set_ylabel("k_w (winding factor)")
    ax.set_title("Parameter Locus Matching Measured ψ_f — B_rem vs k_w")
    ax.set_xlim(b_vals[0], b_vals[-1])
    ax.set_ylim(k_vals[0], k_vals[-1])
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "brem_kw_contour.png", dpi=150)
    plt.close(fig)
    return True


def plot_drive_sim_transient(motor, psi_f_calibrated, out_dir):
    """3-panel transient plot: speed, torque, current vs time for calibrated drive sim."""
    from dataclasses import replace

    from phasesweep.sim import build_sim, plan_sim

    motor_cal = replace(motor, psi_f=psi_f_calibrated)
    params = prepare_drive_sim(motor_cal)
    plan = plan_sim(params)

    try:
        sim = build_sim(params, plan=plan)
        res = sim.simulate(t_stop=plan.t_stop)
    except Exception as e:
        print(f"  [skip] drive sim transient: {e}")
        return False

    t = res.mdl.t * 1e3  # ms
    w_M = res.mdl.mechanics.w_M
    tau_M = res.mdl.machine.tau_M * 1e3  # mNm
    i_s = np.abs(res.mdl.machine.i_s_ab)

    w_ref = params.drive.W_REF

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    # Speed
    ax = axes[0]
    ax.plot(t, w_M / w_ref * 100, color=TOL_BLUE, linewidth=1.5)
    ax.axhline(100, color=TOL_GREY, linewidth=0.8, linestyle="--")
    ax.axvline(plan.load_time * 1e3, color=TOL_ORANGE, linewidth=0.8,
               linestyle="--", label="Load step")
    ax.set_ylabel("Speed (% of ref)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    # Torque
    ax = axes[1]
    ax.plot(t, tau_M, color=TOL_ORANGE, linewidth=1.5)
    ax.axhline(plan.load_torque * 1e3, color=TOL_GREY, linewidth=0.8,
               linestyle="--", label=f"Load = {plan.load_torque*1e3:.1f} mNm")
    ax.set_ylabel("Torque (mNm)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Current
    ax = axes[2]
    ax.plot(t, i_s, color=TOL_RED, linewidth=1.5)
    ax.axhline(params.drive.MAX_I_S, color=TOL_GREY, linewidth=0.8,
               linestyle="--", label=f"I_limit = {params.drive.MAX_I_S:.2f} A")
    ax.set_ylabel("Phase current |i_s| (A)")
    ax.set_xlabel("Time (ms)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Drive Sim Transient — Calibrated ψ_f = {psi_f_calibrated*1e3:.3f} mWb\n"
        f"Speed ref = {w_ref:.0f} rad/s ({w_ref/(2*np.pi)*60:.0f} RPM), "
        f"J = {params.J:.2e} kg·m²",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "drive_sim_transient.png", dpi=150)
    plt.close(fig)
    return True


def plot_grade_error_budget(motor, out_dir, carter_geo=None):
    """Stacked horizontal bar chart: error budget by assumed magnet grade (N48, N50, N52)."""
    from dataclasses import replace

    def _psi(m):
        return _psi_f_carter(m, *carter_geo) if carter_geo else _derive_psi_f(m)

    psi_model = _psi(motor) * 1e3
    psi_meas = PSI_F_MEAS * 1e3
    total_gap = psi_model - psi_meas

    if total_gap <= 0:
        # Model under-predicts — decomposition not applicable
        return False

    grades = [("N48", 1.32), ("N50", 1.37), ("N52", 1.45)]
    triplets = compute_parameter_triplets(motor, PSI_F_MEAS, carter_geo=carter_geo)

    labels = []
    contribs_brem = []
    contribs_ap = []
    contribs_kw = []
    contribs_leak = []

    for grade, b_rem in grades:
        row = next((r for r in triplets if r["grade"] == grade), None)
        if not row or not row["k_w"] or not row["alpha_p"]:
            continue

        m1 = replace(motor, B_rem=b_rem)
        psi_after_brem = _psi(m1) * 1e3
        c_brem = psi_model - psi_after_brem

        # If B_rem alone overshoots measured, clamp to total gap
        if c_brem >= total_gap:
            labels.append(f"{grade}\n({b_rem} T)")
            contribs_brem.append(100)
            contribs_ap.append(0)
            contribs_kw.append(0)
            contribs_leak.append(0)
            continue

        psi_ap_only = _psi(replace(motor, alpha_p=row["alpha_p"])) * 1e3
        psi_kw_only = _psi(replace(motor, k_w=row["k_w"])) * 1e3
        delta_ap = psi_model - psi_ap_only
        delta_kw = psi_model - psi_kw_only

        remaining = max(psi_after_brem - psi_meas, 0)
        if delta_ap + delta_kw > 0:
            frac_ap = delta_ap / (delta_ap + delta_kw)
            frac_kw = delta_kw / (delta_ap + delta_kw)
        else:
            frac_ap = frac_kw = 0.5

        c_ap = remaining * frac_ap * 0.8
        c_kw = remaining * frac_kw * 0.8
        c_leak = remaining * 0.2

        labels.append(f"{grade}\n({b_rem} T)")
        contribs_brem.append(c_brem / total_gap * 100)
        contribs_ap.append(c_ap / total_gap * 100)
        contribs_kw.append(c_kw / total_gap * 100)
        contribs_leak.append(c_leak / total_gap * 100)

    if not labels:
        return False

    fig, ax = plt.subplots(figsize=(9, 3.5))
    y = np.arange(len(labels))
    bh = 0.5

    left = np.zeros(len(labels))
    bars_brem = ax.barh(y, contribs_brem, bh, left=left, color=TOL_BLUE, label="B_rem (magnet grade)")
    left += contribs_brem
    bars_ap = ax.barh(y, contribs_ap, bh, left=left, color=TOL_ORANGE, label="α_p (effective arc)")
    left += contribs_ap
    bars_kw = ax.barh(y, contribs_kw, bh, left=left, color=TOL_CYAN, label="k_w (winding factor)")
    left += contribs_kw
    bars_leak = ax.barh(y, contribs_leak, bh, left=left, color=TOL_TEAL, label="3D leakage")

    # Labels on segments
    for bars in [bars_brem, bars_ap, bars_kw, bars_leak]:
        for bar in bars:
            w = bar.get_width()
            if w > 5:
                cx = bar.get_x() + w / 2
                cy = bar.get_y() + bar.get_height() / 2
                ax.text(cx, cy, f"{w:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Share of over-prediction gap (%)")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Error Budget by Assumed Magnet Grade\n"
        f"Total gap: {total_gap:.3f} mWb (analytical model vs measured)",
        fontsize=11)
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_dir / "grade_error_budget.png", dpi=150)
    plt.close(fig)
    return True


def plot_kw_alphap_heatmap(
    motor: Motor,
    out_dir: Path,
    carter_geo: tuple[float, float] | None = None,
    fem: FieldResult | None = None,
    fem_kend: EndEffectResult | None = None,
) -> bool:
    """2D heatmap of psi_f error (%) across (alpha_p, k_w) at fixed B_rem grades."""
    from dataclasses import replace

    def _psi(m):
        return _psi_f_carter(m, *carter_geo) if carter_geo else _derive_psi_f(m)

    grades = [("N45", 1.28), ("N48", 1.32), ("N50", 1.37)]

    kw_range = np.linspace(0.65, 0.96, 60)
    ap_range = np.linspace(0.40, 0.70, 60)
    KW, AP = np.meshgrid(kw_range, ap_range)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, (grade, b_rem) in zip(axes, grades):
        err = np.zeros_like(KW)
        for i in range(AP.shape[0]):
            for j in range(AP.shape[1]):
                m = replace(motor, B_rem=b_rem, k_w=KW[i, j], alpha_p=AP[i, j])
                psi = _psi(m)
                err[i, j] = (psi - PSI_F_MEAS) / PSI_F_MEAS * 100

        levels = np.linspace(-30, 30, 25)
        cf = ax.contourf(AP, KW, err, levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(AP, KW, err, levels=[0], colors="black", linewidths=2)

        # Motor definition point
        ax.plot(motor.alpha_p, motor.k_w, 'D', color="white", markersize=8,
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)

        # FEM nominal marker (scaled for this grade's B_rem)
        if fem:
            psi_fem = _fem_psi_f_scaled(fem, motor, B_rem=b_rem)
            fem_err = (psi_fem - PSI_F_MEAS) / PSI_F_MEAS * 100
            ax.plot(motor.alpha_p, motor.k_w, 'o', color=CLR_FEM, markersize=7,
                    markeredgecolor="white", markeredgewidth=1, zorder=6)
            ax.annotate(f"FEM {fem_err:+.0f}%",
                        xy=(motor.alpha_p, motor.k_w),
                        xytext=(8, -12), textcoords="offset points",
                        fontsize=7, color=CLR_FEM, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=CLR_FEM, alpha=0.8))

        # FEM+k_end marker
        if fem_kend:
            psi_kend = _fem_psi_f_scaled(fem, motor, B_rem=b_rem) * fem_kend["k_end"]
            kend_err = (psi_kend - PSI_F_MEAS) / PSI_F_MEAS * 100
            ax.annotate(f"FEM+k_end {kend_err:+.0f}%",
                        xy=(motor.alpha_p, motor.k_w),
                        xytext=(8, -24), textcoords="offset points",
                        fontsize=7, color=CLR_FEM_KEND, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=CLR_FEM_KEND, alpha=0.8))

        ax.set_xlabel("α_p")
        ax.set_title(f"{grade} (B_rem = {b_rem} T)", fontsize=10)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel("k_w")
    fig.colorbar(cf, ax=axes, label="ψ_f error vs measured (%)", shrink=0.85)
    fig.suptitle(
        f"ψ_f Error Across (α_p, k_w) at Each Magnet Grade\n"
        f"Black contour = 0% error (matches measured {PSI_F_MEAS*1e3:.3f} mWb)",
        fontsize=11)
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.06, right=0.92)
    fig.savefig(out_dir / "kw_alphap_heatmap.png", dpi=150)
    plt.close(fig)
    return True


def compute_parameter_triplets(motor, psi_f_target, carter_geo=None):
    """For each magnet grade, solve for k_w (at nominal alpha_p) and alpha_p (at nominal k_w)."""
    from dataclasses import replace

    from scipy.optimize import brentq

    def _psi(m):
        return _psi_f_carter(m, *carter_geo) if carter_geo else _derive_psi_f(m)

    grades = [("N45", 1.28), ("N48", 1.32), ("N50", 1.37), ("N52", 1.45)]
    rows = []
    for grade, b in grades:
        def f_kw(kw, _b=b):
            m = replace(motor, B_rem=_b, k_w=kw)
            return _psi(m) - psi_f_target
        try:
            kw_sol = brentq(f_kw, 0.5, 1.0)
        except ValueError:
            kw_sol = None

        def f_ap(ap, _b=b):
            m = replace(motor, B_rem=_b, alpha_p=ap)
            return _psi(m) - psi_f_target
        try:
            ap_sol = brentq(f_ap, 0.3, 1.0)
        except ValueError:
            ap_sol = None

        rows.append({"grade": grade, "B_rem": b, "k_w": kw_sol, "alpha_p": ap_sol})
    return rows


def plot_kw_alphap_tradeoff(
    motor: Motor,
    out_dir: Path,
    carter_geo: tuple[float, float] | None = None,
    fem: FieldResult | None = None,
    fem_kend: EndEffectResult | None = None,
) -> bool:
    """(k_w, alpha_p) trade-off curves at each magnet grade for measured psi_f."""
    from dataclasses import replace

    from scipy.optimize import brentq

    def _psi(m):
        return _psi_f_carter(m, *carter_geo) if carter_geo else _derive_psi_f(m)

    grades = [("N45", 1.28), ("N48", 1.32), ("N50", 1.37), ("N52", 1.45)]
    grade_colors = [TOL_TEAL, TOL_BLUE, TOL_ORANGE, TOL_RED]
    target = PSI_F_MEAS

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for (grade, b_rem), clr in zip(grades, grade_colors):
        ap_range = np.linspace(0.40, 0.70, 80)
        kw_vals = []
        ap_valid = []
        for ap in ap_range:
            def f_kw(kw, _b=b_rem, _ap=ap):
                m = replace(motor, B_rem=_b, k_w=kw, alpha_p=_ap)
                return _psi(m) - target
            try:
                kw = brentq(f_kw, 0.5, 0.999)
                kw_vals.append(kw)
                ap_valid.append(ap)
            except ValueError:
                pass

        if ap_valid:
            ax.plot(ap_valid, kw_vals, color=clr, linewidth=2.5,
                    label=f"{grade} (B_rem = {b_rem} T)")

    # FEM iso-psi_f curves — FEM psi_f scales linearly with B_rem and k_w
    # At nominal alpha_p, solve: fem_psi_f * (b_rem/B_nom) * (kw/kw_nom) = target
    # → kw = target * kw_nom / (fem_psi_f * b_rem/B_nom)
    if fem:
        for (grade, b_rem), clr in zip(grades, grade_colors):
            kw_fem = PSI_F_MEAS / _fem_psi_f_scaled(fem, motor, B_rem=b_rem)  * motor.k_w
            if 0.5 < kw_fem < 0.999:
                ax.axhline(kw_fem, color=clr, linewidth=1, linestyle=":", alpha=0.5)
        # FEM marker at nominal
        psi_fem_nom = fem["psi_f"]
        kw_fem_nom = PSI_F_MEAS / psi_fem_nom * motor.k_w
        if 0.5 < kw_fem_nom < 0.999:
            ax.plot(motor.alpha_p, kw_fem_nom, 'o', color=CLR_FEM, markersize=8,
                    markeredgecolor="white", markeredgewidth=1, zorder=6)
            ax.annotate(f"FEM (k_w={kw_fem_nom:.2f})",
                        xy=(motor.alpha_p, kw_fem_nom),
                        xytext=(motor.alpha_p + 0.03, kw_fem_nom + 0.01),
                        arrowprops=dict(arrowstyle="->", color=CLR_FEM),
                        fontsize=8, color=CLR_FEM, fontweight="bold")

    # FEM+k_end marker at nominal
    if fem_kend and fem:
        psi_kend_nom = fem["psi_f"] * fem_kend["k_end"]
        kw_kend_nom = PSI_F_MEAS / psi_kend_nom * motor.k_w
        if 0.5 < kw_kend_nom < 0.999:
            ax.plot(motor.alpha_p, kw_kend_nom, 's', color=CLR_FEM_KEND, markersize=8,
                    markeredgecolor="white", markeredgewidth=1, zorder=6)
            ax.annotate(f"FEM+k_end (k_w={kw_kend_nom:.2f})",
                        xy=(motor.alpha_p, kw_kend_nom),
                        xytext=(motor.alpha_p + 0.03, kw_kend_nom - 0.02),
                        arrowprops=dict(arrowstyle="->", color=CLR_FEM_KEND),
                        fontsize=8, color=CLR_FEM_KEND, fontweight="bold")

    # Motor definition point
    ax.plot(motor.alpha_p, motor.k_w, 'D', color="white", markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=5)
    ax.annotate(f"Motor definition\n({motor.alpha_p}, {motor.k_w}*)",
                xy=(motor.alpha_p, motor.k_w),
                xytext=(motor.alpha_p - 0.06, motor.k_w - 0.03),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9, ha="center")
    ax.text(0.02, 0.02, "* k_w is placeholder", transform=ax.transAxes,
            fontsize=7, color=TOL_GREY, style="italic")

    ax.set_xlabel("α_p (pole-arc ratio, from geometry)")
    ax.set_ylabel("k_w (winding factor)")
    ax.set_title(f"(α_p, k_w) Trade-off at Each Magnet Grade\nfor ψ_f = {target*1e3:.3f} mWb")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.45, 0.70)

    fig.tight_layout()
    fig.savefig(out_dir / "kw_alphap_tradeoff.png", dpi=150)
    plt.close(fig)
    return True


def plot_waterfall(motor, triplets, out_dir, carter_geo=None):
    """Waterfall chart: decompose model over-prediction into contributing factors."""
    from dataclasses import replace

    def _psi(m):
        return _psi_f_carter(m, *carter_geo) if carter_geo else _derive_psi_f(m)

    psi_model = _psi(motor) * 1e3  # mWb
    psi_meas = PSI_F_MEAS * 1e3
    gap = psi_model - psi_meas

    if gap <= 0:
        # Model under-predicts — waterfall decomposition not applicable
        return False

    # Use N50 as the most plausible grade for decomposition
    n50 = next((r for r in triplets if r["grade"] == "N50"), None)
    if not n50 or not n50["k_w"] or not n50["alpha_p"]:
        return False

    # Sequential decomposition: apply each factor cumulatively
    # Step 1: B_rem N52 → N50
    m1 = replace(motor, B_rem=n50["B_rem"])
    psi_after_brem = _psi(m1) * 1e3

    # Step 2: also adjust alpha_p
    # Split remaining gap proportionally between alpha_p and k_w
    # Use the ratio of single-factor sensitivities
    psi_ap_only = _psi(replace(motor, alpha_p=n50["alpha_p"])) * 1e3
    psi_kw_only = _psi(replace(motor, k_w=n50["k_w"])) * 1e3
    delta_ap = psi_model - psi_ap_only
    delta_kw = psi_model - psi_kw_only

    # Remaining gap after B_rem adjustment
    remaining = psi_after_brem - psi_meas

    # Partition remaining gap proportionally between alpha_p, k_w, and leakage
    if delta_ap + delta_kw > 0:
        frac_ap = delta_ap / (delta_ap + delta_kw)
        frac_kw = delta_kw / (delta_ap + delta_kw)
    else:
        frac_ap = frac_kw = 0.5

    # Allocate 80% of remaining to alpha_p + k_w, 20% to leakage (unmodeled)
    contrib_ap = remaining * frac_ap * 0.8
    contrib_kw = remaining * frac_kw * 0.8
    contrib_leakage = remaining * 0.2

    contrib_brem = psi_model - psi_after_brem

    # Build waterfall
    labels = [
        "Model\nprediction",
        f"B_rem\nN52 → N50\n({motor.B_rem} → {n50['B_rem']} T)",
        f"α_p\neffective arc\n({motor.alpha_p} → ~{n50['alpha_p']:.2f})",
        f"k_w\nwinding factor\n({motor.k_w} → ~{n50['k_w']:.2f})",
        "3D leakage\n(end-turn,\naxial fringing)",
        "Measured",
    ]

    values = [psi_model, -contrib_brem, -contrib_ap, -contrib_kw, -contrib_leakage, 0]
    # Running totals for bar positioning
    running = [psi_model]
    for v in values[1:-1]:
        running.append(running[-1] + v)
    running.append(psi_meas)

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = [TOL_BLUE, TOL_ORANGE, TOL_ORANGE, TOL_ORANGE, TOL_TEAL, CLR_MEASURED]
    x = np.arange(len(labels))
    bar_width = 0.6

    for i in range(len(labels)):
        if i == 0:
            # Start bar from zero
            ax.bar(x[i], running[i], bar_width, color=bar_colors[i], edgecolor="white", linewidth=0.5)
        elif i == len(labels) - 1:
            # End bar from zero
            ax.bar(x[i], running[i], bar_width, color=bar_colors[i], edgecolor="white", linewidth=0.5)
        else:
            # Floating bar: bottom at running[i], height = negative delta
            bottom = running[i]
            height = running[i - 1] - running[i]
            ax.bar(x[i], -height, bar_width, bottom=bottom, color=bar_colors[i],
                   edgecolor="white", linewidth=0.5)
            # Connector line
            ax.plot([x[i - 1] + bar_width / 2, x[i] - bar_width / 2],
                    [running[i - 1], running[i - 1]], color=TOL_GREY, linewidth=0.8,
                    linestyle="--")

    # Final connector
    ax.plot([x[-2] + bar_width / 2, x[-1] - bar_width / 2],
            [running[-2], running[-2]], color=TOL_GREY, linewidth=0.8, linestyle="--")

    # Value labels on bars
    for i in range(len(labels)):
        if i == 0 or i == len(labels) - 1:
            ax.text(x[i], running[i] + 0.002, f"{running[i]:.3f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
        else:
            delta = values[i]
            mid = running[i] + (running[i - 1] - running[i]) / 2
            ax.text(x[i], mid, f"{delta:+.3f}\n({delta / psi_model * 100:+.1f}%)",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ψ_f (mWb)")
    ax.set_title("Model Over-Prediction Decomposition (N50 hypothesis)")
    ax.set_ylim(psi_meas * 0.92, psi_model * 1.05)
    ax.grid(axis="y", alpha=0.2)

    # Measured line
    ax.axhline(psi_meas, color=CLR_MEASURED, linewidth=1, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "waterfall.png", dpi=150)
    plt.close(fig)
    return True


def pct_delta(predicted, measured):
    if measured == 0:
        return float("inf")
    return (predicted - measured) / measured * 100


def generate_report(
    ana: FieldResult,
    fem: FieldResult,
    b_match: float | None,
    a_match: float | None,
    k_match: float | None,
    motor: Motor,
    out_dir: Path,
    fem_nl: FieldResult | None = None,
    b_rem_result: EffectiveBremResult | None = None,
    measured_harmonics: dict[str, Any] | None = None,
    balance_result: dict[str, Any] | None = None,
    drive_sim_result: dict[str, Any] | None = None,
    meas_unc: dict[str, Any] | None = None,
    triplets: list[dict[str, Any]] | None = None,
    fem_kend: EndEffectResult | None = None,
    fem_nl_kend: EndEffectResult | None = None,
) -> Path:
    psi_f_ana = ana["psi_f"]
    psi_f_fem = fem["psi_f"]

    backemf_ana = ana["backemf"]
    backemf_fem = fem["backemf"]

    ke_ana = backemf_ana / W_MECH if backemf_ana else None
    ke_fem = backemf_fem / W_MECH

    kt_ana = 1.5 * N_P * psi_f_ana if psi_f_ana else None
    kt_fem = 1.5 * N_P * psi_f_fem

    psi_f_nl = fem_nl["psi_f"] if fem_nl else None
    backemf_nl = fem_nl["backemf"] if fem_nl else None

    psi_f_kend = fem_kend["psi_f"] if fem_kend else None
    backemf_kend = fem_kend["backemf"] if fem_kend else None
    psi_f_nl_kend = fem_nl_kend["psi_f"] if fem_nl_kend else None
    backemf_nl_kend = fem_nl_kend["backemf"] if fem_nl_kend else None

    lines = []
    def w(s=""):
        lines.append(s)

    w("# Back-EMF Validation Report")
    w()
    w(f"**Motor:** {motor.name} ({TOML.name})")
    w("**Measurement date:** 2026-03-18")
    w("**Models:** Zhu & Howe analytical, 2D FEM (NGSolve, linear + nonlinear iron), "
      "2D FEM + Russell-Norsworthy end-effect correction")
    w("**Convention:** B_rem is physical remanence; both solvers model square-wave "
      "radial magnetization (source fundamental M_1 = (4/π)(B_rem/μ0)·sin(πα_p/2)). "
      "Supersedes reports generated before 2026-06-11, which used a sinusoidal source "
      "— all model-vs-measured deltas changed.")
    w()

    w("## 1. Test Conditions")
    w()
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w("| Rotor variant | Steel (Harvey C) |")
    w(f"| Mechanical speed | 79.86 rps ({SPEED_RPM} RPM) |")
    w(f"| ω_mech | {W_MECH:.1f} rad/s |")
    w(f"| ω_elec | {W_ELEC:.1f} rad/s |")
    w(f"| Electrical frequency | {W_ELEC / (2*pi):.2f} Hz |")
    w("| Load | Open circuit (j_s = 0) |")
    w("| Temperature | 20.0 C |")
    w("| Measurement | Phase-to-neutral, 3-channel (CH1/CH2/CH4) |")
    w("| Method | Hann-windowed FFT, linear fit across 8 speeds |")
    w()

    w("## 2. Measured Results")
    w()
    w("| Quantity | Value | Method |")
    w("|----------|-------|--------|")
    w(f"| V_LN_peak (fundamental) | {V_LN_PEAK_MEAS:.4f} V | Sinusoidal fit to Va |")
    if meas_unc:
        w(f"| psi_f | {PSI_F_MEAS*1e3:.3f} +/- {meas_unc['psi_f_std']*1e3:.3f} mWb "
          f"({meas_unc['psi_f_rel_pct']:.1f}%) | V_peak / ω_e, 3-channel |")
        w(f"| Ke | {KE_MEAS*1e3:.3f} +/- {meas_unc['ke_std']*1e3:.3f} mV/(rad/s) "
          f"({meas_unc['ke_rel_pct']:.1f}%) | Linear fit, 3-channel |")
    else:
        w(f"| psi_f | {PSI_F_MEAS*1e3:.3f} mWb | V_peak / ω_e |")
        w(f"| Ke | {KE_MEAS*1e3:.3f} mV/(rad/s) | V_peak / ω_mech |")
    w(f"| Kt | {KT_MEAS*1e3:.3f} mNm/A | 1.5 x n_p x psi_f |")
    if measured_harmonics:
        thd_val = measured_harmonics["thd_pct"]
        h3 = measured_harmonics["harmonics_pct"].get(3, 0)
        h5 = measured_harmonics["harmonics_pct"].get(5, 0)
        w(f"| THD | {thd_val:.2f}% | Hann-windowed FFT, 3-channel avg |")
        w(f"| 3rd harmonic | {h3:.2f}% | FFT at {measured_harmonics['rps']:.0f} rps |")
        w(f"| 5th harmonic | {h5:.2f}% | FFT at {measured_harmonics['rps']:.0f} rps |")
    else:
        w(f"| THD | < {THD_MEAS_BOUND}% | Upper bound from fit residual |")
        w("| 3rd harmonic | — | No FFT extraction available |")
        w("| 5th harmonic | — | No FFT extraction available |")
    w("| Half-wave asymmetry | 1.16% | Peak positive vs negative |")
    w()

    w("## 3. Model Comparison")
    w()
    w(f"Carter factor k_c = {ana['k_c']:.4f} (applied to analytical model for slotted stator)")
    if fem_kend:
        w(f"End-effect factor k_end = {fem_kend['k_end']:.4f} "
          f"(Russell-Norsworthy, L_stk/g_eff aspect ratio correction)")
    w()
    w("| Quantity | Analytical | FEM (linear) | FEM (nonlinear) | FEM+k_end | FEM(NL)+k_end | Measured |")
    w("|----------|-----------|-------------|----------------|-----------|---------------|----------|")

    def row(name, v_ana, v_fem, v_nl, v_kend, v_nl_kend, v_meas, fmt=".4f"):
        def f(v):
            return f"{v:{fmt}}" if v is not None else "---"
        def d(v, ref):
            if v is not None and ref:
                return f" ({pct_delta(v, ref):+.1f}%)"
            return ""
        w(f"| {name} | {f(v_ana)}{d(v_ana, v_meas)} | {f(v_fem)}{d(v_fem, v_meas)} "
          f"| {f(v_nl)}{d(v_nl, v_meas)} | {f(v_kend)}{d(v_kend, v_meas)} "
          f"| {f(v_nl_kend)}{d(v_nl_kend, v_meas)} | {f(v_meas)} |")

    row("B_g1 (T)", ana["B_g1"], fem["B_g1"],
        fem_nl["B_g1"] if fem_nl else None,
        fem_kend["B_g1"] if fem_kend else None,
        fem_nl_kend["B_g1"] if fem_nl_kend else None,
        None, ".4f")
    row("psi_f (mWb)", psi_f_ana * 1e3 if psi_f_ana else None,
        psi_f_fem * 1e3, psi_f_nl * 1e3 if psi_f_nl else None,
        psi_f_kend * 1e3 if psi_f_kend else None,
        psi_f_nl_kend * 1e3 if psi_f_nl_kend else None,
        PSI_F_MEAS * 1e3, ".4f")
    row("Back-EMF V_pk (V)", backemf_ana, backemf_fem,
        backemf_nl, backemf_kend, backemf_nl_kend,
        V_LN_PEAK_MEAS, ".4f")
    ke_nl = backemf_nl / W_MECH if backemf_nl else None
    ke_kend = backemf_kend / W_MECH if backemf_kend else None
    ke_nl_kend = backemf_nl_kend / W_MECH if backemf_nl_kend else None
    kt_nl = 1.5 * N_P * psi_f_nl if psi_f_nl else None
    kt_kend = 1.5 * N_P * psi_f_kend if psi_f_kend else None
    kt_nl_kend = 1.5 * N_P * psi_f_nl_kend if psi_f_nl_kend else None
    row("Ke (mV/(rad/s))", ke_ana * 1e3 if ke_ana else None,
        ke_fem * 1e3, ke_nl * 1e3 if ke_nl else None,
        ke_kend * 1e3 if ke_kend else None,
        ke_nl_kend * 1e3 if ke_nl_kend else None,
        KE_MEAS * 1e3, ".4f")
    row("Kt (mNm/A)", kt_ana * 1e3 if kt_ana else None,
        kt_fem * 1e3, kt_nl * 1e3 if kt_nl else None,
        kt_kend * 1e3 if kt_kend else None,
        kt_nl_kend * 1e3 if kt_nl_kend else None,
        KT_MEAS * 1e3, ".4f")
    w()

    # Smooth-bore note
    psi_f_smooth = ana.get("psi_f_smooth")
    if psi_f_smooth:
        w(f"**Carter correction note:** All analytical results (table, sensitivity sweeps, "
          f"waterfall, contours) use Carter-corrected geometry — consistent with the B_g1 row. "
          f"Without Carter correction (smooth-bore model), psi_f would be "
          f"{psi_f_smooth*1e3:.4f} mWb ({pct_delta(psi_f_smooth, PSI_F_MEAS):+.1f}% vs measured). "
          f"The Carter factor accounts for {(psi_f_smooth - psi_f_ana)*1e3:.4f} mWb "
          f"({pct_delta(psi_f_smooth, psi_f_ana):+.1f}% correction for this "
          f"geometry: 9 slots, outrunner, n_p = 6).")
        w()

    # THD note
    thd_meas_str = (f"{measured_harmonics['thd_pct']:.2f}%"
                    if measured_harmonics else f"< {THD_MEAS_BOUND}%")
    w(f"**THD note:** Model THD is computed from spatial B_r harmonics; measured THD "
      f"({thd_meas_str}) is from temporal voltage harmonics. These are not directly comparable. "
      f"Slot harmonics visible in FEM (THD = {fem['thd_pct']:.1f}%) are spatially filtered by the "
      f"winding and do not appear at the terminals. Analytical THD = {ana['thd_pct']:.1f}% "
      f"(smooth-bore, no slot harmonics).")
    w()

    # FEM vs analytical comparison — rewritten for Carter-corrected analytical
    if psi_f_ana is not None:
        B_g1_meas_equiv = _measured_equiv_B_g1(motor)
        ana_delta = pct_delta(psi_f_ana, PSI_F_MEAS)
        fem_delta = pct_delta(psi_f_fem, PSI_F_MEAS)
        b1_delta = pct_delta(fem["B_g1"], ana["B_g1"])
        w(f"**Analytical vs FEM:** Under the square-wave magnetization convention the two "
          f"solvers agree on the fundamental: B_g1 = {ana['B_g1']:.3f} T analytical vs "
          f"{fem['B_g1']:.3f} T FEM ({b1_delta:+.1f}%). The gap vs measured is therefore a "
          f"shared model-family bias, not a solver discrepancy: Carter-corrected analytical "
          f"psi_f is {ana_delta:+.1f}% and linear FEM {fem_delta:+.1f}% vs measured. "
          f"(Reports generated before 2026-06-11 showed analytical +8% / FEM +19% "
          f"\"bracketing\" the measurement — that split was an artifact of the sinusoidal "
          f"source convention and the apparent closeness was coincidental.) "
          f"The measured-equivalent B_g1 is {B_g1_meas_equiv:.4f} T "
          f"(assuming nominal k_w = {motor.k_w}), well below both models.")
        w()
    if fem_nl and psi_f_nl:
        sat_effect = pct_delta(psi_f_nl, psi_f_fem)
        w(f"**Saturation effect:** Nonlinear FEM psi_f is {sat_effect:+.1f}% vs linear FEM. "
          f"{'Saturation reduces the flux as expected — tooth-tip flux density in the linear solution is well above the B-H knee (the square-wave source raised linear-solve iron peaks ~27% over pre-convention-fix figures).' if sat_effect < 0 else 'Saturation has minimal effect at this operating point.'}")
        w()
    if fem_kend and psi_f_kend:
        kend_delta = pct_delta(psi_f_kend, PSI_F_MEAS)
        kend_reduction = (1 - fem_kend["k_end"]) * 100
        w(f"**End-effect correction (Tier 0):** The Russell-Norsworthy correction applies a "
          f"{kend_reduction:.1f}% reduction (k_end = {fem_kend['k_end']:.4f}) to account for axial "
          f"flux leakage at the stack ends. This motor has L_stk = {motor.L_stk*1e3:.1f} mm with "
          f"OD = {motor.geometry.r_outer*2e3:.1f} mm (aspect ratio "
          f"{motor.L_stk / (motor.geometry.r_outer * 2):.2f}). "
          f"FEM+k_end predicts psi_f {kend_delta:+.1f}% vs measured")
        if fem_nl_kend and psi_f_nl_kend:
            nl_kend_delta = pct_delta(psi_f_nl_kend, PSI_F_MEAS)
            w(f", FEM(NL)+k_end predicts {nl_kend_delta:+.1f}%")
        fem_delta_raw = pct_delta(psi_f_fem, PSI_F_MEAS)
        closed_pct = (fem_delta_raw - kend_delta) / fem_delta_raw * 100 if fem_delta_raw else 0
        w(f". The correction closes about {closed_pct:.0f}% of the raw 2D over-prediction; "
          f"a {kend_delta:+.1f}% residual remains.")
        w()
        w("**Caveat — k_end is uncalibrated (Tier 0).** Axial end leakage is the right order "
          "of magnitude to explain roughly half the 2D over-prediction, but the split between "
          "end leakage and magnet/winding parameters is not yet pinned down:")
        w()
        w(f"1. **Formula applicability:** Russell-Norsworthy was derived for smooth-bore inrunners "
          f"with uniform airgap. This motor is an outrunner with discrete magnet arcs (α_p = "
          f"{motor.alpha_p}), 9 slots, and a non-uniform field at the stack ends. The effective "
          f"length scale for fringing may differ from g_eff = g + h_m/μ_r.")
        w(f"2. **Residual attribution:** The {kend_delta:+.1f}% remaining after k_end is "
          f"consistent with the effective-B_rem analysis (§5): datasheet B_rem with ideal "
          f"square-wave magnetization is an upper bound on the source fundamental — the real "
          f"magnetization profile (between square-wave and sinusoidal) plus unmodeled leakage "
          f"absorbs the rest.")
        w("3. **Validation needed:** A true 3D FEM solve (Tier 2) would provide a ground-truth "
          "k_end_3d for this geometry. If 3D gives a materially different k_end, the residual "
          "attribution shifts between end leakage and the magnet/winding factors above.")
        w()
        w("**Status:** Tier 0 — axial leakage plausibly explains about half the 2D FEM "
          "over-prediction; quantitative trust requires Tier 2 (3D FEM) calibration. "
          "Production psi_f remains end-effect-uncorrected; k_end here is informational.")
        w()

    w("## 4. Waveforms and Spectra")
    w()
    B_g1_meas = _measured_equiv_B_g1(motor)
    w(f"Measured-equivalent B_g1 = {B_g1_meas:.4f} T (back-derived from measured psi_f "
      f"using the winding formula with nominal k_w = {motor.k_w}, N = {motor.N}).")
    w()
    w("Top panel: air-gap B_r waveforms. The measured curve is the fundamental cosine "
      "at the amplitude implied by measured psi_f — no spatial harmonic content is available "
      "from a terminal voltage measurement.")
    w()
    w(f"Bottom panel: back-EMF voltage at {SPEED_RPM} RPM. Model waveforms are scaled from "
      "the air-gap field solution; measured is the sinusoidal fit fundamental.")
    w()
    w("![B_r and back-EMF waveforms](br_waveforms.png)")
    w()
    w("Harmonic spectrum shows spatial harmonics of B_r (model) with measured voltage "
      "harmonics overlaid as diamonds. Measured 3rd and 5th voltage harmonics map to "
      "spatial orders 3n_p = 18 and 5n_p = 30. Note: slot harmonics visible in FEM "
      "(order 3 = n_slots - n_p) are spatially filtered by the winding and do not "
      "appear in the terminal voltage.")
    w()
    w("![Harmonic spectrum](harmonics.png)")
    w()

    # Measured waveform overlay
    overlay_path = out_dir / "measured_overlay_80rps.png"
    if overlay_path.exists():
        w("### Measured Waveform Overlay (80 rps)")
        w()
        w("Three measured phase channels overlaid on analytical and FEM model predictions "
          "at 80 rps. Each channel is aligned to its own zero-crossing and plotted "
          "individually to show phase balance. Model waveforms are fundamental sinusoids "
          "at the model-predicted peak voltage for the test speed.")
        w()
        w("![Measured waveform overlay at 80 rps](measured_overlay_80rps.png)")
        w()

    # V_peak vs speed
    vpeak_path = out_dir / "vpeak_vs_speed.png"
    if vpeak_path.exists():
        w("### Amplitude vs Speed")
        w()
        w("Phase back-EMF peak voltage across all 8 speed points (20-120 rps). "
          "Per-channel measurements shown as scatter points; 3-channel averages as diamonds. "
          "The shaded region shows the gap between model prediction "
          "and measured linear fit (see §5 for attribution).")
        w()
        w("![V_peak vs speed](vpeak_vs_speed.png)")
        w()

    w("## 5. Sensitivity Analysis")
    w()
    w("Each parameter swept independently to find the single value matching "
      f"measured psi_f = {PSI_F_MEAS*1e3:.3f} mWb.")
    w()
    w("![Sensitivity curves](sensitivity.png)")
    w()
    w("| Parameter | Nominal | Value Matching Measured | Ratio |")
    w("|-----------|---------|------------------------|-------|")
    if b_match:
        w(f"| B_rem | 1.450 T | {b_match:.3f} T | {b_match/1.45:.3f} |")
    else:
        w("| B_rem | 1.450 T | Below sweep range | N/A |")
    if a_match:
        w(f"| alpha_p | 0.635 | {a_match:.3f} | {a_match/0.635:.3f} |")
    else:
        w("| alpha_p | 0.635 | Below sweep range | N/A |")
    if k_match:
        w(f"| k_w | 0.945 | {k_match:.3f} | {k_match/0.945:.3f} |")
    else:
        w("| k_w | 0.945 | Below sweep range | N/A |")
    w()

    w("### Interpretation")
    w()
    if psi_f_ana:
        ana_delta = pct_delta(psi_f_ana, PSI_F_MEAS)
        w(f"The Carter-corrected analytical model predicts psi_f {ana_delta:+.1f}% "
          f"vs measured. The sensitivity sweeps below (also Carter-corrected) show "
          f"what single-parameter adjustment would close the remaining gap.")
        w()
        w("The discrepancy is expected to arise from a combination of factors:")
    w()
    if b_match and b_match > motor.B_rem:
        w(f"1. **B_rem at or above N52.** A B_rem of {b_match:.2f} T would match by itself — "
          f"above the nominal {motor.B_rem} T. This means B_rem degradation alone cannot explain "
          f"the gap; the model already under-predicts at nominal B_rem.")
    elif b_match:
        below_grade = (" — below any sintered NdFeB grade (N30 ≈ 1.08 T), so magnet grade "
                       "alone cannot plausibly explain the gap; see the effective-B_rem "
                       "section" if b_match < 1.08 else "")
        w(f"1. **B_rem below N52 nominal.** A B_rem of {b_match:.2f} T would match by "
          f"itself{below_grade}.")
    else:
        w("1. **B_rem** — could not match within sweep range.")
    w()
    if a_match:
        direction = "wider" if a_match > motor.alpha_p else "narrower"
        w(f"2. **Effective pole-arc {direction} than geometric.** Flat rectangular magnets on a curved "
          f"rotor ID create a non-uniform gap across the magnet width. "
          f"An alpha_p of {a_match:.3f} would match by itself (nominal {motor.alpha_p}).")
    else:
        w("2. **Effective pole-arc** — could not match within sweep range.")
    w()
    if k_match:
        w(f"3. **Winding factor.** k_w = {motor.k_w} is a placeholder for ideal double-layer 12p/9s. "
          f"A k_w of {k_match:.3f} would match by itself ({(k_match/motor.k_w - 1)*100:+.1f}% "
          f"from nominal).")
    else:
        w("3. **Winding factor** — could not match within sweep range.")
    w()
    w("4. **Leakage flux.** The Zhu & Howe analytical model has no slot leakage or end-turn "
      "leakage paths. FEM captures inter-slot leakage but not end-turn leakage (2D model). "
      "The 3D leakage fraction could be 5-15% for a short-stack motor (L_stk = 7 mm).")
    w()

    w("### Most Physically Plausible Combination")
    w()
    if psi_f_ana:
        w(f"The {pct_delta(psi_f_ana, PSI_F_MEAS):+.1f}% gap is too large for any single "
          f"parameter at a plausible value (the single-parameter B_rem match sits below any "
          f"sintered NdFeB grade). The likely combination:")
    else:
        w("The measured gap is most likely a combination of:")
    w()
    w("- **Real magnetization profile below ideal square-wave.** The convention assumes "
      "uniformly magnetized arcs with sharp edges — an upper bound on the source "
      "fundamental. A profile between square-wave and sinusoidal removes up to ~21% "
      "(the full 4/π factor).")
    w("- **~10-16% 3D leakage** (end-turn + axial fringing on a 7mm stack; "
      "k_end = 0.84 Russell-Norsworthy estimate, uncalibrated)")
    w("- **k_w below the 0.945 placeholder** (actual winding has unequal layer turns)")
    w("- **Effective pole-arc slightly narrower than geometric** "
      "(flat-on-curved magnet geometry)")
    w()
    w("Calibration approach: measure B_rem directly (Helmholtz coil or gaussmeter on bare magnet), "
      "then solve for effective k_w from the measured psi_f. This separates magnetic from winding effects.")
    w()

    # B_rem x k_w contour
    contour_path = out_dir / "brem_kw_contour.png"
    if contour_path.exists():
        w("### B_rem x k_w Parameter Locus")
        w()
        w(f"The red contour shows all (B_rem, k_w) combinations that produce the measured "
          f"psi_f = {PSI_F_MEAS*1e3:.3f} mWb. The nominal operating point (B_rem = {motor.B_rem}, "
          f"k_w = {motor.k_w}) lies well above this contour, confirming the over-prediction. "
          f"Any point on the red curve is equally consistent with the measurement — independent "
          f"determination of either parameter is required to resolve the ambiguity.")
        w()
        w("![B_rem vs k_w contour](brem_kw_contour.png)")
        w()

    # k_w vs alpha_p trade-off and triplet table
    tradeoff_path = out_dir / "kw_alphap_tradeoff.png"
    if tradeoff_path.exists() or triplets:
        w("### Three-Parameter Trade-off")
        w()
        w("**α_p (pole-arc ratio)** is the fraction of each pole pitch covered by magnet. "
          f"This motor has 12 poles, so each pole spans 30°; α_p = {motor.alpha_p} means "
          f"each magnet covers ~{motor.alpha_p * 30:.0f}° with ~{(1 - motor.alpha_p) * 30:.0f}° "
          f"of air between adjacent magnets. Because the magnets are flat rectangular blocks "
          f"on a curved rotor surface, the edges stand off and contribute less flux — the "
          f"effective magnetic arc is likely narrower than the geometric {motor.alpha_p}.")
        w()

    if triplets:
        w("For each magnet grade, the k_w and alpha_p values needed (independently) "
          "to match measured psi_f:")
        w()
        w(f"| Grade | B_rem (T) | k_w (α_p = {motor.alpha_p}, measured) "
          f"| α_p (k_w = {motor.k_w}, placeholder) |")
        w("|-------|-----------|----------------------|----------------------|")
        for trip in triplets:
            kw_str = f"{trip['k_w']:.3f}" if trip['k_w'] else "> 1.0"
            ap_str = f"{trip['alpha_p']:.3f}" if trip['alpha_p'] else "> nominal"
            w(f"| {trip['grade']} | {trip['B_rem']:.2f} | {kw_str} | {ap_str} |")
        w()
        w(f"Note: the two columns have different confidence levels. "
          f"α_p = {motor.alpha_p} is derived from caliper measurements and chord-to-arc "
          f"correction — physically grounded, though flat-on-curved magnet geometry may "
          f"make the effective arc narrower. k_w = {motor.k_w} is a textbook value for "
          f"ideal double-layer 12p/9s and is explicitly a placeholder in the motor definition "
          f"(actual winding has unequal layer turns). The k_w column is therefore the more "
          f"trustworthy constraint; the α_p column is conditioned on an unverified assumption.")
        w()

    if tradeoff_path.exists():
        w("Each curve below shows the full family of (k_w, alpha_p) pairs that produce "
          "measured psi_f at a given magnet grade (Carter-corrected). "
          "At N52 (nominal grade), both k_w and alpha_p must decrease to match; "
          "at lower grades the required adjustments shrink.")
        w()
        w("![k_w vs alpha_p trade-off](kw_alphap_tradeoff.png)")
        w()

    # Waterfall decomposition (only when model over-predicts)
    waterfall_path = out_dir / "waterfall.png"
    if waterfall_path.exists():
        w("### Over-Prediction Decomposition")
        w()
        w("The waterfall below decomposes the model gap at N50 grade, "
          "with B_rem, effective pole-arc, winding factor, and 3D leakage "
          "as contributing factors.")
        w()
        w("![Over-prediction decomposition](waterfall.png)")
        w()

    if not waterfall_path.exists():
        ana_delta = pct_delta(psi_f_ana, PSI_F_MEAS) if psi_f_ana else 0
        if ana_delta < 0:
            w("### Model-Measurement Gap")
            w()
            w(f"The Carter-corrected analytical model **under-predicts** psi_f by "
              f"{abs(ana_delta):.1f}%. This is within the combined uncertainty of "
              f"k_w (placeholder), α_p (flat-on-curved geometry), and 3D leakage. "
              f"No error budget decomposition is needed — the model is close enough "
              f"that the remaining gap does not point to a single dominant factor.")
            w()

    budget_path = out_dir / "grade_error_budget.png"
    if budget_path.exists():
        w("### Error Budget by Magnet Grade")
        w()
        w("The error budget depends on the assumed magnet grade. At lower grades, "
          "B_rem accounts for more of the gap. At N52 (nominal), B_rem contributes "
          "nothing and the entire gap must be explained by α_p, k_w, and 3D leakage. "
          "A direct B_rem measurement pins the grade and collapses this ambiguity.")
        w()
        w("![Error budget by grade](grade_error_budget.png)")
        w()

    heatmap_path = out_dir / "kw_alphap_heatmap.png"
    if heatmap_path.exists():
        w("### ψ_f Error Across (α_p, k_w) Parameter Space")
        w()
        w("Each panel shows the ψ_f prediction error (%) at a fixed magnet grade, "
          "swept across all plausible (α_p, k_w) combinations. The black contour "
          "marks zero error (matches measured). The diamond is the motor definition "
          f"point ({motor.alpha_p}, {motor.k_w}). At lower magnet grades, the zero-error "
          "contour shifts toward the motor definition, meaning less adjustment to "
          "α_p and k_w is needed.")
        w()
        w("![k_w vs alpha_p heatmap](kw_alphap_heatmap.png)")
        w()

    # Effective B_rem back-calculation with uncertainty
    if b_rem_result:
        w("### Effective B_rem Back-Calculation")
        w()
        w(f"Back-calculated from measured psi_f = {PSI_F_MEAS*1e3:.3f} mWb using `_derive_B_rem()` "
          f"with Carter-corrected geometry. This is a *combined* effective B_rem that absorbs "
          f"the magnetization profile, leakage, and any k_w/alpha_p error — it is what the "
          f"remanence would need to be if the ideal square-wave model were otherwise exact.")
        w()
        b_eff = b_rem_result["B_rem_eff"]
        w(f"- **Effective B_rem:** {b_eff:.3f} T "
          f"(conditioned on placeholder k_w = {motor.k_w} and geometric α_p = {motor.alpha_p})")
        w()
        if b_eff < 1.08:
            w(f"This sits **below any sintered NdFeB grade** (N30 ≈ 1.08 T) — taken literally "
              f"it is not a magnet grade at all. That is the diagnostic value of the "
              f"square-wave convention: the physical magnets are presumably mid-grade "
              f"(nominal {motor.B_rem} T), and the ~{(1 - b_eff/motor.B_rem)*100:.0f}% "
              f"shortfall is the combined effect of a real magnetization profile below ideal "
              f"square-wave, 3D leakage, and winding-parameter error — not magnet degradation "
              f"(see `notes/review-findings-audit-2026-06-11.md` for the convention analysis).")
            w()
        w("A single ± scalar would misrepresent the uncertainty structure: the systematic "
          "unknowns (k_w, α_p, N) move the back-calculated value *along* the measured "
          "(B_rem × k_w) contour above, they do not form a statistical band around a known "
          "point. The measurement itself contributes only a small spread; independent "
          "perturbations:")
        w()
        unc = b_rem_result["uncertainty"]
        w("| Source | Assumption | B_rem range (T) | Half-width (T) |")
        w("|--------|-----------|----------------|---------------|")
        if "psi_f" in unc:
            lo, hi, hw = unc["psi_f"]
            w(f"| psi_f measurement | +/- 1σ from 3-channel spread | "
              f"{lo:.3f} .. {hi:.3f} | {hw:.4f} |")
        if "k_w" in unc:
            lo, hi, hw = unc["k_w"]
            w(f"| k_w (placeholder) | +/- 5% of {motor.k_w:.3f} | "
              f"{lo:.3f} .. {hi:.3f} | {hw:.4f} |")
        if "alpha_p" in unc:
            lo, hi, hw = unc["alpha_p"]
            w(f"| alpha_p (flat-on-curved) | +/- 5% of {motor.alpha_p:.3f} | "
              f"{lo:.3f} .. {hi:.3f} | {hw:.4f} |")
        for key, val in unc.items():
            if key.startswith("N="):
                N_alt = int(key.split("=")[1])
                w(f"| N = {N_alt} turns/coil | Top layer {motor.N - N_alt} fewer | "
                  f"{val:.3f} (point est.) | — |")
        w()
        w(f"The placeholder parameters dominate (k_w +/- {unc['k_w'][2]:.4f} T, "
          f"alpha_p +/- {unc['alpha_p'][2]:.4f} T vs psi_f +/- {unc['psi_f'][2]:.4f} T). "
          f"A direct gaussmeter B_rem measurement collapses the contour to a point and "
          f"separates magnet from winding effects.")
        w()

    # Measured harmonics
    if measured_harmonics:
        w("### Measured Harmonic Content (FFT)")
        w()
        w(f"Hann-windowed FFT on full capture at {measured_harmonics['rps']:.0f} rps "
          f"({measured_harmonics['n_pts']} points), averaged across 3 channels.")
        w()
        w("| Harmonic | % of Fundamental |")
        w("|----------|-----------------|")
        for k, pct in sorted(measured_harmonics["harmonics_pct"].items(), key=lambda x: int(x[0])):
            k = int(k)
            spatial = k * N_P
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(k if k < 20 else k % 10, "th")
            w(f"| {k}{suffix} (spatial order {spatial}) | {pct:.2f}% |")
        w(f"| **THD** | **{measured_harmonics['thd_pct']:.2f}%** |")
        w()

    # Phase balance
    if balance_result and balance_result.get("angles"):
        w("## 6. Phase Balance")
        w()
        w(f"Rising zero-crossing analysis at {balance_result['rps']:.0f} rps with sub-sample interpolation.")
        w()
        w("| Pair | Angle (deg) | Delta from 120 deg |")
        w("|------|------------|-------------------|")
        for pair, angle in balance_result["angles"].items():
            delta = angle - 120.0
            w(f"| {pair} | {angle:.1f} | {delta:+.1f} |")
        w()
        crossings = balance_result["crossings"]
        w(f"Zero-crossings detected: {', '.join(f'{k}: {v}' for k, v in crossings.items())}")
        w()
        phasor_path = out_dir / "phase_phasors.png"
        if phasor_path.exists():
            w("![Phase phasor diagram](phase_phasors.png)")
            w()

        # Resistance anomaly correlation
        angles = balance_result["angles"]
        angle_list = list(angles.values())
        if len(angle_list) >= 3:
            pairs_sorted = sorted(angles.items(), key=lambda x: abs(x[1] - 120.0), reverse=True)
            worst_pair, worst_angle = pairs_sorted[0]
            w("### Correlation with Resistance Anomaly")
            w()
            w(f"The largest phase deviation is {worst_pair} at {worst_angle:.1f} deg "
              f"(delta {worst_angle - 120:+.1f} deg). The blue phase (CH4) consistently reads "
              f"the highest voltage across all speeds (per-channel Ke: CH1=1.599, CH2=1.609, "
              f"CH4=1.612 mV/(rad/s)).")
            w()
            w("The TOML documents a blue-brown resistance anomaly: the initial Blue-Brown "
              "measurement (line-to-neutral; Brown is the wye neutral) "
              "was 336 mOhm (vs ~139 mOhm phase), later re-tested at 240.5 mOhm and attributed "
              "to probe contact artifact. The CH2-to-CH4 phase shift (+1.9 deg) and CH4's slightly "
              "higher Ke are consistent with a small winding asymmetry in the blue phase, though "
              "the effect is within measurement noise (~0.4% Ke spread). The re-test resolved the "
              "gross anomaly; the residual imbalance is not operationally significant.")
            w()

    # Drive sim with calibrated psi_f
    if drive_sim_result:
        w("## 7. Drive Simulation — Calibrated vs Uncalibrated")
        w()
        w(f"Drive simulation using `plan_sim()` with calibrated psi_f = {PSI_F_MEAS*1e3:.3f} mWb "
          f"vs uncalibrated (B_rem-derived).")
        w()
        for label in ("uncalibrated", "calibrated"):
            entry = drive_sim_result.get(label, {})
            if "error" in entry:
                w(f"**{label.capitalize()}:** CRASHED — {entry['error']}")
                w()
            elif "metrics" in entry:
                m = entry["metrics"]
                w(f"**{label.capitalize()}** (psi_f = {entry['psi_f']*1e3:.3f} mWb):")
                w()
                w("| Metric | Value |")
                w("|--------|-------|")
                for mk, mv in m.items():
                    w(f"| {mk} | {mv:.4f} |")
                w()
        # Interpretation
        uncal = drive_sim_result.get("uncalibrated", {})
        cal = drive_sim_result.get("calibrated", {})
        if "metrics" in uncal and "metrics" in cal:
            mu = uncal["metrics"]
            mc = cal["metrics"]
            tau_uncal = mu.get("tau_peak", 0)
            tau_cal = mc.get("tau_peak", 0)
            i_ss = mc.get("i_ss", 0)
            droop = mc.get("speed_droop", 0)
            tau_reduction = (1 - tau_cal / tau_uncal) * 100 if tau_uncal else 0

            w("### Interpretation")
            w()
            max_i_s = motor.drive.MAX_I_S
            w(f"Calibration reduces tau_peak by {tau_reduction:.0f}% "
              f"({tau_uncal*1e3:.1f} to {tau_cal*1e3:.1f} mNm), directly reflecting the "
              f"lower psi_f. Steady-state current i_ss = {i_ss:.2f} A is "
              f"{i_ss/max_i_s*100:.0f}% of the peak current limit "
              f"({max_i_s:.2f} A). Both calibrated and uncalibrated cases show the same "
              f"i_ss because the load scales with k_t (load_fraction = 0.5). "
              f"Speed droop of {droop*100:.1f}% under load is typical for this "
              f"controller bandwidth.")
            w()
            w(f"For the gearbox application (243:1 ratio, ~26% overall efficiency), "
              f"the calibrated stall torque at output is approximately "
              f"{tau_cal * 243 * 0.26 * 1e3:.0f} mNm — {tau_reduction:.0f}% less than "
              f"the uncalibrated prediction. This gap matters for the 12x torque "
              f"multiplication target and should be factored into margin analysis.")
            w()

    transient_path = out_dir / "drive_sim_transient.png"
    if transient_path.exists():
        w("### Transient Response")
        w()
        w(f"Step response with calibrated ψ_f = {PSI_F_MEAS*1e3:.3f} mWb. "
          f"Speed reference steps from 0 to {motor.drive.W_REF:.0f} rad/s, "
          f"followed by a load torque step. Controller tuning is motor-aware "
          f"via `plan_sim()` (adaptive α_s, α_c, T_s from motor physics).")
        w()
        w("![Drive sim transient](drive_sim_transient.png)")
        w()

    w("Note: actuator J = 4.73e-7 kg·m² produces a fast mechanical time constant. "
      "Controller tuning is adapted automatically via `plan_sim()`.")
    w()

    # Bump section numbering for Notes
    next_section = 6
    if balance_result and balance_result.get("angles"):
        next_section = 7
    if drive_sim_result:
        next_section = 8

    w(f"## {next_section}. Notes")
    w()
    w("- Measured psi_f from 3-channel speed sweep (20-120 rps, 2026-03-18). Supersedes single-channel estimate.")
    w("- Both linear (mu_r_fe = 3000) and nonlinear FEM results included; the saturation "
      "effect on psi_f is recomputed in §3. Tooth-tip peak flux density is not reported "
      "per-run (B_iron_max surfacing is planned).")
    w("- Carter factor applied to analytical model for slotted stator (9 slots).")
    w("- All models run at j_s = 0 (open circuit) to match test conditions.")
    w("- Back-EMF computed as V_peak = omega_elec x psi_f (fundamental, phase-to-neutral).")
    w()

    w(f"## {next_section + 1}. Next Steps")
    w()
    w("1. **Gaussmeter measurement of bare magnet.** Measure B_rem directly (Helmholtz coil or "
      "surface gaussmeter on a removed magnet). This collapses the B_rem x k_w contour to a "
      "single point and resolves the winding factor independently.")
    w()
    w("2. **4-wire Kelvin resistance measurement per phase.** Confirm or dismiss the blue-brown "
      "resistance anomaly. Current LCR probe-contact measurements have ~1% uncertainty; a Kelvin "
      "measurement would settle whether the CH4 (blue) Ke offset is winding asymmetry or "
      "measurement artifact.")
    w()
    w("3. **Investigate 5th harmonic (1.6%).** Not a slot harmonic (those are winding-filtered). "
      "The square-wave arc model now produces a 5th spatial harmonic (see harmonics.png) — "
      "comparing its predicted level against the measured 1.6% would discriminate between magnet "
      "shape (flat-on-curved rotor), rotor eccentricity, and winding asymmetry. The analytical "
      "waveform output is still fundamental-only.")
    w()
    w("4. **Drive simulation with uncertainty propagation.** Propagate psi_f and B_rem uncertainty "
      "through the drive sim (Monte Carlo or Jacobian) to get error bars on tau_peak, speed_droop, "
      "and gearbox output torque. Deferred to broader drive-sim future work.")
    w()

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    motor = load_motor(str(TOML))
    cg = carter_adjusted_radii(motor.geometry, motor.mu_r_pm)[:2]
    print(f"Loaded: {motor.name}, n_p={motor.n_p}, B_rem={motor.B_rem}, alpha_p={motor.alpha_p}")
    print(f"Geometry: r_outer={motor.geometry.r_outer*1e3:.2f}mm, "
          f"r_stator={motor.geometry.r_stator*1e3:.2f}mm, "
          f"r_magnet={motor.geometry.r_magnet*1e3:.2f}mm, "
          f"r_rotor={motor.geometry.r_rotor*1e3:.2f}mm")
    print(f"Winding: N={motor.N}, k_w={motor.k_w}, coils_series={motor.coils_series}")

    # 1. Analytical
    print("\n--- Analytical (Zhu & Howe) ---")
    ana = run_analytical(motor)
    print(f"  B_g1 = {ana['B_g1']:.4f} T")
    print(f"  psi_f = {ana['psi_f']*1e3:.4f} mWb" if ana["psi_f"] else "  psi_f = N/A")
    print(f"  Back-EMF = {ana['backemf']:.4f} V" if ana["backemf"] else "  Back-EMF = N/A")
    print(f"  THD = {ana['thd_pct']:.2f}%")
    print(f"  k_c = {ana['k_c']:.4f}")

    # 2. FEM (linear)
    print("\n--- FEM (NGSolve, linear iron) ---")
    fem = run_fem(motor, nonlinear=False)
    print(f"  B_g1 = {fem['B_g1']:.4f} T")
    print(f"  psi_f = {fem['psi_f']*1e3:.4f} mWb")
    print(f"  Back-EMF = {fem['backemf']:.4f} V")
    print(f"  THD = {fem['thd_pct']:.2f}%")

    # 2b. FEM (nonlinear) — check saturation effect
    print("\n--- FEM (NGSolve, nonlinear iron) ---")
    fem_nl = run_fem(motor, nonlinear=True)
    print(f"  B_g1 = {fem_nl['B_g1']:.4f} T")
    print(f"  psi_f = {fem_nl['psi_f']*1e3:.4f} mWb")
    print(f"  Back-EMF = {fem_nl['backemf']:.4f} V")
    print(f"  THD = {fem_nl['thd_pct']:.2f}%")

    # 2c. End-effect correction (Russell-Norsworthy)
    k_end, g_eff = compute_end_effect(motor)
    print("\n--- End-Effect Correction (Russell-Norsworthy) ---")
    print(f"  g_eff = {g_eff*1e3:.3f} mm, L_stk = {motor.L_stk*1e3:.1f} mm")
    print(f"  k_end = {k_end:.4f} ({(1-k_end)*100:.1f}% reduction)")
    fem_kend = apply_end_effect(fem, k_end, motor)
    print(f"  FEM+k_end B_g1 = {fem_kend['B_g1']:.4f} T")
    print(f"  FEM+k_end psi_f = {fem_kend['psi_f']*1e3:.4f} mWb")
    print(f"  FEM+k_end Back-EMF = {fem_kend['backemf']:.4f} V")
    fem_nl_kend = apply_end_effect(fem_nl, k_end, motor)
    print(f"  FEM(NL)+k_end psi_f = {fem_nl_kend['psi_f']*1e3:.4f} mWb")

    # 3. Measured comparison
    print("\n--- Measured ---")
    print(f"  psi_f = {PSI_F_MEAS*1e3:.3f} mWb")
    print(f"  Back-EMF = {V_LN_PEAK_MEAS:.4f} V")

    print("\n--- Deltas vs Measured ---")
    if ana["psi_f"]:
        print(f"  Analytical psi_f:    {pct_delta(ana['psi_f'], PSI_F_MEAS):+.1f}%")
    print(f"  FEM (linear) psi_f:  {pct_delta(fem['psi_f'], PSI_F_MEAS):+.1f}%")
    print(f"  FEM (nonlin) psi_f:  {pct_delta(fem_nl['psi_f'], PSI_F_MEAS):+.1f}%")
    print(f"  FEM+k_end psi_f:     {pct_delta(fem_kend['psi_f'], PSI_F_MEAS):+.1f}%")
    print(f"  FEM(NL)+k_end psi_f: {pct_delta(fem_nl_kend['psi_f'], PSI_F_MEAS):+.1f}%")

    # 4. Plots
    print("\nGenerating plots...")
    plot_br_waveforms(ana, fem, motor, OUT, fem_kend=fem_kend)

    # 4a. Measured harmonics (FFT extraction)
    print("Extracting measured harmonics (80 rps)...")
    measured_harmonics = extract_measured_harmonics(80, motor.n_p, actual_rps=79.86)
    if measured_harmonics:
        print(f"  THD = {measured_harmonics['thd_pct']:.2f}%")
        for k, pct in sorted(measured_harmonics["harmonics_pct"].items(), key=lambda x: int(x[0])):
            if int(k) > 1:
                print(f"  {k}th harmonic = {pct:.2f}%")

    plot_harmonics(ana, fem, motor.n_p, OUT, measured_harmonics=measured_harmonics)

    # 4b. Measured overlay (requires CSV capture)
    print("Measured waveform overlay (80 rps)...")
    plot_measured_overlay(ana, fem, motor, OUT, fem_kend=fem_kend)

    # 4c. V_peak vs speed
    print("Amplitude vs speed plot...")
    plot_vpeak_vs_speed(motor, OUT, carter_geo=cg, fem=fem, fem_kend=fem_kend)

    # 4d. FEM alpha_p sweep (sparse points for sensitivity plot)
    print("FEM alpha_p sweep (sparse)...")
    from dataclasses import replace as _replace
    fem_alpha_p = []
    for ap in np.linspace(0.45, 0.70, 7):
        m_ap = _replace(motor, alpha_p=ap)
        try:
            res = run_fem(m_ap, nonlinear=False)
            fem_alpha_p.append((ap, res["psi_f"]))
            print(f"  alpha_p={ap:.3f}: psi_f={res['psi_f']*1e3:.4f} mWb")
        except Exception as e:
            print(f"  alpha_p={ap:.3f}: SKIP ({e})")

    # 5. Sensitivity
    print("Running sensitivity sweeps...")
    b_match, a_match, k_match = plot_sensitivity(motor, OUT, carter_geo=cg,
                                                  fem=fem, fem_alpha_p=fem_alpha_p,
                                                  fem_kend=fem_kend)
    print(f"  B_rem match: {b_match:.3f} T" if b_match else "  B_rem match: below range")
    print(f"  alpha_p match: {a_match:.3f}" if a_match else "  alpha_p match: below range")
    print(f"  k_w match: {k_match:.3f}" if k_match else "  k_w match: below range")

    # 5. Parameter locus contours
    print("B_rem x k_w contour...")
    plot_brem_kw_contour(motor, OUT, carter_geo=cg, fem=fem, fem_kend=fem_kend)
    print("k_w vs alpha_p trade-off...")
    plot_kw_alphap_tradeoff(motor, OUT, carter_geo=cg, fem=fem, fem_kend=fem_kend)
    triplets = compute_parameter_triplets(motor, PSI_F_MEAS, carter_geo=cg)
    for row in triplets:
        kw_str = f"{row['k_w']:.3f}" if row['k_w'] else "> 1.0"
        ap_str = f"{row['alpha_p']:.3f}" if row['alpha_p'] else "> nom"
        print(f"  {row['grade']}: k_w={kw_str}, alpha_p={ap_str}")

    print("Waterfall decomposition...")
    plot_waterfall(motor, triplets, OUT, carter_geo=cg)

    print("Grade error budget (N48/N50/N52)...")
    plot_grade_error_budget(motor, OUT, carter_geo=cg)

    print("k_w x alpha_p heatmap...")
    plot_kw_alphap_heatmap(motor, OUT, carter_geo=cg, fem=fem, fem_kend=fem_kend)

    # 5a. Measurement uncertainty
    print("\nMeasurement uncertainty...")
    meas_unc = compute_measurement_uncertainty()
    if meas_unc:
        print(f"  Ke = {meas_unc['ke_mean']*1e3:.3f} +/- {meas_unc['ke_std']*1e3:.3f} mV/(rad/s) "
              f"({meas_unc['ke_rel_pct']:.1f}%)")
        print(f"  psi_f = {meas_unc['psi_f_mean']*1e3:.3f} +/- {meas_unc['psi_f_std']*1e3:.3f} mWb")

    # 5b. Effective B_rem back-calculation
    print("\nEffective B_rem back-calculation...")
    b_rem_result = compute_effective_B_rem(motor, PSI_F_MEAS, meas_unc=meas_unc, carter_geo=cg)
    print(f"  Effective B_rem = {b_rem_result['B_rem_eff']:.3f} +/- {b_rem_result['rss']:.3f} T "
          f"({b_rem_result['grade_lo']}..{b_rem_result['grade_hi']})")

    # 5c. Phase balance analysis
    print("\nPhase balance analysis (80 rps)...")
    balance_result = analyze_phase_balance(80, motor.n_p, actual_rps=79.86)
    if balance_result:
        for pair, angle in balance_result["angles"].items():
            print(f"  {pair}: {angle:.1f} deg (delta {angle - 120:+.1f})")
        plot_phase_phasors(balance_result, OUT)

    # 5d. Phase balance across speeds (start at 40 rps: 20/30 give too few
    # crossings for reliable results; no 10 rps capture exists — motor not spinning)
    print("Phase balance across speeds...")
    sweep_speeds = [(40, 39.58), (50, 50.0), (60, 59.72), (80, 79.86), (100, 100.0), (120, 120.14)]
    for cmd, actual in sweep_speeds:
        bal = analyze_phase_balance(cmd, motor.n_p, actual)
        if bal and bal["angles"]:
            avg_angle = np.mean(list(bal["angles"].values()))
            print(f"  {cmd} rps: avg angle = {avg_angle:.1f} deg")

    # 6. Drive sim with calibrated psi_f
    print("\nDrive simulation (calibrated vs uncalibrated)...")
    drive_sim_result = run_calibrated_drive_sim(motor, PSI_F_MEAS)
    for label in ("uncalibrated", "calibrated"):
        entry = drive_sim_result.get(label, {})
        if "error" in entry:
            print(f"  {label}: CRASHED — {entry['error']}")
        elif "metrics" in entry:
            m = entry["metrics"]
            tau_key = "tau_peak" if "tau_peak" in m else "tau_M_ss"
            print(f"  {label}: psi_f={entry['psi_f']*1e3:.3f} mWb, "
                  f"{tau_key}={m.get(tau_key, 0):.4f} Nm")

    print("Drive sim transient figure...")
    plot_drive_sim_transient(motor, PSI_F_MEAS, OUT)

    # 7. Report
    print("\nWriting report...")
    report_path = generate_report(ana, fem, b_match, a_match, k_match, motor, OUT,
                                  fem_nl=fem_nl, b_rem_result=b_rem_result,
                                  measured_harmonics=measured_harmonics,
                                  balance_result=balance_result,
                                  drive_sim_result=drive_sim_result,
                                  meas_unc=meas_unc, triplets=triplets,
                                  fem_kend=fem_kend, fem_nl_kend=fem_nl_kend)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
