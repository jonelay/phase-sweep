#!/usr/bin/env python3
"""Phase-sweep validation & MVP shakedown report.

Exercises all model types across all reference motors, compares outputs,
cross-validates against measured/published data, and generates a Markdown report.
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phasesweep.configs import load_motor
from phasesweep.crossval import (
    ComparisonRow,
    compare_results,
    diagnose_detailed,
    format_diagnosis,
    format_table,
)
from phasesweep.fem_field import harmonics_1sided
from phasesweep.geometry import inrunner, outrunner
from phasesweep.measured import MeasuredResult
from phasesweep.motor import Motor
from phasesweep.parallel import execute_generic
from phasesweep.perturbation import perturb_motor as _perturb_motor
from phasesweep.plots import plot_geometry
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sim import plan_sim
from phasesweep.solver_params import prepare_drive_sim
from phasesweep.sweep_types import RunConfig, RunResult

# ---------------------------------------------------------------------------
# Motor catalogue and data mapping
# ---------------------------------------------------------------------------

# Maps motor name → (TOML path relative to ROOT, data dir relative to ROOT)
MOTOR_DATA_MAP: dict[str, tuple[str | None, str | None]] = {
    "Actuator Steel Rotor": ("motors/actuator_steel_rotor.toml", "data/actuator_steel_rotor"),
    "Actuator Aluminum Rotor": ("motors/actuator_aluminum_rotor.toml", "data/actuator_aluminum_rotor"),
    "CREATOR Case PMSM": ("motors/creator_case_pmsm.toml", "data/creator_case_pmsm"),
    "Belkhadir 22p/24s ER-PMSM": ("motors/belkhadir_outrunner.toml", "data/belkhadir_outrunner"),
    "Deylami 8p/12s Cooling Fan": ("motors/deylami_fan.toml", "data/deylami_fan"),
    "FEMM LRK 14p/12s": ("motors/lrk_outrunner.toml", None),
}



def build_motor_catalogue() -> dict[str, Motor]:
    """Collect all reference motors from TOML files + paper definitions."""
    motors: dict[str, Motor] = {}

    # TOML-defined motors
    for name, (toml_rel, _) in MOTOR_DATA_MAP.items():
        if toml_rel is None:
            continue
        path = ROOT / toml_rel
        if path.exists():
            motors[name] = load_motor(path)

    # Zhu & Howe 2002 paper motors
    paper_geo = inrunner(
        r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
    )
    motors["Zhu & Howe 8-pole inrunner"] = Motor(
        name="Zhu & Howe 8-pole inrunner",
        geometry=paper_geo,
        n_p=4, B_rem=1.2, mu_r_pm=1.05,
    )

    paper_geo_out = outrunner(
        r_outer=0.060, r_rotor=0.048, r_magnet=0.040,
        r_stator=0.030, r_inner=0.020,
    )
    motors["Zhu & Howe 8-pole outrunner"] = Motor(
        name="Zhu & Howe 8-pole outrunner",
        geometry=paper_geo_out,
        n_p=4, B_rem=1.2, mu_r_pm=1.05,
    )

    return motors


def load_measured_data(motor_name: str, motor: Motor) -> list[RunResult]:
    """Load all measured/published JSON data files for a motor."""
    entry = MOTOR_DATA_MAP.get(motor_name)
    if entry is None:
        return []
    _, data_rel = entry
    if data_rel is None:
        return []
    data_dir = ROOT / data_rel
    if not data_dir.is_dir():
        return []

    results: list[RunResult] = []
    for json_path in sorted(data_dir.glob("*.json")):
        try:
            raw = json.loads(json_path.read_text())
            # Skip files without test_type (e.g. reference_scalars.json)
            if "test_type" not in raw:
                continue
            data = MeasuredResult.from_dict(raw)
            config = RunConfig(motor=motor, model=data.test_type)
            metrics: dict = dict(data.quantities)
            if data.curve_compare:
                metrics["_curve_compare"] = {k: v.to_dict() for k, v in data.curve_compare.items()}
            if data.bound_compare:
                metrics["_bound_compare"] = {k: v.to_dict() for k, v in data.bound_compare.items()}
            if data.key_mapping:
                metrics["_key_mapping"] = {k: v.to_dict() for k, v in data.key_mapping.items()}
            results.append(RunResult(
                config=config,
                model=data.test_type,
                status="OK",
                metrics=metrics,
                elapsed_s=0.0,
                source=data.source,
                tolerances=data.tolerances or None,
            ))
        except Exception as e:
            print(f"WARNING: failed to load {json_path}: {e}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Model eligibility
# ---------------------------------------------------------------------------

def check_model_eligibility(motor: Motor) -> dict[str, str]:
    """Return {model_key: status} for each computed model."""
    results: dict[str, str] = {}
    for key, info in MODEL_REGISTRY.items():
        if info.source != "computed":
            continue
        if info.validate is None:
            results[key] = "no validator"
            continue
        try:
            info.validate(motor)
            results[key] = "eligible"
        except (ValueError, TypeError) as e:
            results[key] = f"SKIP: {e}"
    return results


# ---------------------------------------------------------------------------
# Run models
# ---------------------------------------------------------------------------

def run_model(motor: Motor, model_key: str, **config_kw) -> RunResult | None:
    """Run a single model on a motor. Returns None if ineligible."""
    info = MODEL_REGISTRY[model_key]
    if info.fn is None:
        return None

    config = RunConfig(motor=motor, model=model_key, **config_kw)
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = info.fn(config)
        elapsed = time.perf_counter() - t0
        if caught:
            metrics["_warnings"] = [
                f"{w.category.__name__}: {w.message}" for w in caught
            ]
        return RunResult(
            config=config, model=model_key, status="OK",
            metrics=metrics, elapsed_s=elapsed,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return RunResult(
            config=config, model=model_key, status="ERROR",
            metrics=None, elapsed_s=elapsed, error_msg=str(e),
        )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

# Primary metric per model for sensitivity comparison.
# Rated torque is the main design output — always shown when eligible.
# Field models produce B₁ (Tesla, intensive) as supporting context.
_SENSITIVITY_METRIC: dict[str, str] = {
    "rated_torque": "tau_mtpa",
    "stall_torque": "tau_stall",
    "analytical": "fundamental",
    "fem": "fundamental",
    "drive_sim": "tau_peak",
}


# Display labels for sensitivity tracks (registry keys → plot labels)
_SENSITIVITY_LABEL: dict[str, str] = {
    "rated_torque": "τ_rated",
    "stall_torque": "τ_stall",
    "analytical": "B₁_analytical",
    "fem": "B₁_fem",
    "drive_sim": "τ_sim",
}


def _sensitivity_metric(model_key: str, metrics: dict) -> float | None:
    """Extract the sensitivity metric from model results.

    Field models: B_fundamental (Tesla) — intensive, geometry-independent.
    Torque models: direct torque value (N·m) — extensive.
    """
    metric_key = _SENSITIVITY_METRIC.get(model_key)
    if metric_key is None:
        return None
    val = metrics.get(metric_key)
    if val is None:
        return None
    return float(val)

# Parameter perturbation definitions.
# "OD" models an OEM frame-size lineup: stator and rotor grow together while
# air gap, magnet thickness, and shaft stay fixed.  A bigger frame means a
# larger bore → more flux linkage and torque.  L_m ∝ r_bore/gap.
SENSITIVITY_PARAMS = {
    "OD": {"label": "Motor OD (frame size)", "deltas": [-0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.15, 0.20]},
    "gap": {"label": "Air gap", "deltas": [-0.10, -0.05, 0.0, 0.05, 0.10]},
    "L_stk": {"label": "Stack length", "deltas": [-0.10, -0.05, 0.0, 0.05, 0.10]},
    "B_rem": {"label": "Magnet grade (B_rem)", "deltas": [-0.10, -0.05, 0.0, 0.05, 0.10]},
    # k_w: motors without k_w are skipped. L_d/L_q not scaled (leakage not
    # decomposed — this is a documented limitation). Back-EMF/torque
    # sensitivity is correct; current-loop dynamics sensitivity is not.
    "k_w": {"label": "Winding factor (k_w)", "deltas": [-0.05, -0.025, 0.0, 0.025, 0.05]},
}


def _analytical_valid(motor: Motor) -> bool:
    """Smooth-bore analytical model is valid when slots are absent or numerous."""
    n = motor.geometry.n_slots
    return n == 0 or n > 6




def fig_sensitivity(
    motor_name: str,
    sens_results: dict[str, dict[str, list[float | None]]],
    fig_dir: Path,
) -> str:
    # Filter out params where all tracks are flat zero (no meaningful data)
    params = [
        p for p in sens_results
        if any(
            any(v is not None and abs(v) > 1e-6 for v in curve)
            for curve in sens_results[p].values()
        )
    ]
    if not params:
        return ""
    n = len(params)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f"Sensitivity — {motor_name}", fontsize=13)

    model_colors = {
        "τ_rated": "tab:red", "τ_stall": "tab:green",
        "τ_sim": "tab:purple",
        "B₁_analytical": "tab:blue", "B₁_fem": "tab:orange",
    }

    for idx, param in enumerate(params):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        pdef = SENSITIVITY_PARAMS[param]
        deltas = pdef["deltas"]

        x_vals = [d * 100 for d in deltas]
        x_label = f"{pdef['label']} change (%)"

        model_curves = sens_results[param]
        for model_key, pct_changes in model_curves.items():
            valid_x = [x for x, y in zip(x_vals, pct_changes) if y is not None]
            valid_y = [y for y in pct_changes if y is not None]
            if not valid_y:
                continue
            ax.plot(valid_x, valid_y, "o-", lw=1.5, ms=4,
                    color=model_colors.get(model_key, "gray"),
                    label=model_key)

        ax.axhline(0, color="black", lw=0.4)
        ax.axvline(0, color="black", lw=0.4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Response change (%)")
        ax.set_title(pdef["label"], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Hide unused subplot cells
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return _savefig(fig, fig_dir, f"sensitivity_{_sanitize(motor_name)}.png")


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().replace(" ", "_")).strip("_")


def _savefig(fig: plt.Figure, fig_dir: Path, filename: str) -> str:
    fig.savefig(str(fig_dir / filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def _load_backemf_waveform(
    data_dir: Path, n_p: int, br_fem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and normalize CREATOR-style Back_emf.csv to B_r scale.

    Returns (theta_deg, br_normalized) or None if CSV not found.
    """
    csv_path = (data_dir / "PM_synchronous_motor" / "Measurement_results"
                / "No_load_tests" / "Back_emf.csv")
    if not csv_path.exists():
        return None
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    angle_deg = raw[:, 0]
    phase_u = raw[:, 1]

    n_pts = 1024
    theta_uniform = np.linspace(0, 360, n_pts, endpoint=False)
    emf_uniform = np.interp(theta_uniform, angle_deg, phase_u)

    # FFT to get fundamental amplitudes
    emf_amps = np.abs(np.fft.rfft(emf_uniform)) / n_pts
    emf_amps[1:] *= 2
    emf_fund = emf_amps[n_p] if n_p < len(emf_amps) else 1.0

    fem_amps = harmonics_1sided(br_fem)
    fem_fund = fem_amps[n_p] if n_p < len(fem_amps) else 1.0

    if emf_fund < 1e-12:
        return None
    scale = fem_fund / emf_fund

    br_normalized = emf_uniform * scale

    # Flip sign if anti-correlated with FEM
    fem_interp = np.interp(theta_uniform, np.linspace(0, 360, len(br_fem), endpoint=False), br_fem)
    if np.dot(br_normalized, fem_interp) < 0:
        br_normalized = -br_normalized

    return theta_uniform, br_normalized


def fig_br_waveform_and_harmonics(
    name: str,
    theta_ana: np.ndarray | None, br_ana: np.ndarray | None,
    theta_fem: np.ndarray, br_fem: np.ndarray,
    n_p: int, fig_dir: Path,
    *,
    meas_theta_deg: np.ndarray | None = None,
    meas_br_normalized: np.ndarray | None = None,
    meas_label: str = "Measured (normalized)",
    scalar_br_markers: list[tuple[float, str]] | None = None,
) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Air-gap field — {name}", fontsize=11)

    has_ana = theta_ana is not None and br_ana is not None
    has_meas = meas_theta_deg is not None and meas_br_normalized is not None

    deg_fem = np.degrees(theta_fem)
    if has_ana:
        ax1.plot(np.degrees(theta_ana), br_ana, lw=1.3, label="Analytical (Zhu & Howe)")
    ax1.plot(deg_fem, br_fem, lw=1.0, ls="--", alpha=0.85, label="FEM (NGSolve)")
    if has_meas:
        ax1.plot(meas_theta_deg, meas_br_normalized, lw=0.8, ls=":",
                 color="tab:green", alpha=0.9, label=meas_label)
    if scalar_br_markers:
        for val, label in scalar_br_markers:
            ax1.axhline(val, color="tab:orange", ls="--", lw=1.2, alpha=0.7)
            ax1.axhline(-val, color="tab:orange", ls="--", lw=1.2, alpha=0.3)
            ax1.annotate(label, xy=(1.0, val), fontsize=6.5, color="tab:orange",
                         ha="right", va="bottom",
                         xycoords=("axes fraction", "data"),
                         xytext=(-4, 2), textcoords="offset points")
    ax1.axhline(0, color="black", lw=0.4)
    ax1.set_xlabel("θ (deg)")
    ax1.set_ylabel("B_r (T)")
    ax1.set_xlim(0, 360)
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3)

    # Harmonic spectrum
    amps_fem = harmonics_1sided(br_fem)
    amps_ana = harmonics_1sided(br_ana) if has_ana else None
    amps_meas = harmonics_1sided(meas_br_normalized) if has_meas else None

    n_series = 1 + (1 if has_ana else 0) + (1 if has_meas else 0)
    max_lens = [len(amps_fem)]
    if amps_ana is not None:
        max_lens.append(len(amps_ana))
    if amps_meas is not None:
        max_lens.append(len(amps_meas))
    max_ord = min(n_p * 8 + 2, *max_lens)
    orders = np.arange(max_ord)

    if n_series == 1:
        ax2.bar(orders, amps_fem[:max_ord], width=0.6, alpha=0.75, label="FEM")
    else:
        w = 0.7 / n_series
        idx = 0
        if has_ana:
            ax2.bar(orders + (idx - (n_series - 1) / 2) * w, amps_ana[:max_ord],
                    width=w, alpha=0.85, label="Analytical")
            idx += 1
        ax2.bar(orders + (idx - (n_series - 1) / 2) * w, amps_fem[:max_ord],
                width=w, alpha=0.65, label="FEM")
        idx += 1
        if has_meas:
            ax2.bar(orders + (idx - (n_series - 1) / 2) * w, amps_meas[:max_ord],
                    width=w, alpha=0.55, color="tab:green", label="Measured")
    ax2.set_xlabel("Spatial harmonic order")
    ax2.set_ylabel("Amplitude (T)")
    if n_p < max_ord:
        fund_amp = amps_fem[n_p]
        if amps_ana is not None:
            fund_amp = max(fund_amp, amps_ana[n_p])
        ax2.annotate(f"n={n_p}\n(fund.)",
                     xy=(n_p, fund_amp), xycoords="data",
                     xytext=(max_ord * 0.4, fund_amp * 0.85),
                     textcoords="data", fontsize=7, color="red", ha="center",
                     arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=3)
    return _savefig(fig, fig_dir, f"br_waveform_{_sanitize(name)}.png")


def _field_map_worker(job: dict) -> dict:
    """Module-level worker for process-parallel field map generation.

    Runs the full pipeline in one subprocess: FEM solve → rasterise → plot → save.
    """
    import time as _time
    t0 = _time.perf_counter()
    name = job["name"]
    fig_dir = job["fig_dir"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np

        from phasesweep.fem_field import rasterise_cross_section as _raster
        from phasesweep.fem_field import solve_field_fem as _solve
        from phasesweep.motor import Motor as _Motor
        from phasesweep.solver_params import prepare_fem as _prepare_fem

        motor = _Motor.from_dict(job["motor_dict"])
        params = _prepare_fem(motor)
        geo = params.geometry

        _theta, _Br, mesh, gfu = _solve(
            geo=geo, n_p=params.n_p, B_rem=params.B_rem,
            mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
            nonlinear=True, return_full=True,
            n_slots=geo.n_slots, alpha_p=params.alpha_p,
        )

        xi, yi, Az, Bmag = _raster(mesh, gfu, n_grid=150, r_bound=geo.r_outer)

        fig, axes = _plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"FEM field — {name}", fontsize=11)

        ax = axes[0]
        levels = _np.linspace(_np.nanmin(Az), _np.nanmax(Az), 30)
        ax.contourf(xi * 1e3, yi * 1e3, Az, levels=levels, cmap="RdBu_r")
        ax.contour(xi * 1e3, yi * 1e3, Az, levels=levels, colors="k", linewidths=0.3)
        ax.set_aspect("equal")
        ax.set_title("Vector potential A_z")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax = axes[1]
        im = ax.pcolormesh(xi * 1e3, yi * 1e3, Bmag, cmap="hot", shading="auto",
                           vmin=0, vmax=min(_np.nanmax(Bmag), 2.0))
        ax.set_aspect("equal")
        ax.set_title("|B| (T)")
        ax.set_xlabel("x (mm)")
        fig.colorbar(im, ax=ax, label="|B| (T)")

        import re as _re
        sanitized = _re.sub(r"[^a-z0-9_]", "_", name.lower().replace(" ", "_")).strip("_")
        filename = f"field_{sanitized}.png"
        fig.savefig(str(Path(fig_dir) / filename), dpi=150, bbox_inches="tight")
        _plt.close(fig)

        elapsed = _time.perf_counter() - t0
        return {"tag": name, "status": "OK", "filename": filename, "elapsed_s": elapsed}
    except Exception as e:
        elapsed = _time.perf_counter() - t0
        return {"tag": name, "status": "ERROR", "filename": None,
                "elapsed_s": elapsed, "error_msg": str(e)}


def fig_br_with_peak_marker(
    name: str,
    theta: np.ndarray, br: np.ndarray,
    published_peak: float, source_label: str,
    fig_dir: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.degrees(theta), br, lw=1.2, label="FEM B_r(θ)")
    ax.axhline(published_peak, color="tab:red", ls="--", lw=1.5,
               label=f"Published B_ag_peak = {published_peak:.3f} T")
    ax.axhline(-published_peak, color="tab:red", ls="--", lw=1.5, alpha=0.4)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("B_r (T)")
    ax.set_xlim(0, 360)
    ax.set_title(f"{name} — computed waveform vs published peak\n({source_label})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, fig_dir, f"br_peak_{_sanitize(name)}.png")


def fig_mtpa_gamma(
    name: str,
    I_curve: list[float], gamma_curve: list[float],
    meas_points: list[tuple[float, float, str]],
    fig_dir: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(I_curve, gamma_curve, lw=1.5, label="Computed MTPA γ(I)")
    for I_val, g_val, label in meas_points:
        ax.plot(I_val, g_val, "o", ms=8, color="tab:red", zorder=5)
        ax.annotate(f"{label}\n({g_val:.1f}°)", (I_val, g_val),
                    textcoords="offset points", xytext=(8, 8), fontsize=8,
                    color="tab:red")
    ax.set_xlabel("Stator current I_s (A)")
    ax.set_ylabel("MTPA angle γ (deg)")
    ax.set_title(f"{name} — MTPA angle vs current", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _savefig(fig, fig_dir, f"mtpa_gamma_{_sanitize(name)}.png")


def fig_deviation_dotplot(
    name: str,
    rows: list[ComparisonRow],
    fig_dir: Path,
) -> str:
    """Deviation dot plot: each quantity as a horizontal point ± tolerance band."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8, max(2.5, n * 0.7 + 1.0)))

    y_pos = np.arange(n)
    for i, r in enumerate(rows):
        color = "tab:green" if r.passed else "tab:red"
        # Tolerance band as horizontal error bar
        ax.barh(i, 2 * r.tol_pct, left=-r.tol_pct, height=0.4,
                color=color, alpha=0.12, edgecolor="none")
        ax.plot([-r.tol_pct, r.tol_pct], [i, i], color=color, lw=1.5, alpha=0.4)
        # Data point
        ax.plot(r.rel_pct, i, "o", color=color, ms=9, zorder=5)
        # Annotation: delta and absolute values
        label = f"{r.rel_pct:+.1f}%  ({r.val_b:.4g} vs {r.val_a:.4g})"
        offset = 8 if r.rel_pct >= 0 else -8
        ha = "left" if r.rel_pct >= 0 else "right"
        ax.annotate(label, (r.rel_pct, i), textcoords="offset points",
                    xytext=(offset, 0), fontsize=7.5, va="center", ha=ha,
                    color=color)

    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r.quantity}\n({r.model_a} vs {r.model_b})"
                        for r in rows], fontsize=8)
    ax.set_xlabel("Deviation from reference (%)")
    ax.set_title(f"{name} — validation deviations", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    return _savefig(fig, fig_dir, f"deviation_{_sanitize(name)}.png")


# ---------------------------------------------------------------------------
# Report rendering helpers
# ---------------------------------------------------------------------------

def fmt_val(v, unit="") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 0.01:
            return f"{v:.4e}{' ' + unit if unit else ''}"
        return f"{v:.4f}{' ' + unit if unit else ''}"
    return str(v)


def _render_bound_rows(rows: list[ComparisonRow], w) -> None:
    if not rows:
        return
    w("**Bound checks:**")
    w("")
    for r in rows:
        symbol = "✓" if r.passed else "✗"
        w(f"- {symbol} computed {r.model_b} `{r.quantity}` = {r.val_b:.4g} "
          f"vs {r.val_a:.4g} (margin {r.rel_pct:+.1f}%)")
    w("")


def _render_curve_rows(rows: list[ComparisonRow], w) -> None:
    if not rows:
        return
    w("**Curve comparisons:**")
    w("")
    for r in rows:
        symbol = "✓" if r.passed else "✗"
        extra = " [extrapolated]" if r.extrapolated else ""
        w(f"- {symbol} `{r.quantity}`: computed {r.val_b:.4g} vs published {r.val_a:.4g} "
          f"(Δ {r.rel_pct:.1f}%, tol {r.tol_pct:.0f}%){extra}")
    w("")


def _render_delta_rows(rows: list[ComparisonRow], w) -> None:
    if not rows:
        return
    w("**Delta comparisons:**")
    w("")
    w("```")
    w(format_table(rows))
    w("```")
    w("")


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def _run_sensitivity(
    motors: dict[str, Motor],
    all_results: dict[str, list[RunResult]],
    workers: int,
    cache_dir: str | None,
    timing_log: list[tuple[str, float]],
) -> dict[str, dict[str, dict[str, list[float | None]]]]:
    """Build and execute sensitivity sweep jobs. Returns labelled grids per motor."""
    import dataclasses as dc

    from phasesweep.parallel import execute_parallel

    t_phase = time.perf_counter()

    sens_motors = {
        n: m for n, m in motors.items()
        if n in MOTOR_DATA_MAP or n.startswith("Zhu & Howe")
    }
    sens_params = list(SENSITIVITY_PARAMS.keys())
    motor_eligible: dict[str, list[str]] = {}
    motor_grids: dict[str, dict[str, dict[str, list[float | None]]]] = {}
    baseline_jobs: list[dict] = []
    perturbation_jobs: list[dict] = []

    t_build = time.perf_counter()
    for name, motor in sens_motors.items():
        baseline = all_results.get(name, [])
        if not baseline:
            continue
        eligible = [r.model for r in baseline
                    if r.status == "OK" and r.metrics and r.model in _SENSITIVITY_METRIC]
        if not eligible:
            continue
        motor_eligible[name] = eligible

        baseline_motor = dc.replace(motor, psi_f=None) if motor.B_rem else motor
        for mk in eligible:
            kw = {"nonlinear": True} if mk == "fem" else {}
            if mk == "drive_sim":
                try:
                    kw["sim_plan"] = plan_sim(prepare_drive_sim(baseline_motor))
                except ValueError:
                    continue
            baseline_jobs.append({
                "motor_dict": baseline_motor.to_dict(),
                "model_key": mk, "config_kw": kw,
                "tag": f"baseline:{name}:{mk}",
                "motor_name": name, "is_baseline": True,
            })

        grid: dict[str, dict[str, list[float | None]]] = {}
        for param in sens_params:
            pdef = SENSITIVITY_PARAMS[param]
            deltas = pdef["deltas"]
            curves: dict[str, list[float | None]] = {mk: [None] * len(deltas) for mk in eligible}
            grid[param] = curves

            for di, d in enumerate(deltas):
                if d == 0.0:
                    for mk in eligible:
                        curves[mk][di] = 0.0
                    continue
                perturbed = _perturb_motor(motor, param, d)
                if perturbed is None:
                    continue
                elig = check_model_eligibility(perturbed)
                for mk in eligible:
                    if elig.get(mk) == "eligible":
                        config_kw = {"nonlinear": True} if mk == "fem" else {}
                        if mk == "drive_sim":
                            try:
                                config_kw["sim_plan"] = plan_sim(
                                    prepare_drive_sim(perturbed))
                            except ValueError:
                                continue
                        perturbation_jobs.append({
                            "motor_dict": perturbed.to_dict(),
                            "model_key": mk, "config_kw": config_kw,
                            "tag": f"{name}:{param}:{di}:{mk}",
                            "motor_name": name, "param": param,
                            "delta_idx": di, "is_baseline": False,
                        })
        motor_grids[name] = grid

    all_sens_jobs = baseline_jobs + perturbation_jobs
    timing_log.append(("  sens:build jobs", time.perf_counter() - t_build))
    print(f"Sensitivity: {len(baseline_jobs)} baseline + {len(perturbation_jobs)} perturbation = "
          f"{len(all_sens_jobs)} total jobs across {len(motor_eligible)} motors", flush=True)

    t_exec = time.perf_counter()
    baseline_vals: dict[str, dict[str, float]] = {n: {} for n in motor_eligible}
    per_job_timing: list[tuple[str, float]] = []

    def _on_sens_complete(result: dict, done: int, total: int) -> None:
        tag = result.get("tag", "")
        elapsed = result.get("elapsed_s", 0)
        mk = result["model_key"]
        motor_name = result.get("motor_name", "")
        per_job_timing.append((tag, elapsed))
        if result.get("is_baseline"):
            if result["status"] == "OK" and result["metrics"]:
                val = _sensitivity_metric(mk, result["metrics"])
                if val is not None:
                    baseline_vals[motor_name][mk] = val
            print(f"  [{done}/{total}] baseline {motor_name}:{mk} ({elapsed:.2f}s)", flush=True)
            return
        print(f"  [{done}/{total}] {tag} → {result['status']} ({elapsed:.2f}s)", flush=True)

    all_sens_results = execute_parallel(
        all_sens_jobs, workers=workers, cache_dir=cache_dir,
        on_complete=_on_sens_complete,
    )
    timing_log.append(("  sens:execute", time.perf_counter() - t_exec))

    t_post = time.perf_counter()
    for result in all_sens_results:
        if result.get("is_baseline"):
            continue
        if result["status"] != "OK" or not result.get("metrics"):
            continue
        motor_name = result.get("motor_name", "")
        mk = result["model_key"]
        param = result.get("param", "")
        di = result.get("delta_idx", -1)
        grid = motor_grids.get(motor_name, {})
        curves = grid.get(param, {})
        if mk not in curves or di < 0:
            continue
        base_val = baseline_vals.get(motor_name, {}).get(mk)
        if base_val is None:
            continue
        val = _sensitivity_metric(mk, result["metrics"])
        if val is not None:
            pct = (val - base_val) / abs(base_val) * 100 if abs(base_val) > 1e-12 else 0.0
            curves[mk][di] = pct

    sens_by_motor: dict[str, dict[str, dict[str, list[float | None]]]] = {}
    for name, grid in motor_grids.items():
        labeled_grid: dict[str, dict[str, list[float | None]]] = {}
        for param, curves in grid.items():
            labeled: dict[str, list[float | None]] = {}
            for mk, curve in curves.items():
                labeled[_SENSITIVITY_LABEL.get(mk, mk)] = curve
            labeled_grid[param] = labeled
        sens_by_motor[name] = labeled_grid
    timing_log.append(("  sens:post-process", time.perf_counter() - t_post))

    model_time: dict[str, float] = {}
    for tag, elapsed in per_job_timing:
        parts = tag.split(":")
        mk = parts[-1] if len(parts) >= 3 else parts[1] if len(parts) >= 2 else tag
        model_time[mk] = model_time.get(mk, 0) + elapsed
    for mk, total_t in sorted(model_time.items(), key=lambda x: -x[1]):
        timing_log.append((f"  sens:model:{mk}", total_t))

    timing_log.append(("Sensitivity", time.perf_counter() - t_phase))
    return sens_by_motor


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _compute_all(
    motors: dict[str, Motor],
    fig_dir: Path | None,
    workers: int,
    cache_dir: str | None,
    timing_log: list[tuple[str, float]],
) -> tuple[
    dict[str, list[RunResult]],        # all_results
    dict[str, RunResult],              # linear_fem_results
    dict[str, list[RunResult]],        # all_measured
    dict[str, str | None],             # field_map_filenames
    dict[str, dict[str, dict[str, list[float | None]]]],  # sens_by_motor
]:
    """Run all models, field maps, and sensitivity in bulk.  Returns data dicts."""
    computed_models = [k for k, v in MODEL_REGISTRY.items() if v.source == "computed"]

    # --- Model runs ---
    t_phase = time.perf_counter()
    all_results: dict[str, list[RunResult]] = {}
    linear_fem_results: dict[str, RunResult] = {}

    for name, m in motors.items():
        elig = check_model_eligibility(m)
        motor_results: list[RunResult] = []

        for mk in computed_models:
            if elig.get(mk) != "eligible":
                continue
            kw = {"nonlinear": True} if mk == "fem" else {}
            if mk == "drive_sim":
                try:
                    kw["sim_plan"] = plan_sim(prepare_drive_sim(m))
                except ValueError:
                    continue
            result = run_model(m, mk, **kw)
            if result is not None:
                motor_results.append(result)

        if elig.get("fem") == "eligible":
            lin_r = run_model(m, "fem", nonlinear=False)
            if lin_r and lin_r.status == "OK":
                linear_fem_results[name] = lin_r

        all_results[name] = motor_results

    timing_log.append(("Model runs", time.perf_counter() - t_phase))

    # --- Load measured data ---
    t_phase = time.perf_counter()
    all_measured: dict[str, list[RunResult]] = {}
    for name, m in motors.items():
        meas = load_measured_data(name, m)
        if meas:
            all_measured[name] = meas
    timing_log.append(("Load measured data", time.perf_counter() - t_phase))

    # --- Field maps (parallel) ---
    field_map_filenames: dict[str, str | None] = {}
    if fig_dir:
        t_field = time.perf_counter()
        field_jobs = []
        for name, results in all_results.items():
            fem = next((r for r in results if r.model == "fem"), None)
            if fem is None or fem.status != "OK":
                continue
            field_jobs.append({
                "name": name,
                "motor_dict": motors[name].to_dict(),
                "fig_dir": str(fig_dir),
            })

        if field_jobs:
            def _on_field(result, done, total):
                status = result["status"]
                elapsed = result.get("elapsed_s", 0)
                print(f"  [{done}/{total}] field map {result['tag']} "
                      f"→ {status} ({elapsed:.1f}s)", flush=True)

            field_results = execute_generic(
                _field_map_worker, field_jobs,
                workers=workers, cache_dir=cache_dir,
                on_complete=_on_field,
            )
            for fr in field_results:
                if fr["status"] == "OK" and fr["filename"]:
                    field_map_filenames[fr["tag"]] = fr["filename"]
                elif fr["status"] == "ERROR":
                    print(f"WARNING: field map for {fr['tag']} failed: "
                          f"{fr.get('error_msg', '?')}", file=sys.stderr)

            timing_log.append(("Field maps", time.perf_counter() - t_field))
            for fr in field_results:
                timing_log.append((f"  field:{fr['tag']}", fr.get("elapsed_s", 0)))

    # --- Sensitivity analysis (parallel) ---
    sens_by_motor = _run_sensitivity(
        motors, all_results, workers, cache_dir, timing_log,
    )

    return all_results, linear_fem_results, all_measured, field_map_filenames, sens_by_motor


# ---------------------------------------------------------------------------
# Per-motor rendering
# ---------------------------------------------------------------------------

def _render_motor_section(
    name: str,
    motor: Motor,
    results: list[RunResult],
    linear_fem: RunResult | None,
    measured: list[RunResult],
    field_map_fn: str | None,
    sensitivity: dict[str, dict[str, list[float | None]]] | None,
    fig_dir: Path | None,
    w,
) -> None:
    """Render all content for one motor under a ### heading."""
    geo = motor.geometry

    # --- Header ---
    w(f"### {name}")
    w("")
    w(f"- **Topology:** {geo.topology}")
    w(f"- **Poles/slots:** {2*motor.n_p}p / {geo.n_slots or 'smooth bore'}s")
    w(f"- **Radii (mm):** r_outer={geo.r_outer*1e3:.1f}, r_stator={geo.r_stator*1e3:.1f}, "
      f"r_magnet={geo.r_magnet*1e3:.1f}, r_rotor={geo.r_rotor*1e3:.1f}")
    w("")

    # Geometry figure
    if fig_dir:
        geo_fn = f"geometry_{_sanitize(name)}.png"
        slots_label = f"{geo.n_slots}s" if geo.n_slots else "smooth"
        plot_geometry(
            geo, n_p=motor.n_p, alpha_p=motor.alpha_p,
            title=f"{name} — {2*motor.n_p}p/{slots_label}",
            output_dir=fig_dir, filename=geo_fn,
        )
        w(f"![Geometry — {name}](figures/{geo_fn})")
        w("")

    # --- Model outputs ---
    if results:
        w("#### Model Outputs")
        w("")
        for result in results:
            mk = result.model
            w(f"**{mk}** (elapsed: {result.elapsed_s:.3f}s)")
            w("")
            if result.status == "ERROR":
                w(f"> **ERROR:** {result.error_msg}")
                w("")
            warn_list = (result.metrics or {}).get("_warnings", [])
            if warn_list:
                for wm in warn_list:
                    w(f"> **Warning:** {wm}")
                w("")
            scalars = {k: v for k, v in (result.metrics or {}).items()
                      if isinstance(v, (int, float))
                      and not (k == "thd_pct" and mk == "analytical")}
            if scalars:
                w("| Quantity | Value |")
                w("|----------|-------|")
                for k, v in sorted(scalars.items()):
                    w(f"| {k} | {fmt_val(v)} |")
                w("")

    # --- Analytical vs FEM ---
    ana = next((r for r in results if r.model == "analytical"), None)
    fem = next((r for r in results if r.model == "fem"), None)
    if ana and fem and ana.status == "OK" and fem.status == "OK":
        b1_ana = (ana.metrics or {}).get("fundamental", 0)
        b1_fem = (fem.metrics or {}).get("fundamental", 0)

        if _analytical_valid(motor):
            ref = abs(b1_ana) if abs(b1_ana) > 1e-12 else 1.0
            delta = abs(b1_fem - b1_ana) / ref * 100
            speed = fem.elapsed_s / max(ana.elapsed_s, 1e-6)
            w("#### Analytical vs FEM")
            w("")
            w("| Analytical B1 (T) | FEM B1 (T) | Delta | Speed ratio |")
            w("|-------------------|------------|-------|-------------|")
            w(f"| {b1_ana:.4f} | {b1_fem:.4f} | {delta:.1f}% | {speed:.0f}x |")
            w("")

    # Waveform figure — generate whenever FEM has data (outside ana+fem guard)
    if fem and fem.status == "OK":
        fem_m = fem.metrics or {}
        if fig_dir and "theta_list" in fem_m:
            ana_theta = ana_br = None
            if _analytical_valid(motor) and ana and ana.status == "OK":
                ana_m = ana.metrics or {}
                if "theta_list" in ana_m:
                    ana_theta = np.array(ana_m["theta_list"])
                    ana_br = np.array(ana_m["B_r_list"])

            fem_br = np.array(fem_m["B_r_list"])
            # Load measured back-EMF waveform (CREATOR-style CSV)
            meas_theta = meas_br = None
            entry = MOTOR_DATA_MAP.get(name)
            if entry and entry[1]:
                meas_data = _load_backemf_waveform(
                    ROOT / entry[1], motor.n_p, fem_br,
                )
                if meas_data is not None:
                    meas_theta, meas_br = meas_data

            # Scalar back-EMF markers — use analytical B₁/E₀ ratio (both
            # are computed there). The FEM branch below never fires for E₀:
            # fem doesn't produce backemf_fundamental, so markers are skipped
            # when analytical is unavailable
            scalar_markers: list[tuple[float, str]] = []
            b1_ref = emf_ref = 0.0
            if ana and ana.status == "OK":
                ana_m2 = ana.metrics or {}
                b1_ref = ana_m2.get("fundamental", 0)
                emf_ref = ana_m2.get("backemf_fundamental") or 0
            if not (b1_ref > 0 and emf_ref > 0):
                b1_ref = fem_m.get("fundamental", 0)
                emf_ref = fem_m.get("backemf_fundamental") or 0
            if b1_ref > 0 and emf_ref > 0:
                scale = b1_ref / emf_ref
                for mr in measured:
                    mr_m = mr.metrics or {}
                    for key in ("backemf_fundamental", "backemf_peak"):
                        emf_val = mr_m.get(key)
                        if emf_val is not None and isinstance(emf_val, (int, float)):
                            b_eq = emf_val * scale
                            scalar_markers.append(
                                (b_eq, f"E₀={emf_val:.1f} V → B₁≈{b_eq:.3f} T")
                            )
                            break  # one marker per measured result

            fn = fig_br_waveform_and_harmonics(
                name, ana_theta, ana_br,
                np.array(fem_m["theta_list"]), fem_br,
                motor.n_p, fig_dir,
                meas_theta_deg=meas_theta,
                meas_br_normalized=meas_br,
                scalar_br_markers=scalar_markers or None,
            )
            w(f"![B_r waveform — {name}](figures/{fn})")
            w("")
            if meas_br is not None:
                w("> Measured back-EMF (Phase U) normalized to B_r via"
                  " fundamental matching.")
                w("")
            if scalar_markers:
                w("> Published back-EMF converted to B_r equivalent"
                  " using B₁/E₀ ratio.")
                w("")

    # --- Linear vs nonlinear FEM ---
    if fem and linear_fem and fem.status == "OK" and linear_fem.status == "OK":
        b1_nl = (fem.metrics or {}).get("fundamental", 0)
        b1_lin = (linear_fem.metrics or {}).get("fundamental", 0)
        ref = abs(b1_lin) if abs(b1_lin) > 1e-12 else 1.0
        delta_pct = abs(b1_nl - b1_lin) / ref * 100
        w("#### Linear vs Nonlinear FEM")
        w("")
        w("| B1 linear (T) | B1 nonlinear (T) | Delta |")
        w("|--------------|-----------------|-------|")
        w(f"| {b1_lin:.4f} | {b1_nl:.4f} | {delta_pct:.2f}% |")
        w("")

    # --- Cross-validation ---
    if measured:
        w("#### Cross-Validation")
        w("")
        if any("unvalidated" in (mr.source or "").lower() for mr in measured):
            w("> **⚠ Note:** Some reference data below is self-labeled as"
              " *unvalidated* (e.g. 1-D analytical estimates). Comparison"
              " failures against unvalidated references are expected.")
            w("")
        for mr in measured:
            src_label = mr.source
            qty_keys = [k for k in (mr.metrics or {}) if not k.startswith("_")]
            w(f"- **{mr.model}** ({src_label}): {', '.join(qty_keys)}")
        w("")

        all_cmp_rows: list[ComparisonRow] = []
        emitted_figs: set[str] = set()
        has_rated = any(r.model == "rated_torque" for r in results)

        for mr in measured:
            for cr in results:
                if (cr.model == "stall_torque" and mr.model == "torque_test"
                        and has_rated and mr.metrics
                        and "_curve_compare" in mr.metrics):
                    continue
                rows = compare_results(mr, cr)
                if not rows:
                    continue

                summary = diagnose_detailed([mr, cr])
                w(f"**{mr.model} vs {cr.model}:**")
                w("")
                _render_bound_rows(summary.bound_rows, w)
                _render_curve_rows(summary.curve_rows, w)
                _render_delta_rows(summary.delta_rows, w)
                w(f"**Diagnosis:** {format_diagnosis(summary)}")
                w("")
                all_cmp_rows.extend(rows)

                if not fig_dir:
                    continue

                cr_m = cr.metrics or {}
                mr_m = mr.metrics or {}
                if "I_curve" in cr_m and "gamma_curve_deg" in cr_m:
                    cc_meta = mr_m.get("_curve_compare", {})
                    meas_pts: list[tuple[float, float, str]] = []
                    for qkey, ref in cc_meta.items():
                        if ref.get("curve_y") == "gamma_curve_deg" and ref.get("at_x") is not None:
                            meas_val = mr_m.get(qkey)
                            if meas_val is not None:
                                meas_pts.append((ref["at_x"], meas_val, qkey))
                    if meas_pts and "mtpa_gamma" not in emitted_figs:
                        emitted_figs.add("mtpa_gamma")
                        fn = fig_mtpa_gamma(
                            name, cr_m["I_curve"], cr_m["gamma_curve_deg"],
                            meas_pts, fig_dir,
                        )
                        w(f"![MTPA gamma — {name}](figures/{fn})")
                        w("")

                if cr.model == "fem" and "B_r_list" in cr_m and "theta_list" in cr_m:
                    cc_meta = mr_m.get("_curve_compare", {})
                    for qkey, ref in cc_meta.items():
                        if ref.get("curve_y") == "B_r_list" and ref.get("extract") == "max":
                            pub_val = mr_m.get(qkey)
                            if pub_val is not None and "br_peak" not in emitted_figs:
                                emitted_figs.add("br_peak")
                                fn = fig_br_with_peak_marker(
                                    name,
                                    np.array(cr_m["theta_list"]),
                                    np.array(cr_m["B_r_list"]),
                                    pub_val, mr.source or "published",
                                    fig_dir,
                                )
                                w(f"![B_r with peak — {name}](figures/{fn})")
                                w("")

        if fig_dir and all_cmp_rows:
            seen_qty: set[str] = set()
            deviation_rows: list[ComparisonRow] = []
            for r in all_cmp_rows:
                if r.comparison_type in ("delta", "curve") and r.quantity not in seen_qty:
                    seen_qty.add(r.quantity)
                    deviation_rows.append(r)
            if deviation_rows:
                fn = fig_deviation_dotplot(name, deviation_rows, fig_dir)
                w(f"![Validation deviations — {name}](figures/{fn})")
                w("")

    # --- Field map ---
    if field_map_fn:
        w(f"![FEM field — {name}](figures/{field_map_fn})")
        w("")

    # --- Sensitivity ---
    if sensitivity:
        _render_sensitivity(name, motor, sensitivity, fig_dir, w)


def _render_sensitivity(
    name: str,
    motor: Motor,
    sens: dict[str, dict[str, list[float | None]]],
    fig_dir: Path | None,
    w,
) -> None:
    """Render sensitivity analysis for one motor."""
    w("#### Sensitivity Analysis")
    w("")

    geo = motor.geometry
    bore_od = geo.r_stator / geo.r_outer
    if bore_od < 0.55:
        max_delta = max(SENSITIVITY_PARAMS["OD"]["deltas"])
        bore_pct = max_delta * geo.r_outer / geo.r_stator * 100
        w(f"> **Note:** Bore/OD ratio = {bore_od:.2f}. "
          f"A +{max_delta*100:.0f}% OD change produces a "
          f"+{bore_pct:.0f}% bore change (fixed gap + magnet thickness).")
        w("")
    if not _analytical_valid(motor):
        w("> **Note:** B₁_fem may oppose torque direction under OD perturbation: "
          "slot openings grow with bore circumference (fixed slot geometry) "
          "worsening the Carter factor, while torque still rises because "
          "ψ_f integrates over a larger bore area.")
        w("")

    if not any(any(v is not None for v in c) for curves in sens.values() for c in curves.values()):
        w("*No sensitivity data (insufficient baseline results).*")
        w("")
        return

    # Drop B₁_analytical for motors where smooth-bore assumption is invalid
    if not _analytical_valid(motor):
        for param_curves in sens.values():
            param_curves.pop("B₁_analytical", None)

    if fig_dir:
        fn = fig_sensitivity(name, sens, fig_dir)
        w(f"![Sensitivity — {name}](figures/{fn})")
        w("")

    all_tracks = sorted(set().union(*(mc.keys() for mc in sens.values())))

    for param, model_curves in sens.items():
        pdef = SENSITIVITY_PARAMS[param]
        deltas = pdef["deltas"]
        delta_hdrs = [f"{d*100:+.0f}%" for d in deltas]
        w(f"**{pdef['label']}** (δ = {', '.join(delta_hdrs)})")
        w("")
        w("| Track | " + " | ".join(delta_hdrs) + " | Linearity |")
        w("|-------|" + "|".join(["------:" for _ in deltas]) + "|-----------|")

        for mk in all_tracks:
            curve = model_curves.get(mk, [])
            cells = []
            for v in curve:
                cells.append(f"{v:+.2f}" if v is not None else "—")
            pts = [(d, v) for d, v in zip(deltas, curve)
                   if v is not None and d != 0.0]
            if len(pts) >= 3 and max(abs(v) for _, v in pts) > 0.1:
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                slope, intercept = np.polyfit(xs, ys, 1)
                ss_res = np.sum((ys - (slope * xs + intercept))**2)
                ss_tot = np.sum((ys - np.mean(ys))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
                lin = "linear" if r2 > 0.999 else f"R²={r2:.4f}"
            else:
                lin = "—"
            w(f"| {mk} | " + " | ".join(cells) + f" | {lin} |")
        w("")

    w("**Summary — direction agreement at max perturbation:**")
    w("")
    w("| Parameter | " + " | ".join(all_tracks) + " | Agreement |")
    w("|-----------|" + "|".join(["------:" for _ in all_tracks]) + "|-----------|")

    for param, model_curves in sens.items():
        pdef = SENSITIVITY_PARAMS[param]
        cells = []
        flux_dirs: list[int] = []
        torque_dirs: list[int] = []
        for mk in all_tracks:
            curve = model_curves.get(mk, [])
            last = curve[-1] if curve else None
            if last is not None and last != 0.0:
                cells.append(f"{last:+.1f}%")
                if abs(last) >= 0.5:
                    d = 1 if last > 0 else -1
                    if mk.startswith("B"):
                        flux_dirs.append(d)
                    else:
                        torque_dirs.append(d)
            else:
                cells.append("—")
        all_dirs = flux_dirs + torque_dirs
        if len(all_dirs) >= 2 and len(set(all_dirs)) == 1:
            agree = "YES"
        elif len(all_dirs) < 2:
            agree = "—"
        elif (len(set(flux_dirs)) <= 1 and len(set(torque_dirs)) <= 1
              and flux_dirs and torque_dirs):
            # Flux and torque internally consistent but oppose each other
            f_arrow = "↑" if flux_dirs[0] > 0 else "↓"
            t_arrow = "↑" if torque_dirs[0] > 0 else "↓"
            agree = f"B₁{f_arrow} τ{t_arrow}"
        else:
            agree = "**NO**"
        w(f"| {pdef['label']} | " + " | ".join(cells) + f" | {agree} |")

    w("")


# ---------------------------------------------------------------------------
# Summary tables (cross-motor comparisons)
# ---------------------------------------------------------------------------

def _render_summary_tables(
    motors: dict[str, Motor],
    all_results: dict[str, list[RunResult]],
    linear_fem_results: dict[str, RunResult],
    w,
) -> None:
    """Render cross-motor comparison tables."""
    w("## 3. Cross-Motor Summary Tables")
    w("")

    # Analytical vs FEM
    w("### Analytical vs FEM — Air-Gap B₁")
    w("")
    w("| Motor | Analytical B1 (T) | FEM B1 (T) | Delta (%) | Elapsed ratio |")
    w("|-------|-------------------|------------|-----------|---------------|")
    for name, results in all_results.items():
        if not _analytical_valid(motors[name]):
            continue
        ana = next((r for r in results if r.model == "analytical"), None)
        fem = next((r for r in results if r.model == "fem"), None)
        if ana is None or fem is None:
            continue
        if ana.status != "OK" or fem.status != "OK":
            continue
        b1_ana = (ana.metrics or {}).get("fundamental", 0)
        b1_fem = (fem.metrics or {}).get("fundamental", 0)
        ref = abs(b1_ana) if abs(b1_ana) > 1e-12 else 1.0
        delta = abs(b1_fem - b1_ana) / ref * 100
        speed = fem.elapsed_s / max(ana.elapsed_s, 1e-6)
        w(f"| {name} | {b1_ana:.4f} | {b1_fem:.4f} | {delta:.1f}% | {speed:.0f}x slower |")
    w("")

    # Linear vs nonlinear FEM
    w("### Linear vs Nonlinear FEM")
    w("")
    w("| Motor | B1_linear (T) | B1_nonlinear (T) | Delta (%) |")
    w("|-------|--------------|-----------------|-----------|")
    for name, results in all_results.items():
        nl_fem = next((r for r in results if r.model == "fem"), None)
        lin_fem = linear_fem_results.get(name)
        if nl_fem is None or lin_fem is None:
            continue
        if nl_fem.status != "OK" or lin_fem.status != "OK":
            continue
        b1_nl = (nl_fem.metrics or {}).get("fundamental", 0)
        b1_lin = (lin_fem.metrics or {}).get("fundamental", 0)
        ref = abs(b1_lin) if abs(b1_lin) > 1e-12 else 1.0
        delta_pct = abs(b1_nl - b1_lin) / ref * 100
        w(f"| {name} | {b1_lin:.4f} | {b1_nl:.4f} | {delta_pct:.2f}% |")
    w("")

    # Rated torque
    w("### Rated Torque")
    w("")
    w("| Motor | tau_mtpa (Nm) | k_T (Nm/A) | k_T_rms (Nm/A_rms) | Saliency |")
    w("|-------|---------------|------------|---------------------|----------|")
    for name, results in all_results.items():
        st = next((r for r in results if r.model == "rated_torque"), None)
        if st is None or st.status != "OK":
            continue
        m = st.metrics
        motor = motors[name]
        if motor.L_d and motor.L_q:
            sal = f"salient (Lq/Ld={motor.L_q/motor.L_d:.2f})" if motor.L_q > motor.L_d else "non-salient"
        else:
            sal = "unknown"
        w(f"| {name} | {m['tau_mtpa']:.4f} | {m['k_T']:.4f} | {m['k_T_rms']:.4f} | {sal} |")
    w("")

    # Stall torque
    w("### Stall Torque")
    w("")
    w("| Motor | τ_stall (Nm) | I_stall (A) | τ_stall_em (Nm) | I_stall_em (A) | γ_opt (deg) | Sat. ratio | Warning |")
    w("|-------|-------------|-------------|----------------|----------------|-------------|------------|---------|")
    for name, results in all_results.items():
        st = next((r for r in results if r.model == "stall_torque"), None)
        if st is None or st.status != "OK":
            continue
        m = st.metrics
        sat_ratio = m.get("saturation_ratio")
        sat_str = f"{sat_ratio:.1f}×" if sat_ratio is not None else "—"
        warn_str = "⚠ YES" if m.get("saturation_warning") else "no"
        tau_em = m.get("tau_stall_electromagnetic")
        I_em = m.get("I_stall_electromagnetic")
        tau_em_str = f"{tau_em:.4f}" if tau_em is not None else "—"
        I_em_str = f"{I_em:.1f}" if I_em is not None else "—"
        w(f"| {name} | {m['tau_stall']:.4f} | {m['I_stall']:.2f} | "
          f"{tau_em_str} | {I_em_str} | "
          f"{m['gamma_opt_deg']:.1f} | {sat_str} | {warn_str} |")
    w("")

    # Drive simulation
    w("### Drive Simulation")
    w("")
    w("| Motor | tau_peak (Nm) | i_ss (A) | speed_droop (%) | t_settle (s) |")
    w("|-------|--------------|----------|-----------------|--------------|")
    for name, results in all_results.items():
        ds = next((r for r in results if r.model == "drive_sim"), None)
        if ds is None or ds.status != "OK":
            continue
        m = ds.metrics
        i_ss = m.get("i_ss")
        if i_ss is not None and np.isnan(i_ss):
            continue
        w(f"| {name} | {fmt_val(m.get('tau_peak'))} | {fmt_val(m.get('i_ss'))} | "
          f"{fmt_val(m.get('speed_droop'))} | {fmt_val(m.get('t_settle'))} |")
    w("")


# ---------------------------------------------------------------------------
# Main report orchestrator
# ---------------------------------------------------------------------------

def generate_report(fig_dir: Path | None = None, workers: int = 1) -> str:
    lines: list[str] = []
    w = lines.append
    timing_log: list[tuple[str, float]] = []
    t_report_start = time.perf_counter()

    # Set up disk mesh cache for parallel workers
    cache_dir: str | None = None
    if workers > 1 and fig_dir is not None:
        _cache_path = fig_dir.parent / ".mesh_cache"
        _cache_path.mkdir(parents=True, exist_ok=True)
        cache_dir = str(_cache_path)
        from phasesweep.fem_field import set_disk_cache_dir
        set_disk_cache_dir(cache_dir)

    now = datetime.now()
    w("# Phase-Sweep Validation Report")
    w("")
    w(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M')}")
    w(f"**Python:** {sys.version.split()[0]}")
    w("")

    # Build motor catalogue
    motors = build_motor_catalogue()
    # ===== COMPUTE ALL =====
    (all_results, linear_fem_results, all_measured,
     field_map_filenames, sens_by_motor) = _compute_all(
        motors, fig_dir, workers, cache_dir, timing_log,
    )

    # ===== RENDER =====
    t_render = time.perf_counter()

    # --- §1: Overview ---
    computed_models = [k for k, v in MODEL_REGISTRY.items() if v.source == "computed"]

    w("## 1. Overview")
    w("")
    w("### Reference Motor Catalogue")
    w("")
    w("| Motor | Topology | Poles | B_rem (T) | psi_f (Wb) | Slots | R_s (Ω) |")
    w("|-------|----------|-------|-----------|------------|-------|---------|")
    for name, m in motors.items():
        topo = m.geometry.topology
        poles = 2 * m.n_p
        b_rem = fmt_val(m.B_rem)
        psi = fmt_val(m.psi_f)
        slots = m.geometry.n_slots or "smooth"
        rs = fmt_val(m.R_s)
        w(f"| {name} | {topo} | {poles} | {b_rem} | {psi} | {slots} | {rs} |")
    w("")

    w("### Model Eligibility Matrix")
    w("")
    header = "| Motor | " + " | ".join(computed_models) + " |"
    sep = "|-------|" + "|".join(["-----" for _ in computed_models]) + "|"
    w(header)
    w(sep)
    for name, m in motors.items():
        elig = check_model_eligibility(m)
        cells = ["YES" if elig.get(mk) == "eligible" else "NO" for mk in computed_models]
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")

    # --- §2: Per-motor analysis ---
    w("## 2. Per-Motor Analysis")
    w("")

    for name, motor in motors.items():
        _render_motor_section(
            name, motor,
            results=all_results.get(name, []),
            linear_fem=linear_fem_results.get(name),
            measured=all_measured.get(name, []),
            field_map_fn=field_map_filenames.get(name),
            sensitivity=sens_by_motor.get(name),
            fig_dir=fig_dir,
            w=w,
        )

    # --- §3: Cross-motor summary tables ---
    _render_summary_tables(motors, all_results, linear_fem_results, w)

    # --- §4: Infrastructure ---
    n_computed = sum(1 for v in MODEL_REGISTRY.values() if v.source == "computed")
    n_measured = sum(1 for v in MODEL_REGISTRY.values() if v.source == "measured")
    w("## 4. Infrastructure")
    w("")
    w("| Item | Value |")
    w("|------|-------|")
    w(f"| Registered models | {len(MODEL_REGISTRY)} ({n_computed} computed, {n_measured} measured) |")
    w(f"| Reference motors tested | {len(motors)} |")
    total_runs = sum(len(r) for r in all_results.values())
    ok_runs = sum(1 for rs in all_results.values() for r in rs if r.status == "OK")
    err_runs = total_runs - ok_runs
    w(f"| Total model runs | {total_runs} ({ok_runs} OK, {err_runs} ERROR) |")
    n_data = sum(len(v) for v in all_measured.values())
    w(f"| Measured/published datasets | {n_data} files across {len(all_measured)} motors |")
    w("| Solver param validation | 5 factories (analytical, fem, drive_sim, rated_torque, stall_torque) |")
    w("| B_rem ↔ psi_f derivation | bidirectional, consistency-checked |")
    w("")

    inrunner_count = sum(1 for m in motors.values() if m.geometry.topology == "inrunner")
    outrunner_count = sum(1 for m in motors.values() if m.geometry.topology == "outrunner")
    w(f"- **Inrunner:** {inrunner_count} motors tested")
    w(f"- **Outrunner:** {outrunner_count} motors tested")
    w("")

    timing_log.append(("Render", time.perf_counter() - t_render))

    # --- Appendix: Timing ---
    t_total = time.perf_counter() - t_report_start
    timing_log.append(("**TOTAL**", t_total))

    w("## Appendix: Timing Analysis")
    w("")
    w(f"**Workers:** {workers}")
    w("")
    w("| Phase | Wall time (s) | % of total |")
    w("|-------|--------------|-----------|")
    for label, dt in timing_log:
        indent = label.startswith("  ")
        pct = dt / t_total * 100 if t_total > 0 else 0
        if indent:
            w(f"| {label} | {dt:.2f} | {pct:.1f}% |")
        else:
            w(f"| {label} | **{dt:.2f}** | **{pct:.1f}%** |")
    w("")

    w("---")
    w("*Report generated by `scripts/validation_report.py` — phase-sweep*")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase-sweep validation report")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel workers for sensitivity sweep (default: 1 = serial)")
    args = parser.parse_args()

    out_dir = ROOT / "output" / "validation_report"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = generate_report(fig_dir=fig_dir, workers=args.workers)
    out_path = out_dir / "report.md"
    out_path.write_text(report)
    print(f"Report written to {out_path}")
    print(f"Figures in {fig_dir}")
    print()
    print(report)
