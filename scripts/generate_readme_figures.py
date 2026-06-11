"""Generate three README figures.

1. Hero FEM cross-section: CREATOR 4p/6s inrunner, nonlinear |B| field map
2. Back-EMF waveform: CREATOR measured oscilloscope data vs analytical fundamental
3. Validation chart: Zhu & Howe FEM vs analytical (solver verif.) + CREATOR vs measured (model val.)

Usage:
    uv run python scripts/generate_readme_figures.py
    uv run python scripts/generate_readme_figures.py --fig 1  # single figure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE   = "#1E6EBF"
ORANGE = "#E07B00"
GREEN  = "#1A8F3C"
RED    = "#C0392B"
GRAY   = "#888888"


# ---------------------------------------------------------------------------
# Figure 1: FEM cross-section hero — CREATOR benchmark motor
# ---------------------------------------------------------------------------

def fig_fem_hero() -> Path:
    """CREATOR 4p/6s inrunner, nonlinear |B| field map."""
    from phasesweep.configs import load_motor
    from phasesweep.fem_field import rasterise_cross_section, solve_field_fem
    from phasesweep.solver_params import prepare_fem

    motor = load_motor(ROOT / "motors" / "creator_case_pmsm.toml")
    params = prepare_fem(motor)
    geo = params.geometry

    print("Figure 1: running nonlinear FEM for CREATOR (mesh likely cached)…")
    _theta, _Br, mesh, gfu = solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        nonlinear=True, return_full=True,
        alpha_p=params.alpha_p,
    )
    print("  rasterising…")
    xi, yi, Az, Bmag = rasterise_cross_section(mesh, gfu, n_grid=400, r_bound=geo.r_outer)

    vmax = float(np.nanpercentile(Bmag[~np.isnan(Bmag)], 99))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    im = ax.pcolormesh(
        xi * 1e3, yi * 1e3, Bmag,
        cmap="hot", shading="gouraud", vmin=0, vmax=vmax,
    )

    # Flux tubes — fewer levels for cleaner result
    levels = np.linspace(float(np.nanmin(Az)), float(np.nanmax(Az)), 18)
    ax.contour(xi * 1e3, yi * 1e3, Az, levels=levels,
               colors="white", linewidths=0.4, alpha=0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("|B| (T)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("4-pole 6-slot PMSM — 2D FEM magnetic field (nonlinear iron)",
                 color="white", fontsize=11, pad=8)

    path = OUT_DIR / "fem_cross_section.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: Back-EMF waveform — CREATOR measured vs analytical
# ---------------------------------------------------------------------------

def fig_backemf_waveform() -> Path:
    """CREATOR back-EMF at 2000 RPM: measured oscilloscope data vs analytical fundamental."""
    from phasesweep.configs import load_motor
    from phasesweep.registry import _run_analytical_impl
    from phasesweep.sweep_types import RunConfig

    print("Figure 2: loading CREATOR back-EMF measurement data…")

    csv_path = (ROOT / "data" / "creator_case_pmsm"
                / "PM_synchronous_motor" / "Measurement_results"
                / "No_load_tests" / "Back_emf.csv")
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    angle_mech = raw[:, 0]   # degrees mechanical
    phase_u    = raw[:, 1]   # volts, Phase U

    n_p = 2  # CREATOR is 4-pole

    # Extract measured fundamental via FFT
    N = len(phase_u)
    fft = np.fft.fft(phase_u)
    fund_complex = fft[n_p] * 2 / N
    meas_fund_amp = float(abs(fund_complex))
    meas_fund_phase = float(np.angle(fund_complex))  # rad, in mech-angle frame

    # Analytical fundamental from the model
    motor = load_motor(ROOT / "motors" / "creator_case_pmsm.toml")
    cfg = RunConfig(motor=motor, model="analytical", n_theta=720)
    ana = _run_analytical_impl(cfg)
    ana_fund_amp = float(ana["backemf_fundamental"])

    # Convert to electrical angle; show two electrical periods
    angle_elec = angle_mech * n_p              # degrees
    two_periods = angle_elec <= 720.0
    theta_e = angle_elec[two_periods]
    v_meas  = phase_u[two_periods]

    # Reconstruct fundamentals using cos (FFT convention: angle=0 → cosine)
    theta_e_rad = np.deg2rad(theta_e)
    # fund_phase is already in the right frame for exp(j·n_p·θ_mech) = exp(j·θ_elec)
    fund_phase = meas_fund_phase  # no n_p scaling needed
    v_ana = ana_fund_amp * np.cos(theta_e_rad + fund_phase)
    v_meas_fund = meas_fund_amp * np.cos(theta_e_rad + fund_phase)

    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=150)
    ax.plot(theta_e, v_meas, lw=1.0, color=GRAY, alpha=0.85,
            label="Measured (Phase U, 2000 RPM)")
    ax.plot(theta_e, v_ana, lw=2.0, color=BLUE,
            label=f"Analytical fundamental  ({ana_fund_amp:.1f} V)")
    ax.plot(theta_e, v_meas_fund, lw=1.5, ls="--", color=ORANGE,
            label=f"Measured fundamental  ({meas_fund_amp:.1f} V)")

    ax.set_xlabel("Electrical angle (deg)", fontsize=11)
    ax.set_ylabel("Phase back-EMF (V)", fontsize=11)
    ax.set_title("CREATOR benchmark — no-load back-EMF at 2000 RPM", fontsize=11)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.set_xlim(0, 720)
    ax.set_xticks(range(0, 721, 90))
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "backemf_waveform.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Figure 3: Validation chain — CREATOR only
# ---------------------------------------------------------------------------

def fig_validation_chart() -> Path:
    """Two-step validation chain: solver verification (Zhu & Howe inrunner) then
    model validation (CREATOR measured back-EMF)."""
    from phasesweep.configs import load_motor
    from phasesweep.fem_field import harmonics_1sided, zhu_howe_Br
    from phasesweep.geometry import default_inrunner
    from phasesweep.registry import _run_analytical_impl
    from phasesweep.sweep_types import RunConfig

    # --- Row 1: solver verification — Zhu & Howe 4-pole inrunner, α_p = 1 ---
    # Load the cached FEM result produced by scripts/capture_reference_br.py.
    print("Figure 3: loading Zhu & Howe inrunner B_r (cached FEM)…")
    npz_path = ROOT / "output" / "reference_br" / "zhu_inrunner.npz"
    if not npz_path.exists():
        sys.exit(
            f"missing {npz_path} — run scripts/capture_reference_br.py first"
        )
    data = np.load(npz_path)
    theta = data["theta"]
    Br_fem = data["B_r"]

    geo_zhu = default_inrunner()
    n_p_zhu, B_rem_zhu, mu_r_zhu = 4, 1.2, 1.05
    Br_ana_zhu = zhu_howe_Br(
        theta, n_p_zhu, B_rem_zhu,
        r_stator=geo_zhu.r_stator, r_magnet=geo_zhu.r_magnet, r_rotor=geo_zhu.r_rotor,
        mu_r_pm=mu_r_zhu, alpha_p=1.0,
    )
    amps_fem = harmonics_1sided(Br_fem)
    amps_ana = harmonics_1sided(Br_ana_zhu)
    B1_fem = float(amps_fem[n_p_zhu])
    B1_ana = float(amps_ana[n_p_zhu])
    fem_vs_ana_pct = (B1_fem - B1_ana) / B1_ana * 100

    # --- Row 2: model validation — CREATOR analytical vs oscilloscope ---
    print("  running CREATOR analytical…")
    motor = load_motor(ROOT / "motors" / "creator_case_pmsm.toml")
    cfg = RunConfig(motor=motor, model="analytical", n_theta=720)
    ana = _run_analytical_impl(cfg)
    ana_backemf = float(ana["backemf_fundamental"])
    ana_vs_meas_pct = (ana_backemf - 47.37) / 47.37 * 100

    rows = [
        # (stage_label, metric_label, computed, ref, pct, tol, ref_source)
        ("Solver verification\n(Zhu & Howe 4p inrunner, α_p=1)",
         "FEM B₁ vs analytical",
         B1_fem, B1_ana, fem_vs_ana_pct, 1.0,
         "Zhu & Howe (1993) closed-form"),
        ("Model validation\n(CREATOR 4p/6s ferrite benchmark)",
         "Analytical back-EMF vs oscilloscope",
         ana_backemf, 47.37, ana_vs_meas_pct, 5.0,
         "Measured (oscilloscope, 2000 rpm)"),
    ]

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, 3.0), dpi=150)
    fig.subplots_adjust(left=0.22, right=0.70, top=0.84, bottom=0.22)

    tol_color = "#d9f0da"

    for i, (stage, metric, computed, ref, pct, tol, source) in enumerate(rows):
        y = n - 1 - i
        dot_color  = GREEN if abs(pct) <= tol else RED
        band_color = tol_color

        ax.barh(y,  tol, left=0, height=0.55, color=band_color, zorder=1, alpha=0.85, linewidth=0)
        ax.barh(y, -tol, left=0, height=0.55, color=band_color, zorder=1, alpha=0.85, linewidth=0)
        ax.plot([-tol, tol], [y, y], color=dot_color, lw=1.0, alpha=0.40, zorder=2)
        ax.plot([0, 0], [y - 0.28, y + 0.28], color=GRAY, lw=0.7, zorder=2)
        ax.plot(pct, y, "o", color=dot_color, ms=10, zorder=3, markeredgewidth=0)

        sign = "+" if pct >= 0 else ""
        ax.annotate(f"{sign}{pct:.2f}%",
                    xy=(1.02, (y + 0.5) / n), xycoords=("axes fraction", "axes fraction"),
                    va="center", ha="left", fontsize=11, color=dot_color, fontweight="bold")

        unit = "V" if "EMF" in metric else "T"
        ax.annotate(f"{computed:.4g} vs {ref:.4g} {unit}  ({source})",
                    xy=(1.18, (y + 0.5) / n), xycoords=("axes fraction", "axes fraction"),
                    va="center", ha="left", fontsize=8.5, color=GRAY)

    ytick_labels = [f"{r[0]}\n{r[1]}" for r in reversed(rows)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(ytick_labels, fontsize=9)

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5, zorder=0)
    ax.set_xlabel("Deviation (%)", fontsize=10)
    ax.set_title("CREATOR benchmark — verification and validation",
                 fontsize=11, pad=10)

    ax.set_xlim(-(max(rows, key=lambda r: r[5])[5] + 3),
                 max(rows, key=lambda r: r[5])[5] + 3)
    ax.grid(axis="x", alpha=0.20, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate("Error", xy=(1.02, 1.06), xycoords="axes fraction",
                fontsize=9, color="black", fontweight="bold", ha="left")
    ax.annotate("Computed vs Reference", xy=(1.18, 1.06), xycoords="axes fraction",
                fontsize=9, color="black", fontweight="bold", ha="left")

    path = OUT_DIR / "multimotor_validation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate README figures")
    parser.add_argument("--fig", type=int, choices=[1, 2, 3],
                        help="Generate only this figure (default: all)")
    args = parser.parse_args()

    fns = {1: fig_fem_hero, 2: fig_backemf_waveform, 3: fig_validation_chart}
    targets = [args.fig] if args.fig else [1, 2, 3]
    for n in targets:
        fns[n]()

    print("\nDone.")


if __name__ == "__main__":
    main()
