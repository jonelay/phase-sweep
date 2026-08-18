"""Slotted FEM validation — Deylami 8p/12s outrunner.

Compares smooth-bore vs slotted FEM against published ANSYS Maxwell results.
The published results are saturating, so the comparison run is nonlinear;
linear runs are kept for the internal smooth-vs-slotted contrast.
Generates cross-section figures, field maps, waveform overlays, and STEP files.

Published validation targets (Deylami et al., 2024):
  B_ag_avg  = 0.31 T  (average air-gap flux density, FEM)
  B_ag_peak = 0.49 T  (peak air-gap flux density, FEM)

Usage:
    uv run python scripts/slot_validation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from phasesweep.machines.configs import load_motor
from phasesweep.solvers.fem_field import (
    geometry_to_step,
    rasterise_cross_section,
    solve_field_fem,
)
from phasesweep.solvers.harmonics import compute_thd, harmonics_1sided

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "slot_validation"
MOTOR_PATH = ROOT / "motors" / "deylami_fan.toml"

# Published targets
B_AG_AVG_PUB = 0.31
B_AG_PEAK_PUB = 0.49


def run_fem(motor, slotted: bool, label: str, nonlinear: bool = False):
    """Run FEM and return (theta, B_r, mesh, gfu, metrics_dict)."""
    geo = motor.geometry
    n_p = motor.n_p

    n_slots = geo.n_slots if slotted else 0

    theta, B_r, mesh, gfu = solve_field_fem(
        geo=geo, n_p=n_p, B_rem=motor.B_rem,
        mu_r_pm=motor.mu_r_pm, mu_r_fe=motor.mu_r_fe,
        maxh_fraction=0.03, n_theta=720,
        n_slots=n_slots, return_full=True,
        nonlinear=nonlinear,
        alpha_p=motor.alpha_p,
    )

    amps = harmonics_1sided(B_r)
    if n_p >= len(amps):
        raise ValueError(f"spectrum too short to resolve n_p={n_p} (len={len(amps)})")
    metrics = {
        "peak_Br": float(np.max(np.abs(B_r))),
        "avg_Br": float(np.mean(np.abs(B_r))),
        "fundamental": float(amps[n_p]),
        "thd_pct": compute_thd(amps, n_p),
    }
    if n_slots > 0:
        sh_idx = n_slots - n_p
        if 0 < sh_idx < len(amps):
            metrics["sh_pct"] = float(amps[sh_idx] / max(amps[n_p], 1e-12) * 100)

    print(f"\n{label}:")
    for k, v in metrics.items():
        print(f"  {k:15s} = {v:.4f}")

    return theta, B_r, mesh, gfu, amps, metrics


def fig_cross_section(geo, slotted: bool, filename: str):
    """Draw 2D cross-section with colored annular regions."""
    fig, ax = plt.subplots(figsize=(7, 7))

    region_colors = {
        "yoke": "#888888",
        "pm": "#cc4444",
        "airgap": "#ffffff",
        "stator": "#4488cc",
        "air": "#f0f0f0",
        "slot": "#ffcc44",
    }

    if geo.topology == "outrunner":
        radii = [
            (geo.r_outer, geo.r_rotor, "yoke"),
            (geo.r_rotor, geo.r_magnet, "pm"),
            (geo.r_magnet, geo.r_stator, "airgap"),
            (geo.r_stator, geo.r_inner, "stator"),
            (geo.r_inner, 0, "air"),
        ]
    else:
        radii = [
            (geo.r_outer, geo.r_stator, "stator"),
            (geo.r_stator, geo.r_magnet, "airgap"),
            (geo.r_magnet, geo.r_rotor, "pm"),
            (geo.r_rotor, geo.r_inner, "shaft"),
        ]

    # Draw annular regions
    for r_out, r_in, name in radii:
        color = region_colors.get(name, "#cccccc")
        circle_out = plt.Circle((0, 0), r_out * 1e3, fc=color, ec="black", lw=0.5)
        ax.add_patch(circle_out)

    # Simpler approach: draw from outside in
    ax.clear()
    for r_out, r_in, name in radii:
        color = region_colors.get(name, "#cccccc")
        c = plt.Circle((0, 0), r_out * 1e3, fc=color, ec="black", lw=0.5)
        ax.add_patch(c)

    # Draw slots if slotted
    if slotted and geo.n_slots > 0 and geo.slot_depth > 0:
        delta = geo.slot_width_ratio * 180.0 / geo.n_slots  # half-angle in degrees
        if geo.topology == "outrunner":
            r_slot_outer = geo.r_stator * 1e3
            r_slot_inner = (geo.r_stator - geo.slot_depth) * 1e3
        else:
            r_slot_inner = geo.r_stator * 1e3
            r_slot_outer = (geo.r_stator + geo.slot_depth) * 1e3

        for k in range(geo.n_slots):
            theta_k = 360.0 * k / geo.n_slots
            wedge = Wedge(
                (0, 0), r_slot_outer,
                theta_k - delta, theta_k + delta,
                width=r_slot_outer - r_slot_inner,
                fc=region_colors["slot"], ec="black", lw=0.3,
            )
            ax.add_patch(wedge)

    # Air-gap evaluation radius
    c_ag = plt.Circle((0, 0), geo.r_ag * 1e3, fc="none", ec="green",
                       lw=1.0, ls="--", label=f"r_ag = {geo.r_ag*1e3:.1f} mm")
    ax.add_patch(c_ag)

    r_max = geo.r_outer * 1e3 * 1.05
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    title = "Slotted" if slotted else "Smooth-bore"
    ax.set_title(f"Deylami 8p/12s — {title} cross-section")
    ax.legend(loc="upper right", fontsize=8)

    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_field_map(mesh, gfu, geo, label: str, filename: str):
    """Rasterise and plot |B| field on cross-section."""
    xi, yi, Az, Bmag = rasterise_cross_section(
        mesh, gfu, n_grid=200, r_bound=geo.r_outer,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A_z contour
    ax = axes[0]
    levels = np.linspace(np.nanmin(Az), np.nanmax(Az), 30)
    ax.contourf(xi * 1e3, yi * 1e3, Az, levels=levels, cmap="RdBu_r")
    ax.contour(xi * 1e3, yi * 1e3, Az, levels=levels, colors="k", linewidths=0.3)
    ax.set_aspect("equal")
    ax.set_title(f"{label} — Vector potential A_z")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # |B| map
    ax = axes[1]
    im = ax.pcolormesh(xi * 1e3, yi * 1e3, Bmag, cmap="hot", shading="auto",
                       vmin=0, vmax=min(np.nanmax(Bmag), 2.0))
    ax.set_aspect("equal")
    ax.set_title(f"{label} — |B| (T)")
    ax.set_xlabel("x (mm)")
    fig.colorbar(im, ax=ax, label="|B| (T)")

    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_waveform_comparison(theta_sm, Br_sm, theta_sl, Br_sl,
                            theta_nl: np.ndarray | None = None,
                            Br_nl: np.ndarray | None = None):
    """Overlay smooth-bore vs slotted B_r waveforms."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    deg_sm = np.degrees(theta_sm)
    deg_sl = np.degrees(theta_sl)

    ax = axes[0]
    ax.plot(deg_sm, Br_sm, "b-", lw=1.2, label="Smooth-bore FEM (linear)", alpha=0.8)
    ax.plot(deg_sl, Br_sl, "r-", lw=1.2, label="Slotted FEM (linear)", alpha=0.8)
    if Br_nl is not None:
        ax.plot(np.degrees(theta_nl), Br_nl, "-", color="darkorange", lw=1.2,
                label="Slotted FEM (nonlinear)", alpha=0.8)
    ax.axhline(B_AG_PEAK_PUB, color="green", ls="--", lw=0.8,
               label=f"Published peak = {B_AG_PEAK_PUB} T (saturating ANSYS)")
    ax.axhline(-B_AG_PEAK_PUB, color="green", ls="--", lw=0.8)
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("B_r (T)")
    ax.set_title("Deylami 8p/12s — Air-gap radial flux density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Difference
    ax = axes[1]
    # Interpolate to common grid if needed
    Br_sm_interp = np.interp(theta_sl, theta_sm, Br_sm)
    diff = Br_sl - Br_sm_interp
    ax.plot(deg_sl, diff, "k-", lw=0.8)
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("ΔB_r (T)")
    ax.set_title("Slotted − Smooth-bore")
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / "waveform_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_harmonics(amps_sm, amps_sl, n_p):
    """Bar chart of harmonic amplitudes."""
    n_show = min(40, len(amps_sm), len(amps_sl))
    orders = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax.bar(orders - w / 2, amps_sm[:n_show], w, label="Smooth-bore", color="steelblue", alpha=0.8)
    ax.bar(orders + w / 2, amps_sl[:n_show], w, label="Slotted", color="indianred", alpha=0.8)

    ax.axvline(n_p, color="green", ls="--", lw=0.8, label=f"Fundamental (n_p={n_p})")
    ax.set_xlabel("Harmonic order")
    ax.set_ylabel("Amplitude (T)")
    ax.set_title("Deylami 8p/12s — Harmonic spectrum")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out = OUT_DIR / "harmonics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def export_step_files(motor):
    """Export 3D STEP files for smooth-bore and slotted geometries."""
    geo = motor.geometry

    p_smooth = geometry_to_step(geo, OUT_DIR / "deylami_smooth.step",
                                L_stk=motor.L_stk, slotted=False)
    print(f"  STEP (smooth): {p_smooth}")

    p_slotted = geometry_to_step(geo, OUT_DIR / "deylami_slotted.step",
                                 L_stk=motor.L_stk, slotted=True)
    print(f"  STEP (slotted): {p_slotted}")


def write_summary(m_smooth: dict[str, float], m_slotted: dict[str, float],
                  m_slotted_nl: dict[str, float]):
    """Write a text summary comparing results to published values."""
    lines = [
        "# Deylami 8p/12s Slotted FEM Validation",
        "",
        "## Comparison: smooth-bore vs slotted vs published",
        "",
        "Published values are saturating ANSYS Maxwell results, so Δ is computed",
        "against the nonlinear slotted run; the linear columns are the internal",
        "smooth-vs-slotted contrast. The published averaging convention is",
        "unrecorded — avg_Br here is mean(|B_r(θ)|) over the full circumference",
        "at r_ag, so treat the avg row as indicative only.",
        "",
        f"{'Metric':<20s}  {'Smooth lin':>10s}  {'Slot lin':>10s}  {'Slot NL':>10s}  {'Published':>10s}  {'Δ NL':>10s}",
        f"{'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}",
    ]

    rows = [
        ("B_ag_peak (T)", "peak_Br", B_AG_PEAK_PUB),
        ("B_ag_avg (T)", "avg_Br", B_AG_AVG_PUB),
        ("B₁ fundamental (T)", "fundamental", None),
        ("THD (%)", "thd_pct", None),
    ]
    for label, key, pub in rows:
        vs = m_smooth.get(key, float("nan"))
        vl = m_slotted.get(key, float("nan"))
        vn = m_slotted_nl.get(key, float("nan"))
        pub_str = f"{pub:.4f}" if pub is not None else "—"
        if pub is not None and pub > 0:
            delta = f"{(vn - pub) / pub * 100:+.1f}%"
        else:
            delta = "—"
        lines.append(f"{label:<20s}  {vs:>10.4f}  {vl:>10.4f}  {vn:>10.4f}  {pub_str:>10s}  {delta:>10s}")

    if "sh_pct" in m_slotted_nl:
        lines.append(f"{'Slot harmonic (%)':<20s}  {'—':>10s}  "
                     f"{m_slotted.get('sh_pct', float('nan')):>10.1f}  "
                     f"{m_slotted_nl['sh_pct']:>10.1f}  {'—':>10s}  {'—':>10s}")

    lines.extend(["", "## Output files", ""])
    for f in sorted(OUT_DIR.iterdir()):
        lines.append(f"- `{f.name}`")

    out = OUT_DIR / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n  Summary: {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    motor = load_motor(MOTOR_PATH)
    geo = motor.geometry

    print(f"Motor: {motor.name}")
    print(f"  Topology: {geo.topology}")
    print(f"  n_slots={geo.n_slots}, slot_depth={geo.slot_depth*1e3:.1f} mm, "
          f"slot_width_ratio={geo.slot_width_ratio:.3f}")
    print(f"  B_rem={motor.B_rem} T, α_p={motor.alpha_p}, n_p={motor.n_p}")
    print(f"  mu_r_fe={motor.mu_r_fe}, mu_r_pm={motor.mu_r_pm}")

    # Cross-section figures
    print("\n--- Cross-section figures ---")
    fig_cross_section(geo, slotted=False, filename="cross_section_smooth.png")
    fig_cross_section(geo, slotted=True, filename="cross_section_slotted.png")

    # FEM solves
    print("\n--- FEM solves ---")
    theta_sm, Br_sm, mesh_sm, gfu_sm, amps_sm, m_smooth = run_fem(
        motor, slotted=False, label="Smooth-bore FEM")
    theta_sl, Br_sl, mesh_sl, gfu_sl, amps_sl, m_slotted = run_fem(
        motor, slotted=True, label="Slotted FEM (linear)")
    theta_nl, Br_nl, _, _, _, m_slotted_nl = run_fem(
        motor, slotted=True, label="Slotted FEM (nonlinear)", nonlinear=True)

    # Field maps
    print("\n--- Field maps ---")
    fig_field_map(mesh_sm, gfu_sm, geo, "Smooth-bore", "field_smooth.png")
    fig_field_map(mesh_sl, gfu_sl, geo, "Slotted", "field_slotted.png")

    # Waveform comparison
    print("\n--- Waveform comparison ---")
    fig_waveform_comparison(theta_sm, Br_sm, theta_sl, Br_sl,
                            theta_nl=theta_nl, Br_nl=Br_nl)

    # Harmonics
    print("\n--- Harmonics ---")
    fig_harmonics(amps_sm, amps_sl, motor.n_p)

    # STEP export
    print("\n--- STEP export ---")
    export_step_files(motor)

    # Summary
    write_summary(m_smooth, m_slotted, m_slotted_nl)


if __name__ == "__main__":
    main()
