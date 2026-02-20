from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from phasesweep.configs import (
    CONFIGS, FullMotorConfig, T_STOP, W_REF, MAX_I_S,
    SWEEP_BASE, SWEEP_LOAD, LOAD_T,
    PSI_F_VALS, RATIO_VALS,
)
from phasesweep.fem_field import harmonics_1sided, zhu_howe_Br, _derive_B_rem


def plot_results(results: dict[str, Any]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("PMSM Design Comparison — motulator demo")

    colors = ["tab:blue", "tab:orange", "tab:green"]

    for (name, res), color in zip(results.items(), colors):
        t = res.mdl.t
        w_m = res.mdl.mechanics.w_M
        tau_M = res.mdl.machine.tau_M
        i_s = np.abs(res.mdl.machine.i_s_ab)

        n_p = CONFIGS[name]["n_p"]
        w_e = w_m * n_p

        axes[0].plot(t, w_e / (2 * np.pi), label=name, color=color)
        axes[1].plot(t, tau_M, color=color)
        axes[2].plot(t, i_s, color=color)

    t_ref = [0, 0.2, 0.2, T_STOP]
    w_ref = [0, 0, W_REF / (2 * np.pi), W_REF / (2 * np.pi)]
    axes[0].plot(t_ref, w_ref, "k--", linewidth=0.8, label="w_ref")
    axes[0].axvline(1.2, color="gray", linestyle=":", linewidth=0.8)
    axes[1].axvline(1.2, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axvline(1.2, color="gray", linestyle=":", linewidth=0.8)

    axes[0].set_ylabel("Speed (Hz, electrical)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Torque (Nm)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("|i_s| (A)")
    axes[2].set_xlabel("Time (s)")
    axes[2].axhline(MAX_I_S, color="red", linestyle="--", linewidth=0.8,
                    label=f"i_max = {MAX_I_S} A")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.text(0.5, 0.01, "Dashed vertical line = load step (3 Nm at t=1.2 s)",
             ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    out = "demo_results.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


def plot_sweep(grid: dict[str, NDArray[np.floating]]) -> None:
    metrics = [
        ("t_settle",    "Settling time (s)",           "viridis_r"),
        ("i_ss",        "Steady-state current (A)",    "plasma"),
        ("speed_droop", "Speed droop (fraction)",      "hot"),
        ("tau_peak",    "Peak accel torque (Nm)",      "cividis"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"PMSM sweep: psi_f vs Ld/Lq  |  "
        f"Ld={SWEEP_BASE['L_d']*1e3:.1f} mH, "
        f"load={SWEEP_LOAD} Nm at t={LOAD_T} s",
        fontsize=11,
    )

    for ax, (key, title, cmap) in zip(axes.flat, metrics):
        data = grid[key]
        im = ax.pcolormesh(
            RATIO_VALS, PSI_F_VALS * 1e3, data,
            cmap=cmap, shading="auto",
        )
        ax.set_xlabel("Lq/Ld ratio")
        ax.set_ylabel("psi_f (mVs)")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax)

        for i, pf in enumerate(PSI_F_VALS):
            for j, r in enumerate(RATIO_VALS):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(r, pf * 1e3, f"{val:.2f}",
                            ha="center", va="center", fontsize=6.5,
                            color="white" if val > np.nanmedian(data) else "black")

    plt.tight_layout()
    out = "sweep_results.png"
    plt.savefig(out, dpi=150)
    print(f"Sweep plot saved to {out}")


def plot_field_polar(configs: dict[str, FullMotorConfig]) -> None:
    names = list(configs.keys())
    n_configs = len(names)
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig = plt.figure(figsize=(5 * n_configs, 8))
    fig.suptitle("Air-gap field B_r(θ) — Zhu & Howe analytical",
                 fontsize=11)

    theta = np.linspace(0, 2 * np.pi, 1000)

    for col, (name, color) in enumerate(zip(names, colors)):
        cfg = configs[name]
        n_p = cfg["n_p"]
        B_rem = _derive_B_rem(cfg["psi_f"], n_p, cfg["N"], cfg["k_w"], cfg["L_stk"])
        B_r = zhu_howe_Br(theta, n_p, B_rem)

        ax_p = fig.add_subplot(2, n_configs, col + 1, projection="polar")
        pos = B_r >= 0
        ax_p.plot(theta[pos], B_r[pos], color=color, linewidth=1.2)
        neg_theta = (theta[~pos] + np.pi) % (2 * np.pi)
        ax_p.plot(neg_theta, np.abs(B_r[~pos]), color=color,
                  linewidth=1.2, linestyle="--")
        ax_p.set_title(f"{name}\n({n_p} pole pairs)", fontsize=8, pad=10)
        ax_p.set_yticklabels([])

        ax_l = fig.add_subplot(2, n_configs, n_configs + col + 1)
        ax_l.plot(theta, B_r, color=color, linewidth=1.2)
        ax_l.fill_between(theta, B_r, 0, where=(B_r >= 0).tolist(),
                           color=color, alpha=0.15)
        ax_l.fill_between(theta, B_r, 0, where=(B_r < 0).tolist(),
                           color="tab:red", alpha=0.15)
        ax_l.axhline(0, color="black", linewidth=0.6)
        ax_l.set_xlabel("θ (rad)")
        ax_l.set_ylabel("B_r (T)")
        ax_l.set_xlim(0, 2 * np.pi)
        ax_l.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax_l.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax_l.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "field_results.png"
    plt.savefig(out, dpi=150)
    print(f"Field plot saved to {out}")


def plot_field_comparison(
    configs: dict[str, FullMotorConfig],
    fem_results: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]],
    harmonics: dict[str, NDArray[np.floating]],
) -> None:
    names = list(configs.keys())
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(len(names), 2, figsize=(13, 4 * len(names)))
    fig.suptitle("Air-gap B_r(θ) — Analytical vs FEM  |  GPU harmonic decomp (CuPy)",
                 fontsize=11)

    for row, (name, color) in enumerate(zip(names, colors)):
        cfg = configs[name]
        theta_fem, B_fem = fem_results[name]

        n_p = cfg["n_p"]
        B_rem = _derive_B_rem(cfg["psi_f"], n_p, cfg["N"], cfg["k_w"], cfg["L_stk"])
        B_an = zhu_howe_Br(theta_fem, n_p, B_rem)

        scale = np.max(np.abs(B_an))
        B_an_n = B_an / scale
        B_fem_n = B_fem / (np.nanmax(np.abs(B_fem)) or 1.0)

        ax = axes[row, 0]
        ax.plot(theta_fem, B_an_n, color=color, lw=1.5, label="Zhu & Howe")
        ax.plot(theta_fem, B_fem_n, color=color, lw=1.0, ls="--",
                alpha=0.85, label="FEM (NGSolve)")
        ax.fill_between(theta_fem, B_an_n, 0, where=(B_an_n >= 0),
                        color=color, alpha=0.08)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("B_r (normalised)")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = axes[row, 1]
        an_amps = harmonics_1sided(B_an_n)
        fem_amps = harmonics[name] / (np.nanmax(np.abs(B_fem)) or 1.0) * scale
        fem_amps_n = fem_amps / (np.max(an_amps) or 1.0)
        an_amps_n = an_amps / (np.max(an_amps) or 1.0)

        max_ord = cfg["n_p"] * 9 + 1
        orders = np.arange(min(max_ord, len(an_amps)))
        w = 0.35
        ax2.bar(orders - w / 2, an_amps_n[orders], width=w, color=color,
                alpha=0.85, label="Analytical")
        ax2.bar(orders + w / 2, fem_amps_n[orders], width=w, color=color,
                alpha=0.4, label="FEM")
        ax2.set_xlabel("Harmonic order")
        ax2.set_ylabel("Amplitude (normalised)")
        ax2.set_title(f"Harmonics — {name}", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = "field_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Field comparison saved to {out}")


def plot_field_slotted(
    configs: dict[str, FullMotorConfig],
    slotted_results: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]],
    smooth_results: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]],
    slotted_harmonics: dict[str, NDArray[np.floating]],
) -> None:
    names = list(configs.keys())
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(len(names), 2, figsize=(14, 4 * len(names)))
    fig.suptitle(
        "Air-gap B_r(θ) — Analytical / FEM smooth / FEM slotted+winding  |  slot harmonics",
        fontsize=11)

    for row, (name, color) in enumerate(zip(names, colors)):
        cfg = configs[name]
        theta_fem, B_smooth = smooth_results[name]
        _, B_slot = slotted_results[name]

        n_p = cfg["n_p"]
        B_rem = _derive_B_rem(cfg["psi_f"], n_p, cfg["N"], cfg["k_w"], cfg["L_stk"])
        B_an = zhu_howe_Br(theta_fem, n_p, B_rem)
        scale = np.max(np.abs(B_an)) or 1.0
        B_an_n = B_an / scale
        B_smooth_n = B_smooth / (np.nanmax(np.abs(B_smooth)) or 1.0)
        B_slot_n = B_slot / (np.nanmax(np.abs(B_slot)) or 1.0)

        ax = axes[row, 0]
        ax.plot(theta_fem, B_an_n, color=color, lw=1.5, label="Zhu & Howe (smooth)")
        ax.plot(theta_fem, B_smooth_n, color=color, lw=1.0, ls="--",
                alpha=0.7, label="FEM smooth bore")
        ax.plot(theta_fem, B_slot_n, color="black", lw=0.8, ls="-",
                alpha=0.85, label="FEM slotted+winding")
        ax.fill_between(theta_fem, B_an_n, 0, where=(B_an_n >= 0),
                        color=color, alpha=0.06)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("B_r (normalised)")
        ax.set_title(f"{name}  |  Q={cfg['n_slots']} slots", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax2 = axes[row, 1]
        an_amps = harmonics_1sided(B_an_n)
        sm_amps = harmonics_1sided(B_smooth_n)
        sl_amps = slotted_harmonics[name] / (np.nanmax(np.abs(B_slot)) or 1.0) * scale
        sl_amps = sl_amps / (np.max(an_amps) or 1.0)
        an_amps_n = an_amps / (np.max(an_amps) or 1.0)
        sm_amps_n = sm_amps / (np.max(an_amps) or 1.0)

        max_ord = cfg["n_p"] * 9 + 1
        orders = np.arange(min(max_ord, len(an_amps)))
        w = 0.25
        ax2.bar(orders - w, an_amps_n[orders], width=w, color=color,
                alpha=0.85, label="Analytical")
        ax2.bar(orders, sm_amps_n[orders], width=w, color=color,
                alpha=0.45, label="FEM smooth")
        ax2.bar(orders + w, sl_amps[orders], width=w, color="black",
                alpha=0.55, label="FEM slotted")
        ax2.set_xlabel("Harmonic order")
        ax2.set_ylabel("Amplitude (normalised)")
        ax2.set_title(f"Harmonics — {name}", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = "field_slotted.png"
    plt.savefig(out, dpi=150)
    print(f"Slotted field comparison saved to {out}")


def plot_cross_section(configs: dict[str, FullMotorConfig]) -> None:
    from phasesweep.fem_field import (
        solve_field_fem, rasterise_cross_section,
        _R_S, _R_SI, _R_RO, _R_RI, _R_AG,
    )

    names = list(configs.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle("|B| cross-section — FEM (NGSolve)  |  A_z contours = field lines",
                 fontsize=11)

    theta_c = np.linspace(0, 2 * np.pi, 400)

    for ax, name in zip(axes, names):
        cfg = configs[name]
        print(f"  Cross-section {name}...")
        theta, B_r, mesh, gfu = solve_field_fem(
            n_p=cfg["n_p"], psi_f=cfg["psi_f"],
            L_d=cfg["L_d"], L_q=cfg["L_q"],
            n_slots=cfg.get("n_slots", 0), j_s=cfg.get("j_s", 0.0),
            N=cfg.get("N", 50), k_w=cfg.get("k_w", 0.966),
            L_stk=cfg.get("L_stk", 0.10),
            return_full=True,
        )
        xi, yi, Az, Bmag = rasterise_cross_section(mesh, gfu)

        vmax = np.nanpercentile(Bmag, 98)
        im = ax.pcolormesh(xi, yi, Bmag, cmap="inferno",
                           vmin=0, vmax=vmax, shading="auto")

        az_finite = Az[np.isfinite(Az)]
        if az_finite.size:
            levels = np.linspace(az_finite.min(), az_finite.max(), 20)
            ax.contour(xi, yi, Az, levels=levels,
                       colors="white", linewidths=0.5, alpha=0.6)

        for r, ls, lbl in [
            (_R_S, "-", "stator outer"),
            (_R_SI, "--", "stator/airgap"),
            (_R_RO, "-.", "airgap/PM"),
            (_R_RI, ":", "PM/shaft"),
        ]:
            ax.plot(r * np.cos(theta_c), r * np.sin(theta_c),
                    color="cyan", ls=ls, lw=0.8, label=lbl)

        ax.plot(_R_AG * np.cos(theta_c), _R_AG * np.sin(theta_c),
                color="yellow", ls="--", lw=0.6, alpha=0.6, label="air-gap centre")

        ax.set_aspect("equal")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="|B| (proxy units)")

    axes[-1].legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    out = "cross_section.png"
    plt.savefig(out, dpi=150)
    print(f"Cross-section plot saved to {out}")


def plot_armature_reaction(
    theta: NDArray[np.floating],
    components: dict[str, NDArray[np.floating]],
    harmonics: dict[str, NDArray[np.floating]],
    cfg_name: str,
    Q: int,
    n_p: int,
) -> None:
    colors = {"PM only": "tab:blue", "Winding only": "tab:red", "Combined": "tab:purple"}
    styles = {"PM only": "-", "Winding only": "--", "Combined": "-"}

    scale = np.nanmax(np.abs(components["Combined"])) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Armature Reaction Decomposition — {cfg_name}  (Q={Q}, n_p={n_p})",
                 fontsize=11)

    ax = axes[0]
    for name, B in components.items():
        ax.plot(theta, B / scale, color=colors[name], ls=styles[name],
                lw=1.2, label=name)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("B_r / peak combined")
    ax.set_title("Field decomposition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    max_ord = min(Q * 2 + 5, 40)
    orders = np.arange(max_ord)
    w = 0.25
    for i, (name, amps) in enumerate(harmonics.items()):
        amps_n = amps[:max_ord] / scale
        ax2.bar(orders + (i - 1) * w, amps_n, width=w,
                color=colors[name], alpha=0.75, label=name)

    ymax = ax2.get_ylim()[1]
    for mult in [1, 2]:
        for sign, tag in [(1, f"{mult * Q}+{n_p}"), (-1, f"{mult * Q}-{n_p}")]:
            order = mult * Q + sign * n_p
            if 0 < order < max_ord:
                ax2.axvline(order, color="gray", lw=0.9, ls=":", alpha=0.8)
                ax2.text(order + 0.15, ymax * 0.93, tag,
                         fontsize=6, color="gray", va="top")

    ax2.set_xlabel("Harmonic order")
    ax2.set_ylabel("Amplitude (normalised)")
    ax2.set_title("Harmonic spectrum (dashed = slot harmonic orders)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xlim(-0.5, max_ord - 0.5)

    plt.tight_layout()
    out = "armature_reaction.png"
    plt.savefig(out, dpi=150)
    print(f"Armature reaction plot saved to {out}")


def plot_slot_sweep(
    results: dict[int, NDArray[np.floating]],
    n_p: int,
    cfg_name: str,
) -> None:
    slot_counts = list(results.keys())
    n = len(slot_counts)
    cmap = plt.cm.plasma
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Slot Count Sweep — {cfg_name}  |  Q = {slot_counts}", fontsize=11)

    ax = axes[0]
    max_ord = max(slot_counts) + n_p + 2
    orders = np.arange(max_ord)
    for (Q, amps), col in zip(results.items(), colors):
        fund = amps[n_p] or 1.0
        y_plot = amps[:max_ord] / fund
        ax.plot(orders[:len(y_plot)], y_plot, color=col, lw=1.0, label=f"Q={Q}")
        sh = Q - n_p
        if 0 < sh < max_ord:
            ax.plot(sh, amps[sh] / fund, "x", color=col, ms=6, mew=1.5)

    ax.set_xlabel("Harmonic order")
    ax.set_ylabel("Amplitude / fundamental")
    ax.set_title("Harmonic spectrum  (× = Q−n_p slot harmonic)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, max_ord - 1)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
    ax.grid(True, alpha=0.3, which="both")

    ax2 = axes[1]
    ax2_r = ax2.twinx()

    thd_vals = []
    sh_vals = []
    fund_vals = []
    for Q, amps in results.items():
        fund = amps[n_p] or 1.0
        thd = (np.sqrt(np.sum(amps[1:]**2) - amps[n_p]**2) / fund) * 100
        thd_vals.append(thd)
        sh = Q - n_p
        sh_vals.append((amps[sh] / fund * 100) if sh < len(amps) else 0.0)
        fund_vals.append(amps[n_p])

    ax2.plot(slot_counts, thd_vals, "o-", color="tab:blue", lw=1.5, label="THD (%)")
    ax2.plot(slot_counts, sh_vals, "s--", color="tab:red", lw=1.2, label="Q−n_p harm (%)")
    ax2_r.plot(slot_counts, fund_vals, "^:", color="tab:green", lw=1.2, label="Fundamental")

    ax2.set_xlabel("Slot count Q")
    ax2.set_ylabel("% of fundamental")
    ax2_r.set_ylabel("Fundamental amplitude (proxy units)")
    ax2.set_title("THD and slot harmonic vs Q")
    ax2.grid(True, alpha=0.3)

    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = "slot_sweep.png"
    plt.savefig(out, dpi=150)
    print(f"Slot sweep plot saved to {out}")
