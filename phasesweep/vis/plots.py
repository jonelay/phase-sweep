from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
except ImportError:
    raise ImportError(
        "phasesweep.plots requires the visualization extra: "
        "pip install phasesweep[viz]"
    ) from None

if TYPE_CHECKING:
    from phasesweep.machines.geometry import Geometry

from numpy.typing import NDArray


def _save(fig: plt.Figure, output_dir: Path | str | None, filename: str) -> str:
    if output_dir is not None:
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        out = str(d / filename)
    else:
        out = filename
    fig.savefig(out, dpi=150)
    return out


def plot_geometry(
    geo: Geometry,
    *,
    n_p: int = 2,
    alpha_p: float = 1.0,
    title: str = "",
    output_dir: Path | str | None = None,
    filename: str = "geometry.png",
) -> str:
    """2D cross-section showing annular regions, slots, and magnet arcs."""

    COLORS = {
        "Stator iron": "#4488cc",
        "Rotor iron": "#888888",
        "Magnets": "#cc4444",
        "Air gap": "#ffffff",
        "Windings": "#ffcc44",
        "Shaft / air": "#f0f0f0",
    }

    fig, ax = plt.subplots(figsize=(7, 7))

    # Build region list from outside in (painter's algorithm)
    if geo.topology == "outrunner":
        regions = [
            (geo.r_outer, "Rotor iron"),
            (geo.r_rotor, "Magnets"),
            (geo.r_magnet, "Air gap"),
            (geo.r_stator, "Stator iron"),
            (geo.r_inner, "Shaft / air"),
        ]
    else:
        regions = [
            (geo.r_outer, "Stator iron"),
            (geo.r_stator, "Air gap"),
            (geo.r_magnet, "Magnets"),
            (geo.r_rotor, "Rotor iron"),
        ]
        if geo.r_inner > 0:
            regions.append((geo.r_inner, "Shaft / air"))

    # Draw filled circles from outside in
    for r, label in regions:
        c = plt.Circle((0, 0), r * 1e3, fc=COLORS[label], ec="black", lw=0.5)
        ax.add_patch(c)

    # Discrete magnet arcs for alpha_p < 1
    if alpha_p < 1.0 and n_p >= 2:
        pole_pitch_deg = 180.0 / n_p
        arc_deg = alpha_p * pole_pitch_deg
        n_poles = 2 * n_p

        if geo.topology == "outrunner":
            r_pm_outer = geo.r_rotor
            r_pm_inner = geo.r_magnet
        else:
            r_pm_outer = geo.r_magnet
            r_pm_inner = geo.r_rotor

        # Draw interpole gaps (air color) over the full magnet annulus
        for k in range(n_poles):
            center_deg = k * pole_pitch_deg + pole_pitch_deg / 2
            gap_start = center_deg + arc_deg / 2
            gap_end = center_deg + pole_pitch_deg - arc_deg / 2
            if gap_end - gap_start > 0.1:
                wedge = Wedge(
                    (0, 0), r_pm_outer * 1e3,
                    gap_start, gap_end,
                    width=(r_pm_outer - r_pm_inner) * 1e3,
                    fc=COLORS["Air gap"], ec="black", lw=0.3,
                )
                ax.add_patch(wedge)

    # Draw slots
    if geo.n_slots > 0 and geo.slot_depth > 0:
        delta_body = geo.slot_width_ratio * 180.0 / geo.n_slots
        delta_open = ((geo.slot_opening_ratio if geo.slot_opening_ratio > 0
                        else geo.slot_width_ratio)
                       * 180.0 / geo.n_slots)

        r_bore = geo.r_stator * 1e3
        if geo.topology == "outrunner":
            r_bottom = (geo.r_stator - geo.slot_depth) * 1e3
        else:
            r_bottom = (geo.r_stator + geo.slot_depth) * 1e3

        from phasesweep.defaults import SLOT_OPENING_FRACTION
        stepped = abs(delta_open - delta_body) > 0.01

        for k in range(geo.n_slots):
            theta_k = 360.0 * k / geo.n_slots
            if stepped:
                # Stepped slot: narrow opening near bore, wider body
                if geo.topology == "outrunner":
                    r_step = r_bore - SLOT_OPENING_FRACTION * (r_bore - r_bottom)
                else:
                    r_step = r_bore + SLOT_OPENING_FRACTION * (r_bottom - r_bore)

                # Opening wedge (narrow, near bore)
                r_o1 = max(r_bore, r_step)
                r_i1 = min(r_bore, r_step)
                w1 = Wedge(
                    (0, 0), r_o1,
                    theta_k - delta_open, theta_k + delta_open,
                    width=r_o1 - r_i1,
                    fc=COLORS["Windings"], ec="black", lw=0.3,
                )
                ax.add_patch(w1)

                # Body wedge (wider, toward yoke)
                r_o2 = max(r_step, r_bottom)
                r_i2 = min(r_step, r_bottom)
                w2 = Wedge(
                    (0, 0), r_o2,
                    theta_k - delta_body, theta_k + delta_body,
                    width=r_o2 - r_i2,
                    fc=COLORS["Windings"], ec="black", lw=0.3,
                )
                ax.add_patch(w2)
            else:
                r_outer = max(r_bore, r_bottom)
                r_inner = min(r_bore, r_bottom)
                wedge = Wedge(
                    (0, 0), r_outer,
                    theta_k - delta_body, theta_k + delta_body,
                    width=r_outer - r_inner,
                    fc=COLORS["Windings"], ec="black", lw=0.3,
                )
                ax.add_patch(wedge)

    # Air-gap evaluation circle
    c_ag = plt.Circle((0, 0), geo.r_ag * 1e3, fc="none", ec="green",
                       lw=1.0, ls="--", label=f"r_ag = {geo.r_ag*1e3:.1f} mm")
    ax.add_patch(c_ag)

    # Legend — only show regions present in this geometry
    from matplotlib.artist import Artist
    from matplotlib.patches import Patch
    used_labels = dict.fromkeys(label for _, label in regions)
    if geo.n_slots > 0 and geo.slot_depth > 0:
        used_labels["Windings"] = None
    handles: list[Artist] = [Patch(fc=COLORS[label], ec="black", lw=0.5, label=label)
                             for label in used_labels]
    handles.append(plt.Line2D([0], [0], color="green", ls="--", lw=1.0,
                              label=f"r_ag = {geo.r_ag*1e3:.1f} mm"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    r_max = geo.r_outer * 1e3 * 1.08
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if title:
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    out = _save(fig, output_dir, filename)
    plt.close(fig)
    return out


def plot_sweep_1d(
    param_values: list[float] | NDArray[np.floating],
    metric_values: dict[str, list[float] | NDArray[np.floating]],
    param_label: str,
    metric_label: str = "",
    *,
    title: str = "",
    output_dir: Path | str | None = None,
    filename: str = "sweep_1d.png",
) -> str:
    """1D line plot of one or more metrics vs a swept parameter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if title:
        ax.set_title(title, fontsize=11)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (name, vals) in enumerate(metric_values.items()):
        ax.plot(param_values, vals, "o-", color=colors[i % len(colors)],
                lw=1.5, ms=4, label=name)

    ax.set_xlabel(param_label)
    ax.set_ylabel(metric_label or ", ".join(metric_values.keys()))
    if len(metric_values) > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = _save(fig, output_dir, filename)
    return out


def plot_sweep_2d(
    x_values: list[float] | NDArray[np.floating],
    y_values: list[float] | NDArray[np.floating],
    metric_grid: NDArray[np.floating],
    x_label: str,
    y_label: str,
    metric_label: str,
    *,
    title: str = "",
    cmap: str = "viridis",
    annotate: bool = True,
    output_dir: Path | str | None = None,
    filename: str = "sweep_2d.png",
) -> str:
    """2D heatmap of a metric over two swept parameters.

    metric_grid shape: (len(y_values), len(x_values)).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if title:
        ax.set_title(title, fontsize=11)

    im = ax.pcolormesh(
        x_values, y_values, metric_grid,
        cmap=cmap, shading="auto",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(im, ax=ax, label=metric_label)

    if annotate:
        median = np.nanmedian(metric_grid)
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                val = metric_grid[i, j]
                if np.isfinite(val):
                    color = "white" if val > median else "black"
                    ax.text(x, y, f"{val:.3g}", ha="center", va="center",
                            fontsize=7, color=color)

    plt.tight_layout()
    out = _save(fig, output_dir, filename)
    return out
