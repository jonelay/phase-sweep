"""NGSolve FEM runner: RunConfig in, metrics dict out.

Metrics returned:
  peak_Br      — max |B_r| at air-gap centre (T)
  fundamental  — amplitude at n_p-th harmonic
  thd_pct      — total harmonic distortion (% of fundamental)
  sh_pct       — lower slot-harmonic sideband (|Q - n_p| order) amplitude
                 (% of fundamental)
  sh_upper_pct — upper slot-harmonic sideband (Q + n_p order) amplitude
                 (% of fundamental); slot harmonics come in pairs Q ± n_p
  theta_list   — angle samples (list[float], for reconstruction)
  B_r_list     — radial flux density samples (list[float], for reconstruction)
  B_mag_grid   — |B| on a Cartesian grid over the cross-section (rows = y,
                 cols = x; null outside the motor domain) for the dashboard
                 heatmap
  grid_coords_list — the shared x/y axis of B_mag_grid (m)
  b_iron_max   — peak |B| in iron regions (T)
  tau_maxwell_per_m — armature runs (j_s != 0) only: Maxwell-stress rotor
                 torque per metre of stack (N·m/m, mid-gap contour)
  tau_maxwell  — tau_maxwell_per_m · L_stk (N·m), when Motor.L_stk is set
  picard_*     — nonlinear solves only: iterations, relax_final, residual
                 (raw/unrelaxed step, the stop criterion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from phasesweep.solver_params import (
    reject_induction,
    require_geometry,
    resolve_b_rem,
)
from phasesweep.sweep_types import Magnetization, RunConfig

if TYPE_CHECKING:
    from phasesweep.machines.geometry import Geometry
    from phasesweep.machines.motor import Motor

__all__ = [
    "FemParams",
    "prepare_fem",
    "run_fem",
]

_RASTER_N_GRID = 120


def run_fem(config: RunConfig) -> dict[str, Any]:
    """Run NGSolve FEM solve from RunConfig."""
    from phasesweep.solvers.fem_field import solve_field_fem
    from phasesweep.solvers.harmonics import compute_thd, harmonics_1sided

    params = prepare_fem(config.motor)
    geo = params.geometry

    if config.j_s != 0.0 and config.motor.k_w is None:
        raise ValueError(
            f"Motor {config.motor.name!r}: j_s != 0 requires an explicit "
            "k_w (armature current sheet scales with the winding factor)"
        )
    from phasesweep.solver_params import check_nyquist
    check_nyquist(config.n_theta, params.n_p)

    solve_info: dict[str, float] = {}
    theta, B_r, mesh_obj, gfu = solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        maxh_fraction=config.maxh_fraction,
        n_theta=config.n_theta, nonlinear=config.nonlinear,
        nonlinear_solver=config.nonlinear_solver,
        j_s=config.j_s,
        k_w=config.motor.k_w if config.motor.k_w is not None else 0.966,
        alpha_p=params.alpha_p,
        rotation=config.rotation,
        info=solve_info,
        return_full=True,
        magnetization=params.magnetization,
        halbach_segments=params.halbach_segments,
        halbach_angles_rad=params.halbach_angles_rad,
    )
    torque_info: dict[str, float] = {}
    if config.j_s != 0.0:
        from phasesweep.solvers.fem_sampling import maxwell_stress_torque
        # Torque on the region enclosed by the mid-gap circle = the rotor
        # for an inrunner; sign-flip for an outrunner (rotor outside).
        rotor_sign = 1.0 if geo.topology == "inrunner" else -1.0
        tau_per_m = rotor_sign * maxwell_stress_torque(mesh_obj, gfu, geo.r_ag)
        torque_info["tau_maxwell_per_m"] = tau_per_m
        if config.motor.L_stk is not None:
            torque_info["tau_maxwell"] = tau_per_m * config.motor.L_stk

    from phasesweep.solvers.fem_sampling import rasterise_cross_section
    xi, _yi, _Az, Bmag = rasterise_cross_section(
        mesh_obj, gfu, n_grid=_RASTER_N_GRID, r_bound=geo.r_outer)
    B_mag_grid = [
        [round(float(v), 5) if np.isfinite(v) else None for v in row]
        for row in Bmag
    ]
    grid_coords = np.round(xi[0], 6).tolist()

    amps = harmonics_1sided(B_r)
    n_p = params.n_p
    if np.isnan(B_r).any():
        raise RuntimeError(
            f"FEM B_r sampling produced {int(np.isnan(B_r).sum())} NaN "
            f"point(s) of {B_r.size} — mesh evaluation failed at the "
            "sampling radius"
        )
    peak_B_r = float(np.max(np.abs(B_r)))
    fundamental = float(amps[n_p])
    # Relative, not == 0: FFT round-off leaves ~1e-17 where there is no
    # order-n_p content. A PM solve has fundamental of order peak_B_r.
    if not fundamental > 1e-9 * peak_B_r:
        raise RuntimeError(
            f"FEM B_r fundamental at order n_p={n_p} is {fundamental:.3g} T "
            f"against peak {peak_B_r:.3g} T — no PM field was solved at the "
            "sampling radius"
        )

    thd_pct = compute_thd(amps, n_p)

    sh_pct = 0.0
    sh_upper_pct = 0.0
    n_slots = geo.n_slots
    if n_slots > 0:
        sh_idx = abs(n_slots - n_p)
        if 0 < sh_idx < len(amps):
            sh_pct = float(amps[sh_idx] / fundamental * 100)
        sh_upper_idx = n_slots + n_p
        if sh_upper_idx < len(amps):
            sh_upper_pct = float(amps[sh_upper_idx] / fundamental * 100)

    return {
        "peak_Br": peak_B_r,
        "fundamental": fundamental,
        "thd_pct": thd_pct,
        "sh_pct": sh_pct,
        "sh_upper_pct": sh_upper_pct,
        "theta_list": theta.tolist(),
        "B_r_list": B_r.tolist(),
        "B_mag_grid": B_mag_grid,
        "grid_coords_list": grid_coords,
        **torque_info,
        **solve_info,
    }


@dataclass(frozen=True)
class FemParams:
    """Validated inputs for NGSolve FEM magnetostatic solver.

    geometry in meters; B_rem — magnet remanence (T); mu_r_pm / mu_r_fe —
    relative permeabilities (dimensionless); alpha_p — pole-arc ratio ((0, 1]).

    mu_r_fe sets the iron reluctivity of the LINEAR solve only. With
    nonlinear=True the solver replaces it with the generic B-H table
    — it merely seeds
    Picard iteration 0 and the result is insensitive to its value (inert
    to 6e-6 across 500→4000). It stays validated and hashed either
    way.
    """

    geometry: Geometry
    n_p: int
    B_rem: float
    mu_r_pm: float
    mu_r_fe: float
    alpha_p: float = 1.0
    magnetization: Magnetization = "radial"
    halbach_segments: int | None = None
    halbach_angles_rad: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.B_rem <= 0:
            raise ValueError(f"B_rem must be > 0, got {self.B_rem}")
        if self.mu_r_pm <= 0:
            raise ValueError(f"mu_r_pm must be > 0, got {self.mu_r_pm}")
        if self.mu_r_fe <= 0:
            raise ValueError(f"mu_r_fe must be > 0, got {self.mu_r_fe}")
        if not (0 < self.alpha_p <= 1):
            raise ValueError(f"alpha_p must be in (0, 1], got {self.alpha_p}")
        if self.geometry.magnet_mount == "inset" and self.alpha_p >= 1.0:
            raise ValueError(
                "magnet_mount='inset' requires alpha_p < 1: the interpole "
                f"iron IS the inset feature (got alpha_p={self.alpha_p})"
            )
        if self.magnetization not in ("radial", "halbach"):
            raise ValueError(
                f"magnetization must be 'radial' or 'halbach', "
                f"got {self.magnetization!r}"
            )
        if self.magnetization == "halbach":
            if self.halbach_segments is None or self.halbach_segments < 1:
                raise ValueError(
                    f"halbach_segments must be >= 1 when magnetization='halbach', "
                    f"got {self.halbach_segments}"
                )
            if self.alpha_p != 1.0:
                raise ValueError(
                    f"Halbach magnetization requires alpha_p=1.0, "
                    f"got alpha_p={self.alpha_p}"
                )
            if (
                self.halbach_angles_rad is not None
                and len(self.halbach_angles_rad) != self.halbach_segments
            ):
                raise ValueError(
                    f"halbach_angles_rad length "
                    f"({len(self.halbach_angles_rad)}) must equal "
                    f"halbach_segments ({self.halbach_segments})"
                )
        else:
            if self.halbach_segments is not None:
                raise ValueError(
                    "halbach_segments must be None when magnetization='radial'"
                )
            if self.halbach_angles_rad is not None:
                raise ValueError(
                    "halbach_angles_rad must be None when magnetization='radial'"
                )


def prepare_fem(motor: Motor) -> FemParams:
    """Derive, validate, and extract FEM solver params from a Motor."""
    from math import radians

    reject_induction(motor, "fem")
    geo = require_geometry(motor, "fem")
    b_rem = resolve_b_rem(motor)
    angles_rad = None
    if motor.halbach_angles_deg is not None:
        angles_rad = tuple(radians(a) for a in motor.halbach_angles_deg)
    return FemParams(
        geometry=geo,
        n_p=motor.n_p,
        B_rem=b_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
        magnetization=motor.magnetization,
        halbach_segments=motor.halbach_segments,
        halbach_angles_rad=angles_rad,
    )
