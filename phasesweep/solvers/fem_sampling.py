"""Field sampling, torque integrals, and rasterisation on solved fields.

Every sampler uses the batched ``mesh(xs, ys)`` point locate and
evaluates the field CoefficientFunction directly at the sample points —
no global L2 projection, except the deliberate one in
``rasterise_cross_section`` (its |B| grid is a stored FEM output).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from phasesweep.solvers.fem_materials import MU0
from phasesweep.solvers.fem_mesh import DEFAULT_GEO, pm_radii
from phasesweep.sweep_types import Magnetization

if TYPE_CHECKING:
    from ngsolve import GridFunction, Mesh

    from phasesweep.machines.geometry import Geometry

__all__ = [
    "arkkio_torque",
    "demag_margin",
    "maxwell_interaction_torque_order",
    "maxwell_stress_torque",
    "rasterise_cross_section",
    "rasterise_fields",
    "sample_Br",
    "sample_magnet_Bm",
]

def sample_Br(
    mesh: Mesh, gfu: GridFunction, radius: float, n_theta: int = 360,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Sample B_r = (1/r) ∂A_z/∂θ on a circle of the given radius from a
    solved A_z GridFunction (as returned by solve_field_fem(return_full=True)).

    Evaluates grad(A_z) directly at the sample points (vectorized) — no
    global L2 projection.
    """
    from ngsolve import grad, sqrt, x, y

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf

    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    mips = mesh(radius * np.cos(theta), radius * np.sin(theta))
    B_r = np.asarray(B_r_cf(mips), dtype=float).reshape(-1)
    return theta, B_r


def maxwell_stress_torque(
    mesh: Mesh, gfu: GridFunction, radius: float, L_stk: float = 1.0,
    n_theta: int = 720,
) -> float:
    """Torque (N·m about z) on the material enclosed by a circle of the
    given radius, via the Maxwell stress tensor evaluated on that circle.

        tau = (L_stk · r² / μ0) · ∮ B_r · B_θ dθ

    from the solved A_z (B_r = (1/r)·∂A/∂θ, B_θ = −∂A/∂r). The circle
    must lie in a source-free air region (the air gap): there the
    integral is contour-independent, which doubles as the correctness
    check (see tests). Uniform grid over a periodic integrand — the
    rectangle rule is spectrally accurate. L_stk = 1.0 gives torque per
    metre of stack.

    Sign: torque on the enclosed region — the rotor for an inrunner; for
    an outrunner (rotor outside the sampling circle) the rotor torque is
    the negative.
    """
    B_r, B_t = _sample_B_polar(mesh, gfu, radius, n_theta)
    integral = float(np.sum(B_r * B_t)) * (2 * pi / n_theta)
    return L_stk * radius * radius / MU0 * integral


def arkkio_torque(
    mesh: Mesh, gfu: GridFunction, r_inner: float, r_outer: float,
    L_stk: float = 1.0, n_r: int = 5, n_theta: int = 720,
) -> float:
    """Torque via Arkkio's radial-averaging method (N·m about z).

    Averages the Maxwell-stress contour integral over ``n_r`` radii
    spanning the air-gap annulus [r_inner, r_outer]:

        tau = (1/(r_o - r_i)) * integral_{r_i}^{r_o} tau_contour(r) dr

    The radial average suppresses mesh-noise that a single-contour
    integral picks up (especially on re-meshed cogging sweeps where the
    mesh changes per angle). Both bounds must lie strictly inside the
    source-free gap.

    Sign convention matches ``maxwell_stress_torque``: torque on the
    enclosed region (rotor for inrunner; negate for outrunner rotor).
    """
    if r_outer <= r_inner:
        raise ValueError(
            f"arkkio_torque: r_outer ({r_outer}) must exceed r_inner ({r_inner})"
        )
    radii = np.linspace(r_inner, r_outer, n_r)
    tau_per_r = np.empty(n_r)
    for i, r in enumerate(radii):
        B_r, B_t = _sample_B_polar(mesh, gfu, r, n_theta)
        integral = float(np.sum(B_r * B_t)) * (2 * pi / n_theta)
        tau_per_r[i] = L_stk * r * r / MU0 * integral
    return float(np.trapezoid(tau_per_r, radii) / (r_outer - r_inner))


def _sample_B_polar(
    mesh: Mesh, gfu: GridFunction, radius: float, n_theta: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """(B_r, B_θ) on a uniform circle of the given radius from solved A_z."""
    from ngsolve import grad, sqrt, x, y

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf
    B_t_cf = -(grad(gfu)[0] * x + grad(gfu)[1] * y) / r_cf
    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    mips = mesh(radius * np.cos(theta), radius * np.sin(theta))
    return (np.asarray(B_r_cf(mips), dtype=float).reshape(-1),
            np.asarray(B_t_cf(mips), dtype=float).reshape(-1))


def maxwell_interaction_torque_order(
    mesh: Mesh,
    gfu_a: GridFunction,
    gfu_b: GridFunction,
    radius: float,
    order: int,
    L_stk: float = 1.0,
    n_theta: int = 720,
) -> float:
    """Single-space-harmonic interaction torque (N·m about z) between two
    solved fields on the same mesh, from the Maxwell-stress cross terms:

        tau_n = (2π·L_stk·r²/μ0) · 2·Re[ĉ_r^a(n)·ĉ_θ^b(n)* + ĉ_r^b(n)·ĉ_θ^a(n)*]

    with ĉ the complex Fourier coefficients of B_r/B_θ on the circle.
    At order = n_p between a magnet-only and an armature-only solve this
    is the synchronous mean torque — the piece that closes to the circuit
    tier's 1.5·n_p·ψf·i_q. The full static ``maxwell_stress_torque``
    additionally carries position-locked slot-harmonic ripple (armature
    comb sidebands at m·n_slots ± n_p meeting magnet space harmonics of
    the same order), which averages out under rotation.

    The circle must lie in the source-free gap annulus; both solves must
    share the mesh (same geometry/maxh — the mesh cache guarantees this).
    Sign convention matches maxwell_stress_torque.
    """
    B_r_a, B_t_a = _sample_B_polar(mesh, gfu_a, radius, n_theta)
    B_r_b, B_t_b = _sample_B_polar(mesh, gfu_b, radius, n_theta)
    c_r_a = np.fft.rfft(B_r_a)[order] / n_theta
    c_t_a = np.fft.rfft(B_t_a)[order] / n_theta
    c_r_b = np.fft.rfft(B_r_b)[order] / n_theta
    c_t_b = np.fft.rfft(B_t_b)[order] / n_theta
    cross = 2.0 * np.real(c_r_a * np.conj(c_t_b) + c_r_b * np.conj(c_t_a))
    return float(2 * pi * L_stk * radius * radius / MU0 * cross)


def sample_magnet_Bm(
    mesh: Mesh,
    gfu: GridFunction,
    geo: Geometry,
    n_p: int,
    alpha_p: float = 1.0,
    rotation: float = 0.0,
    n_r: int = 8,
    n_arc: int = 24,
    magnetization: Magnetization = "radial",
    halbach_segments: int | None = None,
    halbach_angles_rad: tuple[float, ...] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating],
           NDArray[np.floating], NDArray[np.floating]]:
    """Flux density along the magnetization direction inside the magnets.

    Samples B_m on a cell-centered polar grid over the PM annulus.
    Returns flat arrays ``(r, theta, B_m, w)`` with ``w`` the fractional
    area weights (proportional to r, summing to 1) for volume-fraction
    statistics.

    For radial magnetization, B_m = sign(M_r) * B_r (n_arc points per
    pole, 2*n_p poles).

    For Halbach magnetization, B_m = B dot M_hat per segment (n_arc
    points per segment, 2*n_p*kappa_m segments), where M_hat is the
    segment's magnetization unit vector — the same direction logic as
    ``solve_field_fem``.

    Magnetization-discontinuity caveat: at every boundary between
    oppositely magnetized regions — 2*n_p pole transitions for radial,
    2*n_p*kappa_m segment transitions for Halbach — the sharp M jump
    drives a locally large opposing self-field. Cell-centering keeps
    sample points away from these boundaries (nearest point is
    half a cell width inside), but the sampled B_m_min can still pin
    at a boundary-adjacent cell rather than tracking the armature
    d-axis field.  This makes a min-based demag criterion
    over-conservative; with discrete arcs (alpha_p < 1, radial only)
    the interpole gap removes the corner.
    """
    from ngsolve import grad, sqrt, x, y

    if n_p < 1:
        raise ValueError("n_p must be >= 1")
    if not 0.0 < alpha_p <= 1.0:
        raise ValueError("alpha_p must be in (0, 1]")

    # Magnet annulus net of the bridge ring — sampling into the bridge
    # would read iron, not magnet, operating points. On both topologies
    # r_pm_inner < r_pm_outer numerically (inrunner: r_rotor .. r_magnet-t;
    # outrunner: r_magnet+t .. r_rotor).
    r_pm_outer, r_pm_inner = pm_radii(geo)
    r_lo, r_hi = r_pm_inner, r_pm_outer

    r_grid = r_lo + (np.arange(n_r) + 0.5) * (r_hi - r_lo) / n_r

    if magnetization == "halbach":
        if halbach_segments is None or halbach_segments < 1:
            raise ValueError(
                "halbach_segments must be >= 1 for Halbach magnetization"
            )
        kappa_m = halbach_segments
        n_seg = 2 * n_p * kappa_m
        seg_pitch = pi / (n_p * kappa_m)
        seg_half = seg_pitch / 2
        sign_h = -1.0 if geo.topology == "inrunner" else 1.0

        t_local = (
            -seg_half + (np.arange(n_arc) + 0.5) * seg_pitch / n_arc
        )

        r_all, t_all, psi_all = [], [], []
        for m in range(n_seg):
            theta_c = rotation + m * seg_pitch
            local_idx = m % (2 * kappa_m)
            if local_idx < kappa_m:
                if halbach_angles_rad is not None:
                    phi_local = halbach_angles_rad[local_idx]
                else:
                    phi_local = sign_h * n_p * local_idx * seg_pitch
            else:
                mirror_idx = local_idx - kappa_m
                if halbach_angles_rad is not None:
                    phi_local = halbach_angles_rad[mirror_idx] - pi
                else:
                    phi_local = sign_h * n_p * mirror_idx * seg_pitch - pi
            psi = theta_c + phi_local

            tt, rr = np.meshgrid(theta_c + t_local, r_grid)
            r_all.append(rr.ravel())
            t_all.append(tt.ravel())
            psi_all.append(np.full(rr.size, psi))

        r = np.concatenate(r_all)
        theta = np.concatenate(t_all)
        psi_arr = np.concatenate(psi_all)

        B_x_cf = grad(gfu)[1]
        B_y_cf = -grad(gfu)[0]
        mips = mesh(r * np.cos(theta), r * np.sin(theta))
        B_x = np.asarray(B_x_cf(mips), dtype=float).reshape(-1)
        B_y = np.asarray(B_y_cf(mips), dtype=float).reshape(-1)
        B_m = B_x * np.cos(psi_arr) + B_y * np.sin(psi_arr)

        w = r / np.sum(r)
        return r, theta, B_m, w

    arc_half = alpha_p * pi / (2 * n_p)
    t_local = -arc_half + (np.arange(n_arc) + 0.5) * (2 * arc_half) / n_arc

    r_all, t_all, s_all = [], [], []
    for k in range(2 * n_p):
        theta_k = pi * k / n_p + rotation
        tt, rr = np.meshgrid(theta_k + t_local, r_grid)
        r_all.append(rr.ravel())
        t_all.append(tt.ravel())
        s_all.append(np.full(rr.size, 1.0 if k % 2 == 0 else -1.0))
    r = np.concatenate(r_all)
    theta = np.concatenate(t_all)
    sign = np.concatenate(s_all)

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf
    mips = mesh(r * np.cos(theta), r * np.sin(theta))
    B_r = np.asarray(B_r_cf(mips), dtype=float).reshape(-1)

    w = r / np.sum(r)
    return r, theta, sign * B_r, w


def demag_margin(
    mesh: Mesh,
    gfu: GridFunction,
    geo: Geometry,
    n_p: int,
    B_knee: float,
    alpha_p: float = 1.0,
    rotation: float = 0.0,
    n_r: int = 8,
    n_arc: int = 24,
    magnetization: Magnetization = "radial",
    halbach_segments: int | None = None,
    halbach_angles_rad: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Worst-point demag margin of the solved field against a knee B_knee.

    Thin statistics layer over :func:`sample_magnet_Bm`: ``B_m_min`` (T,
    worst magnet operating point), ``margin`` (B_m_min − B_knee, negative
    means demag onset somewhere), ``frac_below_knee`` (area fraction of
    magnet cross-section with B_m < B_knee) and the worst point's
    ``r_min`` / ``theta_min``. The caller supplies B_knee at the magnet
    operating temperature (grade data — not derived here).
    """
    r, theta, B_m, w = sample_magnet_Bm(mesh, gfu, geo, n_p,
                                        alpha_p=alpha_p, rotation=rotation,
                                        n_r=n_r, n_arc=n_arc,
                                        magnetization=magnetization,
                                        halbach_segments=halbach_segments,
                                        halbach_angles_rad=halbach_angles_rad)
    i = int(np.argmin(B_m))
    return {
        "B_m_min": float(B_m[i]),
        "margin": float(B_m[i] - B_knee),
        "frac_below_knee": float(np.sum(w[B_m < B_knee])),
        "r_min": float(r[i]),
        "theta_min": float(theta[i]),
    }


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def rasterise_cross_section(
    mesh: Mesh,
    gfu: GridFunction,
    n_grid: int = 100,
    r_bound: float = DEFAULT_GEO.r_outer,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate A_z and |B| on a regular Cartesian grid covering the motor disk."""
    from ngsolve import grad, sqrt

    B_mag_cf = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    xi, yi, fields = rasterise_fields(mesh, {"Az": gfu, "Bmag": B_mag_cf}, n_grid, r_bound)
    return xi, yi, fields["Az"], fields["Bmag"]


def rasterise_fields(
    mesh: Mesh,
    fields: Mapping[str, Any],
    n_grid: int = 100,
    r_bound: float = DEFAULT_GEO.r_outer,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, NDArray[np.floating]]]:
    """Evaluate real scalar fields (GridFunctions or CoefficientFunctions,
    the latter projected onto an order-3 L2 space first) on a regular
    Cartesian grid over the disk r < r_bound; NaN outside the mesh."""
    from ngsolve import L2, GridFunction

    fes_L2 = L2(mesh, order=3)
    gfs: dict[str, Any] = {}
    for name, cf in fields.items():
        if isinstance(cf, GridFunction):
            gfs[name] = cf
            continue
        gf = GridFunction(fes_L2)
        gf.Set(cf)
        gfs[name] = gf

    coords = np.linspace(-r_bound, r_bound, n_grid)
    xi, yi = np.meshgrid(coords, coords)
    out = {name: np.full((n_grid, n_grid), np.nan) for name in gfs}

    r2 = xi**2 + yi**2
    inside = r2 < r_bound**2
    if not np.any(inside):
        return xi, yi, out

    # One batched point-locate for the whole grid; points the mesh does
    # not contain come back with nr == -1 and must be excluded before
    # GridFunction evaluation (which raises on any missing point).
    mips = mesh(xi[inside], yi[inside])
    found = mips["nr"] != -1
    if np.any(found):
        mips_ok = mips[found]
        for name, gf in gfs.items():
            flat = np.full(mips.shape, np.nan)
            flat[found] = np.asarray(gf(mips_ok), dtype=float).reshape(-1)
            out[name][inside] = flat

    failed = int(np.count_nonzero(~found))
    if failed:
        logging.warning("rasterise: %d points failed evaluation", failed)

    return xi, yi, out
