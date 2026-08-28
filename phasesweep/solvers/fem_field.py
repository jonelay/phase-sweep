"""
2D magnetostatic FEM field solver for PMSM air-gap flux density.

Physics: A-formulation, H1 scalar vector potential A_z.
Equation: -∇·(ν ∇A_z) = ∇×(ν_PM B_rem)|_z  in PM region
Source:   square-wave radial magnetization — the curl-curl source is the
          coercive field |H_c| = ν_PM·B_rem = B_rem/(μ0·μr_pm) (reduces to
          B_rem/μ0 only at μr_pm=1), per-pole sign alternation sign(cos(n_p θ))

Uses NGSolve (LGPL) for FEM. Analytical model: phasesweep.analytical;
harmonic decomposition: phasesweep.harmonics.
"""

from __future__ import annotations

import logging
from math import pi
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ngsolve import CoefficientFunction, GridFunction, Mesh
    from ngsolve.comp import FESpace, LinearForm

from phasesweep.machines.geometry import Geometry
from phasesweep.solvers.fem_materials import _MU0, _bh_nu, _bh_nu_prime
from phasesweep.solvers.fem_mesh import (
    _MIN_ARC_GAP_M,
    _arcs_active,
    _circle_spec,
    _get_or_build_mesh,
    _interpole_gap_width,
)
from phasesweep.solvers.fem_sampling import sample_Br

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_MIN_PICARD_RELAX: float = 0.02

# Newton BH solve: stop threshold on the relative algebraic residual
# ||K(nu(u))u - f|| / ||f||. 1e-4 matches or beats Picard's
# shipping-tolerance B1 accuracy on the production nonlinear cases
# (scripts/newton_bh_spike.py). Before the Newton loop, a few relaxed
# Picard nu-updates improve the seed (the Elmer MagnetoDynamics2D /
# Vande Sande hybrid start-up pattern); unrelaxed start-up steps are
# not safe — see the divergence comment in _picard_solve.
_NEWTON_RTOL: float = 1e-4
_NEWTON_STARTUP_PICARD: int = 2


# ---------------------------------------------------------------------------
# FEM solver — new Geometry-aware signature
# ---------------------------------------------------------------------------

def _demoivre(n_p: int) -> tuple[CoefficientFunction, CoefficientFunction]:
    """Return (cos(n_p θ), sin(n_p θ)) as NGSolve CFs via De Moivre."""
    from ngsolve import sqrt, x, y
    r_cf = sqrt(x**2 + y**2)
    cos_t, sin_t = x / r_cf, y / r_cf
    cr, si = cos_t, sin_t
    for _ in range(n_p - 1):
        cr, si = cr * cos_t - si * sin_t, cr * sin_t + si * cos_t
    return cr, si


def _picard_solve(
    mesh: Mesh,
    fes: FESpace,
    lf: LinearForm,
    gfu: GridFunction,
    nu: CoefficientFunction,
    region_names: list[str],
    iron_regions: set[str],
    max_picard: int,
    picard_tol: float,
    picard_relax: float,
    info: dict[str, float] | None,
) -> None:
    """Nonlinear BH solve: Picard fixed-point with adaptive relaxation.

    Solves in place into ``gfu``. Stops on the raw (unrelaxed) nu step;
    damps relax on raw-step growth and recovers it on decay.

    Raises:
        RuntimeError: If the iteration does not converge within
            ``max_picard`` iterations.
    """
    from ngsolve import L2, BilinearForm, GridFunction, dx, grad, sqrt

    u, v = fes.TnT()
    fes_nu = L2(mesh, order=0)
    gfu_nu = GridFunction(fes_nu)
    gfu_nu.Set(nu)

    is_iron_cf: dict[str, float] = {
        name: (1.0 if name in iron_regions else 0.0)
        for name in region_names
    }
    is_iron = mesh.MaterialCF(is_iron_cf)
    iron_indicator = GridFunction(fes_nu)
    iron_indicator.Set(is_iron)
    iron_mask = iron_indicator.vec.FV().NumPy() > 0.5

    a = BilinearForm(gfu_nu * grad(u) * grad(v) * dx)

    delta_raw_rel = float("inf")
    prev_raw_rel = float("inf")
    prev_applied_rel = float("inf")
    relax = picard_relax
    for _picard_i in range(max_picard):
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

        B_mag_cf = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
        gfu_Bmag = GridFunction(fes_nu)
        gfu_Bmag.Set(B_mag_cf)
        B_vals = gfu_Bmag.vec.FV().NumPy().copy()

        nu_new = _bh_nu(B_vals)
        nu_old = gfu_nu.vec.FV().NumPy().copy()

        # Stop criterion on the raw (unrelaxed) step: relax-independent,
        # so the effective tolerance cannot loosen as relax shrinks
        # (the old relax-scaled criterion accepted raw residual 0.40 at
        # the relax floor).
        scale = np.max(np.abs(nu_old[iron_mask]))
        delta_raw_rel = (
            np.max(np.abs(nu_new[iron_mask] - nu_old[iron_mask])) / scale
        )

        if delta_raw_rel < picard_tol:
            # gfu was solved with nu_old and nu_new ~ nu_old: the pair
            # is a fixed point within tol — stop without updating.
            break

        if delta_raw_rel > prev_raw_rel:
            # Diverging: damp, and cap the applied step at the previous
            # applied step. This damping must stay — relax >= 0.35
            # diverges without it on saturating geometries.
            relax = max(relax * 0.5, _MIN_PICARD_RELAX)
            while (relax * delta_raw_rel > prev_applied_rel
                   and relax > _MIN_PICARD_RELAX):
                relax = max(relax * 0.5, _MIN_PICARD_RELAX)
        else:
            # Converging: recover toward the configured relaxation so a
            # saturation transient cannot pin relax at the floor for
            # the rest of the solve.
            relax = min(relax * 1.5, picard_relax)

        nu_update = nu_old.copy()
        nu_update[iron_mask] = (
            relax * nu_new[iron_mask]
            + (1 - relax) * nu_old[iron_mask]
        )
        gfu_nu.vec.FV().NumPy()[:] = nu_update
        prev_applied_rel = relax * delta_raw_rel
        prev_raw_rel = delta_raw_rel
    else:
        raise RuntimeError(
            f"Picard iteration did not converge in {max_picard} iterations "
            f"(residual: {delta_raw_rel:.4e}, tol: {picard_tol})"
        )
    if info is not None:
        info["picard_iterations"] = float(_picard_i + 1)
        info["picard_relax_final"] = float(relax)
        info["picard_residual"] = float(delta_raw_rel)


def _newton_solve(
    mesh: Mesh,
    fes: FESpace,
    lf: LinearForm,
    gfu: GridFunction,
    nu: CoefficientFunction,
    region_names: list[str],
    iron_regions: set[str],
    max_newton: int,
    info: dict[str, float] | None,
) -> None:
    """Nonlinear BH solve: Newton iteration with line-search damping.

    Same discrete system as ``_picard_solve`` (element-wise L2^0 nu from
    the BH table, element-averaged |B|), so both solvers converge to the
    same solution — verified to ~1e-6 T on the production cases by
    scripts/newton_bh_spike.py. Stops on the algebraic residual
    ``_NEWTON_RTOL``, NOT on a nu-step criterion: the line search leaves
    nu at nu(u), so a nu-step check here would compare nu(u) with itself
    and report false convergence after one step.

    The Jacobian uses the normalized SPD tangent form
    nu_t I + (nu_l - nu_t) n n^T (n = grad A/|grad A| pointwise,
    nu_l = dH/dB per element — see the jac comment below). nu_l is
    evaluated at the element-averaged |B| while n is pointwise — an
    inconsistent Jacobian, so the asymptotic rate is linear rather than
    quadratic; irrelevant at the 1e-4 stop (several times fewer
    iterations than Picard on the production cases,
    scripts/newton_bh_spike.py).

    Raises:
        RuntimeError: If the line search stalls or ``max_newton`` is
            exhausted. Callers fall back to ``_picard_solve``.
    """
    from ngsolve import L2, BilinearForm, GridFunction, dx, grad, sqrt

    u, v = fes.TnT()
    fes_nu = L2(mesh, order=0)
    gfu_nu = GridFunction(fes_nu)
    gfu_nu.Set(nu)

    is_iron_cf: dict[str, float] = {
        name: (1.0 if name in iron_regions else 0.0)
        for name in region_names
    }
    iron_indicator = GridFunction(fes_nu)
    iron_indicator.Set(mesh.MaterialCF(is_iron_cf))
    iron_mask = iron_indicator.vec.FV().NumPy() > 0.5

    free = fes.FreeDofs()
    free_mask = np.array([free[i] for i in range(fes.ndof)])
    lf_np = lf.vec.FV().NumPy().copy()

    gg = grad(gfu)
    gfu_coeff = GridFunction(fes_nu)
    a = BilinearForm(gfu_nu * grad(u) * grad(v) * dx)
    # Normalized SPD tangent form (ported from the wedge Newton):
    # nu_t I + (nu_l - nu_t) n n^T with n = grad A/|grad A| pointwise,
    # nu_t = nu(B_e) and nu_l = dH/dB(B_e) per element. Pointwise
    # eigenvalues are exactly {nu_l, nu_t, nu_t} > 0 (the monotone BH
    # spline keeps dH/dB > 0), so jac stays SPD even where nu' < 0 —
    # which the PCHIP secant makes common at low B. The old
    # un-normalized (nu'/|B_e|)(grad A . grad u)(grad A . grad v) term
    # went indefinite there (pointwise |grad A| >> B_e in corners) and
    # stalled convergence on saturating geometries. The 1e-30 T^2
    # regularizer only matters at exactly-zero gradient, where the
    # numerator vanishes too.
    jac = BilinearForm(
        gfu_nu * grad(u) * grad(v) * dx
        + gfu_coeff * (gg * grad(u)) * (gg * grad(v)) / (gg * gg + 1e-30) * dx
    )

    tmp = gfu.vec.CreateVector()
    rvec = gfu.vec.CreateVector()
    step = gfu.vec.CreateVector()

    def project_Bmag() -> NDArray:
        gfu_B = GridFunction(fes_nu)
        gfu_B.Set(sqrt(gg[0]**2 + gg[1]**2))
        return gfu_B.vec.FV().NumPy().copy()

    def residual_np() -> NDArray:
        a.Assemble()
        tmp.data = a.mat * gfu.vec
        r = tmp.FV().NumPy() - lf_np
        r[~free_mask] = 0.0
        return r

    # Iteration 0: linear solve with the seed nu (identical to Picard's
    # first solve).
    a.Assemble()
    gfu.vec.data = a.mat.Inverse(free, inverse="sparsecholesky") * lf.vec

    nu_seed = gfu_nu.vec.FV().NumPy().copy()

    # Picard start-up: a few relaxed nu-updates before Newton (hybrid
    # Picard->Newton pattern). Relaxed with the shipping Picard factor —
    # unrelaxed steps diverge on saturating geometries (see
    # _picard_solve). Capped by the shared iteration budget, always
    # reserving at least one Newton iteration.
    from phasesweep.defaults import PICARD_RELAXATION
    n_startup = min(_NEWTON_STARTUP_PICARD, max(0, max_newton - 1))
    for _ in range(n_startup):
        nu_cur = gfu_nu.vec.FV().NumPy().copy()
        nu_upd = nu_cur.copy()
        nu_upd[iron_mask] = (
            PICARD_RELAXATION * _bh_nu(project_Bmag())[iron_mask]
            + (1 - PICARD_RELAXATION) * nu_cur[iron_mask]
        )
        gfu_nu.vec.FV().NumPy()[:] = nu_upd
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(free, inverse="sparsecholesky") * lf.vec

    f_norm = float(np.linalg.norm(lf_np[free_mask]))
    n_backtracks = 0
    rel_res = float("inf")
    for _newton_i in range(max(1, max_newton - n_startup)):
        B_vals = project_Bmag()
        nu_upd = nu_seed.copy()
        nu_upd[iron_mask] = _bh_nu(B_vals)[iron_mask]
        gfu_nu.vec.FV().NumPy()[:] = nu_upd
        r_np = residual_np()
        rnorm = float(np.linalg.norm(r_np))
        rel_res = rnorm / f_norm
        if rel_res < _NEWTON_RTOL:
            break

        # Tangent slope nu_l = dH/dB of the effective clipped curve,
        # nu + B*nu' — analytic from the BH spline (equals nu on the
        # clip plateau, where the correction vanishes). Positive by
        # monotonicity; floored well below nu_0 as pure safety.
        nu_p = _bh_nu_prime(B_vals)
        nu_l = np.maximum(nu_upd + B_vals * nu_p, 1.0)
        gfu_coeff.vec.FV().NumPy()[:] = np.where(
            iron_mask, nu_l - nu_upd, 0.0)

        jac.Assemble()
        jinv = jac.mat.Inverse(free, inverse="sparsecholesky")

        rvec.FV().NumPy()[:] = r_np
        step.data = jinv * rvec
        step_np = step.FV().NumPy().copy()

        u0 = gfu.vec.FV().NumPy().copy()
        alpha, accepted = 1.0, False
        for _ in range(8):
            gfu.vec.FV().NumPy()[:] = u0 - alpha * step_np
            nu_trial = nu_seed.copy()
            nu_trial[iron_mask] = _bh_nu(project_Bmag())[iron_mask]
            gfu_nu.vec.FV().NumPy()[:] = nu_trial
            if float(np.linalg.norm(residual_np())) < rnorm:
                accepted = True
                break
            n_backtracks += 1
            alpha *= 0.5
        if not accepted:
            gfu.vec.FV().NumPy()[:] = u0
            raise RuntimeError(
                f"Newton line search stalled at iteration {_newton_i + 1} "
                f"(rel residual: {rel_res:.4e})"
            )
    else:
        raise RuntimeError(
            f"Newton iteration did not converge in {max_newton} iterations "
            f"({n_startup} Picard start-up + {max_newton - n_startup} Newton; "
            f"rel residual: {rel_res:.4e}, tol: {_NEWTON_RTOL})"
        )
    if info is not None:
        info["newton_iterations"] = float(_newton_i + 1)
        info["newton_startup_iterations"] = float(n_startup)
        info["newton_backtracks"] = float(n_backtracks)
        info["newton_rel_residual"] = float(rel_res)


def _check_iron_saturation(
    mesh: Mesh,
    gfu: GridFunction,
    region_names: list[str],
    iron_regions: set[str],
    info: dict[str, float] | None,
) -> None:
    """Record peak |B| in iron into ``info``; warn above 1.8 T."""
    from ngsolve import L2, GridFunction, grad, sqrt

    B_mag_check = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    fes_L2_check = L2(mesh, order=0)
    gfu_Bcheck = GridFunction(fes_L2_check)
    gfu_Bcheck.Set(B_mag_check)
    iron_ind = GridFunction(fes_L2_check)
    iron_cf_check: dict[str, float] = {
        name: (1.0 if name in iron_regions else 0.0)
        for name in region_names
    }
    iron_ind.Set(mesh.MaterialCF(iron_cf_check))
    B_iron = gfu_Bcheck.vec.FV().NumPy()[iron_ind.vec.FV().NumPy() > 0.5]
    b_iron_max = float(np.max(B_iron)) if len(B_iron) > 0 else 0.0
    if info is not None:
        info["b_iron_max"] = b_iron_max
    if b_iron_max > 1.8:
        logging.warning(
            "peak |B| in stator iron = %.2f T (> 1.8 T saturation threshold)",
            b_iron_max,
        )


@overload
def solve_field_fem(
    geo: Geometry, n_p: int, B_rem: float, mu_r_pm: float, mu_r_fe: float,
    n_theta: int = ..., maxh_fraction: float = ...,
    return_full: Literal[False] = ...,
    n_slots: int | None = ..., j_s: float = ..., k_w: float = ...,
    slot_width_ratio: float | None = ..., nonlinear: bool = ...,
    max_picard: int = ..., picard_tol: float = ..., picard_relax: float = ...,
    nonlinear_solver: str = ...,
    alpha_p: float = ..., rotation: float = ..., sheet_phase: float = ...,
    info: dict[str, float] | None = ...,
    magnetization: str = ..., halbach_segments: int | None = ...,
    halbach_angles_rad: tuple[float, ...] | None = ...,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

@overload
def solve_field_fem(
    geo: Geometry, n_p: int, B_rem: float, mu_r_pm: float, mu_r_fe: float,
    n_theta: int = ..., maxh_fraction: float = ...,
    return_full: Literal[True] = ...,
    n_slots: int | None = ..., j_s: float = ..., k_w: float = ...,
    slot_width_ratio: float | None = ..., nonlinear: bool = ...,
    max_picard: int = ..., picard_tol: float = ..., picard_relax: float = ...,
    nonlinear_solver: str = ...,
    alpha_p: float = ..., rotation: float = ..., sheet_phase: float = ...,
    info: dict[str, float] | None = ...,
    magnetization: str = ..., halbach_segments: int | None = ...,
    halbach_angles_rad: tuple[float, ...] | None = ...,
) -> tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]: ...

def solve_field_fem(
    geo: Geometry,
    n_p: int,
    B_rem: float,
    mu_r_pm: float,
    mu_r_fe: float,
    n_theta: int = 360,
    maxh_fraction: float = 0.05,
    return_full: bool = False,
    n_slots: int | None = None,
    j_s: float = 0.0,
    k_w: float = 0.966,
    slot_width_ratio: float | None = None,
    nonlinear: bool = False,
    max_picard: int | None = None,
    picard_tol: float | None = None,
    picard_relax: float | None = None,
    nonlinear_solver: str = "newton",
    alpha_p: float = 1.0,
    rotation: float = 0.0,
    sheet_phase: float = 0.0,
    info: dict[str, float] | None = None,
    magnetization: str = "radial",
    halbach_segments: int | None = None,
    halbach_angles_rad: tuple[float, ...] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]:
    """Geometry-aware FEM solver with discrete magnet arcs.

    Inrunner idealization: everything inside ``r_rotor`` is solid iron —
    ``geo.r_inner`` does not enter the mesh (valid for magnetic steel
    shafts; hollow or non-magnetic shafts are not represented).

    ``mu_r_fe`` sets the iron reluctivity of the linear solve only; with
    ``nonlinear=True`` iron follows the generic B-H table
    and ``mu_r_fe`` merely seeds Picard iteration 0 — the converged
    result is insensitive to its value.

    When ``alpha_p < 1.0``, builds discrete magnet arcs with interpole gaps
    named ``"pm_gap"`` (air reluctivity, no source). The square-wave source
    term (uniform magnitude, per-pole sign) integrates only over ``"pm"``
    regions. Pass physical remanence as ``B_rem`` (no sin(πα_p/2) or 4/π
    pre-scaling needed). Interpole gaps narrower than 0.5 mm
    (``_MIN_ARC_GAP_M``) cannot be cut reliably by OCC: the arcs collapse
    to a full PM ring and the solve is effectively ``alpha_p = 1.0`` — a
    warning is logged and ``info["arcs_collapsed"]`` is set. This fires
    for ``alpha_p >= 1 - _MIN_ARC_GAP_M * n_p / (pi * r_pm_inner)``, where
    ``r_pm_inner`` is the inner PM-annulus radius (``r_rotor`` inrunner,
    ``r_magnet`` outrunner) — the gap wedge is narrowest there.

    Inset rotors (``geo.magnet_mount == "inset"``): the interpole gaps
    become rotor iron instead of air, and ``geo.bridge_thickness`` (when
    set) inserts an iron "bridge" annulus between the magnet faces and
    the airgap — the magnet arcs then stop ``bridge_thickness`` short of
    the airgap. Both regions are nonlinear iron; ``nonlinear=True`` is
    mandatory (the leakage shunt through them saturates by design, a
    linear solve grossly overestimates it) and the solve raises if the
    magnet arcs cannot be built (``alpha_p = 1.0`` or a sub-threshold
    interpole gap).

    ``n_slots`` and ``slot_width_ratio`` default to the geometry's values;
    pass an explicit value (including 0) to override, e.g. ``n_slots=0``
    for a smooth-bore solve of a slotted geometry.

    Armature reaction (``j_s != 0``) is an equivalent current sheet
    ``J_z = -j_s * k_w * cos(n_p * θ - sheet_phase)`` integrated over the
    slot regions: ``k_w`` scales the sheet to the effective (distributed)
    winding, and ``sheet_phase`` (electrical radians) sets the current
    angle. At ``sheet_phase = 0`` the MMF lies purely on the q-axis (the
    magnet d-axis sits at θ=0) — the torque-producing sheet. At
    ``sheet_phase = ±π/2`` it lies on the d-axis and produces no
    order-n_p interaction torque; ``+π/2`` opposes the magnet flux
    (demagnetizing / flux-weakening) and ``-π/2`` aids it, on both
    topologies (sign pinned numerically, see tests). The j_s ↔
    phase-current mapping is phase-invariant: the uniform slot comb
    rotates the sampled fundamental phasor without changing its
    amplitude, so the same ``j_s`` represents the same current magnitude
    at any angle. Requires slots: ``j_s != 0`` with ``n_slots == 0``
    raises ValueError.

    ``rotation`` rotates the magnet pattern (arc positions and the
    per-pole magnetization sign, i.e. the whole rotor) by a mechanical
    angle in radians; the stator (slots, current sheet) stays fixed.
    Sweeping ``rotation`` over one cogging period 2π/lcm(n_slots, 2·n_p)
    with ``j_s = 0`` and ``arkkio_torque`` per position yields the
    cogging waveform (see ``cogging.py``). A combined ``j_s != 0`` + ``rotation != 0``
    solve is a fixed-current locked-rotor snapshot, not a synchronous
    operating point; setting ``sheet_phase = n_p * rotation + γ`` holds
    the sheet MMF at a fixed electrical angle γ from the rotated magnet
    axis. Each distinct rotation builds and caches its own mesh, in memory
    and (when a disk cache is configured) on disk — unless the geometry has
    no discrete magnet arcs, in which case all rotations share one mesh.

    ``nonlinear_solver`` selects the BH iteration (ignored when
    ``nonlinear=False``, like the ``picard_*`` knobs): ``"newton"``
    (default — line-search damped Newton with a short relaxed-Picard
    start-up, same discrete system and converged solution as Picard,
    several times fewer iterations on the production nonlinear cases;
    see scripts/newton_bh_spike.py) or ``"picard"``
    (adaptive-relaxation fixed point). Newton ignores ``picard_tol`` and
    ``picard_relax`` (it stops on the algebraic residual,
    ``_NEWTON_RTOL``) but shares the ``max_picard`` iteration budget;
    if it stalls, the solve falls back to Picard automatically and sets
    ``info["newton_fell_back"]``.

    ``info``, when provided, is populated with solve diagnostics:
    ``b_iron_max`` (peak |B| in iron regions, T), ``arcs_collapsed``
    (only when a sub-0.5 mm interpole gap collapsed the magnet arcs, see
    above) and — nonlinear only — ``picard_iterations``,
    ``picard_relax_final`` and ``picard_residual`` (raw, i.e. unrelaxed,
    step — the stop criterion), or for the Newton path
    ``newton_iterations``, ``newton_backtracks`` and
    ``newton_rel_residual``.

    Contract:
        Preconditions:
            - ``n_p`` >= 1.
            - ``maxh_fraction`` > 0.
            - ``picard_relax`` > 0 when ``nonlinear=True``.

        Raises:
            RuntimeError: If ``nonlinear=True`` and Picard iteration
                does not converge within ``max_picard`` iterations.
    """
    from phasesweep.defaults import (
        PICARD_MAX_ITERATIONS,
        PICARD_RELAXATION,
        PICARD_TOLERANCE,
    )
    if max_picard is None:
        max_picard = PICARD_MAX_ITERATIONS
    if picard_tol is None:
        picard_tol = PICARD_TOLERANCE
    if picard_relax is None:
        picard_relax = PICARD_RELAXATION
    if n_slots is None:
        n_slots = geo.n_slots
    if slot_width_ratio is None:
        slot_width_ratio = geo.slot_width_ratio
    if j_s != 0.0 and n_slots == 0:
        raise ValueError(
            "j_s != 0 requires n_slots > 0: the armature current sheet "
            "integrates over slot regions and would be silently ignored "
            "on a smooth-bore geometry"
        )
    if nonlinear_solver not in ("picard", "newton"):
        raise ValueError(
            f"nonlinear_solver must be 'picard' or 'newton', "
            f"got {nonlinear_solver!r}"
        )

    _is_inset = geo.magnet_mount == "inset"
    _has_bridge = geo.bridge_thickness is not None
    if _is_inset and not nonlinear:
        raise ValueError(
            "magnet_mount='inset' requires nonlinear=True: the interpole "
            "iron (and bridge ring, when set) saturates by design — a "
            "linear solve grossly overestimates the leakage shunt"
        )
    if _has_bridge and not _is_inset and not nonlinear:
        raise ValueError(
            "bridge_thickness requires nonlinear=True: the iron bridge "
            "ring saturates by design — a linear solve grossly "
            "overestimates the shunt flux through it"
        )

    # Determine if discrete arcs will actually be built (same threshold as
    # _build_motor_geometry). A sub-threshold gap collapses the arcs to a
    # full PM ring — the FEM then solves α_p = 1.0, not the requested α_p.
    _wants_arcs = alpha_p < 1.0 and n_p > 0
    _has_arcs = _arcs_active(geo, alpha_p, n_p)
    if _is_inset and not _has_arcs:
        raise ValueError(
            "magnet_mount='inset' needs discrete magnet arcs, but "
            + ("alpha_p=1.0 builds none"
               if not _wants_arcs else
               f"the interpole gap "
               f"{_interpole_gap_width(geo, alpha_p, n_p) * 1e3:.3g} mm is "
               f"below the {_MIN_ARC_GAP_M * 1e3:.3g} mm OCC limit and the "
               f"arcs would silently collapse to a full ring")
        )
    if _wants_arcs and not _has_arcs:
        logging.warning(
            "interpole gap %.3g mm < %.3g mm OCC limit: magnet arcs "
            "collapsed, FEM solves alpha_p=1.0 instead of requested "
            "alpha_p=%.4g (fundamental error ~1/sin(pi*alpha_p/2) - 1)",
            _interpole_gap_width(geo, alpha_p, n_p) * 1e3,
            _MIN_ARC_GAP_M * 1e3, alpha_p,
        )
        if info is not None:
            info["arcs_collapsed"] = 1.0

    if magnetization not in ("radial", "halbach"):
        raise ValueError(
            f"magnetization must be 'radial' or 'halbach', "
            f"got {magnetization!r}"
        )
    if magnetization == "halbach":
        if halbach_segments is None or halbach_segments < 1:
            raise ValueError(
                f"halbach_segments must be >= 1 when magnetization='halbach', "
                f"got {halbach_segments}"
            )
        if alpha_p != 1.0:
            raise ValueError(
                f"Halbach magnetization requires alpha_p=1.0, "
                f"got alpha_p={alpha_p}"
            )
        if (
            halbach_angles_rad is not None
            and len(halbach_angles_rad) != halbach_segments
        ):
            raise ValueError(
                f"halbach_angles_rad length ({len(halbach_angles_rad)}) "
                f"must equal halbach_segments ({halbach_segments})"
            )

    # Imported after the validation gates so they stay loud without
    # phasesweep[fem] installed.
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        IfPos,
        LinearForm,
        dx,
        grad,
        sqrt,
        x,
        y,
    )

    _is_halbach = magnetization == "halbach" and halbach_segments is not None
    mesh = _get_or_build_mesh(geo, maxh_fraction, n_slots, slot_width_ratio,
                              alpha_p, n_p, _has_arcs, rotation,
                              magnetization=magnetization,
                              halbach_segments=halbach_segments)

    if j_s != 0.0 and "slot" not in set(mesh.GetMaterials()):
        # n_slots > 0 alone does not guarantee slot faces — a zero
        # slot_depth builds none, and the current sheet would integrate
        # over an empty region (this silently ignored j_s before).
        raise ValueError(
            "j_s != 0 but the mesh has no slot regions (slot_depth == 0?): "
            "the armature current sheet would be silently ignored"
        )

    # Build material property dicts from circle spec regions
    region_names = [name for _, name in _circle_spec(geo)]
    if n_slots > 0:
        region_names.append("slot")
    if _has_arcs:
        region_names.append("pm_gap")

    # Halbach: replace "pm" with per-segment region names
    if _is_halbach:
        assert halbach_segments is not None  # _is_halbach guarantees this
        pm_region_names = [f"pmh{m}" for m in range(2 * n_p * halbach_segments)]
        region_names = [n for n in region_names if n != "pm"]
        region_names.extend(pm_region_names)
    else:
        pm_region_names = ["pm"]

    iron_regions = {"stator", "shaft", "yoke", "back_iron"}
    if _is_inset:
        iron_regions.add("pm_gap")
    if _has_bridge:
        iron_regions.add("bridge")

    nu_cf: dict[str, float] = {}
    for name in region_names:
        if name in pm_region_names:
            nu_cf[name] = 1 / (_MU0 * mu_r_pm)
        elif name in iron_regions:
            nu_cf[name] = 1 / (_MU0 * mu_r_fe)
        else:
            nu_cf[name] = 1 / _MU0
    nu = mesh.MaterialCF(nu_cf)

    nu_pm = 1 / (_MU0 * mu_r_pm)
    cos_np, sin_np = _demoivre(n_p)

    r_cf = sqrt(x**2 + y**2)
    cos_t = x / r_cf
    sin_t = y / r_cf

    mx_cf: dict[str, CoefficientFunction | float] = {}
    my_cf: dict[str, CoefficientFunction | float] = {}

    if _is_halbach:
        # Per-segment constant magnetization: each segment m carries M of
        # magnitude nu_pm*B_rem in absolute direction psi(m).
        # theta_c(m) = rotation + m*pi/(n_p*kappa_m)
        # phi_local: ideal rule or custom angles (half-wave symmetry for
        # the second pole in each pole pair).
        M_mag = nu_pm * B_rem
        assert halbach_segments is not None  # _is_halbach guarantees this
        kappa_m = halbach_segments
        seg_pitch = pi / (n_p * kappa_m)
        sign = -1.0 if geo.topology == "inrunner" else 1.0
        for m, name in enumerate(pm_region_names):
            theta_c = rotation + m * seg_pitch
            local_idx = m % (2 * kappa_m)
            if local_idx < kappa_m:
                if halbach_angles_rad is not None:
                    phi_local = halbach_angles_rad[local_idx]
                else:
                    phi_local = sign * n_p * local_idx * seg_pitch
            else:
                mirror_idx = local_idx - kappa_m
                if halbach_angles_rad is not None:
                    phi_local = halbach_angles_rad[mirror_idx] - pi
                else:
                    phi_local = sign * n_p * mirror_idx * seg_pitch - pi
            psi = theta_c + phi_local
            mx_cf[name] = M_mag * float(np.cos(psi))
            my_cf[name] = M_mag * float(np.sin(psi))
        for name in region_names:
            if name not in pm_region_names:
                mx_cf[name] = 0.
                my_cf[name] = 0.
    else:
        # Radial square-wave magnetization (bit-identical to pre-Halbach)
        if rotation != 0.0:
            cos_np_rot = (cos_np * float(np.cos(n_p * rotation))
                          + sin_np * float(np.sin(n_p * rotation)))
        else:
            cos_np_rot = cos_np
        M_r = nu_pm * B_rem * IfPos(cos_np_rot, 1.0, -1.0)
        for name in region_names:
            if name == "pm":
                mx_cf[name] = M_r * cos_t
                my_cf[name] = M_r * sin_t
            else:
                mx_cf[name] = 0.
                my_cf[name] = 0.

    Mx = mesh.MaterialCF(mx_cf)
    My = mesh.MaterialCF(my_cf)

    fes = H1(mesh, order=3, dirichlet="outer")
    u, v = fes.TnT()

    pm_dx_region = "|".join(pm_region_names)
    lf_expr = (Mx * grad(v)[1] - My * grad(v)[0]) * dx(pm_dx_region)
    if n_slots > 0 and j_s != 0.0:
        # Shift the sheet wave by the electrical angle sheet_phase:
        # cos(n_p·θ − β) via De Moivre, same pattern as rotation above.
        # The sheet_phase == 0 branch keeps the CF bit-identical.
        if sheet_phase != 0.0:
            cos_np_sheet = (cos_np * float(np.cos(sheet_phase))
                            + sin_np * float(np.sin(sheet_phase)))
        else:
            cos_np_sheet = cos_np
        lf_expr = lf_expr + (-j_s * k_w * cos_np_sheet) * v * dx("slot")
    lf = LinearForm(lf_expr)
    lf.Assemble()
    gfu = GridFunction(fes)

    if not nonlinear:
        a = BilinearForm(nu * grad(u) * grad(v) * dx)
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec
    elif nonlinear_solver == "newton":
        try:
            _newton_solve(mesh, fes, lf, gfu, nu, region_names,
                          iron_regions, max_picard, info)
        except RuntimeError as e:
            logging.warning("Newton BH solve failed (%s); falling back "
                            "to Picard", e)
            if info is not None:
                info["newton_fell_back"] = 1.0
            _picard_solve(mesh, fes, lf, gfu, nu, region_names,
                          iron_regions, max_picard, picard_tol,
                          picard_relax, info)
    else:
        _picard_solve(mesh, fes, lf, gfu, nu, region_names, iron_regions,
                      max_picard, picard_tol, picard_relax, info)

    _check_iron_saturation(mesh, gfu, region_names, iron_regions, info)

    theta, B_r = sample_Br(mesh, gfu, geo.r_ag, n_theta)
    if return_full:
        return theta, B_r, mesh, gfu
    return theta, B_r
