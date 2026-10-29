"""
2D time-harmonic eddy-current solver (complex A_z) for induction machines.

Phasors e^{jωt} in the stator frame, linear materials:

    -∇·(ν ∇A) + jωσ A + σ Ω ∂A/∂θ = J_s          solid conductors
    -∇·(ν ∇A) + j s ω σ A - σ u_k  = 0           cage bar k

The velocity term σ Ω ∂A/∂θ is the exact stator-frame form of the
v × B induction for a rotor that is uniform in θ (every stator space
harmonic sees its own slip); it is what TEAM Problem 30a needs.
A discrete cage is not uniform in θ, so its bars carry the standard
rotor-frame fundamental form j s ω σ instead, and each bar has one
axial voltage gradient u_k = -∂φ/∂z as an extra scalar unknown
(eliminated by static condensation, see solve_harmonic) fixed by the
cage circuit: prescribed bar currents,
an ideal end ring (common u, ΣI_k = 0), or ring segments of impedance
Z_r between neighbouring bars at both ends.

The magnetostatic PM solver (fem_field.py) is untouched; this module
shares only the mesh builders and the material constants.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ngsolve import CoefficientFunction, GridFunction, Mesh
    from ngsolve.comp import FESpace

from phasesweep.solvers.fem_materials import MU0

__all__ = [
    "CageCircuit",
    "HarmonicResult",
    "HarmonicSpec",
    "solve_harmonic",
]

_DIRECT_SOLVERS: tuple[str, ...] = ("pardiso", "umfpack")


@dataclass(frozen=True)
class CageCircuit:
    """Cage-bar circuit closing the per-bar constraints.

    bars      — mesh regions carrying one bar each (order = ring order)
    L_stk     — axial length (m) turning u_k into a bar voltage u_k·L_stk
    currents  — prescribed complex bar currents (A, peak); None = solve
    Z_ring    — impedance (Ω) of ONE end-ring segment between adjacent
                bars at ONE end, at the rotor frequency; None = ideal
                ring (all bars at a common u, ΣI_k = 0)
    """

    bars: tuple[str, ...]
    L_stk: float = 1.0
    currents: tuple[complex, ...] | None = None
    Z_ring: complex | None = None

    def __post_init__(self) -> None:
        from math import isfinite
        if len(self.bars) < 1:
            raise ValueError("CageCircuit needs at least one bar region")
        if len(set(self.bars)) != len(self.bars):
            raise ValueError("CageCircuit bar regions must be distinct")
        if not (isfinite(self.L_stk) and self.L_stk > 0):
            raise ValueError(f"L_stk={self.L_stk} must be > 0 and finite")
        if self.currents is not None and len(self.currents) != len(self.bars):
            raise ValueError(
                f"{len(self.currents)} prescribed currents for {len(self.bars)} bars"
            )
        if self.Z_ring is not None:
            if self.currents is not None:
                raise ValueError("prescribed currents and Z_ring are exclusive")
            if self.Z_ring == 0:
                raise ValueError("Z_ring=0 is the ideal ring: pass Z_ring=None")
            if len(self.bars) < 2:
                raise ValueError("an end ring needs at least two bars")

    @property
    def per_bar_unknowns(self) -> bool:
        return self.currents is not None or self.Z_ring is not None


@dataclass(frozen=True)
class HarmonicSpec:
    """One time-harmonic solve on a named-region mesh.

    omega      — supply angular frequency (rad/s)
    nu         — reluctivity per region (1/H·m, a number or a
                 CoefficientFunction such as an element-wise field from
                 a saturation fixed point); default vacuum
    sigma      — conductivity per eddy region (S/m); regions listed in
                 ``cage.bars`` are bars, the rest solid conductors
    J_source   — imposed current density per stranded region (A/m²,
                 complex peak amplitude or CoefficientFunction)
    slip       — rotor slip for the bar regions (j s ω σ)
    omega_mech — rotor angular velocity (rad/s, counterclockwise) for
                 the solid-conductor velocity term
    order      — H1 polynomial order of A_z
    dirichlet  — boundary name carrying A_z = 0
    periodic_phase — Bloch factor for a mesh whose sides are identified
                 (fem_mesh.build_bar_cell_mesh): A on the image side =
                 phase × A on the source side. None = no periodicity.
    """

    omega: float
    nu: Mapping[str, Any] = field(default_factory=dict)
    sigma: Mapping[str, float] = field(default_factory=dict)
    J_source: Mapping[str, Any] = field(default_factory=dict)
    slip: float = 1.0
    omega_mech: float = 0.0
    cage: CageCircuit | None = None
    order: int = 3
    dirichlet: str = "outer"
    periodic_phase: complex | None = None

    def __post_init__(self) -> None:
        from math import isfinite
        if not (isfinite(self.omega) and self.omega > 0):
            raise ValueError(f"omega={self.omega} must be > 0 and finite")
        if not isfinite(self.slip):
            raise ValueError(f"slip={self.slip} must be finite")
        if not isfinite(self.omega_mech):
            raise ValueError(f"omega_mech={self.omega_mech} must be finite")
        if self.order < 1:
            raise ValueError(f"order={self.order} must be >= 1")
        for name, s in self.sigma.items():
            if not (isfinite(s) and s > 0):
                raise ValueError(f"sigma[{name!r}]={s} must be > 0 and finite")
        for name, n in self.nu.items():
            if isinstance(n, (int, float)) and not (isfinite(n) and n > 0):
                raise ValueError(f"nu[{name!r}]={n} must be > 0 and finite")
        overlap = set(self.J_source) & set(self.sigma)
        if overlap:
            raise ValueError(
                f"regions {sorted(overlap)} carry both a source and a "
                "conductivity: stranded sources have no eddy currents"
            )
        if self.cage is not None:
            missing = [b for b in self.cage.bars if b not in self.sigma]
            if missing:
                raise ValueError(f"bar regions {missing} have no conductivity")

    @property
    def bars(self) -> tuple[str, ...]:
        return self.cage.bars if self.cage is not None else ()

    @property
    def solid(self) -> tuple[str, ...]:
        return tuple(r for r in self.sigma if r not in self.bars)


@dataclass
class HarmonicResult:
    """Solved field plus the derived quantities the gates and models read."""

    spec: HarmonicSpec
    mesh: Mesh
    fes: FESpace
    gf: GridFunction
    A: GridFunction
    u: dict[str, complex]       # bar axial voltage gradient u_k (V/m)
    I_bar: dict[str, complex]   # bar currents (A, peak)

    def J(self, region: str) -> CoefficientFunction:
        """Total current density phasor in an eddy region (A/m²)."""
        from ngsolve import CoefficientFunction, grad, x, y

        spec = self.spec
        if region not in spec.sigma:
            raise ValueError(f"{region!r} is not an eddy region")
        sig = spec.sigma[region]
        if region in spec.bars:
            u_k = CoefficientFunction(self.u[region])
            return sig * (u_k - 1j * spec.slip * spec.omega * self.A)
        J = -1j * spec.omega * sig * self.A
        if spec.omega_mech != 0.0:
            J = J - sig * spec.omega_mech * (x * grad(self.A)[1] - y * grad(self.A)[0])
        return J

    def loss(self, regions: str | Sequence[str]) -> float:
        """Time-averaged Joule loss (W per metre of depth) in eddy regions."""
        from ngsolve import Conj, Integrate

        names = [regions] if isinstance(regions, str) else list(regions)
        total = 0.0
        for r in names:
            J = self.J(r)
            val = Integrate(0.5 * J * Conj(J) / self.spec.sigma[r], self.mesh,
                            definedon=self.mesh.Materials(r))
            total += float(np.real(val))
        return total

    def ring_loss(self) -> float:
        """Time-averaged Joule loss in both end rings of a ``Z_ring``
        cage: Σ_k Re(Z_ring)|J_k|² with the segment current
        J_k = L(u_k - u_{k+1})/(2 Z_ring) (the KVL loop through bars k,
        k+1 and one segment at each end). Not part of ``loss()``, which
        is the volumetric bar term only — and unlike ``loss()`` and
        ``input_power()`` this is the WHOLE stack in W, not W per metre:
        the balance reads input_power·L_stk = loss·L_stk + ring_loss."""
        cage = self.spec.cage
        if cage is None or cage.Z_ring is None:
            raise ValueError("ring_loss needs a cage closed through Z_ring")
        bars = cage.bars
        n = len(bars)
        total = 0.0
        for k in range(n):
            J_k = cage.L_stk * (self.u[bars[k]] - self.u[bars[(k + 1) % n]]) / (2 * cage.Z_ring)
            total += cage.Z_ring.real * abs(J_k) ** 2
        return total

    def input_power(self) -> float:
        """Time-averaged power delivered by the imposed sources (W/m):
        -½ Re ∫ J_s·conj(E) with E = -jωA."""
        from ngsolve import Conj, Integrate

        total = 0.0
        for r, J in self.spec.J_source.items():
            val = Integrate(J * Conj(self.A), self.mesh, definedon=self.mesh.Materials(r))
            total += 0.5 * self.spec.omega * float(np.imag(val))
        return total

    def flux_linkage(self, plus: str, minus: str) -> complex:
        """Per-turn flux linkage per metre of depth (Wb/m) of a single
        turn distributed over the ``plus`` region and returning through
        ``minus``: area-averaged A_z difference."""
        from ngsolve import CoefficientFunction, Integrate

        def avg(name: str) -> complex:
            area = Integrate(CoefficientFunction(1), self.mesh,
                             definedon=self.mesh.Materials(name))
            return complex(Integrate(self.A, self.mesh,
                                     definedon=self.mesh.Materials(name))) / area

        return avg(plus) - avg(minus)

    def B_polar(self, radius: float, n_theta: int = 720,
                ) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        """(B_r, B_θ) phasors on a uniform circle of the given radius."""
        from ngsolve import grad, sqrt, x, y

        r_cf = sqrt(x**2 + y**2)
        B_r_cf = (grad(self.A)[1] * x - grad(self.A)[0] * y) / r_cf
        B_t_cf = -(grad(self.A)[0] * x + grad(self.A)[1] * y) / r_cf
        theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
        mips = self.mesh(radius * np.cos(theta), radius * np.sin(theta))
        return (np.asarray(B_r_cf(mips), dtype=complex).reshape(-1),
                np.asarray(B_t_cf(mips), dtype=complex).reshape(-1))

    def gap_power(self, radius: float, n_theta: int = 720, L_stk: float = 1.0) -> float:
        """Time-averaged power (W) crossing the circle inward, from the
        Poynting flux ½ Re(E × H*)·r̂ with E = -jωA and H_θ = ν0 B_θ in
        source-free air: the air-gap power. Equals the source power when
        the circle encloses every eddy region and the sources sit outside
        it, up to discretisation."""
        from ngsolve import grad, sqrt, x, y

        r_cf = sqrt(x**2 + y**2)
        B_t_cf = -(grad(self.A)[0] * x + grad(self.A)[1] * y) / r_cf
        theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
        mips = self.mesh(radius * np.cos(theta), radius * np.sin(theta))
        A = np.asarray(self.A(mips), dtype=complex).reshape(-1)
        B_t = np.asarray(B_t_cf(mips), dtype=complex).reshape(-1)
        flux = np.sum(0.5 * np.real(-1j * self.spec.omega * A * np.conj(B_t))) / MU0
        return L_stk * radius * float(flux) * (2 * pi / n_theta)

    def torque(self, radius: float, n_theta: int = 720, L_stk: float = 1.0) -> float:
        """Time-averaged Maxwell-stress torque (N·m about +z) on the
        material enclosed by the circle, which must lie in source-free
        air: τ = (L r²/μ0) ∮ ½ Re(B_r conj(B_θ)) dθ."""
        B_r, B_t = self.B_polar(radius, n_theta)
        integral = float(np.sum(0.5 * np.real(B_r * np.conj(B_t)))) * (2 * pi / n_theta)
        return L_stk * radius * radius / MU0 * integral


def _matches_any(pattern: str, names: set[str]) -> bool:
    """Region names are NGSolve regex patterns (``slot.*`` covers every
    per-index slot of a cage mesh); a plain name matches itself."""
    return pattern in names or any(re.fullmatch(pattern, n) for n in names)


def _region_area(mesh: Mesh, name: str) -> float:
    from ngsolve import CoefficientFunction, Integrate

    return float(Integrate(CoefficientFunction(1), mesh, definedon=mesh.Materials(name)))


def _invert(a: Any, fes: FESpace) -> Any:
    errors: list[str] = []
    for inv in _DIRECT_SOLVERS:
        try:
            return a.mat.Inverse(fes.FreeDofs(), inverse=inv)
        except Exception as e:  # ngsolve raises bare RuntimeError
            errors.append(f"{inv}: {e}")
    raise RuntimeError(
        "no direct solver for the complex non-symmetric harmonic system "
        "(sparsecholesky needs symmetry): " + "; ".join(errors)
    )


def solve_harmonic(mesh: Mesh, spec: HarmonicSpec) -> HarmonicResult:
    """Assemble and solve one time-harmonic problem on a named-region mesh.

    The bar unknowns u_k are eliminated by static condensation rather
    than carried as NumberSpace dofs: a global scalar dof couples to every
    A dof of the compound space and the direct solver's fill-in on those
    dense rows costs minutes on a production cage mesh (255 s against
    0.8 s for the field system alone on the 36/28 CREATOR mesh). The
    field system K is factorized once; A = A_p + Σ_k u_k A_k with
    K A_p = f and K A_k = σ_k m_k (m_k = ∫_bar_k φ_i), and the bar
    constraints σ_k S_k u_k − j s ω σ_k ∫_k A − ring(u)_k = I_k become a
    dense n_bars × n_bars system for the u_k. Same solution as the
    coupled formulation to solver precision.
    """
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        LinearForm,
        Periodic,
        dx,
        grad,
        x,
        y,
    )

    mats = set(mesh.GetMaterials())
    for group, names in (("nu", spec.nu), ("sigma", spec.sigma), ("J_source", spec.J_source)):
        missing = [n for n in names if not _matches_any(n, mats)]
        if missing:
            raise ValueError(f"{group} names regions {missing} absent from the mesh")
    if spec.dirichlet not in set(mesh.GetBoundaries()):
        raise ValueError(f"boundary {spec.dirichlet!r} absent from the mesh")

    nu = mesh.MaterialCF(dict(spec.nu), default=1 / MU0)
    fes = H1(mesh, order=spec.order, complex=True, dirichlet=spec.dirichlet)
    if spec.periodic_phase is not None:
        fes = Periodic(fes, phase=[spec.periodic_phase])
    A, v = fes.TnT()

    a = BilinearForm(fes)
    a += nu * grad(A) * grad(v) * dx
    for r in spec.solid:
        sig = spec.sigma[r]
        term = 1j * spec.omega * sig * A * v
        if spec.omega_mech != 0.0:
            term = term + sig * spec.omega_mech * (x * grad(A)[1] - y * grad(A)[0]) * v
        a += term * dx(r)
    cage = spec.cage
    sw = spec.slip * spec.omega
    for r in spec.bars:
        a += 1j * sw * spec.sigma[r] * A * v * dx(r)

    f = LinearForm(fes)
    for r, J in spec.J_source.items():
        f += J * v * dx(r)

    a.Assemble()
    f.Assemble()
    inv = _invert(a, fes)
    gf = GridFunction(fes)
    gf.vec.data = inv * f.vec

    u: dict[str, complex] = {}
    if cage is not None:
        u = _condense_cage(mesh, fes, inv, gf, spec)

    result = HarmonicResult(spec=spec, mesh=mesh, fes=fes, gf=gf, A=gf, u=u, I_bar={})
    if cage is not None:
        from ngsolve import Integrate

        for r in cage.bars:
            result.I_bar[r] = complex(Integrate(result.J(r), mesh, definedon=mesh.Materials(r)))
    return result


def _condense_cage(mesh: Mesh, fes: FESpace, inv: Any, gf: GridFunction,
                   spec: HarmonicSpec) -> dict[str, complex]:
    """Solve the bar circuit on the condensed system and add Σ u_k A_k to
    the particular solution held in ``gf``. Returns the u_k per bar."""
    from ngsolve import GridFunction, LinearForm, dx

    cage = spec.cage
    assert cage is not None
    bars = cage.bars
    n_b = len(bars)
    sw = spec.slip * spec.omega
    _trial, v = fes.TnT()

    m_vecs: list[NDArray[np.complexfloating]] = []
    responses: list[GridFunction] = []
    areas: list[float] = []
    A_p = gf.vec.FV().NumPy()
    for r in bars:
        lf = LinearForm(fes)
        lf += v * dx(r)
        lf.Assemble()
        m_vecs.append(lf.vec.FV().NumPy().copy())
        resp = GridFunction(fes)
        resp.vec.data = inv * (spec.sigma[r] * lf.vec)
        responses.append(resp)
        areas.append(_region_area(mesh, r))

    # constraint k: σ_k S_k u_k − j s ω σ_k ∫_k A − ring(u)_k = I_k
    M = np.zeros((n_b, n_b), dtype=complex)
    rhs = np.zeros(n_b, dtype=complex)
    for k, r in enumerate(bars):
        sig = spec.sigma[r]
        M[k, k] += sig * areas[k]
        for j in range(n_b):
            M[k, j] -= 1j * sw * sig * np.dot(m_vecs[k], responses[j].vec.FV().NumPy())
        rhs[k] = 1j * sw * sig * np.dot(m_vecs[k], A_p)
    if cage.currents is not None:
        rhs += np.asarray(cage.currents, dtype=complex)
        u_vec = np.linalg.solve(M, rhs)
    elif cage.Z_ring is not None:
        # KCL at the ring nodes with KVL around adjacent bar loops:
        #   I_k = L (u_{k+1} - 2 u_k + u_{k-1}) / (2 Z_r)
        c = cage.L_stk / (2 * cage.Z_ring)
        lap = -2 * np.eye(n_b) + np.roll(np.eye(n_b), 1, axis=1) + np.roll(np.eye(n_b), -1, axis=1)
        u_vec = np.linalg.solve(M - c * lap, rhs)
    else:
        # ideal ring: one common u, ΣI_k = 0 (rows summed)
        u_common = rhs.sum() / M.sum()
        u_vec = np.full(n_b, u_common)

    for j in range(n_b):
        gf.vec.data += complex(u_vec[j]) * responses[j].vec
    return {r: complex(u_vec[k]) for k, r in enumerate(bars)}
