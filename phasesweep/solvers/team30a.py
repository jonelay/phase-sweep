"""
TEAM Problem 30a — Davey, "Induction Motor Analyses" (COMPUMAG TEAM
workshop; problem sheet at compumag.org, problem30a.pdf). Exact
verification target for the time-harmonic solver: a 2-D exposed-winding
induction motor with a solid conducting rotor, solved analytically.

Geometry (m): rotor steel r < R1, aluminium R1..R2, air gap R2..R3,
winding annulus R3..R4 (45° coil belts, air between), stator steel
R4..R5, then open air. Steel μr = 30 both sides; the stator steel is
laminated (σ = 0), the rotor steel conducts. Coils carry 310 A/cm² at
60 Hz. Three-phase: six belts A, -B, C, -A, B, -C on 60° centres with
B and C lagging A by 120° and 240°. Single-phase: belts A, -A at
0° and 180°.

Conventions pinned by reproducing the published tables (both were
needed; neither is stated on the problem sheet):
  * the current density is an rms phasor — torque and loss are
    time averages (peak/√2 amplitudes), the voltage is rms;
  * the exterior beyond R5 is open (no A = 0 at the stator surface).
Speeds are positive in the direction the stator field travels. With
Davey's phase order and the belts laid out counterclockwise from +x,
that field travels clockwise, so the solver sees omega_mech = -speed;
this keeps Table II's signed B_r/H_θ on the +x axis intact.

``analytic_team30a`` is the layered-cylinder series solution (modified
Bessel functions in the conductors, each space harmonic n at its own
frequency ω + nΩ); ``solve_team30a`` runs the FE solver on a mesh from
``build_team30a_mesh``. Reference tables ship in ``data/team30a/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import iv, ivp, kv, kvp

from phasesweep.solvers.fem_harmonic import HarmonicSpec, solve_harmonic
from phasesweep.solvers.fem_materials import MU0

if TYPE_CHECKING:
    from ngsolve import Mesh

    from phasesweep.solvers.fem_harmonic import HarmonicResult

__all__ = [
    "COIL_HALF_ANGLE",
    "F_HZ",
    "J_RMS",
    "MU_R_STEEL",
    "OMEGA",
    "R_TORQUE",
    "SIGMA_ALUMINUM",
    "SIGMA_ROTOR_STEEL",
    "SPEEDS_SINGLE_PHASE",
    "SPEEDS_THREE_PHASE",
    "Belt",
    "analytic_field_on_axis",
    "analytic_team30a",
    "build_team30a_mesh",
    "coil_belts",
    "fe_field_on_axis",
    "solve_team30a",
    "team30a_spec",
]

R1, R2, R3, R4, R5 = 0.02, 0.03, 0.032, 0.052, 0.057
MU_R_STEEL = 30.0
SIGMA_ROTOR_STEEL = 1.6e6
SIGMA_ALUMINUM = 3.72e7
J_RMS = 3.1e6
F_HZ = 60.0
OMEGA = 2 * pi * F_HZ
COIL_HALF_ANGLE = pi / 8
R_TORQUE = 0.5 * (R2 + R3)
SPEEDS_THREE_PHASE = (0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0)
SPEEDS_SINGLE_PHASE = (0.0, 39.79351, 79.58701, 119.3805, 159.174, 198.9675,
                       238.761, 278.5546, 318.3481, 358.1416)

_PHASE = {"A": 0.0, "B": -2 * pi / 3, "C": -4 * pi / 3}


@dataclass(frozen=True)
class Belt:
    name: str
    theta: float
    factor: complex


def coil_belts(three_phase: bool) -> tuple[Belt, ...]:
    """Coil belts (region name, centre angle, complex current factor)."""
    if not three_phase:
        return (Belt("Ap", 0.0, 1.0), Belt("Am", pi, -1.0))
    layout = (("A", 1), ("B", -1), ("C", 1), ("A", -1), ("B", 1), ("C", -1))
    return tuple(
        Belt(f"{ph}{'p' if s > 0 else 'm'}", k * pi / 3, s * np.exp(1j * _PHASE[ph]))
        for k, (ph, s) in enumerate(layout)
    )


def build_team30a_mesh(three_phase: bool, maxh: float = 0.003,
                       r_far: float = 20 * R5) -> Mesh:
    """Named-region mesh: rotor / alu / airgap / coil belts / wair /
    stator / air_out, boundary ``outer`` at ``r_far``."""
    from netgen.occ import Glue, OCCGeometry, WorkPlane
    from ngsolve import Mesh

    def disk(r: float) -> Any:
        return WorkPlane().Circle(r).Face()

    def sector(th0: float, th1: float, r_o: float, r_i: float) -> Any:
        R = 2 * r_far
        wp = WorkPlane()
        wp.MoveTo(0, 0)
        wp.LineTo(R * cos(th0), R * sin(th0))
        wp.LineTo(R * cos(th1), R * sin(th1))
        wp.Close()
        return (wp.Face() * disk(r_o)) - disk(r_i)

    faces = []
    for name, r_o, r_i in (("rotor", R1, 0.0), ("alu", R2, R1), ("airgap", R3, R2),
                           ("stator", R5, R4), ("air_out", r_far, R5)):
        f = disk(r_o) - disk(r_i) if r_i > 0 else disk(r_o)
        f.faces.name = name
        faces.append(f)
    winding = disk(R4) - disk(R3)
    for belt in coil_belts(three_phase):
        f = sector(belt.theta - COIL_HALF_ANGLE, belt.theta + COIL_HALF_ANGLE, R4, R3)
        f.faces.name = belt.name
        winding = winding - f
        faces.append(f)
    winding.faces.name = "wair"
    faces.append(winding)
    geo = Glue(faces)
    for e in geo.edges:
        if abs(e.mass - 2 * pi * r_far) < 1e-6 * r_far:
            e.name = "outer"
    geo.faces["airgap"].maxh = (R3 - R2) / 3
    for name in ("rotor", "alu"):
        geo.faces[name].maxh = maxh / 2
    for name in ("stator", "wair", *(b.name for b in coil_belts(three_phase))):
        geo.faces[name].maxh = maxh
    return Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=r_far / 6))


def team30a_spec(three_phase: bool, speed: float, order: int = 3) -> HarmonicSpec:
    """Solver spec at a rotor speed (rad/s, along the field)."""
    J_peak = sqrt(2) * J_RMS
    nu_fe = 1 / (MU_R_STEEL * MU0)
    return HarmonicSpec(
        omega=OMEGA,
        nu={"rotor": nu_fe, "stator": nu_fe},
        sigma={"rotor": SIGMA_ROTOR_STEEL, "alu": SIGMA_ALUMINUM},
        J_source={b.name: complex(J_peak * b.factor) for b in coil_belts(three_phase)},
        omega_mech=-speed,
        order=order,
    )


def solve_team30a(mesh: Mesh, three_phase: bool, speed: float, order: int = 3,
                  n_theta: int = 1440) -> dict[str, Any]:
    """FE solution in the published conventions: torque (N·m/m, along the
    field), voltage (V/m/turn rms, phase A single turn), rotor and rotor
    steel loss (W/m)."""
    res = solve_harmonic(mesh, team30a_spec(three_phase, speed, order))
    return {
        "torque": -res.torque(R_TORQUE, n_theta=n_theta),
        "voltage": OMEGA * abs(res.flux_linkage("Ap", "Am")) / sqrt(2),
        "rotor_loss": res.loss(("rotor", "alu")),
        "steel_loss": res.loss("rotor"),
        "input_power": res.input_power(),
        "result": res,
    }


# ---------------------------------------------------------------------------
# Exact layered-cylinder solution
# ---------------------------------------------------------------------------

class _Layer:
    """One annulus r_in..r_out with basis functions normalised at its
    own radii so high harmonics stay conditioned."""

    def __init__(self, r_in: float, r_out: float, mu: float, sigma: float,
                 m: int, we: float, source: bool) -> None:
        self.r_in, self.r_out, self.mu, self.source = r_in, r_out, mu, source
        k2 = 1j * we * mu * sigma
        if abs(k2) > 0:
            k = np.sqrt(k2)
            n1 = iv(m, k * r_out)
            n2 = kv(m, k * r_in) if r_in > 0 else 1.0
            self.f1 = lambda r: iv(m, k * r) / n1
            self.df1 = lambda r: k * ivp(m, k * r) / n1
            self.f2 = lambda r: kv(m, k * r) / n2
            self.df2 = lambda r: k * kvp(m, k * r) / n2
        else:
            self.f1 = lambda r: (r / r_out) ** m
            self.df1 = lambda r: m * r ** (m - 1) / r_out**m
            self.f2 = lambda r: (r_in / r) ** m if r_in > 0 else 0.0
            self.df2 = lambda r: -m * r_in**m / r ** (m + 1) if r_in > 0 else 0.0
        self.regular = r_in == 0.0
        self.n_coef = 1 if self.regular else 2


def _particular(m: int, Jn: complex) -> tuple[Any, Any]:
    if m == 2:
        return (lambda r: -MU0 * Jn * r**2 * np.log(r) / 4,
                lambda r: -MU0 * Jn * (2 * r * np.log(r) + r) / 4)
    return (lambda r: -MU0 * Jn * r**2 / (4 - m**2),
            lambda r: -MU0 * Jn * 2 * r / (4 - m**2))


class _ModeSolution:
    """A_n(r) for one azimuthal harmonic, layer by layer."""

    def __init__(self, n: int, Jn: complex, omega: float, Om: float,
                 r_far: float | None) -> None:
        m = abs(n)
        we = omega + n * Om
        mu_fe = MU_R_STEEL * MU0
        radii = [(0.0, R1, mu_fe, SIGMA_ROTOR_STEEL, False), (R1, R2, MU0, SIGMA_ALUMINUM, False),
                 (R2, R3, MU0, 0.0, False), (R3, R4, MU0, 0.0, True), (R4, R5, mu_fe, 0.0, False)]
        if r_far is None:
            radii.append((R5, np.inf, MU0, 0.0, False))
        else:
            radii.append((R5, r_far, MU0, 0.0, False))
        self.layers = [_Layer(ri, ro, mu, sg, m, we, src) for ri, ro, mu, sg, src in radii]
        outer = self.layers[-1]
        if r_far is None:
            outer.f1 = lambda r: 0.0
            outer.df1 = lambda r: 0.0
            outer.n_coef = 1
        self.Ap, self.dAp = _particular(m, Jn)
        self.n, self.we = n, we
        self._solve(r_far)

    def _cols(self, i: int) -> list[int]:
        start = sum(L.n_coef for L in self.layers[:i])
        return list(range(start, start + self.layers[i].n_coef))

    def _basis(self, i: int, r: float) -> tuple[list[complex], list[complex]]:
        L = self.layers[i]
        if L.regular:
            return [L.f1(r)], [L.df1(r)]
        if L.n_coef == 1:
            return [L.f2(r)], [L.df2(r)]
        return [L.f1(r), L.f2(r)], [L.df1(r), L.df2(r)]

    def _solve(self, r_far: float | None) -> None:
        n_unknown = sum(L.n_coef for L in self.layers)
        M = np.zeros((n_unknown, n_unknown), complex)
        rhs = np.zeros(n_unknown, complex)
        row = 0
        for i in range(len(self.layers) - 1):
            La, Lb = self.layers[i], self.layers[i + 1]
            r = La.r_out
            fa, dfa = self._basis(i, r)
            fb, dfb = self._basis(i + 1, r)
            for c, val in zip(self._cols(i), fa):
                M[row, c] += val
            for c, val in zip(self._cols(i + 1), fb):
                M[row, c] -= val
            rhs[row] = (self.Ap(r) if Lb.source else 0) - (self.Ap(r) if La.source else 0)
            row += 1
            for c, val in zip(self._cols(i), dfa):
                M[row, c] += val / La.mu
            for c, val in zip(self._cols(i + 1), dfb):
                M[row, c] -= val / Lb.mu
            rhs[row] = ((self.dAp(r) / Lb.mu if Lb.source else 0)
                        - (self.dAp(r) / La.mu if La.source else 0))
            row += 1
        if r_far is not None:
            f, _ = self._basis(len(self.layers) - 1, r_far)
            for c, val in zip(self._cols(len(self.layers) - 1), f):
                M[row, c] += val
            row += 1
        self.coef = np.linalg.solve(M, rhs)

    def layer_index(self, r: float) -> int:
        for i, L in enumerate(self.layers):
            if r < L.r_out or i == len(self.layers) - 1:
                return i
        return len(self.layers) - 1

    def A(self, r: float, i: int | None = None) -> complex:
        i = self.layer_index(r) if i is None else i
        f, _ = self._basis(i, r)
        val = sum(self.coef[c] * v for c, v in zip(self._cols(i), f))
        return val + (self.Ap(r) if self.layers[i].source else 0)

    def dA(self, r: float, i: int | None = None) -> complex:
        i = self.layer_index(r) if i is None else i
        _, df = self._basis(i, r)
        val = sum(self.coef[c] * v for c, v in zip(self._cols(i), df))
        return val + (self.dAp(r) if self.layers[i].source else 0)


def _mode_sources(three_phase: bool, n_max: int) -> dict[int, complex]:
    """Fourier coefficients J_n of the coil-belt current density (rms)."""
    belts = coil_belts(three_phase)
    out: dict[int, complex] = {}
    for n in range(-n_max, n_max + 1):
        if n == 0:
            continue
        Jn = sum(b.factor * J_RMS * sin(n * COIL_HALF_ANGLE) / (pi * n)
                 * np.exp(-1j * n * b.theta) for b in belts)
        if abs(Jn) > 1e-9:
            out[n] = complex(Jn)
    return out


def _layer_loss(sol: _ModeSolution, i: int, sigma: float, n_r: int = 400) -> float:
    L = sol.layers[i]
    r = np.linspace(max(L.r_in, 1e-9), L.r_out, n_r)
    a = np.array([sol.A(x, i) for x in r])
    return float(np.trapezoid(sigma * sol.we**2 * np.abs(a) ** 2 * 2 * pi * r, r))


def analytic_team30a(three_phase: bool, speed: float, r_far: float | None = None,
                     n_max: int = 60) -> dict[str, float]:
    """Exact series solution in the published conventions (rms phasors).
    ``r_far`` None = open exterior (the published case); a radius puts
    A = 0 there, matching a truncated FE domain."""
    Om = -speed
    rotor_loss = steel_loss = torque = 0.0
    flux = 0j
    rr = np.linspace(R3, R4, 200)
    for n, Jn in _mode_sources(three_phase, n_max).items():
        sol = _ModeSolution(n, Jn, OMEGA, Om, r_far)
        Ps = _layer_loss(sol, 0, SIGMA_ROTOR_STEEL)
        rotor_loss += Ps + _layer_loss(sol, 1, SIGMA_ALUMINUM)
        steel_loss += Ps
        B_r = 1j * n / R_TORQUE * sol.A(R_TORQUE, 2)
        B_t = -sol.dA(R_TORQUE, 2)
        torque += R_TORQUE**2 / MU0 * 2 * pi * float(np.real(B_r * np.conj(B_t)))
        radial = np.trapezoid(np.array([sol.A(x, 3) for x in rr]) * rr, rr)

        def ang(th0: float, n: int = n) -> complex:
            return (np.exp(1j * n * (th0 + COIL_HALF_ANGLE))
                    - np.exp(1j * n * (th0 - COIL_HALF_ANGLE))) / (1j * n)

        flux += complex(radial * (ang(0.0) - ang(pi)))
    area = COIL_HALF_ANGLE * (R4**2 - R3**2)
    return {
        "torque": -torque,
        "voltage": OMEGA * abs(flux) / area,
        "rotor_loss": rotor_loss,
        "steel_loss": steel_loss,
    }


def analytic_field_on_axis(speed: float, x_points: NDArray[np.floating],
                           three_phase: bool = True, r_far: float | None = None,
                           n_max: int = 60,
                           ) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """(B_r, H_θ) rms phasors on the +x axis (Table II)."""
    Om = -speed
    xs = np.asarray(x_points, dtype=float)
    if xs.size and xs.max() > R4:
        raise ValueError(
            f"analytic_field_on_axis covers the machine out to the winding "
            f"outer radius R4 = {R4} m; r = {xs.max():.6g} m falls in the "
            f"stator steel (mu_r = 30) or the exterior, whose coefficients "
            f"this clamp would not use")
    B_r = np.zeros(len(xs), complex)
    H_t = np.zeros(len(xs), complex)
    for n, Jn in _mode_sources(three_phase, n_max).items():
        sol = _ModeSolution(n, Jn, OMEGA, Om, r_far)
        for i, r in enumerate(xs):
            # r == R4 takes the winding-side limit of the interface, which is
            # the side fe_field_on_axis samples (it evaluates at R4 - eps).
            li = sol.layer_index(r) if r < R4 else 3
            B_r[i] += 1j * n / r * sol.A(r, li)
            H_t[i] += -sol.dA(r, li) / sol.layers[li].mu
    return B_r, H_t


def fe_field_on_axis(res: HarmonicResult, x_points: NDArray[np.floating],
                     ) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """(B_r, H_θ) rms phasors on the +x axis from an FE result, evaluated
    on the winding side of r = R4."""
    xs = np.asarray(x_points, dtype=float)
    if xs.size and xs.max() > R4:
        raise ValueError(
            f"fe_field_on_axis samples the winding side of R4 = {R4} m; "
            f"r = {xs.max():.6g} m falls in the stator steel or the exterior, "
            f"and the R4 clamp would return the interface value there")
    from ngsolve import grad

    eps = 1e-5
    B_r = np.zeros(len(xs), complex)
    H_t = np.zeros(len(xs), complex)
    for i, r in enumerate(xs):
        mip = res.mesh(min(r, R4 - eps), 0.0)
        B_r[i] = complex(grad(res.A)[1](mip)) / sqrt(2)
        H_t[i] = -complex(grad(res.A)[0](mip)) / MU0 / sqrt(2)
    return B_r, H_t
