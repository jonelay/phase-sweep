"""Deep-bar rotor impedance of a squirrel cage from a periodic
single-slot-pitch harmonic solve.

One rotor slot pitch (fem_mesh.build_bar_cell_mesh) with A_z = 0 on the
stator bore and the cage's bar-to-bar phase progression e^{-j 2π n_p /
n_bars} across the cell sides, driven by a prescribed bar current at
each rotor frequency f_r. The bore condition removes the magnetizing
flux, so u/I is the bar's LEAKAGE impedance per metre — slot leakage
plus the tooth-top flux that crosses the gap and returns through the
neighbouring teeth — with the skin effect resolved by the field solve.

Circuit assembly (peak phasors, one bar and its ring share):

    Z_be(f_r) = Z_bar(f_r)·L_stk + Z_seg(f_r) / (2 sin²(π n_p / n_bars))

The ring term is the F2 ring stencil I_k = L(u_{k+1} - 2u_k + u_{k-1}) /
(2 Z_ring) evaluated on the progressive-phase mode u_k ∝ e^{-jαk}: the
circulant Laplacian eigenvalue -4 sin²(α/2) turns one segment impedance
per end into the series term above. Z_seg = R_seg + jω L_seg with the
DC segment resistance from the ring section and the Alger / Boldea
end-ring leakage permeance. Referral to the stator uses the standard
cage turns ratio 4 m (N_eff k_w)² / n_bars (loss and MMF equivalence).

What the prediction excludes, so the comparison with a measured table
is read correctly: differential (harmonic) rotor leakage — reported
separately as σ_d·L_m when the motor carries L_m — skew leakage, the
ring's own skin effect, and the bar length outside the stack (the bar
ends inside the ring body are booked as ring). Linear iron at mu_r_fe:
the bridge and tooth tips do not saturate here, which is right for the
low-current deep-bar tables and wrong for starting-current leakage.
Temperature: everything at the reference temperature of sigma_bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10, pi, sin
from typing import TYPE_CHECKING, Any

import numpy as np

from phasesweep.solver_params import n_eff, require_geometry
from phasesweep.solvers.fem_materials import MU0

if TYPE_CHECKING:
    from phasesweep.machines.geometry import Geometry
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "ImBarParams",
    "differential_leakage_coefficient",
    "prepare_im_bar_impedance",
    "referral_factor",
    "ring_factor",
    "ring_segment",
    "rotor_frequency_grid",
    "run_im_bar_impedance",
    "solve_bar_cell",
]

# Rotor-frequency grid as fractions of f_rated; the motor's measured
# table frequencies are merged in so predicted and measured points
# coincide. Not a RunConfig knob.
_F_GRID_REL = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
# The f_r = 0 point is solved at this frequency: for any real bar
# h/δ < 0.01 there, i.e. R_ac/R_dc - 1 < 1e-9.
_F_DC = 1e-3
_M_PHASES = 3


@dataclass(frozen=True)
class ImBarParams:
    """Validated inputs of the bar-impedance solve."""

    geometry: Geometry
    n_p: int
    sigma_bar: float
    mu_r_fe: float
    L_stk: float
    N_eff: int
    k_w: float
    f_rated: float
    f_table: tuple[float, ...]
    L_m: float | None

    def __post_init__(self) -> None:
        if self.geometry.n_bars <= 0:
            raise ValueError("ImBarParams needs a cage rotor (n_bars > 0)")
        for name in ("sigma_bar", "mu_r_fe", "L_stk", "k_w", "f_rated"):
            v = getattr(self, name)
            # NaN fails every comparison, so `v <= 0` alone lets it through.
            if not isfinite(v) or v <= 0:
                raise ValueError(f"{name}={v} must be > 0 and finite")
        if self.N_eff <= 0:
            raise ValueError(f"N_eff={self.N_eff} must be > 0")
        if self.n_p % self.geometry.n_bars == 0:
            raise ValueError(
                f"n_p={self.n_p} with n_bars={self.geometry.n_bars}: the cage's "
                f"bar-to-bar phase exp(-2j pi n_p/n_bars) is 1, so a single "
                f"slot pitch is no longer a periodic cell of the field. The "
                f"ring stencil and the differential-leakage sum are both "
                f"singular there (1/sin^2(pi n_p/n_bars)); this model cannot "
                f"represent such a cage")


def ring_factor(n_p: int, n_bars: int) -> float:
    """Series share of ONE ring segment per bar, both ends: 1/(2 sin²(π n_p/n_bars))."""
    return 1.0 / (2.0 * sin(pi * n_p / n_bars) ** 2)


def referral_factor(N_eff: int, k_w: float, n_bars: int, m: int = _M_PHASES) -> float:
    """Cage-to-stator impedance ratio 4 m (N_eff k_w)² / n_bars."""
    return 4.0 * m * (N_eff * k_w) ** 2 / n_bars


def differential_leakage_coefficient(n_p: int, n_bars: int) -> float:
    """Rotor differential (harmonic) leakage per unit L_m for an unskewed
    cage: Σ_{k≠0} (n_p/(k n_bars + n_p))² = (πa)²/sin²(πa) - 1, a = n_p/n_bars."""
    a = pi * n_p / n_bars
    return a * a / sin(a) ** 2 - 1.0


def ring_segment(geo: Geometry, sigma_ring: float) -> tuple[float, float]:
    """(R_seg, L_seg) of one end-ring segment between adjacent bars.

    The ring's outer radius is taken at r_rotor (the geometry carries the
    section, not the ring radii). R_seg is the DC value of the mean
    circumference over the section. L_seg is the Alger end-ring leakage
    permeance (Boldea & Nasar, The Induction Machine Handbook, ch. 6),
    written per segment: μ0·2.3 D/(2 n_bars)·log10(4.7 D/(a + 2b)) with
    D the mean ring diameter, a the axial length and b the radial height.
    """
    a, b = geo.end_ring_axial_length, geo.end_ring_radial_height
    if a is None or b is None:
        raise ValueError("ring_segment needs end_ring_axial_length and end_ring_radial_height")
    D = 2.0 * (geo.r_rotor - b / 2)
    # Alger's permeance is a log of a ratio: outside its range the formula
    # returns a negative leakage inductance rather than failing.
    if 4.7 * D <= a + 2 * b:
        raise ValueError(
            f"end-ring section (axial {a} m, radial {b} m) is too large for a "
            f"mean ring diameter of {D:.6g} m: the Alger end-ring permeance "
            f"log10(4.7 D/(a + 2b)) is not positive, so L_seg would come out "
            f"negative")
    R_seg = pi * D / (geo.n_bars * sigma_ring * a * b)
    L_seg = MU0 * 2.3 * D / (2 * geo.n_bars) * log10(4.7 * D / (a + 2 * b))
    return R_seg, L_seg


def solve_bar_cell(mesh: Any, p: ImBarParams, f_rotor: float, order: int = 3) -> complex:
    """Bar leakage impedance per metre (Ω/m) at rotor frequency f_rotor
    on a build_bar_cell_mesh mesh: u/I for a prescribed unit current."""
    from phasesweep.solvers.fem_harmonic import CageCircuit, HarmonicSpec, solve_harmonic

    geo = p.geometry
    spec = HarmonicSpec(
        omega=2 * pi * f_rotor,
        nu={"rotor": 1.0 / (MU0 * p.mu_r_fe)},
        sigma={"bar": p.sigma_bar},
        slip=1.0,
        cage=CageCircuit(bars=("bar",), currents=(1.0,)),
        order=order,
        periodic_phase=complex(np.exp(-2j * pi * p.n_p / geo.n_bars)),
    )
    return solve_harmonic(mesh, spec).u["bar"]


def rotor_frequency_grid(p: ImBarParams) -> list[float]:
    grid = {round(r * p.f_rated, 9) for r in _F_GRID_REL} | {round(f, 9) for f in p.f_table}
    return sorted(grid)


def prepare_im_bar_impedance(motor: Motor) -> ImBarParams:
    """Validate and extract the bar-impedance inputs from a Motor."""
    ind = motor.induction
    if ind is None:
        raise ValueError(
            f"Motor '{motor.name}': im_bar_impedance requires an [induction] "
            f"section (squirrel-cage rotor); this motor is a synchronous machine"
        )
    geo = require_geometry(motor, "im_bar_impedance")
    # Motor already rejects [induction] with a PM-rotor geometry, so
    # n_bars > 0 holds here; ImBarParams re-checks it for direct callers.
    missing = [f for f in ("end_ring_axial_length", "end_ring_radial_height")
               if getattr(geo, f) is None]
    if missing:
        raise ValueError(
            f"Motor '{motor.name}': im_bar_impedance needs [geometry] "
            f"{', '.join(missing)} for the end-ring correction"
        )
    if ind.sigma_bar is None:
        raise ValueError(f"Motor '{motor.name}': im_bar_impedance needs [induction] sigma_bar")
    if ind.f_rated is None:
        raise ValueError(f"Motor '{motor.name}': im_bar_impedance needs [induction] f_rated")
    if motor.k_w is None:
        raise ValueError(f"Motor '{motor.name}': im_bar_impedance needs [winding] k_w for the referral")
    if motor.L_stk is None:
        raise ValueError(f"Motor '{motor.name}': im_bar_impedance needs L_stk")
    f_table: list[float] = []
    for table in (ind.R_2_vs_f, ind.L_sigma_r_vs_f):
        if table is not None:
            f_table.extend(f for f, _v in table)
    return ImBarParams(
        geometry=geo,
        n_p=motor.n_p,
        sigma_bar=ind.sigma_bar,
        mu_r_fe=motor.mu_r_fe,
        L_stk=motor.L_stk,
        N_eff=n_eff(motor),
        k_w=motor.k_w,
        f_rated=ind.f_rated,
        f_table=tuple(f_table),
        L_m=ind.L_m,
    )


def run_im_bar_impedance(config: RunConfig) -> dict[str, Any]:
    """Predicted deep-bar R_2(f_r) / L_sigma_r(f_r) tables in the circuit
    tier's shape (stator-referred, rotor frequency in Hz)."""
    from phasesweep.solvers.fem_mesh import build_bar_cell_mesh

    p = prepare_im_bar_impedance(config.motor)
    geo = p.geometry
    mesh = build_bar_cell_mesh(geo, config.maxh_fraction)
    R_bar_dc = p.L_stk / (p.sigma_bar * geo.bar_area)
    R_seg, L_seg = ring_segment(geo, p.sigma_bar)
    k_ring = ring_factor(p.n_p, geo.n_bars)
    k_ref = referral_factor(p.N_eff, p.k_w, geo.n_bars)

    f_list = rotor_frequency_grid(p)
    R_2, L_2, k_R, k_x, share = [], [], [], [], []
    for f in f_list:
        omega = 2 * pi * (f if f > 0 else _F_DC)
        Z_bar = solve_bar_cell(mesh, p, f if f > 0 else _F_DC) * p.L_stk
        Z_ring = complex(R_seg, omega * L_seg) * k_ring
        Z_be = Z_bar + Z_ring
        R_2.append(k_ref * Z_be.real)
        L_2.append(k_ref * Z_be.imag / omega)
        k_R.append(Z_bar.real / R_bar_dc)
        k_x.append(Z_bar.imag / omega)
        share.append(Z_ring.real / Z_be.real)

    # k_x: bar leakage inductance (slot + tooth-top) normalised to DC.
    L_bar_dc = k_x[0]
    if not (isfinite(L_bar_dc) and L_bar_dc > 0):
        raise ValueError(
            f"DC bar leakage inductance {L_bar_dc} is not positive and "
            f"finite — the quasi-DC FEM solve returned an invalid impedance")
    k_x = [v / L_bar_dc for v in k_x]

    # --- Layer 2 invariants (Boldea Ch 6 eq 6.71-72, derivation Ch 9) ---
    # K_R >= 1 and non-decreasing; K_x <= 1 and non-increasing.
    # k_R is normalised to the analytic R_bar_dc (exact bar_area), not the
    # FEM DC solve, so k_R[0] departs from 1 by the polygonal-vs-curved
    # boundary area ratio of the meshed bar (~3e-4 for 8 bars). This is a
    # systematic bias independent of maxh_fraction, not stochastic mesh noise.
    _KR_FLOOR = 1.0 - 1e-3
    _KX_CEIL = 1.0 + 1e-9
    _MONO_TOL = 1e-6
    for i, (kr, kx) in enumerate(zip(k_R, k_x)):
        if kr < _KR_FLOOR:
            raise ValueError(
                f"k_R[{i}]={kr:.9g} < 1 at f_r={f_list[i]} Hz: skin effect "
                f"cannot reduce bar resistance below DC")
        if kx > _KX_CEIL:
            raise ValueError(
                f"k_x[{i}]={kx:.9g} > 1 at f_r={f_list[i]} Hz: skin effect "
                f"cannot increase bar slot leakage above DC")
    for i in range(1, len(f_list)):
        if k_R[i] < k_R[i - 1] - _MONO_TOL:
            raise ValueError(
                f"k_R non-monotonic: k_R[{i}]={k_R[i]:.9g} < "
                f"k_R[{i-1}]={k_R[i-1]:.9g} at f_r={f_list[i]} Hz")
        if k_x[i] > k_x[i - 1] + _MONO_TOL:
            raise ValueError(
                f"k_x non-monotonic: k_x[{i}]={k_x[i]:.9g} > "
                f"k_x[{i-1}]={k_x[i-1]:.9g} at f_r={f_list[i]} Hz")

    # --- Layer 2 invariant: leakage permeance non-negativity ---
    # Every leakage component must be >= 0; a negative value means a sign
    # error in the FEM solve or the assembly algebra.
    for i, (r_val, l_val) in enumerate(zip(R_2, L_2)):
        if r_val < 0:
            raise ValueError(
                f"R_2[{i}]={r_val:.6g} < 0 at f_r={f_list[i]} Hz: "
                f"referred bar+ring resistance cannot be negative")
        if l_val < 0:
            raise ValueError(
                f"L_sigma_r[{i}]={l_val:.6g} < 0 at f_r={f_list[i]} Hz: "
                f"referred bar+ring leakage inductance cannot be negative")

    out: dict[str, Any] = {
        "f_rotor_list": f_list,
        "R_2_list": R_2,
        "L_sigma_r_list": L_2,
        "k_R_list": k_R,
        "k_x_list": k_x,
        "ring_share_list": share,
        "R_2_dc": R_2[0],
        "L_sigma_r_dc": L_2[0],
        "R_bar_dc": R_bar_dc,
        "R_ring_seg": R_seg,
        "L_ring_seg": L_seg,
        "referral_factor": k_ref,
        "sigma_diff": differential_leakage_coefficient(p.n_p, geo.n_bars),
    }
    if p.L_m is not None:
        out["L_sigma_r_diff"] = out["sigma_diff"] * p.L_m
    return out
