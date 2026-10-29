"""Full-machine time-harmonic induction FEM: the cage mesh (fem_mesh.get_or_build_cage_mesh) with a
per-slot three-phase winding, solved by fem_harmonic at one slip.

Two registry models share one solve routine:

  im_fem_harmonic      — the operating point at RunConfig.slip and the
                         rated supply: stator current, power factor,
                         torque, the loss split, the field magnitude
                         grid for the dashboard heatmap.
  im_fem_locked_rotor  — virtual blocked-rotor tests (s = 1) at the
                         motor's table frequencies, fed through the
                         locked-rotor step of the IEEE-112 identification
                         (im_circuit.identify_locked_rotor) so the FEM
                         yields R_2(f_r) and leakage the way a measured
                         test does; plus the linear-iron no-load current
                         as a labeled diagnostic.

Winding. Integral-slot double-layer layout from n_slots, n_p and the
coil pitch: 60° phase belts of q = n_slots/(2 n_p m) slots in the top
layer, the return side coil_pitch_slots further on in the bottom layer.
Both layers of a slot are smeared over the whole slot face (opening and
body, like the PM path's j_s), so the slot current density is
J_k = (N_eff m / n_slots) Σ_ph C[ph, k] I_ph / S_k with C the signed
coil-side count — only N_eff = N·coils_series enters, parallel branches
cancel. The layout's fundamental winding factor must reproduce k_w.

Circuit. Current-driven, linear: one solve at 1 A rms per phase
(balanced positive sequence) gives the FEM internal impedance
Z_int = jω ψ/I with ψ = L_stk (N_eff m/n_slots) Σ_k C ⟨A⟩_k the 2D flux
linkage, and every power quantity scales with |I|². Operating points
carry Z_int only; operating_point_at_voltage alone applies the external
network — Z_s = R_1 + jωL_ew in series with Z_int (∥ R_fe when set) —
and every terminal quantity comes from it. R_1 is the motor's R_s at its
reference temperature; L_ew is a Boldea end-winding leakage (stator coil
ends, not modeled in 2D). The virtual identification tests use
R_1 + Z_int, and L_ew is added to the identified L_sigma_s after the
IEEE-112 split, so it stays on the stator side.

Torque. `tau` is the Maxwell-stress torque on the mid-gap circle; the
discrete cage treats every stator space harmonic at the fundamental
slip, so it is reported beside
`tau_power` = n_p P_ag/ω, which the single-slip model makes exact for
its own power balance. The two differ by the harmonic torque share;
neither is gated on the other — the anchor is the measured locked-rotor
data.

Saturation (RunConfig.nonlinear). Iron is linear
at mu_r_fe by default, and the linear no-load current is only a
labeled diagnostic. With nonlinear=True the operating point is solved
by an energy-equivalent effective-reluctivity fixed point
(solve_saturated_operating_point): each iteration is one linear
harmonic solve at 1 A rms on an element-wise reluctivity field, the
current through the FEM branch follows from the supply voltage
through the same circuit, the phasor field scaled to
that current gives every iron element its cycle-averaged reluctivity
ν_eff = ⟨ν(|B|)|B|²⟩/⟨|B|²⟩ over the elliptic locus (BHCurve.nu_harmonic
— the secant for a rotating field, the fundamental of H for a pulsating
one), and the field is relaxed toward it until the raw step is below
tolerance. Not harmonic balance: one frequency, one effective linear
medium per element. The curve is the motor's [materials] bh_curve or
the generic one. The locked-rotor identification stays linear (the
measured tests are at reduced voltage); nonlinear=True on
im_fem_locked_rotor turns the no-load diagnostic into a voltage sweep
whose points go through identify_from_tests, so L_m comes out on the
saturation curve the way a measured no-load test yields it.

Skew (Geometry.skew_angle). A skewed cage is
solved as _N_SKEW_SLICES decoupled 2D slices: slice j is the same
machine with the whole bar pattern rotated by the midpoint-rule offset
skew·((j + ½)/N − ½) (one cage mesh per offset, cached like any other),
and the slices are averaged — Z, torque and powers are means (a stack
of N series slices of length L/N, each solved at full L, sums to the
mean), phase_asym and the worst bar current are maxima.

What this captures: each slice sees the bars at a different angular
position relative to the fixed stator MMF (current-driven), so the
discrete bar/slot interaction varies per slice and the average smooths
out the rotor-position-dependent slotting effects. This is a real
physical effect and reduces torque.

What this does NOT capture: the classical sin(x)/x fundamental skew
factor arises from axial bar-current continuity — each physical bar
must carry ONE current through all slices, constraining the rotor's
reaction field. The decoupled formulation lets each slice close its
own bar circuit independently, so it does not model this coupling.
The torque reduction from position averaging is real but distinct from
the sinc mechanism; it may happen to be numerically similar for some
geometries but this is not guaranteed.

The unskewed path (skew_angle = None) is the single solve,
bit-identical to the unskewed models. Skew with nonlinear=True is rejected: the
saturation fixed point would need one stator current common to all
slices, which is not implemented. The other 3D effect, the end-ring
impedance, is in every solve already
(im_bar_impedance.ring_segment → ImFemParams.ring); skew adds nothing
there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from phasesweep.solver_params import n_eff, require_geometry
from phasesweep.solvers.analytical import end_winding_leakage
from phasesweep.solvers.fem_materials import GENERIC_BH, MU0, BHCurve, bh_curve_for

if TYPE_CHECKING:
    from phasesweep.machines.geometry import Geometry
    from phasesweep.machines.motor import Motor
    from phasesweep.solvers.fem_harmonic import HarmonicResult
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "ImFemOperatingPoint",
    "ImFemParams",
    "SaturationState",
    "balanced_currents",
    "cycle_moments",
    "field_grid",
    "iron_peak_B",
    "layout_winding_factor",
    "locked_rotor_frequencies",
    "no_load_sweep",
    "operating_point_at_voltage",
    "prepare_im_fem_harmonic",
    "run_im_fem_harmonic",
    "run_im_fem_locked_rotor",
    "skew_offsets",
    "solve_operating_point",
    "solve_operating_point_multislice",
    "solve_saturated_operating_point",
    "winding_matrix",
]

_log = logging.getLogger(__name__)
_M_PHASES = 3
_ORDER = 2
_N_THETA_TORQUE = 1440
_RASTER_N_GRID = 120
_K_W_TOL = 0.01
_IRON_REGIONS = ("stator", "rotor")
_SAT_TOL = 1e-2          # permeance-weighted μ step over the iron (see solver)
_SAT_TOL_I = 1e-3        # relative stator-current change per iteration
_SAT_MAX_ITER = 100
_SAT_RELAX = 0.5
_SAT_RELAX_FLOOR = 0.05
_SAT_MAX_RATIO = 2.0     # per-element trust region: ν may change at most 2× per iteration
# Slices of the decoupled multi-slice skew solve. Odd puts a slice
# at the stack center. A model constant like _F_GRID_REL, not a RunConfig
# knob.
_N_SKEW_SLICES = 5
# no-load sweep of im_fem_locked_rotor, fractions of the rated phase voltage
_NO_LOAD_FRACTIONS = (0.3, 0.5, 0.7, 0.85, 1.0, 1.1)
# The sequence of 60° belts around one pole pair, top layer: +A −C +B −A +C −B
_BELT_SEQUENCE = ((0, 1), (2, -1), (1, 1), (0, -1), (2, 1), (1, -1))


def winding_matrix(n_slots: int, n_p: int, coil_pitch_slots: int,
                   m: int = _M_PHASES) -> NDArray[np.int_]:
    """Signed coil-side count per phase and slot, C[ph, k] ∈ {−2..2}, of
    the integral-slot double-layer layout. Raises on a fractional slot
    count or a coil pitch outside (0, pole pitch]."""
    if n_slots <= 0 or n_p <= 0:
        raise ValueError("winding_matrix needs n_slots > 0 and n_p > 0")
    if m != _M_PHASES:
        raise ValueError(
            f"m={m}: the belt sequence this lays down is the three-phase 60 "
            f"degree one, so any other phase count would return a matrix whose "
            f"extra rows are all zero. Give this function its own belt table "
            f"before widening it")
    per_belt, rem = divmod(n_slots, 2 * n_p * m)
    if rem or per_belt == 0:
        raise ValueError(
            f"n_slots={n_slots} with n_p={n_p} is not an integral-slot "
            f"{m}-phase winding (slots per pole per phase = "
            f"{n_slots / (2 * n_p * m):.3g})")
    pole_pitch = n_slots // (2 * n_p)
    if not 0 < coil_pitch_slots <= pole_pitch:
        raise ValueError(
            f"coil_pitch_slots={coil_pitch_slots} outside (0, pole pitch "
            f"= {pole_pitch}]")
    C = np.zeros((m, n_slots), dtype=int)
    for k in range(n_slots):
        ph, sign = _BELT_SEQUENCE[(k // per_belt) % len(_BELT_SEQUENCE)]
        C[ph, k] += sign
        C[ph, (k + coil_pitch_slots) % n_slots] -= sign
    return C


def layout_winding_factor(C: NDArray[np.int_], n_p: int) -> float:
    """Fundamental winding factor of phase A of a winding matrix:
    |Σ_k C[0,k] e^{j n_p θ_k}| / Σ_k |C[0,k]|."""
    n_slots = C.shape[1]
    theta = 2 * pi * np.arange(n_slots) / n_slots
    return float(abs(np.sum(C[0] * np.exp(1j * n_p * theta))) / np.abs(C[0]).sum())


def balanced_currents(I_rms: float, m: int = _M_PHASES) -> NDArray[np.complexfloating]:
    """Positive-sequence peak phasors of a balanced m-phase set."""
    return sqrt(2) * I_rms * np.exp(-2j * pi * np.arange(m) / m)


@dataclass(frozen=True)
class ImFemParams:
    """Validated inputs of the full-machine harmonic solve."""

    geometry: Geometry
    n_p: int
    sigma_bar: float
    mu_r_fe: float
    L_stk: float
    N_eff: int
    k_w: float
    R_1: float
    f_rated: float
    V_ph_rms: float
    C: NDArray[np.int_]
    f_table: tuple[float, ...]
    leakage_split: float | str
    ring: tuple[float, float] | None    # (R_seg, L_seg) or None = ideal ring
    L_ew: float = 0.0                   # stator end-winding leakage (H)
    R_fe: float | None = None           # core-loss shunt across L_m (Ω)
    bh: BHCurve = GENERIC_BH            # saturation curve (nonlinear runs)

    def __post_init__(self) -> None:
        geo = self.geometry
        if geo.n_bars <= 0:
            raise ValueError("ImFemParams needs a cage rotor (n_bars > 0)")
        if geo.n_slots <= 0 or geo.slot_depth <= 0:
            raise ValueError("ImFemParams needs stator slots (n_slots > 0, slot_depth > 0)")
        for name in ("sigma_bar", "mu_r_fe", "L_stk", "k_w", "R_1", "f_rated", "V_ph_rms"):
            v = getattr(self, name)
            # NaN fails every comparison, so `v <= 0` alone lets it through.
            if not isfinite(v) or v <= 0:
                raise ValueError(f"{name}={v} must be > 0 and finite")
        if self.R_fe is not None and (not isfinite(self.R_fe) or self.R_fe <= 0):
            raise ValueError(f"R_fe must be > 0 and finite, got {self.R_fe}")
        if self.N_eff <= 0:
            raise ValueError(f"N_eff={self.N_eff} must be > 0")
        if self.C.shape != (_M_PHASES, geo.n_slots):
            raise ValueError(f"winding matrix shape {self.C.shape} != ({_M_PHASES}, {geo.n_slots})")
        k_w_layout = layout_winding_factor(self.C, self.n_p)
        if abs(k_w_layout / self.k_w - 1) > _K_W_TOL:
            raise ValueError(
                f"winding layout k_w = {k_w_layout:.4f} disagrees with the "
                f"motor's k_w = {self.k_w} by more than {_K_W_TOL:.0%}: check "
                f"coil_pitch_slots / n_slots / n_p")

    @property
    def k_w_layout(self) -> float:
        return layout_winding_factor(self.C, self.n_p)

    @property
    def turns_per_side(self) -> float:
        """N_c / a — series turns per coil side per parallel branch."""
        return self.N_eff * _M_PHASES / self.geometry.n_slots


def prepare_im_fem_harmonic(motor: Motor, caller: str = "im_fem_harmonic") -> ImFemParams:
    """Validate and extract the full-machine harmonic inputs from a Motor."""
    from phasesweep.models.im_bar_impedance import ring_segment

    ind = motor.induction
    if ind is None:
        raise ValueError(
            f"Motor '{motor.name}': {caller} requires an [induction] section "
            f"(squirrel-cage rotor); this motor is a synchronous machine")
    geo = require_geometry(motor, caller)
    if geo.n_slots <= 0 or geo.slot_depth <= 0:
        raise ValueError(
            f"Motor '{motor.name}': {caller} needs stator slots ([geometry] "
            f"n_slots and slot_depth) to carry the winding")
    for field_name, value in (("sigma_bar", ind.sigma_bar), ("f_rated", ind.f_rated),
                              ("V_LL_rated", ind.V_LL_rated)):
        if value is None:
            raise ValueError(f"Motor '{motor.name}': {caller} needs [induction] {field_name}")
    if motor.R_s is None:
        raise ValueError(f"Motor '{motor.name}': {caller} needs [circuit] R_s (stator R_1)")
    if motor.k_w is None:
        raise ValueError(f"Motor '{motor.name}': {caller} needs [winding] k_w")
    if motor.L_stk is None:
        raise ValueError(f"Motor '{motor.name}': {caller} needs L_stk")
    pitch = motor.coil_pitch_slots
    if pitch is None:
        pitch, rem = divmod(geo.n_slots, 2 * motor.n_p)
        if rem:
            raise ValueError(
                f"Motor '{motor.name}': {caller} needs [winding] coil_pitch_slots "
                f"(n_slots={geo.n_slots} is not a multiple of 2·n_p={2 * motor.n_p})")
    try:
        C = winding_matrix(geo.n_slots, motor.n_p, pitch)
    except ValueError as e:
        raise ValueError(f"Motor '{motor.name}': {caller}: {e}") from e
    assert ind.sigma_bar is not None and ind.f_rated is not None and ind.V_LL_rated is not None
    # An ideal (zero-impedance) ring would drop the ring's own I^2R, 5-15 %
    # of rotor copper loss on a typical cage, so the factory requires the
    # section as im_bar_impedance does. Direct construction may pass
    # ring=None for the superconducting limit.
    missing = [f for f in ("end_ring_axial_length", "end_ring_radial_height")
               if getattr(geo, f) is None]
    if missing:
        raise ValueError(
            f"Motor '{motor.name}': {caller} needs [geometry] "
            f"{', '.join(missing)} for the end-ring impedance")
    ring = ring_segment(geo, ind.sigma_bar)
    f_table: list[float] = []
    for table in (ind.R_2_vs_f, ind.L_sigma_r_vs_f):
        if table is not None:
            f_table.extend(f for f, _v in table)
    N_eff = n_eff(motor)  # raises the named error for missing N / coils_series
    assert motor.N is not None and motor.coils_series is not None
    L_ew = end_winding_leakage(
        r_stator=geo.r_stator, n_slots=geo.n_slots, n_p=motor.n_p,
        coil_pitch_slots=pitch, N=motor.N, coils_series=motor.coils_series,
    )
    try:
        return ImFemParams(
            geometry=geo, n_p=motor.n_p, sigma_bar=ind.sigma_bar, mu_r_fe=motor.mu_r_fe,
            L_stk=motor.L_stk, N_eff=N_eff, k_w=motor.k_w, R_1=motor.R_s,
            f_rated=ind.f_rated, V_ph_rms=ind.V_LL_rated / sqrt(3), C=C,
            f_table=tuple(f_table),
            leakage_split=ind.design_letter if ind.design_letter is not None else 0.5,
            ring=ring,
            L_ew=L_ew,
            R_fe=ind.R_fe,
            bh=bh_curve_for(motor),
        )
    except ValueError as e:
        raise ValueError(f"Motor '{motor.name}': {caller}: {e}") from e


@dataclass(frozen=True)
class ImFemOperatingPoint:
    """One solve at 1 A rms per phase; every extensive quantity is per
    A²rms of stator current (linear model), whole stack.

    Z_int        — FEM internal impedance jωψ/I (Ω): 2D flux linkage only,
                   no R_1, L_ew or R_fe (operating_point_at_voltage adds them)
    phase_asym   — max spread of Z_int across the three phases, relative
                   to |R_1 + Z_int|
    tau_per_A2   — Maxwell-stress torque (N·m/A²)
    p_field_per_A2 — power the winding delivers to the field (W/A²):
                   equals p_bar + p_ring + p_mech in the discrete model
    p_bar_per_A2, p_ring_per_A2 — cage losses (W/A²)
    p_gap_per_A2 — Poynting flux through the mid-gap circle (W/A²)
    I_bar_rms_max_per_A — largest bar current (A rms per A rms)
    """

    f_supply: float
    slip: float
    Z_int: complex
    phase_asym: float
    tau_per_A2: float
    p_field_per_A2: float
    p_bar_per_A2: float
    p_ring_per_A2: float
    p_gap_per_A2: float
    I_bar_rms_max_per_A: float
    result: HarmonicResult

    @property
    def omega(self) -> float:
        return 2 * pi * self.f_supply


def solve_operating_point(mesh: Any, p: ImFemParams, f_supply: float, slip: float,
                          order: int = _ORDER, nu_iron: Any = None) -> ImFemOperatingPoint:
    """Full-machine solve at 1 A rms per phase, supply frequency f_supply
    and the given slip (0 = synchronous, 1 = locked rotor). ``nu_iron``
    replaces the constant 1/(μ0 mu_r_fe) of the stator and rotor iron
    with a reluctivity CoefficientFunction (the saturation fixed point)."""
    from ngsolve import CoefficientFunction, Integrate

    from phasesweep.solvers.fem_harmonic import CageCircuit, HarmonicSpec, solve_harmonic

    geo = p.geometry
    omega = 2 * pi * f_supply
    bars = tuple(f"bar{k}" for k in range(geo.n_bars))
    slots = [f"slot{k}" for k in range(geo.n_slots)]
    areas = np.array([float(Integrate(CoefficientFunction(1), mesh,
                                      definedon=mesh.Materials(r))) for r in slots])
    I_ph = balanced_currents(1.0)
    turns = p.turns_per_side
    J_slot = turns * (p.C.T @ I_ph) / areas          # peak A/m² per slot
    Z_ring = None
    if p.ring is not None:
        R_seg, L_seg = p.ring
        Z_ring = complex(R_seg, slip * omega * L_seg)
    nu_fe = nu_iron if nu_iron is not None else 1.0 / (MU0 * p.mu_r_fe)
    spec = HarmonicSpec(
        omega=omega,
        nu={r: nu_fe for r in _IRON_REGIONS},
        sigma={b: p.sigma_bar for b in bars},
        J_source={r: complex(J) for r, J in zip(slots, J_slot)},
        slip=slip,
        cage=CageCircuit(bars=bars, L_stk=p.L_stk, Z_ring=Z_ring),
        order=order,
    )
    res = solve_harmonic(mesh, spec)

    A_avg = np.array([complex(Integrate(res.A, mesh, definedon=mesh.Materials(r)))
                      for r in slots]) / areas
    psi = p.L_stk * turns * (p.C @ A_avg)            # peak Wb per phase
    Z_all = 1j * omega * psi / I_ph
    Z = complex(np.mean(Z_all))
    # relative to R_1 + Z_int, the V/I a measured per-phase test reads;
    # against Z_int alone a small locked-rotor Z_int inflates the ratio
    asym = float(np.max(np.abs(Z_all - Z)) / abs(p.R_1 + Z))

    L = p.L_stk
    p_field = res.input_power() * L
    p_bar = res.loss(bars) * L
    p_ring = res.ring_loss() if Z_ring is not None else 0.0
    return ImFemOperatingPoint(
        f_supply=f_supply, slip=slip, Z_int=Z, phase_asym=asym,
        tau_per_A2=res.torque(geo.r_ag, n_theta=_N_THETA_TORQUE, L_stk=L),
        p_field_per_A2=p_field, p_bar_per_A2=p_bar, p_ring_per_A2=p_ring,
        p_gap_per_A2=res.gap_power(geo.r_ag, n_theta=_N_THETA_TORQUE, L_stk=L),
        I_bar_rms_max_per_A=max(abs(i) for i in res.I_bar.values()) / sqrt(2),
        result=res,
    )


def skew_offsets(skew_angle: float | None,
                 n_slices: int = _N_SKEW_SLICES) -> tuple[float, ...]:
    """Bar-pattern rotation of each skew slice (mechanical rad).

    Midpoint rule over the stack, centered: offset_j = skew·((j + ½)/N − ½),
    the same placement as fem_axial._plane_radii. The discrete average of
    e^{j n_p δ} over these offsets equals sin(x)/(N sin(x/N)) with
    x = n_p·skew/2, a property of the placement formula — but the
    decoupled solver does not enforce the axial bar-current constraint
    that would make this the physical skew factor (see module docstring).
    Returns (0.0,) for an unskewed cage (skew_angle None): one slice,
    the plain F4 solve."""
    if skew_angle is None:
        return (0.0,)
    if n_slices < 2:
        raise ValueError(f"n_slices={n_slices}: a skewed solve needs >= 2 slices")
    return tuple(skew_angle * ((j + 0.5) / n_slices - 0.5) for j in range(n_slices))


def _slice_meshes(p: ImFemParams, maxh_fraction: float) -> list[Any]:
    """One cage mesh per skew slice; a single plain mesh when unskewed."""
    from phasesweep.solvers.fem_mesh import get_or_build_cage_mesh

    return [get_or_build_cage_mesh(p.geometry, maxh_fraction, bar_offset=d)
            for d in skew_offsets(p.geometry.skew_angle)]


def solve_operating_point_multislice(
        meshes: list[Any], p: ImFemParams, f_supply: float, slip: float,
        order: int = _ORDER) -> ImFemOperatingPoint:
    """Decoupled multi-slice solve: average of solve_operating_point over
    the slice meshes. Z, torque and every power are means (N series
    slices of length L/N, each solved at full stack length, sum to the
    mean); phase_asym and the worst bar current are maxima (a reported
    spread must not be averaged away); ``result`` is the center slice's
    field for the dashboard grid. One mesh returns the plain solve
    unchanged (the unskewed path is bit-identical to F4)."""
    if len(meshes) == 1:
        return solve_operating_point(meshes[0], p, f_supply, slip, order)
    ops = [solve_operating_point(m, p, f_supply, slip, order) for m in meshes]
    mid = ops[len(ops) // 2]
    return ImFemOperatingPoint(
        f_supply=f_supply, slip=slip,
        Z_int=complex(np.mean([op.Z_int for op in ops])),
        phase_asym=max(op.phase_asym for op in ops),
        tau_per_A2=float(np.mean([op.tau_per_A2 for op in ops])),
        p_field_per_A2=float(np.mean([op.p_field_per_A2 for op in ops])),
        p_bar_per_A2=float(np.mean([op.p_bar_per_A2 for op in ops])),
        p_ring_per_A2=float(np.mean([op.p_ring_per_A2 for op in ops])),
        p_gap_per_A2=float(np.mean([op.p_gap_per_A2 for op in ops])),
        I_bar_rms_max_per_A=max(op.I_bar_rms_max_per_A for op in ops),
        result=mid.result,
    )


def _reject_skewed_nonlinear(p: ImFemParams, caller: str) -> None:
    if p.geometry.skew_angle is not None:
        raise ValueError(
            f"{caller}: nonlinear=True with a skewed cage (skew_angle="
            f"{p.geometry.skew_angle}) is not implemented — the saturation "
            f"fixed point needs one stator current common to all skew "
            f"slices. Run the skewed machine linear, or the saturated "
            f"machine unskewed (skew_angle = None)")


@dataclass(frozen=True)
class SaturationState:
    """Reluctivity field of a converged (or warm-start) saturation solve.

    nu           — L2 order-0 GridFunction, element-wise ν used by the
                   solve held beside it (air/bar/slot elements carry a
                   placeholder; only ``iron_mask`` elements are read)
    iterations   — fixed-point iterations run (0 = linear)
    residual     — last raw step: area-integrated |Δμ| over the iron
                   relative to its area-integrated μ
    I_rms        — stator current the field was scaled to (A rms)
    B_peak_iron_max — largest element-mean peak-over-cycle |B| in iron (T)
    """

    nu: Any
    iron_mask: NDArray[np.bool_]
    iterations: int
    residual: float
    I_rms: float
    B_peak_iron_max: float


def _element_areas(mesh: Any) -> NDArray[np.floating]:
    from ngsolve import CoefficientFunction, Integrate

    return np.asarray(Integrate(CoefficientFunction(1), mesh, element_wise=True), dtype=float)


def _iron_mask(mesh: Any, fes_nu: Any) -> NDArray[np.bool_]:
    from ngsolve import GridFunction

    ind = GridFunction(fes_nu)
    ind.Set(mesh.MaterialCF({r: 1.0 for r in _IRON_REGIONS}, default=0.0))
    return ind.vec.FV().NumPy() > 0.5


def cycle_moments(res: HarmonicResult, fes_nu: Any, fes_cx: Any,
                  ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Element means of S = |B̂_x|² + |B̂_y|² and D = |B̂_x² + B̂_y²| of the
    phasor field: |B(t)|² = ½(S + D cos 2ωt), peak² = ½(S + D)."""
    from ngsolve import Conj, GridFunction, grad

    Bx, By = grad(res.A)[1], -grad(res.A)[0]
    gS = GridFunction(fes_nu)
    gS.Set((Bx * Conj(Bx) + By * Conj(By)).real)
    gP = GridFunction(fes_cx)
    gP.Set(Bx * Bx + By * By)
    S = np.maximum(gS.vec.FV().NumPy().copy(), 0.0)
    D = np.minimum(np.abs(gP.vec.FV().NumPy()), S)
    return S, D


def solve_saturated_operating_point(
    mesh: Any, p: ImFemParams, f_supply: float, slip: float, V_ph_rms: float, *,
    order: int = _ORDER, tol: float = _SAT_TOL, tol_current: float = _SAT_TOL_I,
    max_iter: int = _SAT_MAX_ITER, relax: float = _SAT_RELAX,
    state: SaturationState | None = None, I_rms: float | None = None,
) -> tuple[ImFemOperatingPoint, SaturationState]:
    """Energy-equivalent saturation fixed point at a supply voltage.

    Iterates: linear solve at 1 A rms on the current ν field → FEM
    branch current I_int from the circuit owner (circuit_currents, the
    arithmetic of operating_point_at_voltage: R_1, L_ew and R_fe all
    included) → cycle moments of the field scaled to I →
    ν_eff per iron element from the curve (BHCurve.nu_harmonic) →
    log-space relaxed update ν ← ν (ν_eff/ν)^β, each element's ratio
    limited to a factor 2 per iteration (a trust region: a knee element
    whose secant wants to jump 10³ walks there over several iterations
    while its neighbours adjust, instead of flipping the field around
    it) and clipped to the curve's reluctivity range (ratios of 10³
    between a saturated bridge and the tooth beside it make an
    arithmetic step one-sided). Stops when the
    raw (unrelaxed) step, measured as the area-integrated change of
    permeability over the iron relative to its area-integrated
    permeability, Σ a|Δμ| / Σ aμ, is below ``tol`` AND the stator
    current moved less than ``tol_current`` since the previous
    iteration (a step below 1e-6 alone suffices: the linear case). The
    permeance weighting is deliberate: past the knee the secant map is
    not a contraction, and at the CREATOR rated point a fringe of
    deeply saturated tooth-tip and bridge elements settles into a stable
    two-cycle, ν flipping ~30× between the saturated and the vacuum end
    of the curve, while the current is still to 5e-4 — the flux avoids
    those elements either way, so a change of μ near μ0 is weightless
    in this norm and a tooth crossing the knee is not. The returned
    operating point
    was solved on the ν field it is reported with, so
    operating_point_at_voltage at the same voltage is exact for it.
    Relaxation halves whenever the raw step has not fallen by 10 % against
    the best of the last three iterations (growth or a plateau
    oscillation) and recovers on decay (floor 0.05). ``state`` warm-starts
    from a previous solve (a rising voltage sweep).

    Raises RuntimeError when ``max_iter`` is exhausted; a non-converged
    field is never returned.
    """
    from ngsolve import L2, GridFunction

    fes_nu = L2(mesh, order=0)
    fes_cx = L2(mesh, order=0, complex=True)
    mask = _iron_mask(mesh, fes_nu)
    gf_nu = GridFunction(fes_nu)
    nu_vals = gf_nu.vec.FV().NumPy()
    if state is None:
        nu_vals[:] = 1.0 / (MU0 * p.mu_r_fe)
    else:
        nu_vals[:] = state.nu.vec.FV().NumPy()
    nu_lo = float(p.bh.nu_secant(np.linspace(0.0, p.bh.B[-1], 400)).min())
    nu_hi = 1.0 / MU0
    weights = _element_areas(mesh)[mask]
    weights /= weights.sum()

    history: list[float] = []
    # a warm start at the same voltage is converged on its first step
    prev_I = float("nan") if state is None else state.I_rms
    beta = relax
    for it in range(1, max_iter + 1):
        op = solve_operating_point(mesh, p, f_supply, slip, order, nu_iron=gf_nu)
        I_target = abs(circuit_currents(op, p, V_ph_rms)["I_int"])
        I_used = I_target if I_rms is None else I_rms      # current-driven variant
        S, D = cycle_moments(op.result, fes_nu, fes_cx)
        S, D = S * I_used**2, D * I_used**2
        x = nu_vals[mask].copy()
        f = p.bh.nu_harmonic(S[mask], D[mask]) - x
        delta = float(np.sum(weights * np.abs(1 / (x + f) - 1 / x)) / np.sum(weights / x))
        B_peak = float(np.sqrt(0.5 * (S[mask] + D[mask])).max())
        dI = abs(I_target / prev_I - 1)
        _log.debug("saturation it %d: mu step %.3g (max %.3g) dI %.2g relax %.3g I %.4f A "
                   "B_peak %.3f T", it, delta, np.max(np.abs(f / x)), dI, beta, I_target, B_peak)
        if delta < tol and (dI < tol_current or delta < 1e-6):
            return op, SaturationState(nu=gf_nu, iron_mask=mask, iterations=it,
                                       residual=delta, I_rms=I_used, B_peak_iron_max=B_peak)
        # damp on growth or stagnation of the raw step, recover on decay
        stalled = len(history) >= 3 and delta > 0.9 * min(history[-3:])
        beta = (max(beta * 0.5, _SAT_RELAX_FLOOR) if stalled
                else min(beta * 1.2, relax))
        history.append(delta)
        prev_I = I_target
        ratio = np.clip(((x + f) / x) ** beta, 1 / _SAT_MAX_RATIO, _SAT_MAX_RATIO)
        nu_vals[mask] = np.clip(x * ratio, nu_lo, nu_hi)   # log-space step, trust region
    raise RuntimeError(
        f"saturation fixed point did not converge in {max_iter} iterations "
        f"(last raw step {delta:.3g} > {tol}, relax {beta:.3g}) at "
        f"{V_ph_rms:.1f} V / {f_supply:g} Hz / s = {slip:g}")


def iron_peak_B(op: ImFemOperatingPoint, I_rms: float) -> float:
    """Largest element-mean peak-over-cycle |B| in the iron (T) of a
    unit-current solve scaled to ``I_rms`` (the linear-model counterpart
    of SaturationState.B_peak_iron_max)."""
    from ngsolve import L2

    mesh = op.result.mesh
    fes_nu = L2(mesh, order=0)
    mask = _iron_mask(mesh, fes_nu)
    S, D = cycle_moments(op.result, fes_nu, L2(mesh, order=0, complex=True))
    return float(I_rms * np.sqrt(0.5 * (S[mask] + D[mask])).max())


def circuit_currents(op: ImFemOperatingPoint, p: ImFemParams,
                     V_ph_rms: float) -> dict[str, complex]:
    """The stator circuit, the one place it is built: Z_s = R_1 + jωL_ew
    in series with Z_int, R_fe (when set) shunting Z_int. Returns the rms
    phasors I_1 (terminal), I_int (FEM branch), V_m (across Z_int) and
    the terminal impedance Z_terminal = Z_s + (Z_int ∥ R_fe)."""
    Z_s = complex(p.R_1, op.omega * p.L_ew)
    Z_par = op.Z_int if p.R_fe is None else 1.0 / (1.0 / op.Z_int + 1.0 / p.R_fe)
    Z_terminal = Z_s + Z_par
    I_1 = V_ph_rms / Z_terminal
    V_m = I_1 * Z_par
    return {"I_1": I_1, "I_int": V_m / op.Z_int, "V_m": V_m, "Z_terminal": Z_terminal}


def operating_point_at_voltage(op: ImFemOperatingPoint, p: ImFemParams,
                               V_ph_rms: float) -> dict[str, Any]:
    """Scale the unit-current solve to a supply voltage (per-phase rms).

    The circuit is circuit_currents: V_m across Z_int ∥ R_fe,
    I_int = V_m/Z_int, I_fe = V_m/R_fe, I_1 = I_int + I_fe. Terminal
    quantities (I_1_rms, pf, p_in, p_cu_s, Z_phase_re/_im) include R_1,
    L_ew and I_fe; air-gap quantities (tau, p_ag, p_cu_r, I_bar_rms_max)
    scale with I_int. p_in = p_cu_s + p_fe + p_ag;
    eta_circuit = p_mech / p_in.
    """
    c = circuit_currents(op, p, V_ph_rms)
    I_1, I_int, V_m = c["I_1"], c["I_int"], c["V_m"]
    p_fe = 0.0 if p.R_fe is None else 3 * abs(V_m)**2 / p.R_fe

    # I_int is the current through the FEM branch — it drives the
    # air-gap field, cage currents, and torque.
    I_int_rms = abs(I_int)
    k_int = I_int_rms**2
    p_ag = op.p_field_per_A2 * k_int
    p_cu_r = (op.p_bar_per_A2 + op.p_ring_per_A2) * k_int
    p_mech = p_ag - p_cu_r

    I_rms = abs(I_1)
    p_in = 3 * (V_ph_rms * I_1.conjugate()).real
    pf = p_in / (3 * V_ph_rms * I_rms) if I_rms > 0 else 0.0
    p_cu_s = 3 * p.R_1 * I_rms**2

    r_fe_over_x_m: float | None = None
    if p.R_fe is not None and op.Z_int.imag > 0:
        r_fe_over_x_m = p.R_fe / op.Z_int.imag

    return {
        "V_phase_rms": V_ph_rms,
        "I_1_rms": I_rms,
        "pf": pf,
        "p_in": p_in,
        "p_cu_s": p_cu_s,
        "p_ag": p_ag,
        "p_cu_r": p_cu_r,
        "p_ring": op.p_ring_per_A2 * k_int,
        "p_fe": p_fe,
        "p_mech": p_mech,
        "eta_circuit": p_mech / p_in if p_in > 0 else 0.0,
        "tau": op.tau_per_A2 * k_int,
        "tau_power": p.n_p * p_ag / op.omega,
        "I_bar_rms_max": op.I_bar_rms_max_per_A * I_int_rms,
        "I_int_rms": I_int_rms,
        "R_fe_used": p.R_fe,
        "r_fe_over_x_m": r_fe_over_x_m,
        "Z_phase_re": c["Z_terminal"].real,
        "Z_phase_im": c["Z_terminal"].imag,
    }


def _b_peak_cf(A: Any) -> Any:
    """Peak-over-cycle |B| of the phasor field: for B = Re(B̂ e^{jωt}),
    max_t |B(t)|² = ½(|B̂_x|² + |B̂_y|² + |B̂_x² + B̂_y²|)."""
    from ngsolve import Conj, Norm, grad, sqrt

    Bx, By = grad(A)[1], -grad(A)[0]
    S = (Bx * Conj(Bx) + By * Conj(By)).real
    D = Norm(Bx * Bx + By * By)
    return sqrt(0.5 * (S + D))


def field_grid(op: ImFemOperatingPoint, I_rms: float, r_bound: float,
               ) -> tuple[list[list[float | None]], list[float]]:
    """Peak |B| (T) on the dashboard raster at the given stator current."""
    from phasesweep.solvers.fem_sampling import rasterise_fields

    scale = I_rms          # solved at 1 A rms; B is linear in I
    xi, _yi, fields = rasterise_fields(op.result.mesh, {"B": scale * _b_peak_cf(op.result.A)},
                                       n_grid=_RASTER_N_GRID, r_bound=r_bound)
    grid = [[round(float(v), 5) if np.isfinite(v) else None for v in row]
            for row in fields["B"]]
    return grid, np.round(xi[0], 6).tolist()


def _require_slip(config: RunConfig) -> float:
    if config.slip is None:
        raise ValueError(
            "im_fem_harmonic needs RunConfig.slip: the operating slip is a "
            "deliberate choice (rated slip, standstill, ...)")
    if not 0 <= config.slip <= 1:
        raise ValueError(f"slip={config.slip} must be in [0, 1]")
    return config.slip


def run_im_fem_harmonic(config: RunConfig) -> dict[str, Any]:
    """Operating point at RunConfig.slip and the rated supply."""
    slip = _require_slip(config)
    p = prepare_im_fem_harmonic(config.motor)
    if config.nonlinear:
        _reject_skewed_nonlinear(p, "im_fem_harmonic")
    meshes = _slice_meshes(p, config.maxh_fraction)
    if config.nonlinear:
        op, sat = solve_saturated_operating_point(meshes[0], p, p.f_rated, slip, p.V_ph_rms)
        iterations, B_peak = sat.iterations, sat.B_peak_iron_max
    else:
        op = solve_operating_point_multislice(meshes, p, p.f_rated, slip)
        # skewed: peak field of the CENTER slice (the reported result)
        iterations, B_peak = 0, None  # deferred to after operating_point_at_voltage
    out = operating_point_at_voltage(op, p, p.V_ph_rms)
    I_int = out["I_int_rms"]
    if B_peak is None:
        B_peak = iron_peak_B(op, I_int)
    grid, coords = field_grid(op, I_int, p.geometry.r_outer)
    return {
        "slip": slip,
        "f_supply_Hz": p.f_rated,
        "f_rotor_Hz": slip * p.f_rated,
        **out,
        "phase_asymmetry": op.phase_asym,
        "p_gap_poynting": op.p_gap_per_A2 * I_int ** 2,
        "k_w_layout": p.k_w_layout,
        "ring_model": "segment" if p.ring is not None else "ideal",
        "iron_model": "bh" if config.nonlinear else "linear",
        "skew_model": "multi-slice" if len(meshes) > 1 else "none",
        "n_skew_slices": len(meshes) if len(meshes) > 1 else 0,
        "L_ew_H": p.L_ew,
        "bh_iterations": iterations,
        "B_peak_iron_max": B_peak,
        "B_mag_grid": grid,
        "grid_coords_list": coords,
    }


def locked_rotor_frequencies(p: ImFemParams) -> list[float]:
    grid = {round(f, 9) for f in p.f_table} | {round(p.f_rated, 9)}
    return sorted(grid)


def no_load_sweep(meshes: list[Any], p: ImFemParams, lr_point: tuple[float, float, float],
                  nonlinear: bool) -> dict[str, list[float]]:
    """Virtual no-load tests at f_rated over _NO_LOAD_FRACTIONS of the
    rated phase voltage, each fed with the f_rated locked-rotor point to
    identify_from_tests (the IEEE-112 no-load step: L_m = (X_nl − X_1)/ω
    at that saturation state). Reported V/I/L are terminal (L_ew in);
    the identification point drops the jωL_ew·I_1 drop from V, matching
    the 2D locked-rotor point, so L_ew cannot leak into L_m. Saturated solves warm-start up the
    voltage ladder; the linear model scales one (slice-averaged) solve.
    ``meshes`` are the skew-slice meshes; the nonlinear ladder only ever
    sees one (skew + nonlinear is rejected by the runner)."""
    from phasesweep.models.im_circuit import identify_from_tests

    mesh = meshes[0]
    state: SaturationState | None = None
    lin = None if nonlinear else solve_operating_point_multislice(meshes, p, p.f_rated, 0.0)
    keys = ("V_no_load_list", "I_no_load_rms_list", "L_no_load_list", "L_m_list",
            "I_m_rms_list", "B_peak_iron_no_load_list", "bh_iterations_list")
    out: dict[str, list[float]] = {k: [] for k in keys}
    for frac in _NO_LOAD_FRACTIONS:
        V = frac * p.V_ph_rms
        if nonlinear:
            op, state = solve_saturated_operating_point(mesh, p, p.f_rated, 0.0, V, state=state)
            B_peak, iterations = state.B_peak_iron_max, state.iterations
        else:
            assert lin is not None
            op = lin
        pt = operating_point_at_voltage(op, p, V)
        if not nonlinear:
            B_peak, iterations = iron_peak_B(op, pt["I_int_rms"]), 0
        I_rms = pt["I_1_rms"]
        P = pt["p_cu_s"] + max(pt["p_ag"], 0.0) + pt["p_fe"]
        I_1 = circuit_currents(op, p, V)["I_1"]
        V_id = abs(V - 1j * op.omega * p.L_ew * I_1)       # the 2D virtual test's voltage
        ident = identify_from_tests(
            R_1=p.R_1, no_load=(V_id, I_rms, P), locked_rotor=lr_point,
            f_no_load=p.f_rated, f_locked_rotor=p.f_rated, leakage_split=p.leakage_split)
        for k, v in zip(keys, (V, I_rms, pt["Z_phase_im"] / op.omega, ident.L_m,
                               ident.I_m_rms, B_peak, float(iterations))):
            out[k].append(v)
    return out


def run_im_fem_locked_rotor(config: RunConfig) -> dict[str, Any]:
    """Virtual blocked-rotor tests through the IEEE-112 locked-rotor
    step (linear iron), the linear-iron no-load current diagnostic, and
    the no-load voltage sweep — an L_m(I_m) identification when
    RunConfig.nonlinear is set, the flat linear value otherwise."""
    from phasesweep.models.im_circuit import identify_locked_rotor

    p = prepare_im_fem_harmonic(config.motor, "im_fem_locked_rotor")
    if config.nonlinear:
        _reject_skewed_nonlinear(p, "im_fem_locked_rotor")
    meshes = _slice_meshes(p, config.maxh_fraction)
    f_list = locked_rotor_frequencies(p)
    R_2, L_s, L_r, R_lr, X_lr, tau_A2, Z_re, Z_im, asym = ([] for _ in range(9))
    for f in f_list:
        op = solve_operating_point_multislice(meshes, p, f, 1.0)
        Z = p.R_1 + op.Z_int
        # a 1 A rms virtual test point on the 2D machine (L_ew added after
        # the split below): V = |Z|, P = 3 Re Z
        lr = identify_locked_rotor(R_1=p.R_1, locked_rotor=(abs(Z), 1.0, 3 * Z.real),
                                   f_locked_rotor=f, leakage_split=p.leakage_split)
        R_2.append(lr.R_2)
        L_s.append(lr.L_sigma_s + p.L_ew)
        L_r.append(lr.L_sigma_r)
        R_lr.append(lr.R_lr)
        X_lr.append(lr.X_lr)
        tau_A2.append(op.tau_per_A2)
        Z_re.append(Z.real)
        Z_im.append(Z.imag)
        asym.append(op.phase_asym)

    nl = solve_operating_point_multislice(meshes, p, p.f_rated, 0.0)
    nl_out = operating_point_at_voltage(nl, p, p.V_ph_rms)
    # f_list is rounded to 9 decimals; f_rated = 50/3 is not in it unrounded
    i_rated = f_list.index(round(p.f_rated, 9))
    Z_rated = complex(Z_re[i_rated], Z_im[i_rated])
    sweep = no_load_sweep(meshes, p, (abs(Z_rated), 1.0, 3 * Z_rated.real), config.nonlinear)
    return {
        "f_locked_rotor_list": f_list,
        "R_2_list": R_2,
        "L_sigma_s_list": L_s,
        "L_sigma_r_list": L_r,
        "R_lr_list": R_lr,
        "X_lr_list": X_lr,
        "tau_per_A2_list": tau_A2,
        "Z_re_list": Z_re,
        "Z_im_list": Z_im,
        "phase_asymmetry_max": max(asym),
        "leakage_split_used": (identify_locked_rotor(
            R_1=p.R_1, locked_rotor=(abs(Z), 1.0, 3 * Z.real), f_locked_rotor=f_list[-1],
            leakage_split=p.leakage_split).stator_leakage_fraction),
        "R_1": p.R_1,
        "k_w_layout": p.k_w_layout,
        "ring_model": "segment" if p.ring is not None else "ideal",
        "skew_model": "multi-slice" if len(meshes) > 1 else "none",
        "n_skew_slices": len(meshes) if len(meshes) > 1 else 0,
        "L_ew_H": p.L_ew,
        # linear-iron no-load diagnostic: NOT an L_m identification (F5)
        "I_no_load_rms_linear_iron": nl_out["I_1_rms"],
        "L_no_load_linear_iron": nl_out["Z_phase_im"] / nl.omega,
        "f_no_load_Hz": p.f_rated,
        "V_phase_rms": p.V_ph_rms,
        "iron_model": "bh" if config.nonlinear else "linear",
        **sweep,
    }
