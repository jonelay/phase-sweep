"""Steady-state per-phase equivalent circuit for squirrel-cage induction
machines.

The IEEE/Steinmetz circuit, rotor referred to the stator:

    V_1 ── R_1 ── jX_1 ──┬── R_fe ──┐
                         │          │
                         ├── jX_m ──┤
                         │          │
                         └── R_2/s ── jX_2

R_fe is a parallel core-loss shunt across the magnetizing inductance
(Boldea Ch 7 sect 7.1, Ch 11 sect 11.2). When R_fe is None (the default),
the shunt is open and the circuit reduces to the classic lossless-shunt
form — backward compatible with all existing motors.

Air-gap power P_ag = 3·I_2²·R_2/s; core-loss power P_fe = 3·|V_m|²/R_fe
(0 when the shunt is open); electromagnetic torque τ = P_ag / ω_s, where
ω_s = 2πf/n_p is the synchronous MECHANICAL speed. P_fe is dissipated in
the shunt branch upstream of the rotor, so P_mech = (1-s)·P_ag — core loss
does not come out of the rotor's mechanical output. eta_circuit includes
copper and (when present) core loss — friction, windage, and stray load
loss remain excluded.

Two parameter refinements are supported because real data demands them
(CREATOR IM: R_r +66% across 10-50 Hz rotor frequency, L_m −52% across
the magnetizing sweep):

  - deep-bar tables R_2(f_r) / L_sigma_r(f_r), interpolated at the ROTOR
    frequency f_r = s·f with the scalar reference value anchoring f = 0
    (a cage's resistance approaches its DC value as the rotor field
    slows), clamped above the last tabulated point;
  - a stator-leakage table L_sigma_s(f) at the SUPPLY frequency, same
    anchoring;
  - magnetizing saturation L_m(I_m), resolved by damped fixed-point
    iteration on the magnetizing-branch rms current.

All voltages/currents in this module are per-phase rms (the natural
frame for a mains machine and for every published IM test) — outputs are
suffixed _rms to keep them from being confused with the repo's
peak-valued PM convention.

identify_from_tests() recovers the circuit parameter set
from the standard DC / no-load / locked-rotor tests, the IEEE-112
route; separate_no_load_losses() is its friction+windage companion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "ImCircuitParams",
    "ImLockedRotorIdentification",
    "ImOperatingPoint",
    "ImTestIdentification",
    "StrayLoadResidual",
    "compute_stray_load_residual",
    "identify_from_tests",
    "identify_locked_rotor",
    "prepare_im_circuit",
    "run_im_equivalent_circuit",
    "run_im_torque_slip",
    "separate_no_load_losses",
    "solve_im_circuit",
    "thevenin_breakdown",
]


@dataclass(frozen=True)
class ImOperatingPoint:
    """One solved circuit operating point (per-phase rms quantities).

    When R_fe is present (core-loss shunt across L_m), p_fe carries the
    core-loss power 3·|V_m|^2/R_fe. P_fe is dissipated in the shunt
    upstream of the rotor (not subtracted from p_mech).
    eta_circuit then includes core loss via the larger p_in.
    """

    slip: float
    f_supply: float
    f_rotor: float
    V_phase_rms: float
    I_1_rms: float
    I_2_rms: float
    I_m_rms: float
    pf: float
    tau: float
    p_in: float
    p_ag: float
    p_cu_s: float
    p_cu_r: float
    p_fe: float
    p_mech: float
    eta_circuit: float
    R_2_used: float
    R_fe_used: float | None
    L_m_used: float
    L_sigma_s_used: float
    L_sigma_r_used: float


def _interp_anchored(table, x: float, y_ref: float) -> float:
    """Piecewise-linear interpolation with an implicit (0, y_ref) anchor,
    clamped at the last tabulated x."""
    if table is None:
        return y_ref
    if x >= table[-1][0]:
        return table[-1][1]
    xs = [0.0] + [row[0] for row in table]
    ys = [y_ref] + [row[1] for row in table]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def solve_im_circuit(
    p: ImCircuitParams, slip: float, *,
    V_ph_rms: float | None = None, f_supply: float | None = None,
) -> ImOperatingPoint:
    """Solve the per-phase circuit at one slip.

    V_ph_rms / f_supply override the rated supply (reduced-voltage
    locked-rotor tests are the canonical use). slip must be in (0, 1] —
    the motoring range; generator/plugging operation is out of scope for
    v1 and rejected loudly.
    """
    if not (0 < slip <= 1):
        raise ValueError(
            f"slip must be in (0, 1] (motoring range), got {slip} — "
            f"generator (s < 0) and synchronous (s = 0) operation are not "
            f"modeled"
        )
    f = p.f_supply if f_supply is None else f_supply
    V = p.V_LL / sqrt(3) if V_ph_rms is None else V_ph_rms
    if f <= 0 or V <= 0:
        raise ValueError(f"supply must be positive, got V={V}, f={f}")

    f_r = slip * f
    w = 2 * pi * f
    R_2 = _interp_anchored(p.R_2_vs_f, f_r, p.R_2)
    L_ss = _interp_anchored(p.L_sigma_s_vs_f, f, p.L_sigma_s)
    L_sr = _interp_anchored(p.L_sigma_r_vs_f, f_r, p.L_sigma_r)

    Z_1 = complex(p.R_1, w * L_ss)
    Z_2 = complex(R_2 / slip, w * L_sr)

    def _z_mag(L_m_val: float) -> complex:
        """Magnetizing branch impedance: jX_m in parallel with R_fe
        when the core-loss shunt is present, pure jX_m otherwise."""
        Z_jxm = complex(0.0, w * L_m_val)
        if p.R_fe is None:
            return Z_jxm
        return Z_jxm * p.R_fe / (Z_jxm + p.R_fe)

    # Magnetizing saturation: damped fixed point on the branch rms current.
    # I_m is the current through the jX_m inductance alone (not the total
    # magnetizing-branch current which includes I_fe when R_fe is present).
    L_m = p.L_m
    for _ in range(100):
        Z_m = _z_mag(L_m)
        Z = Z_1 + Z_m * Z_2 / (Z_m + Z_2)
        I_1 = V / Z
        V_m = V - I_1 * Z_1
        I_m = abs(V_m) / (w * L_m)
        if p.L_m_vs_i is None:
            break
        L_m_new = _interp_anchored(p.L_m_vs_i, I_m, p.L_m_vs_i[0][1])
        if abs(L_m_new - L_m) <= 1e-9 * p.L_m:
            L_m = L_m_new
            break
        L_m = 0.5 * (L_m + L_m_new)
    else:
        raise RuntimeError(
            "solve_im_circuit: L_m(I_m) fixed point did not converge in "
            "100 iterations"
        )
    Z_m = _z_mag(L_m)
    Z = Z_1 + Z_m * Z_2 / (Z_m + Z_2)
    I_1 = V / Z
    V_m = V - I_1 * Z_1
    I_2 = V_m / Z_2
    # I_m_c is the current through jX_m (the inductive magnetizing
    # component), used for the saturation table key and the I_m_rms output.
    I_m_c = V_m / complex(0.0, w * L_m)

    w_s = 2 * pi * f / p.n_p  # synchronous mechanical speed
    p_ag = 3 * abs(I_2) ** 2 * R_2 / slip
    tau = p_ag / w_s
    p_cu_s = 3 * abs(I_1) ** 2 * p.R_1
    p_cu_r = slip * p_ag
    p_fe = 3 * abs(V_m) ** 2 / p.R_fe if p.R_fe is not None else 0.0
    p_mech = (1 - slip) * p_ag
    p_in = 3 * (V * I_1.conjugate()).real
    pf = p_in / (3 * V * abs(I_1)) if abs(I_1) > 0 else 0.0
    eta = p_mech / p_in if p_in > 0 else 0.0

    return ImOperatingPoint(
        slip=slip, f_supply=f, f_rotor=f_r, V_phase_rms=V,
        I_1_rms=abs(I_1), I_2_rms=abs(I_2), I_m_rms=abs(I_m_c),
        pf=pf, tau=tau, p_in=p_in, p_ag=p_ag, p_cu_s=p_cu_s,
        p_cu_r=p_cu_r, p_fe=p_fe, p_mech=p_mech, eta_circuit=eta,
        R_2_used=R_2, R_fe_used=p.R_fe, L_m_used=L_m,
        L_sigma_s_used=L_ss, L_sigma_r_used=L_sr,
    )


def thevenin_breakdown(
    p: ImCircuitParams, *, V_ph_rms: float | None = None,
    f_supply: float | None = None,
) -> tuple[float, float]:
    """(s_max, tau_max) from the Thévenin-reduced closed form,
    CONSTANT reference parameters (no deep-bar / saturation tables).

    When R_fe is present, the magnetizing impedance is jX_m || R_fe;
    this changes both the Thévenin source impedance and voltage.

    tau_max is independent of R_2 — the classic analytic sanity check on
    a torque-slip implementation (R_2 moves the breakdown slip along the
    curve, never its height). Pinned by a unit test.
    """
    f = p.f_supply if f_supply is None else f_supply
    V = p.V_LL / sqrt(3) if V_ph_rms is None else V_ph_rms
    w = 2 * pi * f
    X_1 = w * p.L_sigma_s
    X_2 = w * p.L_sigma_r
    X_m = w * p.L_m
    Z_jxm = complex(0, X_m)
    Z_mag = Z_jxm * p.R_fe / (Z_jxm + p.R_fe) if p.R_fe is not None else Z_jxm
    Z_1 = complex(p.R_1, X_1)
    Z_th = Z_mag * Z_1 / (Z_mag + Z_1)
    V_th = V * abs(Z_mag / (Z_mag + Z_1))
    R_th, X_th = Z_th.real, Z_th.imag
    s_max = p.R_2 / sqrt(R_th**2 + (X_th + X_2) ** 2)
    w_s = 2 * pi * f / p.n_p
    tau_max = (3 / (2 * w_s)) * V_th**2 / (
        R_th + sqrt(R_th**2 + (X_th + X_2) ** 2))
    return s_max, tau_max


# ---------------------------------------------------------------------------
# Registry runners
# ---------------------------------------------------------------------------

def run_im_equivalent_circuit(config: RunConfig) -> dict[str, Any]:
    """Solve the equivalent circuit at RunConfig.slip, rated supply."""

    p = prepare_im_circuit(config.motor)
    if config.slip is None:
        raise ValueError(
            "im_equivalent_circuit requires RunConfig.slip — the operating "
            "slip is a deliberate choice (rated slip, standstill, ...); "
            "there is no safe default"
        )
    op = solve_im_circuit(p, config.slip)
    # Boldea diagnostic: R_fe / X_m should be ~20 (range 10-40 typical,
    # soft warning outside 5-100). None when no core-loss shunt.
    r_fe_over_x_m: float | None = None
    if op.R_fe_used is not None and op.L_m_used > 0:
        X_m = 2 * pi * op.f_supply * op.L_m_used
        if X_m > 0:
            r_fe_over_x_m = op.R_fe_used / X_m
    return {
        "slip": op.slip,
        "f_supply_Hz": op.f_supply,
        "f_rotor_Hz": op.f_rotor,
        "V_phase_rms": op.V_phase_rms,
        "I_1_rms": op.I_1_rms,
        "I_2_rms": op.I_2_rms,
        "I_m_rms": op.I_m_rms,
        "pf": op.pf,
        "tau": op.tau,
        "p_in": op.p_in,
        "p_ag": op.p_ag,
        "p_cu_s": op.p_cu_s,
        "p_cu_r": op.p_cu_r,
        "p_fe": op.p_fe,
        "p_mech": op.p_mech,
        "eta_circuit": op.eta_circuit,
        "R_2_used": op.R_2_used,
        "R_fe_used": op.R_fe_used,
        "r_fe_over_x_m": r_fe_over_x_m,
        "L_m_used": op.L_m_used,
        "L_sigma_s_used": op.L_sigma_s_used,
        "L_sigma_r_used": op.L_sigma_r_used,
    }


# Fixed sweep grid: dense at low slip where the curve is steep. Not a
# RunConfig knob (and so not in hash_fields) — changing it is a
# version-bumping model change.
_N_SLIP = 400
_SLIP_MIN = 1e-3


def run_im_torque_slip(config: RunConfig) -> dict[str, Any]:
    """Torque-slip sweep at the rated supply (V_LL_rated, f_rated)."""
    import numpy as np


    p = prepare_im_circuit(config.motor)
    slips = np.geomspace(_SLIP_MIN, 1.0, _N_SLIP)
    ops = [solve_im_circuit(p, float(s)) for s in slips]
    tau = np.array([o.tau for o in ops])
    i_max = int(np.argmax(tau))

    s_max_th, tau_max_th = thevenin_breakdown(p)

    out: dict[str, Any] = {
        "slip_list": slips.tolist(),
        "tau_slip_list": tau.tolist(),
        "I_1_rms_list": [o.I_1_rms for o in ops],
        "pf_list": [o.pf for o in ops],
        "eta_circuit_list": [o.eta_circuit for o in ops],
        "p_fe_list": [o.p_fe for o in ops],
        "tau_breakdown": float(tau[i_max]),
        "slip_breakdown": float(slips[i_max]),
        "tau_starting": ops[-1].tau,
        "I_locked_rotor_rms": ops[-1].I_1_rms,
        "R_fe_used": p.R_fe,
        # Constant-parameter Thévenin closed form, for reference. When
        # frequency tables are present the swept breakdown above is the
        # honest value — the gap comes mostly from the LEAKAGE tables
        # (lower leakage at the breakdown rotor frequency raises the
        # peak), not from R_2(f): breakdown torque is R_2-invariant.
        "slip_breakdown_thevenin": s_max_th,
        "tau_breakdown_thevenin": tau_max_th,
        "tau_rated": None,
        "I_1_rated_rms": None,
        "slip_rated": p.s_rated,
    }
    if p.s_rated is not None:
        op_r = solve_im_circuit(p, p.s_rated)
        out["tau_rated"] = op_r.tau
        out["I_1_rated_rms"] = op_r.I_1_rms
    return out


@dataclass(frozen=True)
class ImCircuitParams:
    """Validated inputs for the induction-machine equivalent-circuit
    models (im_equivalent_circuit / im_torque_slip).

    All rotor quantities referred to the stator. R_1 — stator phase
    resistance (Ω); R_2 — rotor resistance (Ω, DC/low-rotor-frequency
    reference); L_sigma_s / L_sigma_r — leakage inductances (H); L_m —
    magnetizing inductance (H, rated-point reference); V_LL — supply
    line-line rms voltage (V); f_supply — supply frequency (Hz).

    Optional tables (see InductionParams): deep-bar R_2_vs_f /
    L_sigma_r_vs_f keyed on ROTOR frequency, L_sigma_s_vs_f keyed on
    supply frequency, L_m_vs_i keyed on magnetizing rms current. s_rated
    is carried for the sweep model's rated-point outputs.

    R_fe — core-loss equivalent resistance (Ω), parallel shunt across
    L_m. None = branch open (lossless magnetizing shunt — backward
    compatible default).
    """

    n_p: int
    R_1: float
    R_2: float
    L_sigma_s: float
    L_sigma_r: float
    L_m: float
    V_LL: float
    f_supply: float
    s_rated: float | None = None
    R_fe: float | None = None
    R_2_vs_f: tuple[tuple[float, float], ...] | None = None
    L_sigma_s_vs_f: tuple[tuple[float, float], ...] | None = None
    L_sigma_r_vs_f: tuple[tuple[float, float], ...] | None = None
    L_m_vs_i: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.n_p < 1:
            raise ValueError(f"n_p must be >= 1, got {self.n_p}")
        for f in ("R_1", "R_2", "L_sigma_s", "L_sigma_r", "L_m", "V_LL",
                  "f_supply"):
            v = getattr(self, f)
            if v <= 0 or not isfinite(v):
                raise ValueError(f"{f} must be > 0 and finite, got {v}")
        if self.R_fe is not None and (self.R_fe <= 0 or not isfinite(self.R_fe)):
            raise ValueError(f"R_fe must be > 0 and finite, got {self.R_fe}")
        if self.s_rated is not None and not (0 < self.s_rated < 1):
            raise ValueError(f"s_rated must be in (0, 1), got {self.s_rated}")
        from phasesweep.machines.motor import check_xy_table
        for name in ("R_2_vs_f", "L_sigma_s_vs_f", "L_sigma_r_vs_f",
                     "L_m_vs_i"):
            check_xy_table(name, getattr(self, name))


def prepare_im_circuit(motor: Motor) -> ImCircuitParams:
    """Validate and extract induction-machine circuit params from a Motor.

    The mirror of reject_induction: a Motor WITHOUT an [induction]
    section is a synchronous machine and is rejected loudly. Requires
    motor.R_s (> 0 — the stator branch drop is structural to the
    circuit) and the full [induction] parameter set plus the rated
    supply (V_LL_rated, f_rated).
    """
    ind = motor.induction
    if ind is None:
        raise ValueError(
            f"Motor '{motor.name}': the im_* models require an [induction] "
            f"section (squirrel-cage rotor); this motor is a synchronous "
            f"machine — use the analytical/fem/rated_torque/torque_speed "
            f"model families instead."
        )
    if motor.R_s is None or motor.R_s <= 0:
        raise ValueError(
            f"Motor '{motor.name}': the im_* models need R_s > 0 "
            f"([circuit] TOML section)"
        )
    if (ind.R_2 is None or ind.L_m is None or ind.L_sigma_s is None
            or ind.L_sigma_r is None or ind.V_LL_rated is None
            or ind.f_rated is None):
        missing = [f for f in ("R_2", "L_m", "L_sigma_s", "L_sigma_r",
                               "V_LL_rated", "f_rated")
                   if getattr(ind, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': the im_* models need [induction] "
            f"{', '.join(missing)}"
        )
    return ImCircuitParams(
        n_p=motor.n_p,
        R_1=motor.R_s,
        R_2=ind.R_2,
        L_sigma_s=ind.L_sigma_s,
        L_sigma_r=ind.L_sigma_r,
        L_m=ind.L_m,
        V_LL=ind.V_LL_rated,
        f_supply=ind.f_rated,
        s_rated=ind.s_rated,
        R_fe=ind.R_fe,
        R_2_vs_f=ind.R_2_vs_f,
        L_sigma_s_vs_f=ind.L_sigma_s_vs_f,
        L_sigma_r_vs_f=ind.L_sigma_r_vs_f,
        L_m_vs_i=ind.L_m_vs_i,
    )


# ---------------------------------------------------------------------------
# Parameter identification from standard tests
# ---------------------------------------------------------------------------

# IEEE-112 empirical stator fraction of the locked-rotor leakage
# reactance, by NEMA design letter (0.5 is also the standard default
# when the letter is unknown, and for wound rotors).
_NEMA_X1_FRACTION = {"A": 0.5, "B": 0.4, "C": 0.3, "D": 0.5}


@dataclass(frozen=True)
class ImTestIdentification:
    """Circuit parameters identified from DC / no-load / locked-rotor
    tests (identify_from_tests). Per-phase rms conventions throughout.

    R_1/R_2/L_sigma_s/L_sigma_r/L_m map directly onto the [induction]
    parameter set; the rest are diagnostics of the identification
    itself. L_m is the value at the no-load test's saturation state —
    I_m_rms records the magnetizing current it belongs to, as the
    REACTIVE component I·sin(φ) of the no-load current (the standard
    saturation-curve key; in the series route the full no-load current
    flows through X_m, ~1% higher at CREATOR's no-load power factor).
    (CREATOR IM: the identified L_m lands on the published L_m(I_m)
    curve at that current, ~9% below the rated-point scalar reference.) P_rot_W is
    core + friction/windage at the no-load point (total power minus
    stator copper); it splits into P_core_W/P_fw_W only when the caller
    supplied a friction+windage power (see separate_no_load_losses).
    """

    R_1: float
    R_2: float
    L_sigma_s: float
    L_sigma_r: float
    L_m: float
    I_m_rms: float
    P_rot_W: float
    P_core_W: float | None
    P_fw_W: float | None
    pf_no_load: float
    pf_locked_rotor: float
    stator_leakage_fraction: float
    f_no_load: float
    f_locked_rotor: float


def _check_test_point(caller: str, name: str,
                      point: tuple[float, float, float],
                      ) -> tuple[float, float, float]:
    V, I_rms, P = (float(x) for x in point)
    for label, v in (("V_ph_rms", V), ("I_rms", I_rms), ("P_W", P)):
        if v <= 0 or not isfinite(v):
            raise ValueError(
                f"{caller}: {name} {label} must be > 0 and "
                f"finite, got {v}")
    if P >= 3 * V * I_rms:
        raise ValueError(
            f"{caller}: {name} power {P} W implies power "
            f"factor >= 1 (3·V·I = {3 * V * I_rms:.1f} VA) — check the "
            f"per-phase-rms voltage convention")
    return V, I_rms, P


@dataclass(frozen=True)
class ImLockedRotorIdentification:
    """Locked-rotor step of the IEEE-112 identification on its own
    (identify_locked_rotor): the rotor resistance and the split leakage
    from one blocked-rotor point, no no-load test involved. Per-phase
    rms conventions. R_lr / X_lr are the measured per-phase locked-rotor
    resistance and reactance before the split."""

    R_2: float
    L_sigma_s: float
    L_sigma_r: float
    R_lr: float
    X_lr: float
    pf_locked_rotor: float
    stator_leakage_fraction: float
    f_locked_rotor: float


def _leakage_fraction(caller: str, leakage_split: float | str) -> float:
    if isinstance(leakage_split, str):
        letter = leakage_split.upper()
        if letter not in _NEMA_X1_FRACTION:
            raise ValueError(
                f"{caller}: unknown NEMA design letter "
                f"{leakage_split!r} (expected one of "
                f"{sorted(_NEMA_X1_FRACTION)})")
        return _NEMA_X1_FRACTION[letter]
    frac = float(leakage_split)
    if not 0 < frac < 1 or not isfinite(frac):
        raise ValueError(
            f"{caller}: leakage_split must be in (0, 1), got {frac}")
    return frac


def _check_positive(caller: str, **values: float) -> None:
    for name, v in values.items():
        if v <= 0 or not isfinite(v):
            raise ValueError(f"{caller}: {name} must be > 0, got {v}")


def _locked_rotor_step(caller: str, R_1: float,
                       point: tuple[float, float, float],
                       f_locked_rotor: float, frac: float,
                       ) -> ImLockedRotorIdentification:
    V_lr, I_lr, P_lr = point
    Z_lr = V_lr / I_lr
    R_lr = P_lr / (3 * I_lr**2)
    X_lr = sqrt(Z_lr**2 - R_lr**2)
    R_2 = R_lr - R_1
    if R_2 <= 0:
        raise ValueError(
            f"{caller}: locked-rotor resistance {R_lr:.4f} Ω "
            f"does not exceed R_1 = {R_1} Ω — inconsistent test data")
    w_lr = 2 * pi * f_locked_rotor
    return ImLockedRotorIdentification(
        R_2=R_2, L_sigma_s=frac * X_lr / w_lr,
        L_sigma_r=(1 - frac) * X_lr / w_lr,
        R_lr=R_lr, X_lr=X_lr,
        pf_locked_rotor=P_lr / (3 * V_lr * I_lr),
        stator_leakage_fraction=frac, f_locked_rotor=f_locked_rotor,
    )


def identify_locked_rotor(
    *,
    R_1: float,
    locked_rotor: tuple[float, float, float],
    f_locked_rotor: float,
    leakage_split: float | str = 0.5,
) -> ImLockedRotorIdentification:
    """The locked-rotor step of identify_from_tests on its own: R_2 =
    P/(3I²) − R_1 and the total leakage reactance split by
    `leakage_split` (stator fraction or NEMA letter), from one
    (V_ph_rms, I_rms, P_W) blocked-rotor point at `f_locked_rotor`.
    identify_from_tests runs exactly this step before the no-load one;
    the induction FEM tier feeds its virtual locked-rotor solves through
    it so a FEM-derived R_2 is identified the way a measured one is —
    including the no-core-loss-branch bias the shared step carries.
    """
    frac = _leakage_fraction("identify_locked_rotor", leakage_split)
    _check_positive("identify_locked_rotor", R_1=R_1,
                    f_locked_rotor=f_locked_rotor)
    point = _check_test_point("identify_locked_rotor", "locked_rotor",
                              locked_rotor)
    return _locked_rotor_step("identify_locked_rotor", R_1, point,
                              f_locked_rotor, frac)


def identify_from_tests(
    *,
    R_1: float,
    no_load: tuple[float, float, float],
    locked_rotor: tuple[float, float, float],
    f_no_load: float,
    f_locked_rotor: float,
    leakage_split: float | str = 0.5,
    P_fw: float | None = None,
) -> ImTestIdentification:
    """IEEE-112 short-form circuit identification from the three
    standard tests. Every test point is (V_ph_rms, I_rms, P_W) with V
    per-phase rms and P the total three-phase power.

    R_1 is the per-phase stator resistance from the DC test (halve a
    line-to-line wye measurement). The locked-rotor test (s = 1, rotor
    frequency = supply frequency) gives R_1 + R_2 and the total leakage
    X, split by `leakage_split` — a stator fraction in (0, 1) or a NEMA
    design letter ("A"/"B"/"C"/"D"); run it at reduced frequency (IEEE
    112 recommends ~25% of rated) so the deep-bar R_2 lands near its
    running value, or at several frequencies to build the R_2_vs_f /
    leakage tables point by point. The no-load test at rated voltage
    gives X_m = X_nl − X_1 (series route — consistent with the
    core-loss-branch-free circuit this module solves, unlike the
    textbook shortcut X_m = V/I_m that skips the stator drop).

    Two known biases, pinned against the CREATOR IM raw data: the
    magnetizing shunt is ignored at locked rotor, so R_2 is a slight
    under-estimate (factor ≈ (X_m/(X_m + X_2))², −2.7..−8.1% across
    the CREATOR band, worst where X_m is smallest); and L_m carries
    the no-load test's saturation state (see ImTestIdentification).
    """
    frac = _leakage_fraction("identify_from_tests", leakage_split)
    _check_positive("identify_from_tests", R_1=R_1, f_no_load=f_no_load,
                    f_locked_rotor=f_locked_rotor)
    V_nl, I_nl, P_nl = _check_test_point(
        "identify_from_tests", "no_load", no_load)
    V_lr, I_lr, P_lr = _check_test_point(
        "identify_from_tests", "locked_rotor", locked_rotor)

    lr = _locked_rotor_step("identify_from_tests", R_1, (V_lr, I_lr, P_lr),
                            f_locked_rotor, frac)
    R_2, L_sigma_s, L_sigma_r = lr.R_2, lr.L_sigma_s, lr.L_sigma_r

    Z_nl = V_nl / I_nl
    R_nl = P_nl / (3 * I_nl**2)
    X_nl = sqrt(Z_nl**2 - R_nl**2)
    X_1 = 2 * pi * f_no_load * L_sigma_s
    if X_nl <= X_1:
        raise ValueError(
            f"identify_from_tests: no-load reactance {X_nl:.3f} Ω does "
            f"not exceed the stator leakage {X_1:.3f} Ω — inconsistent "
            f"test data (or a wrong leakage_split)")
    L_m = (X_nl - X_1) / (2 * pi * f_no_load)

    P_rot = P_nl - 3 * I_nl**2 * R_1
    if P_rot < -1e-6 * max(P_nl, 1.0):
        raise ValueError(
            f"identify_from_tests: no-load power {P_nl} W is below the "
            f"stator copper loss {3 * I_nl**2 * R_1:.1f} W — "
            f"inconsistent test data")
    P_rot = max(P_rot, 0.0)
    P_core: float | None = None
    if P_fw is not None:
        if not 0 <= P_fw < P_rot:
            raise ValueError(
                f"identify_from_tests: P_fw must be in [0, P_rot = "
                f"{P_rot:.1f} W), got {P_fw}")
        P_core = P_rot - P_fw

    return ImTestIdentification(
        R_1=R_1, R_2=R_2, L_sigma_s=L_sigma_s, L_sigma_r=L_sigma_r,
        L_m=L_m,
        I_m_rms=sqrt(I_nl**2 - (P_nl / (3 * V_nl)) ** 2),
        P_rot_W=P_rot, P_core_W=P_core, P_fw_W=P_fw,
        pf_no_load=P_nl / (3 * V_nl * I_nl),
        pf_locked_rotor=lr.pf_locked_rotor,
        stator_leakage_fraction=frac,
        f_no_load=f_no_load, f_locked_rotor=f_locked_rotor,
    )


def separate_no_load_losses(
    points: list[tuple[float, float, float]] | tuple[
        tuple[float, float, float], ...],
    R_1: float,
) -> float:
    """Friction + windage power (W) from a constant-frequency no-load
    voltage sweep, by the IEEE-112 separation: P_rot = P − 3·I²·R_1 is
    fit linearly against V² and extrapolated to V = 0 (core loss scales
    ~V², friction and windage do not). Points are (V_ph_rms, I_rms,
    P_W) rows at ONE supply frequency; the caller chooses the fit
    window (IEEE 112 drops the saturated top of the sweep — on the
    CREATOR IM the full-sweep and low-voltage-only intercepts differ
    by ~5%).
    """
    if R_1 <= 0 or not isfinite(R_1):
        raise ValueError(
            f"separate_no_load_losses: R_1 must be > 0, got {R_1}")
    if len(points) < 2:
        raise ValueError(
            "separate_no_load_losses: need at least 2 points at "
            "distinct voltages")
    xs, ys = [], []
    for point in points:
        V, I_rms, P = _check_test_point(
            "separate_no_load_losses", "no_load", point)
        xs.append(V**2)
        ys.append(P - 3 * I_rms**2 * R_1)
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    var = sum((x - x_bar) ** 2 for x in xs)
    if var == 0:
        raise ValueError(
            "separate_no_load_losses: all points at the same voltage")
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / var
    P_fw = y_bar - slope * x_bar
    if P_fw < 0:
        raise ValueError(
            f"separate_no_load_losses: negative V²-intercept "
            f"({P_fw:.1f} W) — the sweep does not follow the "
            f"core-loss ~ V² separation model")
    return P_fw


# ---------------------------------------------------------------------------
# IEEE-112 stray load loss residual (Boldea Layer 1b)
# ---------------------------------------------------------------------------

def _ieee_112_assumed_sll_pct(P_rated_kW: float) -> float:
    """IEEE-112 assumed stray load loss as a percentage of rated output,
    by power rating (Table 9-2 in IEEE Std 112-2017)."""
    if P_rated_kW <= 90:
        return 1.8
    if P_rated_kW <= 375:
        return 1.5
    if P_rated_kW <= 1850:
        return 1.2
    return 0.9


@dataclass(frozen=True)
class StrayLoadResidual:
    """IEEE-112 method-B stray load loss residual at one load point.

    P_sll = P_in − P_out − P_cu_s − P_cu_r − P_fe − P_fw captures
    what the segregated loss model does not: leakage-flux pulsation,
    high-order MMF harmonics, and surface losses. A significantly
    negative P_sll indicates inconsistent segregation data.
    """

    P_sll_W: float
    P_sll_pct: float
    P_in_W: float
    P_out_W: float
    P_cu_s_W: float
    P_cu_r_W: float
    P_fe_W: float
    P_fw_W: float
    ieee_112_assumed_pct: float | None


def compute_stray_load_residual(
    *,
    P_in: float,
    I_1_rms: float,
    slip: float,
    P_out: float,
    R_1: float,
    P_fe: float,
    P_fw: float,
    P_rated_kW: float | None = None,
) -> StrayLoadResidual:
    """IEEE-112 method-B stray load loss residual at one load point.

    Inputs are measured or identified quantities — the function does
    the loss segregation (P_cu_s, P_ag, P_cu_r) and returns the
    unaccounted residual.

    Parameters
    ----------
    P_in : float
        Measured three-phase input power (W).
    I_1_rms : float
        Measured per-phase stator rms current (A).
    slip : float
        Measured slip, in (0, 1].
    P_out : float
        Measured shaft output power (W), typically T_shaft × ω_r.
    R_1 : float
        Per-phase stator resistance (Ω), from DC test.
    P_fe : float
        Core loss (W), from no-load test / R_fe identification.
    P_fw : float
        Friction + windage (W), from voltage-sweep separation.
    P_rated_kW : float or None
        Rated output power (kW) for IEEE-112 assumed-value lookup.

    Raises
    ------
    ValueError
        If the residual is significantly negative (inconsistent
        segregation — same class as identify_from_tests rejections).
    """
    _check_positive("compute_stray_load_residual",
                    P_in=P_in, I_1_rms=I_1_rms, R_1=R_1)
    for name, v in (("P_out", P_out), ("P_fe", P_fe), ("P_fw", P_fw),
                    ("slip", slip)):
        if not isfinite(v):
            raise ValueError(
                f"compute_stray_load_residual: {name} must be finite, "
                f"got {v}")
    if not (0 < slip <= 1):
        raise ValueError(
            f"compute_stray_load_residual: slip must be in (0, 1], "
            f"got {slip}")
    if P_out < 0:
        raise ValueError(
            f"compute_stray_load_residual: P_out must be >= 0, "
            f"got {P_out}")
    if P_fe < 0 or P_fw < 0:
        raise ValueError(
            f"compute_stray_load_residual: P_fe and P_fw must be >= 0, "
            f"got P_fe={P_fe}, P_fw={P_fw}")

    P_cu_s = 3 * I_1_rms**2 * R_1
    P_ag = P_in - P_cu_s - P_fe
    P_cu_r = slip * P_ag
    P_sll = P_in - P_out - P_cu_s - P_cu_r - P_fe - P_fw

    tol = 0.005 * P_in
    if P_sll < -tol:
        pct_str = (f" (= {P_sll / P_out * 100:.1f}% of P_out)"
                   if P_out > 0 else "")
        raise ValueError(
            f"compute_stray_load_residual: negative residual "
            f"P_sll = {P_sll:.1f} W{pct_str} — the loss segregation is "
            f"inconsistent (P_in={P_in:.1f}, P_out={P_out:.1f}, "
            f"P_cu_s={P_cu_s:.1f}, P_cu_r={P_cu_r:.1f}, "
            f"P_fe={P_fe:.1f}, P_fw={P_fw:.1f})")

    P_sll_pct = P_sll / P_out * 100 if P_out > 0 else 0.0
    assumed = (_ieee_112_assumed_sll_pct(P_rated_kW)
               if P_rated_kW is not None else None)

    return StrayLoadResidual(
        P_sll_W=P_sll, P_sll_pct=P_sll_pct,
        P_in_W=P_in, P_out_W=P_out,
        P_cu_s_W=P_cu_s, P_cu_r_W=P_cu_r,
        P_fe_W=P_fe, P_fw_W=P_fw,
        ieee_112_assumed_pct=assumed,
    )
