"""Validated solver parameter types — type witnesses for model readiness.

Each params type is a frozen dataclass with self-validating __post_init__.
Construction guarantees all fields are present and physically valid.

Factory functions (prepare_analytical, prepare_fem, prepare_drive_sim,
prepare_rated_torque, prepare_stall_torque) derive missing fields from a
Motor, validate completeness, and extract the params.
Direct construction is also legal for cases like armature decomposition
(B_rem overrides) or tests.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from phasesweep.geometry import Geometry

if TYPE_CHECKING:
    from phasesweep.motor import DriveParams, Motor


# ---------------------------------------------------------------------------
# Parameter types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalyticalParams:
    """Validated inputs for Zhu & Howe analytical air-gap field solver.

    geometry in meters; B_rem — magnet remanence (T); mu_r_pm / mu_r_fe —
    relative permeabilities (dimensionless); alpha_p — pole-arc ratio ((0, 1]).

    mu_r_fe is validated and hashed but does not enter the analytical
    solution — Zhu & Howe assumes infinitely permeable iron. It is kept for
    parity with FemParams (where it enters the linear solve) and config_id
    stability.
    """

    geometry: Geometry
    n_p: int
    B_rem: float
    mu_r_pm: float
    mu_r_fe: float
    alpha_p: float = 1.0

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


@dataclass(frozen=True)
class FemParams:
    """Validated inputs for NGSolve FEM magnetostatic solver.

    geometry in meters; B_rem — magnet remanence (T); mu_r_pm / mu_r_fe —
    relative permeabilities (dimensionless); alpha_p — pole-arc ratio ((0, 1]).

    mu_r_fe sets the iron reluctivity of the LINEAR solve only. With
    nonlinear=True the solver replaces it with the generic B-H table
    (see Documented Limitations) — it merely seeds
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


@dataclass(frozen=True)
class DriveSimParams:
    """Validated inputs for motulator PMSM drive simulation.

    R_s — stator resistance (Ω); L_d / L_q — inductances (H);
    psi_f — PM flux linkage (Wb, peak); J — rotor inertia (kg·m²);
    drive — DriveParams operating point (V / A peak / mechanical rad/s).

    `geometry` is optional — `build_sim()` does not read it, it exists for
    phase-sweep's hashing/traceability pipeline. Dynamics-only consumers
    (e.g. cross-checks against an external plant model) may omit it.
    """

    n_p: int
    R_s: float
    L_d: float
    L_q: float
    psi_f: float
    J: float
    drive: DriveParams
    geometry: Geometry | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.R_s < 0:
            raise ValueError(f"R_s must be >= 0, got {self.R_s}")
        if self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")
        if self.psi_f <= 0:
            raise ValueError(f"psi_f must be > 0, got {self.psi_f}")
        if self.J <= 0:
            raise ValueError(f"J must be > 0, got {self.J}")


@dataclass(frozen=True)
class RatedTorqueParams:
    """Validated inputs for rated torque (MTPA at I_rated) computation.

    psi_f — PM flux linkage (Wb, peak); I_rated — rated current (A, peak);
    L_d / L_q — inductances (H), both required for the salient form.
    """

    n_p: int
    psi_f: float
    I_rated: float
    L_d: float | None = None
    L_q: float | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.psi_f <= 0:
            raise ValueError(f"psi_f must be > 0, got {self.psi_f}")
        if self.I_rated <= 0:
            raise ValueError(f"I_rated must be > 0, got {self.I_rated}")
        if self.L_d is not None and self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q is not None and self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")


@dataclass(frozen=True)
class StallTorqueParams:
    """Validated inputs for stall torque (MTPA at I_stall) computation.

    psi_f — PM flux linkage (Wb, peak); L_d / L_q — inductances (H).
    I_stall: drive current limit, I_LIMIT or MAX_I_S (A, peak).
    I_stall_em: electromagnetic stall current U_DC / (sqrt(3) * R_s)
        (A, peak), None when R_s = 0.
    """

    n_p: int
    psi_f: float
    I_stall: float
    I_stall_em: float | None = None
    L_d: float | None = None
    L_q: float | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.psi_f <= 0:
            raise ValueError(f"psi_f must be > 0, got {self.psi_f}")
        if self.I_stall <= 0:
            raise ValueError(f"I_stall must be > 0, got {self.I_stall}")
        if self.I_stall_em is not None and self.I_stall_em <= 0:
            raise ValueError(f"I_stall_em must be > 0, got {self.I_stall_em}")
        if self.L_d is not None and self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q is not None and self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")


@dataclass(frozen=True)
class TorqueSpeedParams:
    """Validated inputs for the torque-speed envelope computation.

    psi_f — PM flux linkage (Wb, peak); R_s — stator resistance (Ω, >= 0);
    L_d / L_q — inductances (H), both required (the voltage limit above
    base speed is inductance-dominated; omitting them would silently give
    a wildly wrong field-weakening range).
    I_peak — drive current limit, I_LIMIT or MAX_I_S (A, peak);
    I_cont — continuous rating I_rated (A, peak), None to skip the
    continuous envelope; U_max — phase peak voltage limit U_DC/sqrt(3).
    """

    n_p: int
    psi_f: float
    R_s: float
    L_d: float
    L_q: float
    I_peak: float
    U_max: float
    I_cont: float | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.psi_f <= 0:
            raise ValueError(f"psi_f must be > 0, got {self.psi_f}")
        if self.R_s < 0:
            raise ValueError(f"R_s must be >= 0, got {self.R_s}")
        if self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")
        if self.I_peak <= 0:
            raise ValueError(f"I_peak must be > 0, got {self.I_peak}")
        if self.I_cont is not None and self.I_cont <= 0:
            raise ValueError(f"I_cont must be > 0, got {self.I_cont}")
        if self.U_max <= self.R_s * self.I_peak:
            raise ValueError(
                f"U_max={self.U_max:.4g} V does not exceed the resistive "
                f"drop R_s*I_peak={self.R_s * self.I_peak:.4g} V — "
                f"drive voltage and motor are mismatched"
            )


@dataclass(frozen=True)
class IronLossParams:
    """Validated inputs for the lumped Bertotti iron-loss model.

    k_h / k_e — per-kg hysteresis / eddy coefficients (steel grade +
    lamination specific); alpha_fe — Steinmetz exponent; m_core — core
    mass (kg) assumed to cycle at B_core (T, peak); W_REF — mechanical
    speed (rad/s) setting f_e = n_p·W_REF/2π.
    """

    n_p: int
    k_h: float
    k_e: float
    alpha_fe: float
    m_core: float
    B_core: float
    W_REF: float

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        for f in ("k_h", "k_e", "alpha_fe", "m_core", "B_core", "W_REF"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be > 0, got {getattr(self, f)}")


@dataclass(frozen=True)
class DemagScreenParams:
    """Validated inputs for the FEM demag screen.

    fem — field-solver params with B_rem derated to magnet_temp when the
    derating fields are set; B_knee — effective knee (T) at that same
    temperature; k_w — winding factor scaling the current sheet.
    """

    fem: FemParams
    B_knee: float
    k_w: float


@dataclass(frozen=True)
class ThermalDutyParams:
    """Validated inputs for thermal-duty (copper-loss budget) computation.

    n_p — pole pairs; psi_f — PM flux linkage (Wb, peak) at the magnet
    operating temperature (derated to magnet_temp when alpha_Br is known,
    else the 20 °C reference value); R_s — stator
    phase resistance (Ω, > 0) at the winding operating temperature (derated
    to winding_temp_limit when known, else the cold datasheet value);
    L_d / L_q — inductances (H) for the salient torque→current inverse
    (None → non-salient k_T mapping). Both saliency signs take the MTPA
    locus; reverse saliency (L_d > L_q) yields gamma < 0 (magnetizing i_d).

    p_s1_budget — S1 continuous loss budget (W). budget_source —
    "thermal_resistance" (from r_th + temps) or "rated_current" (from
    I_rated). duty_profile — torque-time segments ((torque_Nm, duration_s),
    durations > 0). p_fe — lumped Bertotti iron loss at W_REF (W, >= 0;
    0.0 when the Motor [iron] fields are unset): added to the duty
    consumption on both budget paths, and to the rated_current budget
    (the nameplate S1 condition is copper at I_rated PLUS iron at rated
    speed — a copper-only budget with iron-inclusive consumption would
    read the motor over budget at its own nameplate point).

    thermal_time_constant — winding τ_th (s, > 0, None to skip the
    first-order transient march). ambient_temp / temp_rise_limit (°C / K)
    are set together only on the thermal_resistance path, where the
    normalized transient peak converts to an absolute winding temperature.
    """

    n_p: int
    psi_f: float
    R_s: float
    p_s1_budget: float
    budget_source: str
    duty_profile: tuple[tuple[float, float], ...]
    L_d: float | None = None
    L_q: float | None = None
    p_fe: float = 0.0
    thermal_time_constant: float | None = None
    ambient_temp: float | None = None
    temp_rise_limit: float | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.psi_f <= 0:
            raise ValueError(f"psi_f must be > 0, got {self.psi_f}")
        if self.R_s <= 0:
            raise ValueError(f"R_s must be > 0, got {self.R_s}")
        if self.p_s1_budget <= 0:
            raise ValueError(f"p_s1_budget must be > 0, got {self.p_s1_budget}")
        if not self.duty_profile:
            raise ValueError("duty_profile must have at least one segment")
        for tau, dt in self.duty_profile:
            if dt <= 0:
                raise ValueError(f"duty segment duration must be > 0, got {dt}")
        if self.L_d is not None and self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q is not None and self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")
        if self.p_fe < 0:
            raise ValueError(f"p_fe must be >= 0, got {self.p_fe}")
        if self.thermal_time_constant is not None and self.thermal_time_constant <= 0:
            raise ValueError(
                f"thermal_time_constant must be > 0, "
                f"got {self.thermal_time_constant}"
            )
        if self.temp_rise_limit is not None and self.temp_rise_limit <= 0:
            raise ValueError(
                f"temp_rise_limit must be > 0, got {self.temp_rise_limit}"
            )


# ---------------------------------------------------------------------------
# B_rem <-> psi_f derivation (shared by analytical/FEM factories)
# ---------------------------------------------------------------------------

def _coils_per_phase(motor: Motor) -> int:
    """Coils connected in series per phase.

    Requires motor.coils_series to be set explicitly. Auto-derivation
    was removed because n_slots//3 computes coils_per_phase, not
    coils_series — the two differ for motors with parallel branches.
    Silent wrong N_eff is worse than a loud error.

    Set coils_series in [winding] TOML section:
        coils_series = coils_per_phase / parallel_branches
    For smooth-bore motors where N is already total series turns per
    phase, set coils_series = 1.
    """
    if motor.coils_series is not None:
        return motor.coils_series
    raise ValueError(
        f"Motor '{motor.name}': coils_series is required when N and k_w are "
        f"set (needed for N_eff = N × coils_series). "
        f"Add coils_series to [winding] in the TOML "
        f"(= coils_per_phase / parallel_branches)."
    )


def n_eff(motor: Motor) -> int:
    """Total series turns per phase: N (turns per coil) × coils_series.

    Raises ValueError when coils_series is unset — never silently
    derives from n_slots//3, which is wrong for parallel branches.
    """
    if motor.N is None:
        raise ValueError(f"Motor '{motor.name}': N is required for N_eff")
    return motor.N * _coils_per_phase(motor)


def winding_transfer(motor: Motor) -> float:
    """psi_f per unit peak fundamental air-gap B (winding formula at the
    original bore): 2 * N_eff * k_w * r_stator * L_stk / n_p.
    """
    if motor.geometry is None or motor.k_w is None or motor.L_stk is None:
        raise ValueError(
            f"Motor '{motor.name}': winding_transfer needs geometry, k_w, L_stk"
        )
    return (
        2 * n_eff(motor) * motor.k_w * motor.geometry.r_stator
        * motor.L_stk / motor.n_p
    )


def j_s_from_phase_current(
    motor: Motor, i_peak: float, *, slot_width_ratio: float | None = None,
) -> float:
    """FEM slot current-density amplitude j_s (A/m²) for a peak phase
    current, matching the winding's fundamental MMF.

    The FEM armature source J_z = -j_s·k_w·cos(n_p·θ) over the slot faces
    has ampere-turns-per-angle fundamental j_s·k_w·S (S = the slot faces'
    radial moment, ``fem_field.slot_source_moment``); a balanced 3-phase
    winding with N_eff series turns per phase at peak current Î has
    3·k_w·N_eff·Î/π. Equating gives

        j_s = 3 · N_eff · Î / (π · S)

    k_w cancels — the solver applies it to the sheet itself, so pass the
    raw phase current here. The mapping is phase-invariant: the
    solver's ``sheet_phase`` rotates the sheet's fundamental without
    changing its amplitude, so Î is the current magnitude at any current
    angle (sheet_phase = 0: pure i_q; ±π/2: pure i_d). At sheet_phase = 0
    the Maxwell-stress torque closes to the circuit tier's
    1.5·n_p·ψf·i_q analytically.

    Pass ``slot_width_ratio`` only when overriding it in solve_field_fem
    too — the mapping must integrate the same slot faces the mesh builds.
    """
    from math import pi

    from phasesweep.fem_field import slot_source_moment

    geo = _require_geometry(motor, "j_s_from_phase_current")
    S = slot_source_moment(geo, slot_width_ratio=slot_width_ratio,
                           n_p=motor.n_p)
    return 3 * n_eff(motor) * i_peak / (pi * S)


def phase_current_from_j_s(
    motor: Motor, j_s: float, *, slot_width_ratio: float | None = None,
) -> float:
    """Peak phase current (A) equivalent to a FEM sheet amplitude j_s —
    exact inverse of ``j_s_from_phase_current``."""
    from math import pi

    from phasesweep.fem_field import slot_source_moment

    geo = _require_geometry(motor, "phase_current_from_j_s")
    S = slot_source_moment(geo, slot_width_ratio=slot_width_ratio,
                           n_p=motor.n_p)
    return j_s * pi * S / (3 * n_eff(motor))


def _derive_B_rem_from_psi_f(motor: Motor) -> float | None:
    """Derive B_rem from psi_f via Zhu & Howe transfer ratio.

    Carter-consistent: inverts the same model psi_f_carter evaluates —
    Carter-adjusted radii for the field, original bore for the winding —
    so psi_f → B_rem → flux_linkage_peak round-trips exactly.
    Returns derived B_rem, or None if derivation inputs are missing.
    N_eff = N (turns per coil) × coils_series = total series turns per phase.
    """
    from phasesweep.analytical import _derive_B_rem, carter_adjusted_radii

    if motor.psi_f is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None:
        return None
    r_s_c, r_m_c, _k_c = carter_adjusted_radii(motor.geometry, motor.mu_r_pm)
    return _derive_B_rem(
        motor.psi_f, motor.n_p, n_eff(motor), motor.k_w, motor.L_stk,
        r_stator=motor.geometry.r_stator,
        r_magnet=motor.geometry.r_magnet,
        r_rotor=motor.geometry.r_rotor,
        mu_r_pm=motor.mu_r_pm,
        alpha_p=motor.alpha_p,
        r_stator_c=r_s_c,
        r_magnet_c=r_m_c,
    )


def derive_psi_f_smooth(motor: Motor) -> float | None:
    """Derive psi_f from B_rem via Zhu & Howe — smooth-bore (no Carter).

    Explicit baseline helper: slotting is deliberately ignored. Production
    psi_f resolution uses the Carter-consistent _derive_psi_f below.
    Returns derived psi_f, or None if derivation inputs are missing.
    N_eff = N (turns per coil) × coils_series = total series turns per phase.
    """
    import numpy as np

    from phasesweep.analytical import zhu_howe_Br

    if motor.B_rem is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None:
        return None
    geo = motor.geometry
    B_peak = zhu_howe_Br(
        np.array([0.0]), motor.n_p, motor.B_rem, r_eval=geo.r_stator,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=motor.mu_r_pm, alpha_p=motor.alpha_p,
    )[0]
    return B_peak * winding_transfer(motor)


def _derive_psi_f(motor: Motor) -> float | None:
    """Derive psi_f from B_rem, Carter-consistently.

    Exact inverse of _derive_B_rem_from_psi_f and identical to the registry's
    flux_linkage_peak: Carter-adjusted radii for the field, original bore
    for the evaluation point and winding formula. B_rem → psi_f → B_rem
    round-trips exactly.
    Returns derived psi_f, or None if derivation inputs are missing.
    """
    from phasesweep.analytical import carter_adjusted_radii

    if motor.B_rem is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None:
        return None
    r_s_c, r_m_c, _k_c = carter_adjusted_radii(motor.geometry, motor.mu_r_pm)
    return psi_f_carter(motor, r_s_c, r_m_c, motor.B_rem)


def psi_f_carter(
    motor: Motor, r_stator_c: float, r_magnet_c: float, B_rem: float,
) -> float | None:
    """psi_f (Wb, peak) with Carter-adjusted radii for the field solution,
    original bore for both the evaluation point and the winding formula.

    B_rem is passed explicitly so callers can supply the resolved value
    (which may be derived from psi_f) rather than motor.B_rem.
    Returns None if winding inputs (N, k_w, L_stk) are missing.
    """
    import numpy as np

    from phasesweep.analytical import zhu_howe_Br

    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None:
        return None
    geo = motor.geometry
    B_peak = zhu_howe_Br(
        np.array([0.0]), motor.n_p, B_rem, r_eval=geo.r_stator,
        r_stator=r_stator_c, r_magnet=r_magnet_c, r_rotor=geo.r_rotor,
        mu_r_pm=motor.mu_r_pm, alpha_p=motor.alpha_p,
    )[0]
    return B_peak * winding_transfer(motor)


def _resolve_b_rem(motor: Motor) -> float:
    """Get or derive B_rem from motor. Raises ValueError if impossible."""
    if motor.B_rem is not None:
        _check_consistency(motor)
        return motor.B_rem

    derived = _derive_B_rem_from_psi_f(motor)
    if derived is not None:
        return derived

    missing = []
    if motor.B_rem is None:
        missing.append("B_rem")
    if motor.psi_f is None:
        missing.append("psi_f (for derivation)")
    for f in ("N", "k_w", "L_stk"):
        if getattr(motor, f) is None:
            missing.append(f"{f} (for derivation)")
    raise ValueError(
        f"Motor '{motor.name}': cannot determine B_rem — "
        f"missing {', '.join(missing)}"
    )


def _resolve_psi_f(motor: Motor) -> float:
    """Get or derive psi_f from motor. Raises ValueError if impossible."""
    if motor.psi_f is not None:
        _check_consistency(motor)
        return motor.psi_f

    derived = _derive_psi_f(motor)
    if derived is not None:
        return derived

    missing = []
    if motor.psi_f is None:
        missing.append("psi_f")
    if motor.B_rem is None:
        missing.append("B_rem (for derivation)")
    for f in ("N", "k_w", "L_stk"):
        if getattr(motor, f) is None:
            missing.append(f"{f} (for derivation)")
    raise ValueError(
        f"Motor '{motor.name}': cannot determine psi_f — "
        f"missing {', '.join(missing)}"
    )


def _check_consistency(motor: Motor) -> None:
    """Warn if both B_rem and psi_f are present and disagree significantly.

    The B_rem <-> psi_f derivation uses a simplified winding formula
    (N_eff * k_w * Phi_1) which is ~20% accurate for FSCW motors.
    Threshold is 50% to avoid false warnings from model limitations.
    """
    if motor.B_rem is None or motor.psi_f is None:
        return
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return
    derived = _derive_B_rem_from_psi_f(motor)
    if derived is None:
        return
    rel_err = abs(derived - motor.B_rem) / max(abs(motor.B_rem), 1e-12)
    if rel_err > 0.50:
        warnings.warn(
            f"B_rem from psi_f={derived:.4f} disagrees with "
            f"B_rem={motor.B_rem:.4f} by "
            f"{rel_err * 100:.1f}%",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _require_geometry(motor: Motor, model: str) -> Geometry:
    if motor.geometry is None:
        raise ValueError(f"Motor '{motor.name}': {model} needs geometry")
    return motor.geometry


def _salient_pair(motor: Motor) -> tuple[float | None, float | None]:
    """(L_d, L_q) for the salient torque form, or (None, None) unless both set."""
    if motor.L_d is not None and motor.L_q is not None:
        return motor.L_d, motor.L_q
    return None, None


def prepare_analytical(motor: Motor) -> AnalyticalParams:
    """Derive, validate, and extract analytical solver params from a Motor."""
    geo = _require_geometry(motor, "analytical")
    b_rem = _resolve_b_rem(motor)
    return AnalyticalParams(
        geometry=geo,
        n_p=motor.n_p,
        B_rem=b_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
    )


def prepare_fem(motor: Motor) -> FemParams:
    """Derive, validate, and extract FEM solver params from a Motor."""
    geo = _require_geometry(motor, "fem")
    b_rem = _resolve_b_rem(motor)
    return FemParams(
        geometry=geo,
        n_p=motor.n_p,
        B_rem=b_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
    )


def _require_drive_fields(
    motor: Motor, model: str, fields: tuple[str, ...],
) -> None:
    """Raise ValueError if any named DriveParams fields are None."""
    missing = [f for f in fields if getattr(motor.drive, f) is None]
    if missing:
        raise ValueError(
            f"Motor '{motor.name}': {model} needs [drive] "
            f"{', '.join(missing)}"
        )


def prepare_drive_sim(motor: Motor) -> DriveSimParams:
    """Derive, validate, and extract drive sim params from a Motor."""
    psi_f = _resolve_psi_f(motor)

    if motor.R_s is None or motor.L_d is None or motor.L_q is None or motor.J is None:
        missing = [f for f in ("R_s", "L_d", "L_q", "J")
                   if getattr(motor, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': drive_sim needs {', '.join(missing)}"
        )
    _require_drive_fields(motor, "drive_sim", ("U_DC", "MAX_I_S", "W_REF"))

    return DriveSimParams(
        geometry=motor.geometry,
        n_p=motor.n_p,
        R_s=motor.R_s,
        L_d=motor.L_d,
        L_q=motor.L_q,
        psi_f=psi_f,
        J=motor.J,
        drive=motor.drive,
    )


def prepare_rated_torque(motor: Motor) -> RatedTorqueParams:
    """Derive, validate, and extract rated torque params from a Motor."""
    psi_f = _resolve_psi_f(motor)

    if motor.I_rated is None:
        raise ValueError(
            f"Motor '{motor.name}': rated_torque needs I_rated"
        )

    L_d, L_q = _salient_pair(motor)

    return RatedTorqueParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        I_rated=motor.I_rated,
        L_d=L_d,
        L_q=L_q,
    )


def prepare_stall_torque(motor: Motor) -> StallTorqueParams:
    """Derive, validate, and extract stall torque params from a Motor.

    I_stall = I_LIMIT (or MAX_I_S when I_LIMIT is not set).
    I_stall_em = U_DC / (sqrt(3) * R_s) — electromagnetic stall (no drive limit).
    """
    from math import sqrt

    psi_f = _resolve_psi_f(motor)

    if motor.R_s is None:
        raise ValueError(
            f"Motor '{motor.name}': stall_torque needs R_s"
        )

    drive = motor.drive
    if drive.I_LIMIT is None and drive.MAX_I_S is None:
        raise ValueError(
            f"Motor '{motor.name}': stall_torque needs [drive] "
            f"MAX_I_S (or I_LIMIT)"
        )
    I_stall = drive.I_LIMIT if drive.I_LIMIT is not None else drive.MAX_I_S
    assert I_stall is not None  # narrowed above
    if drive.U_DC is None:
        raise ValueError(
            f"Motor '{motor.name}': stall_torque needs [drive] U_DC"
        )
    I_stall_em = drive.U_DC / (sqrt(3) * motor.R_s) if motor.R_s > 0 else None

    L_d, L_q = _salient_pair(motor)

    return StallTorqueParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        I_stall=I_stall,
        I_stall_em=I_stall_em,
        L_d=L_d,
        L_q=L_q,
    )


def prepare_torque_speed(motor: Motor) -> TorqueSpeedParams:
    """Derive, validate, and extract torque-speed envelope params from a Motor.

    I_peak = I_LIMIT (or MAX_I_S when I_LIMIT is not set) — same convention
    as prepare_stall_torque. I_cont = I_rated when set. U_max = U_DC/sqrt(3)
    (SVPWM linear region, phase peak).
    """
    from math import sqrt

    psi_f = _resolve_psi_f(motor)

    R_s, L_d, L_q = motor.R_s, motor.L_d, motor.L_q
    if R_s is None or L_d is None or L_q is None:
        fields = (("R_s", R_s), ("L_d", L_d), ("L_q", L_q))
        missing = [f for f, v in fields if v is None]
        raise ValueError(
            f"Motor '{motor.name}': torque_speed needs {', '.join(missing)}"
        )

    drive = motor.drive
    if drive.I_LIMIT is None and drive.MAX_I_S is None:
        raise ValueError(
            f"Motor '{motor.name}': torque_speed needs [drive] "
            f"MAX_I_S (or I_LIMIT)"
        )
    I_peak = drive.I_LIMIT if drive.I_LIMIT is not None else drive.MAX_I_S
    assert I_peak is not None  # narrowed above
    if drive.U_DC is None:
        raise ValueError(
            f"Motor '{motor.name}': torque_speed needs [drive] U_DC"
        )

    return TorqueSpeedParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        R_s=R_s,
        L_d=L_d,
        L_q=L_q,
        I_peak=I_peak,
        U_max=drive.U_DC / sqrt(3),
        I_cont=motor.I_rated,
    )


def prepare_iron_loss(motor: Motor) -> IronLossParams:
    """Validate and extract lumped iron-loss params from a Motor."""
    k_h, k_e, alpha_fe = motor.k_h, motor.k_e, motor.alpha_fe
    m_core, B_core = motor.m_core, motor.B_core
    if k_h is None or k_e is None or alpha_fe is None or m_core is None or B_core is None:
        missing = [f for f in _IRON_FIELDS if getattr(motor, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': iron_loss needs {', '.join(missing)} "
            f"([iron] TOML section)"
        )

    _require_drive_fields(motor, "iron_loss", ("W_REF",))

    return IronLossParams(
        n_p=motor.n_p,
        k_h=k_h,
        k_e=k_e,
        alpha_fe=alpha_fe,
        m_core=m_core,
        B_core=B_core,
        W_REF=motor.drive.W_REF,  # type: ignore[arg-type]  # validated above
    )


def prepare_demag_screen(motor: Motor) -> DemagScreenParams:
    """Validate and extract demag-screen params from a Motor.

    Needs the FEM inputs plus [materials] B_knee, a slotted geometry (the
    d-axis current sheet integrates over slot faces) and the winding
    fields behind the j_s ↔ phase-current mapping (N, coils_series, k_w).
    When magnet_temp is set, BOTH alpha_Br and alpha_B_knee must be set —
    remanence and knee are evaluated at one consistent temperature:
    B_rem·[1 + alpha_Br·(T − 20)] and B_knee + alpha_B_knee·(T − 20)
    (the knee slope is absolute T/K — the knee can cross zero).
    """
    from phasesweep.thermal_duty import psi_f_at_magnet_temp

    geo = _require_geometry(motor, "demag_screen")
    if geo.n_slots == 0 or geo.slot_depth == 0:
        raise ValueError(
            f"Motor '{motor.name}': demag_screen needs a slotted geometry "
            f"(n_slots > 0, slot_depth > 0) — the d-axis current sheet "
            f"integrates over slot faces"
        )
    b_rem = _resolve_b_rem(motor)
    if motor.B_knee is None:
        raise ValueError(
            f"Motor '{motor.name}': demag_screen needs B_knee "
            f"([materials] TOML section)"
        )
    if motor.k_w is None:
        raise ValueError(
            f"Motor '{motor.name}': demag_screen needs an explicit k_w "
            f"(the current sheet scales with the winding factor)"
        )
    n_eff(motor)  # loud on missing N / coils_series (the j_s mapping inputs)

    b_knee = motor.B_knee
    if motor.magnet_temp is not None:
        alpha_Br, alpha_B_knee = motor.alpha_Br, motor.alpha_B_knee
        if alpha_Br is None or alpha_B_knee is None:
            fields = (("alpha_Br", alpha_Br), ("alpha_B_knee", alpha_B_knee))
            missing = [f for f, v in fields if v is None]
            raise ValueError(
                f"Motor '{motor.name}': magnet_temp is set but "
                f"{', '.join(missing)} is not — the demag screen evaluates "
                f"B_rem and B_knee at the same temperature (assuming the "
                f"20 °C knee at operating temperature is optimistic for "
                f"NdFeB)"
            )
        b_rem = psi_f_at_magnet_temp(b_rem, motor.magnet_temp, alpha_Br)
        b_knee = b_knee + alpha_B_knee * (motor.magnet_temp - 20.0)

    return DemagScreenParams(
        fem=FemParams(
            geometry=geo,
            n_p=motor.n_p,
            B_rem=b_rem,
            mu_r_pm=motor.mu_r_pm,
            mu_r_fe=motor.mu_r_fe,
            alpha_p=motor.alpha_p,
        ),
        B_knee=b_knee,
        k_w=motor.k_w,
    )


def _resolve_thermal_budget(
    motor: Motor, R_s_op: float, p_fe: float,
) -> tuple[float, str]:
    """S1 continuous loss budget (W) and its source.

    R_s_op is the operating-temperature stator resistance (already derated);
    p_fe the lumped iron loss at W_REF (0.0 when [iron] is unset).

    Thermal-resistance path (preferred when r_th + both temps are set):
        P_S1 = (winding_temp_limit - ambient_temp) / r_th.
    This is a total-dissipation budget — p_fe joins the consumption side
    only (run_thermal_duty), never the budget.
    Otherwise the rated-current fallback: I_rated encodes the
    manufacturer's S1 thermal limit as a current, and the nameplate S1
    operating point dissipates copper AND iron, so
        P_S1 = 1.5 * R_s_op * I_rated² + p_fe  (3-phase, peak convention).
    A duty at exactly the nameplate point then reads over_budget_ratio = 1
    whether or not [iron] is set; iron-aware consumption against a
    copper-only budget would read the motor over budget at its own rating.
    Using the same R_s_op as the duty consumption keeps over_budget_ratio
    invariant to derating on this path (only R_s cancels).
    Raises ValueError when neither path is satisfiable.
    """
    if (
        motor.r_th is not None
        and motor.winding_temp_limit is not None
        and motor.ambient_temp is not None
    ):
        # winding_temp_limit > ambient_temp guaranteed by Motor.__post_init__
        return (motor.winding_temp_limit - motor.ambient_temp) / motor.r_th, "thermal_resistance"

    if motor.I_rated is not None:
        from phasesweep.thermal_duty import copper_loss
        return copper_loss(R_s_op, motor.I_rated) + p_fe, "rated_current"

    raise ValueError(
        f"Motor '{motor.name}': thermal_duty needs an S1 budget — either "
        f"(r_th, winding_temp_limit, ambient_temp) or (I_rated with R_s > 0)"
    )


_IRON_FIELDS = ("k_h", "k_e", "alpha_fe", "m_core", "B_core")


def _thermal_duty_p_fe(motor: Motor) -> float:
    """Lumped Bertotti iron loss at W_REF (W) for the thermal-duty screen.

    All five [iron] fields set → m_core · p_fe_density(f_e, B_core) with
    f_e = n_p·W_REF/2π — the same number the iron_loss model reports.
    None set → 0.0 (copper-only screen, bit-identical to pre-iron
    behaviour). A partial set is a loud error, mirroring the
    magnet_temp-without-alpha_Br rule: silently dropping iron loss because
    one coefficient is missing would un-tighten the screen without a trace.
    """
    from math import pi

    k_h, k_e, alpha_fe = motor.k_h, motor.k_e, motor.alpha_fe
    m_core, B_core = motor.m_core, motor.B_core
    if k_h is None or k_e is None or alpha_fe is None or m_core is None or B_core is None:
        values = [getattr(motor, f) for f in _IRON_FIELDS]
        if all(v is None for v in values):
            return 0.0
        missing = [f for f, v in zip(_IRON_FIELDS, values) if v is None]
        raise ValueError(
            f"Motor '{motor.name}': [iron] is partially set — thermal_duty "
            f"needs all of {', '.join(_IRON_FIELDS)} to include iron loss "
            f"(missing: {', '.join(missing)}); set them or clear the section"
        )
    from phasesweep.iron_loss import bertotti_loss_density
    if motor.drive.W_REF is None:
        raise ValueError(
            f"Motor '{motor.name}': thermal_duty with [iron] needs "
            f"[drive] W_REF"
        )
    f_e = motor.n_p * motor.drive.W_REF / (2 * pi)
    return m_core * bertotti_loss_density(f_e, B_core, k_h, k_e, alpha_fe)


def _check_thermal_duty_motor(
    motor: Motor,
) -> tuple[float, float, float, str, float]:
    """Resolve psi_f (magnet-temp derated), operating R_s, S1 budget, p_fe.

    Raises on gaps. magnet_temp without alpha_Br is a loud error rather than
    a silent skip — the coefficient is grade-specific (NdFeB ≈ -0.0012/K,
    SmCo ≈ -0.0003/K), so there is no safe default.
    """
    from phasesweep.thermal_duty import psi_f_at_magnet_temp, r_s_at_operating_temp
    psi_f = _resolve_psi_f(motor)
    if motor.magnet_temp is not None and motor.alpha_Br is None:
        raise ValueError(
            f"Motor '{motor.name}': magnet_temp is set but alpha_Br is not — "
            f"the B_rem temperature coefficient is grade-specific "
            f"(per K, e.g. NdFeB N42 -0.0012, SmCo -0.0003); set alpha_Br "
            f"or drop magnet_temp"
        )
    psi_f = psi_f_at_magnet_temp(psi_f, motor.magnet_temp, motor.alpha_Br)
    if motor.R_s is None or motor.R_s <= 0:
        raise ValueError(
            f"Motor '{motor.name}': thermal_duty needs R_s > 0"
        )
    R_s_op = r_s_at_operating_temp(motor.R_s, motor.winding_temp_limit)
    p_fe = _thermal_duty_p_fe(motor)
    budget, source = _resolve_thermal_budget(motor, R_s_op, p_fe)
    return psi_f, R_s_op, budget, source, p_fe


def prepare_thermal_duty(
    motor: Motor, duty_profile: tuple[tuple[float, float], ...],
) -> ThermalDutyParams:
    """Derive, validate, and extract thermal-duty params from a Motor + profile."""
    psi_f, R_s, budget, source, p_fe = _check_thermal_duty_motor(motor)

    L_d, L_q = _salient_pair(motor)

    ambient = None
    rise = None
    if source == "thermal_resistance":
        # Guaranteed set (and limit > ambient) by _resolve_thermal_budget
        assert motor.ambient_temp is not None and motor.winding_temp_limit is not None
        ambient = motor.ambient_temp
        rise = motor.winding_temp_limit - motor.ambient_temp

    return ThermalDutyParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        R_s=R_s,
        p_s1_budget=budget,
        budget_source=source,
        duty_profile=tuple((float(t), float(d)) for t, d in duty_profile),
        L_d=L_d,
        L_q=L_q,
        p_fe=p_fe,
        thermal_time_constant=motor.thermal_time_constant,
        ambient_temp=ambient,
        temp_rise_limit=rise,
    )
