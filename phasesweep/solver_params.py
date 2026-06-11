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
    return (
        2 * n_eff(motor) * motor.k_w * motor.geometry.r_stator
        * motor.L_stk / motor.n_p
    )


def _derive_b_rem(motor: Motor) -> float | None:
    """Derive B_rem from psi_f via Zhu & Howe transfer ratio.

    Carter-consistent: inverts the same model psi_f_carter evaluates —
    Carter-adjusted radii for the field, original bore for the winding —
    so psi_f → B_rem → flux_linkage_peak round-trips exactly.
    Returns derived B_rem, or None if derivation inputs are missing.
    N_eff = N (turns per coil) × coils_series = total series turns per phase.
    """
    from phasesweep.fem_field import _derive_B_rem, carter_adjusted_radii

    if motor.psi_f is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
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


def _derive_psi_f(motor: Motor) -> float | None:
    """Derive psi_f from B_rem via Zhu & Howe analytical model.

    Returns derived psi_f, or None if derivation inputs are missing.
    N_eff = N (turns per coil) × coils_series = total series turns per phase.
    """
    import numpy as np

    from phasesweep.fem_field import zhu_howe_Br

    if motor.B_rem is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    geo = motor.geometry
    B_peak = zhu_howe_Br(
        np.array([0.0]), motor.n_p, motor.B_rem, r_eval=geo.r_stator,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=motor.mu_r_pm, alpha_p=motor.alpha_p,
    )[0]
    return B_peak * winding_transfer(motor)


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

    from phasesweep.fem_field import zhu_howe_Br

    if motor.N is None or motor.k_w is None or motor.L_stk is None:
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

    derived = _derive_b_rem(motor)
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
    derived = _derive_b_rem(motor)
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

def prepare_analytical(motor: Motor) -> AnalyticalParams:
    """Derive, validate, and extract analytical solver params from a Motor."""
    b_rem = _resolve_b_rem(motor)
    return AnalyticalParams(
        geometry=motor.geometry,
        n_p=motor.n_p,
        B_rem=b_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
    )


def prepare_fem(motor: Motor) -> FemParams:
    """Derive, validate, and extract FEM solver params from a Motor."""
    b_rem = _resolve_b_rem(motor)
    return FemParams(
        geometry=motor.geometry,
        n_p=motor.n_p,
        B_rem=b_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
    )


def prepare_drive_sim(motor: Motor) -> DriveSimParams:
    """Derive, validate, and extract drive sim params from a Motor."""
    psi_f = _resolve_psi_f(motor)

    missing = []
    for field in ("R_s", "L_d", "L_q", "J"):
        if getattr(motor, field) is None:
            missing.append(field)
    if missing:
        raise ValueError(
            f"Motor '{motor.name}': drive_sim needs {', '.join(missing)}"
        )

    return DriveSimParams(
        geometry=motor.geometry,
        n_p=motor.n_p,
        R_s=motor.R_s,  # type: ignore[arg-type]
        L_d=motor.L_d,  # type: ignore[arg-type]
        L_q=motor.L_q,  # type: ignore[arg-type]
        psi_f=psi_f,
        J=motor.J,  # type: ignore[arg-type]
        drive=motor.drive,
    )


def prepare_rated_torque(motor: Motor) -> RatedTorqueParams:
    """Derive, validate, and extract rated torque params from a Motor."""
    psi_f = _resolve_psi_f(motor)

    if motor.I_rated is None:
        raise ValueError(
            f"Motor '{motor.name}': rated_torque needs I_rated"
        )

    L_d = motor.L_d if (motor.L_d is not None and motor.L_q is not None) else None
    L_q = motor.L_q if (motor.L_d is not None and motor.L_q is not None) else None

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
    I_stall = drive.I_LIMIT if drive.I_LIMIT is not None else drive.MAX_I_S
    I_stall_em = drive.U_DC / (sqrt(3) * motor.R_s) if motor.R_s > 0 else None

    L_d = motor.L_d if (motor.L_d is not None and motor.L_q is not None) else None
    L_q = motor.L_q if (motor.L_d is not None and motor.L_q is not None) else None

    return StallTorqueParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        I_stall=I_stall,
        I_stall_em=I_stall_em,
        L_d=L_d,
        L_q=L_q,
    )
