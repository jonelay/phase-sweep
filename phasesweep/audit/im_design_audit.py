"""Induction machine design audit — Boldea-derived diagnostic bands.

Soft diagnostics that compare design parameters and computed performance
against typical ranges from Boldea & Nasar, *The Induction Machines Design
Handbook*, 2nd ed., chapters 5–7.  Results are advisory: a verdict of
"warn" flags an unusual value, not a bug.  Bands depend on motor class
(power rating, pole count, NEMA design letter) and come from the textbook
tables, not from the motor's own specification.

Usage:
    from phasesweep.audit.im_design_audit import im_design_audit
    results = im_design_audit(motor, torque_slip=sweep_output)
    for r in results:
        print(f"{r.verdict:4s}  {r.name:20s}  {r.value:.4g}  {r.motor_class}")
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor

__all__ = [
    "AuditResult",
    "im_design_audit",
]


@dataclass(frozen=True)
class AuditResult:
    """One audit check result."""

    name: str
    value: float
    band: tuple[float, float] | None
    motor_class: str
    verdict: str  # "pass", "info", "warn"


def _verdict(value: float, band: tuple[float, float] | None) -> str:
    if band is None:
        return "info"
    lo, hi = band
    return "pass" if lo <= value <= hi else "warn"


def _rated_power_kw(
    motor: Motor, torque_slip: dict[str, Any] | None,
) -> float | None:
    """Estimate rated mechanical power in kW from torque-slip output."""
    if torque_slip is None:
        return None
    tau = torque_slip.get("tau_rated")
    s = torque_slip.get("slip_rated")
    ind = motor.induction
    if tau is None or s is None or ind is None or ind.f_rated is None:
        return None
    omega_sync = 2 * pi * ind.f_rated / motor.n_p
    return tau * omega_sync * (1 - s) / 1000.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_rated_slip(
    motor: Motor, p_kw: float | None,
) -> AuditResult | None:
    """S_n rated slip (Boldea Ch 7 sect 7.6)."""
    ind = motor.induction
    if ind is None or ind.s_rated is None:
        return None
    s = ind.s_rated
    s_pct = s * 100
    if p_kw is None:
        band = None
        cls = "power unknown"
    elif p_kw < 1:
        band = (3.0, 8.0)
        cls = f"P={p_kw:.2f} kW (<1 kW)"
    elif p_kw < 10:
        band = (1.5, 5.0)
        cls = f"P={p_kw:.1f} kW (1–10 kW)"
    else:
        band = (0.6, 3.0)
        cls = f"P={p_kw:.0f} kW (>10 kW)"
    return AuditResult("S_n_pct", s_pct, band, cls, _verdict(s_pct, band))


def _check_breakdown_ratio(
    motor: Motor, torque_slip: dict[str, Any] | None,
) -> AuditResult | None:
    """T_ek / T_en breakdown-to-rated torque (Boldea Ch 7 eq 7.74)."""
    if torque_slip is None:
        return None
    tau_bk = torque_slip.get("tau_breakdown")
    tau_r = torque_slip.get("tau_rated")
    if tau_bk is None or tau_r is None or tau_r == 0:
        return None
    ratio = tau_bk / tau_r
    ind = motor.induction
    dl = ind.design_letter if ind is not None else None
    if dl == "B":
        band = (1.75, 4.0)
        cls = "NEMA B"
    elif dl == "C":
        band = (1.9, 4.0)
        cls = "NEMA C"
    elif dl == "D":
        band = (2.75, 5.0)
        cls = "NEMA D"
    elif dl == "A":
        band = (1.5, 4.0)
        cls = "NEMA A"
    else:
        band = (1.5, 5.0)
        cls = "design letter unknown"
    return AuditResult("T_ek_over_T_en", ratio, band, cls, _verdict(ratio, band))


def _check_starting_current_ratio(
    motor: Motor, torque_slip: dict[str, Any] | None,
) -> AuditResult | None:
    """I_start / I_n starting-to-rated current (Boldea Ch 7 eq 7.35)."""
    if torque_slip is None:
        return None
    i_lr = torque_slip.get("I_locked_rotor_rms")
    i_r = torque_slip.get("I_1_rated_rms")
    if i_lr is None or i_r is None or i_r == 0:
        return None
    ratio = i_lr / i_r
    ind = motor.induction
    dl = ind.design_letter if ind is not None else None
    if dl == "B":
        band = (5.0, 7.0)
        cls = "NEMA B"
    elif dl == "D":
        band = (3.0, 5.0)
        cls = "NEMA D"
    elif dl == "C":
        band = (5.0, 7.0)
        cls = "NEMA C"
    elif dl == "A":
        band = (5.0, 10.0)
        cls = "NEMA A"
    else:
        band = (3.0, 10.0)
        cls = "design letter unknown"
    return AuditResult("I_start_over_I_n", ratio, band, cls, _verdict(ratio, band))


def _check_no_load_current_ratio(
    motor: Motor, p_kw: float | None,
) -> AuditResult | None:
    """I_0 / I_n no-load to rated current (Boldea Ch 5 sect 5.4)."""
    ind = motor.induction
    if ind is None or ind.L_m is None or ind.V_LL_rated is None:
        return None
    if ind.f_rated is None or motor.R_s is None:
        return None
    if motor.I_rated is None:
        return None
    v_phase = ind.V_LL_rated / sqrt(3)
    omega = 2 * pi * ind.f_rated
    x_m = omega * ind.L_m
    i_0 = v_phase / sqrt(motor.R_s**2 + x_m**2)
    i_rated_rms = motor.I_rated / sqrt(2)
    if i_rated_rms == 0:
        return None
    ratio = i_0 / i_rated_rms
    if p_kw is None:
        band = (0.2, 0.9)
        cls = "power unknown"
    elif p_kw < 1:
        band = (0.5, 0.85)
        cls = f"P={p_kw:.2f} kW (<1 kW)"
    elif p_kw < 10:
        band = (0.35, 0.7)
        cls = f"P={p_kw:.1f} kW (1–10 kW)"
    elif p_kw < 50:
        band = (0.25, 0.5)
        cls = f"P={p_kw:.0f} kW (10–50 kW)"
    else:
        band = (0.2, 0.4)
        cls = f"P={p_kw:.0f} kW (>50 kW)"
    return AuditResult("I_0_over_I_n", ratio, band, cls, _verdict(ratio, band))


def _check_airgap_flux(motor: Motor) -> AuditResult | None:
    """B_g airgap fundamental flux density, peak (Boldea Ch 5 sect 5.4).

    From rated voltage and the magnetizing branch: E = 4.44 f N_eff k_w Φ
    with pole flux Φ = (2/π) B_g_peak τ_p L_stk, τ_p = π r_s / n_p, so
    B_g_peak ≈ (π/2) V_phase / (4.44 f N_eff k_w τ_p L_stk). Without the
    π/2 the value is the pole-average flux density.
    """
    ind = motor.induction
    if ind is None or ind.V_LL_rated is None or ind.f_rated is None:
        return None
    geo = motor.geometry
    if geo is None or motor.k_w is None or motor.L_stk is None:
        return None
    if not hasattr(geo, "r_stator"):
        return None
    from phasesweep.solver_params import n_eff
    try:
        n = n_eff(motor)
    except (ValueError, AttributeError):
        return None
    v_phase = ind.V_LL_rated / sqrt(3)
    tau_p = pi * geo.r_stator / motor.n_p  # type: ignore[union-attr]
    b_avg = v_phase / (4.44 * ind.f_rated * n * motor.k_w * tau_p * motor.L_stk)
    b_g = b_avg * pi / 2
    if ind.f_rated <= 60:
        band = (0.4, 0.9)
        cls = f"f={ind.f_rated} Hz (50/60 Hz class)"
    else:
        band = (0.2, 0.7)
        cls = f"f={ind.f_rated} Hz (high-freq)"
    return AuditResult("B_g_T", b_g, band, cls, _verdict(b_g, band))


def _check_saturation_factor(motor: Motor) -> AuditResult | None:
    """K_s saturation factor (Boldea Ch 5 sect 5.4) — informational."""
    ind = motor.induction
    if ind is None or ind.L_m_vs_i is None or ind.L_m is None:
        return None
    if len(ind.L_m_vs_i) < 2:
        return None
    l_unsat = ind.L_m_vs_i[0][1]
    if l_unsat == 0:
        return None
    k_s = l_unsat / ind.L_m
    return AuditResult("K_s", k_s, None, "informational", "info")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def im_design_audit(
    motor: Motor,
    torque_slip: dict[str, Any] | None = None,
) -> list[AuditResult]:
    """Run the induction machine design audit.

    Parameters
    ----------
    motor : Motor
        Must have an ``induction`` section.
    torque_slip : dict, optional
        Output of ``run_im_torque_slip``.  When provided, enables the
        performance checks (breakdown ratio, starting current, rated slip
        band by power).

    Returns
    -------
    list[AuditResult]
        One entry per check that had enough data to run. Checks with
        missing inputs are silently skipped.
    """
    if motor.induction is None:
        raise ValueError("im_design_audit requires an [induction] section")
    p_kw = _rated_power_kw(motor, torque_slip)
    results: list[AuditResult] = []
    for check in (
        _check_rated_slip(motor, p_kw),
        _check_breakdown_ratio(motor, torque_slip),
        _check_starting_current_ratio(motor, torque_slip),
        _check_no_load_current_ratio(motor, p_kw),
        _check_airgap_flux(motor),
        _check_saturation_factor(motor),
    ):
        if check is not None:
            results.append(check)
    return results
