"""Thermal-duty model: copper-loss budget over a torque-time profile.

Maps each torque segment to the minimum stator current on the MTPA locus,
converts to 3-phase copper loss (1.5·R_s·I_peak²), and compares the
time-averaged loss against an S1 continuous budget. Pure circuit model —
no geometry required.

When the Motor [iron] fields are set, the lumped Bertotti iron loss at
W_REF joins the consumption on both budget paths (iron loss runs whenever
the machine spins, independent of the torque duty), and joins the
rated_current budget (the nameplate S1 point dissipates copper AND iron
— see _resolve_thermal_budget). The duty profile carries no speed, so
p_fe is a single W_REF-rate figure across all segments.

Copper loss uses R_s at the winding operating temperature (the S1 steady
state sits at winding_temp_limit), not the cold datasheet value — see
r_s_at_operating_temp. When Motor.magnet_temp and alpha_Br are set, psi_f
is likewise derated to the magnet operating temperature — see
psi_f_at_magnet_temp; unlike R_s this does not cancel on the rated_current
budget path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from phasesweep.models.rated_torque import magnet_torque_constant, mtpa_gamma, mtpa_torque

if TYPE_CHECKING:
    from phasesweep.sweep_types import RunConfig

# Copper resistivity temperature coefficient (per K) and the reference
# temperature of a datasheet R_s. 20 °C matches the back-EMF derating
# reference; a 25 °C datasheet differs by ~2%, negligible
# beside the ~50% rise to a class-F winding limit.
from phasesweep.defaults import COPPER_TEMP_COEFF

R_S_REF_TEMP_C: float = 20.0

# Reference temperature of a datasheet B_rem / psi_f (magnet derating).
# Same 20 °C convention as R_S_REF_TEMP_C; unlike copper's
# universal alpha, the magnet coefficient is grade-specific and lives on
# Motor.alpha_Br.
MAGNET_REF_TEMP_C: float = 20.0


def psi_f_at_magnet_temp(
    psi_f_ref: float, magnet_temp: float | None, alpha_Br: float | None,
) -> float:
    """psi_f (Wb, peak) derated to the magnet operating temperature.

    psi_f(T) = psi_f_ref · [1 + alpha_Br·(T_magnet − T_ref)] — B_rem and
    psi_f share the linear coefficient since psi_f is linear in B_rem.
    With no magnet_temp there is nothing to derate to (cold reference
    returned); alpha_Br alone is inert material data. The magnet runs
    cooler than the winding hot-spot, so magnet_temp is its own field,
    not winding_temp_limit. The (-0.01, 0) range check on alpha_Br lives
    in Motor.__post_init__; combined with plausible temperatures the
    factor stays well above zero.
    """
    if magnet_temp is None or alpha_Br is None:
        return psi_f_ref
    return psi_f_ref * (1.0 + alpha_Br * (magnet_temp - MAGNET_REF_TEMP_C))


def r_s_at_operating_temp(
    R_s_ref: float, winding_temp_limit: float | None,
) -> float:
    """R_s (Ω) derated to the winding temperature limit (worst-case S1).

    R_s(T) = R_s_ref · [1 + α_cu·(T_limit − T_ref)]. At the S1 thermal
    steady state the winding sits at winding_temp_limit, so this is the
    conservative operating resistance with no fixed-point solve. With no
    winding_temp_limit there is nothing to derate to, so R_s_ref is returned
    unchanged (cold). winding_temp_limit is an insulation hot-spot, marginally
    above the mean winding temperature that sets bulk R_s, so this slightly
    over-estimates R_s (conservative). Where derating affects the verdict vs.
    merely rescaling watts is documented at _resolve_thermal_budget.
    """
    if winding_temp_limit is None:
        return R_s_ref
    return R_s_ref * (1.0 + COPPER_TEMP_COEFF * (winding_temp_limit - R_S_REF_TEMP_C))


def copper_loss(R_s: float, i_peak: float) -> float:
    """3-phase copper loss (W) at peak phase current: 1.5·R_s·I_peak².

    Single source of the peak-current convention shared by the duty
    consumption and the I_rated S1 budget — they must agree or
    over_budget_ratio is meaningless.
    """
    return 1.5 * R_s * i_peak**2


def current_for_torque(
    n_p: int, psi_f: float, L_d: float | None, L_q: float | None, tau: float,
) -> float:
    """Minimum stator current (A, peak) to produce |tau| on the MTPA locus.

    Non-salient (or missing L_d/L_q): |tau| / k_T.
    Salient (both saliency signs): bisect the MTPA torque-vs-current curve
    (monotone increasing); the non-salient current is an upper bound since
    MTPA reluctance torque only adds to the magnet torque.
    """
    tau = abs(tau)
    k_T = magnet_torque_constant(n_p, psi_f)
    if tau == 0.0:
        return 0.0
    i_nonsalient = tau / k_T
    if L_d is None or L_q is None or L_q == L_d:
        return i_nonsalient

    lo, hi = 0.0, i_nonsalient
    tol = 1e-12 * i_nonsalient
    for _ in range(60):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        g = mtpa_gamma(psi_f, L_d, L_q, mid)
        if mtpa_torque(n_p, psi_f, L_d, L_q, mid, g) < tau:
            lo = mid
        else:
            hi = mid
    return hi


def transient_peak(
    u: list[float], dts: list[float], tau: float,
) -> float:
    """Peak of the steady-periodic first-order thermal response.

    x is the winding temperature rise as a fraction of the S1 rise; each
    segment k holds a constant normalized loss u_k (segment consumption /
    S1 budget), so within it dx/dt = (u_k − x)/τ and
    x_end = u_k + (x_start − u_k)·exp(−dt_k/τ). The cycle map x0 → x(T)
    is affine with slope exp(−T/τ) < 1, giving the periodic start point
    in closed form; the response is monotone within a segment, so the
    cycle peak is the max over segment endpoints. No integration error.

    τ much longer than the cycle recovers the cycle-average criterion
    (x → mean u = over_budget_ratio); segments much longer than τ recover
    the per-segment peak criterion (x → max u).
    """
    from math import exp

    a_tot = 1.0
    b_tot = 0.0
    factors = []
    for u_k, dt in zip(u, dts):
        a_k = exp(-dt / tau)
        factors.append(a_k)
        b_tot = a_k * b_tot + u_k * (1.0 - a_k)
        a_tot *= a_k
    if a_tot == 1.0:
        # tau >> cycle time beyond float resolution: x is flat at the
        # time-averaged loading
        total = sum(dts)
        x = sum(u_k * dt for u_k, dt in zip(u, dts)) / total
        return x
    x = b_tot / (1.0 - a_tot)
    x_peak = x
    for u_k, a_k in zip(u, factors):
        x = u_k + (x - u_k) * a_k
        x_peak = max(x_peak, x)
    return x_peak


def run_thermal_duty(config: RunConfig) -> dict[str, Any]:
    """Copper-loss duty analysis over a torque-time profile.

    The reported losses are DC copper loss at the worst-case winding
    temperature (cold R_s when no winding_temp_limit), plus lumped iron
    loss at W_REF when [iron] is set — still a screen, not an equilibrium
    prediction. See the budget-path and loss-mechanism caveats (AC copper,
    magnet eddy and friction/windage remain non-goals;
    rated_current is the self-consistent path).

    within_s1 / over_budget_ratio are cycle-average criteria: valid only when
    the duty-cycle period is short relative to the winding thermal time
    constant. Segments comparable to or longer than it can overheat the
    winding while the average still passes. With Motor.thermal_time_constant
    set, the first-order transient march computes this directly:
    transient_peak_ratio / within_s1_transient (and winding_temp_peak on
    the thermal_resistance path) replace the caveat with a number; without
    it, check p_cu_peak / sustainable_duty_fraction.
    """
    from phasesweep.solver_params import prepare_thermal_duty

    if config.duty_profile is None:
        raise ValueError("thermal_duty needs config.duty_profile")

    p = prepare_thermal_duty(config.motor, config.duty_profile)

    seg_losses: list[float] = []
    energy = 0.0
    total_time = 0.0
    for tau, dt in p.duty_profile:
        i_s = current_for_torque(p.n_p, p.psi_f, p.L_d, p.L_q, tau)
        p_cu = copper_loss(p.R_s, i_s)
        seg_losses.append(p_cu)
        energy += p_cu * dt
        total_time += dt

    p_cu_avg = energy / total_time
    p_cu_peak = max(seg_losses)
    p_total_avg = p_cu_avg + p.p_fe
    over_budget_ratio = p_total_avg / p.p_s1_budget
    # Iron loss is set by speed, not torque, so it spends budget at every
    # duty fraction — only the remainder is scalable by the copper duty.
    if p_cu_peak > 0:
        sustainable_duty_fraction = min(
            1.0, max(0.0, (p.p_s1_budget - p.p_fe) / p_cu_peak)
        )
    else:
        sustainable_duty_fraction = 1.0 if p.p_fe <= p.p_s1_budget else 0.0

    result: dict[str, Any] = {
        "p_cu_avg": p_cu_avg,
        "p_cu_peak": p_cu_peak,
        "p_fe": p.p_fe,
        "p_total_avg": p_total_avg,
        "p_s1_budget": p.p_s1_budget,
        "budget_source": p.budget_source,
        "over_budget_ratio": over_budget_ratio,
        "within_s1": over_budget_ratio <= 1.0,
        "sustainable_duty_fraction": sustainable_duty_fraction,
        "total_cycle_time": total_time,
    }

    if p.thermal_time_constant is not None:
        u = [(loss + p.p_fe) / p.p_s1_budget for loss in seg_losses]
        dts = [dt for _, dt in p.duty_profile]
        x_peak = transient_peak(u, dts, p.thermal_time_constant)
        result["transient_peak_ratio"] = x_peak
        result["within_s1_transient"] = x_peak <= 1.0
        if p.temp_rise_limit is not None and p.ambient_temp is not None:
            result["winding_temp_peak"] = (
                p.ambient_temp + x_peak * p.temp_rise_limit
            )

    return result
