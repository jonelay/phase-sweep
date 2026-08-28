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

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Any

from phasesweep.models.rated_torque import magnet_torque_constant, mtpa_gamma, mtpa_torque
from phasesweep.solver_params import _reject_induction, _resolve_psi_f, _salient_pair

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

# Copper resistivity temperature coefficient (per K) and the reference
# temperature of a datasheet R_s. 20 °C matches the back-EMF derating
# reference; a 25 °C datasheet differs by ~2%, negligible
# beside the ~50% rise to a class-F winding limit.
from phasesweep.defaults import COPPER_TEMP_COEFF

R_S_REF_TEMP_C: float = 20.0

# Reference temperature of a datasheet B_rem / psi_f (magnet derating). Same 20 °C convention as R_S_REF_TEMP_C; unlike copper's
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
    psi_f = 0 (SynRM): reluctance-only, MTPA sits at 45° from the q-axis,
    so tau = 0.75·n_p·|L_d − L_q|·I² inverts in closed form.
    """
    tau = abs(tau)
    if tau == 0.0:
        return 0.0
    k_T = magnet_torque_constant(n_p, psi_f)
    if L_d is None or L_q is None or L_q == L_d:
        if psi_f == 0.0:
            raise ValueError(
                "psi_f = 0 with no saliency (L_d == L_q or unset): "
                "machine cannot produce torque"
            )
        return tau / k_T
    if psi_f == 0.0:
        return sqrt(tau / (0.75 * n_p * abs(L_q - L_d)))
    i_nonsalient = tau / k_T

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
    prediction. Note the budget-path and loss-mechanism
    caveats (AC copper, magnet eddy and friction/windage remain non-goals;
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
        if self.psi_f < 0:
            raise ValueError(f"psi_f must be >= 0, got {self.psi_f}")
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
        return copper_loss(R_s_op, motor.I_rated) + p_fe, "rated_current"

    raise ValueError(
        f"Motor '{motor.name}': thermal_duty needs an S1 budget — either "
        f"(r_th, winding_temp_limit, ambient_temp) or (I_rated with R_s > 0)"
    )


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

    from phasesweep.models.iron_loss import _IRON_FIELDS, bertotti_loss_density

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
    _reject_induction(motor, "thermal_duty")
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
