"""Tests for the thermal-duty model (copper-loss S1 budget).

The model maps each torque segment to the minimum MTPA current, converts
to 3-phase copper loss (1.5·R_s·I_peak²), and compares the time-averaged
loss against an S1 continuous budget. The budget comes from a winding
thermal resistance when available, else from I_rated (which already
encodes the manufacturer's S1 thermal limit as a current).
"""

import pytest

from phasesweep.machines.motor import DriveParams, Motor
from phasesweep.models.thermal_duty import (
    COPPER_TEMP_COEFF,
    current_for_torque,
    psi_f_at_magnet_temp,
    r_s_at_operating_temp,
    run_thermal_duty,
)
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solver_params import prepare_thermal_duty
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import REPO_ROOT

PROFILE = ((5.0, 10.0), (1.0, 50.0))  # 5 Nm for 10 s, 1 Nm for 50 s


def _motor(**kw):
    base = dict(name="ipm", geometry=None, n_p=2, R_s=0.5,
                L_d=5e-3, L_q=1.05e-2, psi_f=0.1, I_rated=10.0,
                drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.159265358979))
    base.update(kw)
    return Motor(**base)


def _config(motor, profile=PROFILE):
    return RunConfig(motor=motor, model="thermal_duty", duty_profile=profile)


# --- current_for_torque ---

def test_current_nonsalient_handcalc():
    # k_T = 1.5 * n_p * psi_f = 1.5*2*0.1 = 0.3 Nm/A
    i = current_for_torque(2, 0.1, None, None, 3.0)
    assert i == pytest.approx(3.0 / 0.3, rel=1e-12)


def test_current_zero_torque():
    assert current_for_torque(2, 0.1, 5e-3, 1e-2, 0.0) == 0.0


def test_current_salient_less_than_nonsalient():
    # Reluctance torque (L_q > L_d) means less current for the same torque.
    i_sal = current_for_torque(2, 0.1, 5e-3, 1.05e-2, 5.0)
    i_non = 5.0 / (1.5 * 2 * 0.1)
    assert i_sal < i_non


def test_current_for_torque_inverts_mtpa_torque():
    from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque
    i = current_for_torque(2, 0.1, 5e-3, 1.05e-2, 4.0)
    g = mtpa_gamma(0.1, 5e-3, 1.05e-2, i)
    assert mtpa_torque(2, 0.1, 5e-3, 1.05e-2, i, g) == pytest.approx(4.0, rel=1e-6)


def test_current_reverse_salient_roundtrip():
    # Reverse saliency (L_d > L_q): MTPA locus (gamma < 0, magnetizing i_d)
    # still needs less current than the non-salient mapping.
    from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque
    i = current_for_torque(2, 0.1, 1.05e-2, 5e-3, 4.0)
    assert i < 4.0 / (1.5 * 2 * 0.1)
    g = mtpa_gamma(0.1, 1.05e-2, 5e-3, i)
    assert g < 0
    assert mtpa_torque(2, 0.1, 1.05e-2, 5e-3, i, g) == pytest.approx(4.0, rel=1e-6)


# --- budget resolution ---

def test_budget_from_rated_current():
    p = prepare_thermal_duty(_motor(), PROFILE)
    assert p.budget_source == "rated_current"
    assert p.p_s1_budget == pytest.approx(1.5 * 0.5 * 10.0**2, rel=1e-12)


def test_budget_from_thermal_resistance():
    m = _motor(winding_temp_limit=155.0, ambient_temp=40.0, r_th=0.5)
    p = prepare_thermal_duty(m, PROFILE)
    assert p.budget_source == "thermal_resistance"
    assert p.p_s1_budget == pytest.approx((155.0 - 40.0) / 0.5, rel=1e-12)


def test_thermal_resistance_preferred_over_rated_current():
    # Both available → thermal-resistance path wins.
    m = _motor(I_rated=10.0, winding_temp_limit=155.0, ambient_temp=40.0, r_th=0.5)
    p = prepare_thermal_duty(m, PROFILE)
    assert p.budget_source == "thermal_resistance"


def test_no_budget_raises():
    m = _motor(I_rated=None)  # no thermal fields, no I_rated
    with pytest.raises(ValueError, match="S1 budget"):
        prepare_thermal_duty(m, PROFILE)


# --- loss math ---

def test_loss_and_budget_math():
    r = run_thermal_duty(_config(_motor()))
    # Peak segment is 5 Nm.
    i_peak = current_for_torque(2, 0.1, 5e-3, 1.05e-2, 5.0)
    i_low = current_for_torque(2, 0.1, 5e-3, 1.05e-2, 1.0)
    p_peak = 1.5 * 0.5 * i_peak**2
    p_low = 1.5 * 0.5 * i_low**2
    avg = (p_peak * 10.0 + p_low * 50.0) / 60.0
    assert r["p_cu_peak"] == pytest.approx(p_peak, rel=1e-9)
    assert r["p_cu_avg"] == pytest.approx(avg, rel=1e-9)
    assert r["total_cycle_time"] == 60.0
    assert r["over_budget_ratio"] == pytest.approx(avg / 75.0, rel=1e-9)
    assert r["sustainable_duty_fraction"] == pytest.approx(
        min(1.0, 75.0 / p_peak), rel=1e-9)


def test_within_s1_true():
    r = run_thermal_duty(_config(_motor()))
    assert r["within_s1"] is (r["over_budget_ratio"] <= 1.0)
    assert r["within_s1"] is True


def test_over_budget_when_loss_exceeds_s1():
    # Tiny budget (small I_rated) → over budget.
    r = run_thermal_duty(_config(_motor(I_rated=1.0)))
    assert r["over_budget_ratio"] > 1.0
    assert r["within_s1"] is False
    assert r["sustainable_duty_fraction"] < 1.0


def test_light_load_fully_sustainable():
    # All segments below the continuous budget → duty fraction clamps to 1.
    r = run_thermal_duty(_config(_motor(I_rated=100.0)))
    assert r["sustainable_duty_fraction"] == 1.0
    assert r["within_s1"] is True


# --- operating-temperature R_s derating ---

def test_r_s_at_operating_temp_handcalc():
    # 0.5 Ω at the 20 °C reference, derated to a 155 °C class-F limit.
    assert r_s_at_operating_temp(0.5, 155.0) == pytest.approx(
        0.5 * (1 + 0.00393 * (155.0 - 20.0)), rel=1e-12)


def test_r_s_no_thermal_limit_stays_cold():
    assert r_s_at_operating_temp(0.5, None) == 0.5


def test_derating_scales_thermal_resistance_verdict():
    # On the thermal_resistance path the budget is R_s-independent, so the
    # over-budget ratio rises by exactly the derating factor vs cold copper —
    # the ~50% optimism the derating corrects.
    m = _motor(winding_temp_limit=155.0, ambient_temp=40.0, r_th=0.5)
    r = run_thermal_duty(_config(m))
    factor = 1 + 0.00393 * (155.0 - 20.0)
    i_peak = current_for_torque(2, 0.1, 5e-3, 1.05e-2, 5.0)
    i_low = current_for_torque(2, 0.1, 5e-3, 1.05e-2, 1.0)
    cold_avg = (1.5 * 0.5 * i_peak**2 * 10.0 + 1.5 * 0.5 * i_low**2 * 50.0) / 60.0
    budget = (155.0 - 40.0) / 0.5
    assert r["over_budget_ratio"] == pytest.approx(
        factor * cold_avg / budget, rel=1e-9)


def test_rated_current_ratio_invariant_to_derating():
    # winding_temp_limit set but no r_th → rated_current path. R_s_op
    # scales both budget and consumption equally, so the verdict matches
    # the cold no-thermal-info motor.
    hot = run_thermal_duty(_config(
        _motor(winding_temp_limit=155.0, ambient_temp=40.0)))
    cold = run_thermal_duty(_config(_motor()))
    assert hot["budget_source"] == "rated_current"
    assert hot["over_budget_ratio"] == pytest.approx(
        cold["over_budget_ratio"], rel=1e-12)
    assert hot["within_s1"] is cold["within_s1"]
    # absolute watts do rise with the hotter copper
    assert hot["p_cu_avg"] > cold["p_cu_avg"]


# --- magnet-temperature psi_f derating ---

def test_psi_f_at_magnet_temp_handcalc():
    # 0.1 Wb at the 20 °C reference, NdFeB N42 coefficient, 100 °C magnet.
    assert psi_f_at_magnet_temp(0.1, 100.0, -0.0012) == pytest.approx(
        0.1 * (1 - 0.0012 * 80.0), rel=1e-12)


def test_psi_f_no_magnet_temp_stays_cold():
    assert psi_f_at_magnet_temp(0.1, None, -0.0012) == 0.1
    assert psi_f_at_magnet_temp(0.1, None, None) == 0.1


def test_alpha_br_alone_is_inert():
    # Material data without an operating temperature derates nothing.
    r = run_thermal_duty(_config(_motor(alpha_Br=-0.0012)))
    assert r["p_cu_avg"] == run_thermal_duty(_config(_motor()))["p_cu_avg"]


def test_magnet_temp_without_alpha_br_raises():
    # Loud error, not a silent skip — the coefficient is grade-specific.
    with pytest.raises(ValueError, match="alpha_Br"):
        prepare_thermal_duty(_motor(magnet_temp=100.0), PROFILE)


def test_alpha_br_percent_trap_rejected():
    # A datasheet "-0.12 %/°C" entered verbatim is caught by Motor validation.
    with pytest.raises(ValueError, match="divide by 100"):
        _motor(alpha_Br=-0.12)
    with pytest.raises(ValueError, match="alpha_Br"):
        _motor(alpha_Br=0.0012)


def test_magnet_derating_raises_ratio_on_rated_current_path():
    # Unlike R_s, a hot magnet does NOT cancel on the rated_current path:
    # the budget is fixed by I_rated while the MTPA current for each torque
    # rises as 1/psi_f-ish, so the verdict gets strictly less optimistic.
    hot = run_thermal_duty(_config(_motor(alpha_Br=-0.0012, magnet_temp=100.0)))
    cold = run_thermal_duty(_config(_motor()))
    assert hot["budget_source"] == "rated_current"
    assert hot["over_budget_ratio"] > cold["over_budget_ratio"]
    assert hot["p_s1_budget"] == pytest.approx(cold["p_s1_budget"], rel=1e-12)


def test_magnet_derating_nonsalient_exact_factor():
    # Non-salient: i = tau/(1.5·n_p·psi_f), so losses scale as 1/factor².
    factor = 1 - 0.0012 * 80.0
    hot = run_thermal_duty(_config(
        _motor(L_d=None, L_q=None, alpha_Br=-0.0012, magnet_temp=100.0)))
    cold = run_thermal_duty(_config(_motor(L_d=None, L_q=None)))
    assert hot["p_cu_avg"] == pytest.approx(
        cold["p_cu_avg"] / factor**2, rel=1e-9)


def test_magnet_fields_change_config_id():
    base = _motor()
    assert _motor(alpha_Br=-0.0012).config_id != base.config_id
    assert _motor(alpha_Br=-0.0012, magnet_temp=100.0).config_id != \
        _motor(alpha_Br=-0.0012).config_id


def test_magnet_fields_roundtrip_dict():
    m = _motor(alpha_Br=-0.0003, magnet_temp=80.0)
    m2 = Motor.from_dict(m.to_dict())
    assert m2.alpha_Br == -0.0003
    assert m2.magnet_temp == 80.0


# --- error cases ---

def test_missing_r_s():
    with pytest.raises(ValueError, match="R_s"):
        prepare_thermal_duty(_motor(R_s=None), PROFILE)


def test_zero_r_s():
    with pytest.raises(ValueError, match="R_s"):
        prepare_thermal_duty(_motor(R_s=0.0), PROFILE)


def test_missing_psi_f():
    with pytest.raises(ValueError, match="psi_f"):
        prepare_thermal_duty(_motor(psi_f=None), PROFILE)


def test_empty_profile():
    with pytest.raises(ValueError, match="at least one segment"):
        prepare_thermal_duty(_motor(), ())


def test_bad_segment_duration():
    with pytest.raises(ValueError, match="duration"):
        prepare_thermal_duty(_motor(), ((5.0, 0.0),))


def test_run_without_duty_profile():
    cfg = RunConfig(motor=_motor(), model="thermal_duty")
    with pytest.raises(ValueError, match="duty_profile"):
        run_thermal_duty(cfg)


# --- registry + hashing ---

def test_registry_entry():
    info = MODEL_REGISTRY["thermal_duty"]
    assert info.cost == "fast"
    assert info.source == "computed"
    assert {"p_cu_avg", "over_budget_ratio", "sustainable_duty_fraction",
            "within_s1", "p_s1_budget"}.issubset(info.produces)
    assert "R_s" in info.needs
    info.validate(_motor())  # motor-only readiness check passes
    r = info.fn(_config(_motor()))
    assert "p_cu_avg" in r


def test_validate_rejects_no_budget():
    info = MODEL_REGISTRY["thermal_duty"]
    with pytest.raises(ValueError):
        info.validate(_motor(I_rated=None))


def test_run_id_depends_on_profile():
    a = compute_run_id(_config(_motor(), profile=((5.0, 10.0), (1.0, 50.0))))
    b = compute_run_id(_config(_motor(), profile=((5.0, 20.0), (1.0, 50.0))))
    assert a != b


def test_run_id_stable_for_same_profile():
    a = compute_run_id(_config(_motor()))
    b = compute_run_id(_config(_motor()))
    assert a == b


# --- TOML integration (circuit-only motor with [thermal]) ---

def test_toml_circuit_only_with_thermal(tmp_path):
    from phasesweep.machines.configs import load_motor
    toml = tmp_path / "ipm.toml"
    toml.write_text(
        "[motor]\nname = 'datasheet-ipm'\n\n"
        "[circuit]\nn_p = 2\nR_s = 0.5\nL_d = 0.005\nL_q = 0.0105\n"
        "psi_f = 0.1\nI_rated = 10.0\n\n"
        "[thermal]\nwinding_temp_limit = 155.0\nambient_temp = 40.0\n"
        "r_th = 0.5\ninsulation_class = 'F'\n"
    )
    m = load_motor(toml)
    assert m.geometry is None
    assert m.r_th == 0.5
    assert m.insulation_class == "F"
    r = MODEL_REGISTRY["thermal_duty"].fn(_config(m))
    assert r["budget_source"] == "thermal_resistance"
    assert r["p_s1_budget"] == pytest.approx(230.0, rel=1e-12)


def test_toml_magnet_derating_fields(tmp_path):
    # alpha_Br rides in [materials] (grade property, like B_rem);
    # magnet_temp in [thermal] (operating condition).
    from phasesweep.machines.configs import load_motor
    toml = tmp_path / "ipm.toml"
    toml.write_text(
        "[motor]\nname = 'datasheet-ipm'\n\n"
        "[circuit]\nn_p = 2\nR_s = 0.5\nL_d = 0.005\nL_q = 0.0105\n"
        "psi_f = 0.1\nI_rated = 10.0\n\n"
        "[materials]\nalpha_Br = -0.0012\n\n"
        "[thermal]\nmagnet_temp = 100.0\n"
    )
    m = load_motor(toml)
    assert m.alpha_Br == -0.0012
    assert m.magnet_temp == 100.0
    p = prepare_thermal_duty(m, PROFILE)
    assert p.psi_f == pytest.approx(0.1 * (1 - 0.0012 * 80.0), rel=1e-12)


# --- published-motor S1 nameplate anchors ---
#
# Three independent S1-rated IPMs from published characterizations:
# - ABB M2BJ 100L (Awan): 2.2 kW catalog motor, 6-pole, 1500 rpm,
#   saliency 1.42, nameplate 14 Nm / 6.08 A peak.
# - Magnetic S.r.l. BLQ-40: 754 W industrial servo, 6-pole, 4000 rpm,
#   saliency 1.47 (Caruso et al. 2016, IMEKO TC4).
# - Kollmorgen Goldline B-104-B: 1.1 kW servo, 4-pole, 7500 rpm, the most
#   salient (Lq/Ld = 2.06); electrical model from a measured thesis table,
#   S1 nameplate from the fingerprint-matched datasheet column.
#
# Two checks per anchor: (1) the MTPA torque->current inverse is physically
# sound (MTPA is the minimum current, so <= nameplate); (2) the screen is
# optimistic at nameplate load because consumption assumes ideal MTPA while
# the budget is anchored to the higher nameplate current.

# (toml, rated_torque_Nm, MTPA/nameplate current ratio)
_NAMEPLATE_ANCHORS = [
    ("data/awan_ipm/awan_2p2kw_ipm.toml", 14.0, 0.928),
    ("data/magnetic_blq40/magnetic_blq40_ipm.toml", 1.8, 0.919),
    ("data/kollmorgen_b104b/kollmorgen_b104b_ipm.toml", 1.57, 0.916),
]


@pytest.mark.parametrize("toml,tau,ratio", _NAMEPLATE_ANCHORS)
def test_anchor_mtpa_current_below_nameplate(toml, tau, ratio):
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    i_mtpa = current_for_torque(m.n_p, m.psi_f, m.L_d, m.L_q, tau)
    # MTPA is the minimum current for the torque, so it cannot exceed the
    # actual nameplate current.
    assert i_mtpa < m.I_rated
    assert i_mtpa / m.I_rated == pytest.approx(ratio, abs=0.01)


@pytest.mark.parametrize("toml,tau,ratio", _NAMEPLATE_ANCHORS)
def test_anchor_continuous_at_rated_is_optimistic(toml, tau, ratio):
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    r = run_thermal_duty(_config(m, profile=((tau, 60.0),)))
    i_mtpa = current_for_torque(m.n_p, m.psi_f, m.L_d, m.L_q, tau)
    # An S1-rated motor at nameplate torque should sit at ~1.0; the model
    # reads below 1.0 by exactly (I_mtpa / I_rated)**2 — the ideal-MTPA
    # optimism. within_s1 True is a false positive by that headroom.
    assert r["over_budget_ratio"] == pytest.approx((i_mtpa / m.I_rated) ** 2, rel=1e-9)
    assert 0.80 < r["over_budget_ratio"] < 0.90
    assert r["within_s1"] is True


def test_magnet_derating_cannot_explain_blq40_band():
    # Re-check of the 0.92 band against magnet temperature:
    # closing MTPA/nameplate to 1.0 needs the magnet at ~83-101 C on the
    # NdFeB-class anchors (plausible) but ~313 C on BLQ-40's SmCo
    # (alpha ≈ -0.0003/K, family thermal trip 135 C — impossible). Even at
    # the trip temperature the SmCo ratio stays below 0.95, so the flat
    # 0.92±0.01 band across chemistry cannot be a magnet-temperature
    # artifact — the manufacturer-margin attribution stands.
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / "data/magnetic_blq40/magnetic_blq40_ipm.toml")
    psi_hot = psi_f_at_magnet_temp(m.psi_f, 135.0, -0.0003)
    r = current_for_torque(m.n_p, psi_hot, m.L_d, m.L_q, 1.8) / m.I_rated
    assert r < 0.95


def test_mtpa_nameplate_optimism_is_systematic():
    # All three anchors land in 0.90-0.93 on MTPA/nameplate current and
    # within ~1.5% of each other: the ideal-MTPA optimism is systematic
    # across machine, manufacturer, power class, speed, pole count, and
    # saliency — not motor-specific scatter. If a future anchor breaks
    # this band, the "systematic ~8% optimism" claim
    # needs revisiting.
    from phasesweep.machines.configs import load_motor
    ratios = []
    for toml, tau, _ in _NAMEPLATE_ANCHORS:
        m = load_motor(REPO_ROOT / toml)
        r = current_for_torque(m.n_p, m.psi_f, m.L_d, m.L_q, tau) / m.I_rated
        assert 0.90 < r < 0.93, f"{toml}: ratio {r:.3f} outside band"
        ratios.append(r)
    assert max(ratios) - min(ratios) < 0.02


# --- ETEL TMB thermal_resistance-path anchors (water-cooled datasheets) ---
#
# First external validation of the thermal_resistance budget path: ETEL
# torque-motor datasheets publish r_th, coil temp limit, ambient AND the
# continuous power dissipation Pc on one page. Five frames (22 to 88 poles,
# all reverse-salient L_d > L_q). Deliberately NOT in _NAMEPLATE_ANCHORS:
# that band is air-cooled S1 servos on the rated_current path; these
# water-cooled ratings sit lower and saturate harder (see TOML notes).

# (toml, Pc_W published dissipation @ Ic, Tc_Nm, over_budget_ratio at Tc)
_ETEL_RTH_ANCHORS = [
    ("data/etel_tmb/etel_tmb0140_030_ra.toml", 652.0, 21.6, 0.6372),
    ("data/etel_tmb/etel_tmb0210_030_ta.toml", 1110.0, 69.1, 0.7931),
    ("data/etel_tmb/etel_tmb0290_030_ra.toml", 1570.0, 133.0, 0.7894),
    ("data/etel_tmb/etel_tmb0360_030_ta.toml", 1980.0, 231.0, 0.6299),
    ("data/etel_tmb/etel_tmb0450_030_va.toml", 2460.0, 375.0, 0.5411),
]


@pytest.mark.parametrize("toml,pc,tau,over", _ETEL_RTH_ANCHORS)
def test_etel_budget_from_thermal_resistance(toml, pc, tau, over):
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    p = prepare_thermal_duty(m, ((tau, 60.0),))
    assert p.budget_source == "thermal_resistance"
    assert p.p_s1_budget == pytest.approx(
        (m.winding_temp_limit - m.ambient_temp) / m.r_th, rel=1e-12
    )


@pytest.mark.parametrize("toml,pc,tau,over", _ETEL_RTH_ANCHORS)
def test_etel_copper_loss_matches_datasheet_pc(toml, pc, tau, over):
    # Pure loss-model check, no thermal inference: 3-phase copper loss at
    # the datasheet continuous current with R_s at the 130 C coil limit vs
    # the datasheet's own published dissipation Pc. Lands within 2% on all
    # five frames (+0.1% to +1.9%) — iron loss is negligible at torque-motor
    # speeds, so Pc is almost entirely hot copper loss.
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    r_s_hot = r_s_at_operating_temp(m.R_s, m.winding_temp_limit)
    p_cu = 1.5 * r_s_hot * m.I_rated**2
    assert p_cu / pc == pytest.approx(1.0, abs=0.02)
    # ... and the dT/r_th budget sits a consistent 2.5-3.6% ABOVE the hot
    # copper loss across the family (ratio 0.964-0.975, n=5): the
    # manufacturer's Ic is copper-loss-limited through exactly the r_th
    # relation the thermal_resistance path models, with a small uniform
    # rating margin. Was stated as "within ~3%" at n=2; the band below is
    # the n=5 restatement, one-sided like the data.
    budget = (m.winding_temp_limit - m.ambient_temp) / m.r_th
    assert 0.955 < p_cu / budget < 0.985


@pytest.mark.parametrize("toml,pc,tau,over", _ETEL_RTH_ANCHORS)
def test_etel_duty_at_continuous_torque(toml, pc, tau, over):
    # Duty verdict at the nameplate continuous torque, exercising the
    # reverse-salient MTPA torque->current inverse (gamma < 0). Reads well
    # under 1.0: ideal-MTPA optimism plus saturation (Tc < Kt*Ic on these
    # frames), largest on the 66-pole frame. Documented, not a model target.
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    i_mtpa = current_for_torque(m.n_p, m.psi_f, m.L_d, m.L_q, tau)
    assert i_mtpa < m.I_rated  # MTPA current below nameplate Ic
    r = run_thermal_duty(_config(m, profile=((tau, 60.0),)))
    assert r["budget_source"] == "thermal_resistance"
    assert r["over_budget_ratio"] == pytest.approx(over, abs=1e-3)
    assert r["within_s1"] is True


# --- Kollmorgen B-104-B air-cooled r_th consistency (two-ambient ratings) ---
#
# First AIR-COOLED check of the thermal_resistance relation (ETEL above is
# water-cooled). The Goldline datasheet prints no rating winding temperature
# — but it prints continuous stall torque at TWO ambients (1.67 N.m @ 25 C,
# 1.57 N.m @ 40 C), one Ic (4.20 A rms, pairing with 40 C: Kt = 0.37 printed
# = 1.57/4.20), Rth = 1.07 K/W at stall, Class H insulation (180 C) and a
# 170 C thermostat. Rating both ambients to the same winding temperature
# theta through dT = Rth * P_cu(theta) makes theta over-determined:
#   ratio:    (Tc25/Tc40)^2 = (theta - 25)/(theta - 40)   [R_s, Rth cancel]
#   absolute: theta = 40 + Rth * P_cu(theta) at Ic        [uses Rth + R_s(T)]
# Agreement of the two estimates (and both landing below the thermostat)
# is the air-cooled validation: the published Rth, R_s, Ic and the two
# torque ratings cohere through exactly the relation the thermal_resistance
# budget path models — but only with R_s(T) derating (cold R_s puts the
# absolute estimate 36 K away from the ratio estimate).

_KOLL_TOML = REPO_ROOT / "data/kollmorgen_b104b/kollmorgen_b104b_ipm.toml"
_KOLL_RTH = 1.07     # K/W, datasheet Rth at stall (specified heatsink)
_KOLL_TC40 = 1.57    # N.m continuous stall torque @ 40 C ambient (pairs with Ic)
_KOLL_TC25 = 1.67    # N.m continuous stall torque @ 25 C ambient
_KOLL_THERMOSTAT = 170.0  # C, built-in thermostat opening temperature


def _koll_theta_ratio():
    # theta from the two-ambient torque ratio alone (R_s and Rth cancel)
    r2 = (_KOLL_TC25 / _KOLL_TC40) ** 2
    return (25.0 - 40.0 * r2) / (1.0 - r2)


def _koll_theta_absolute(m, derate=True):
    # fixed point of theta = 40 + Rth * P_cu(theta) at the datasheet Ic
    from phasesweep.models.thermal_duty import copper_loss
    theta = 100.0
    for _ in range(60):
        r_s = r_s_at_operating_temp(m.R_s, theta) if derate else m.R_s
        theta = 40.0 + _KOLL_RTH * copper_loss(r_s, m.I_rated)
    return theta


def test_kollmorgen_rth_predicts_25C_rating_from_40C():
    # Star check: fit theta from the 40 C rating (Ic + Rth + R_s(T)), then
    # predict the 25 C rating through the same relation. Lands within 0.4%
    # of the published 1.67 N.m; with cold (underated) R_s it misses by
    # +2.7% — the R_s(T) derating is what makes the datasheet cohere.
    from phasesweep.machines.configs import load_motor
    m = load_motor(_KOLL_TOML)
    theta = _koll_theta_absolute(m)
    pred = _KOLL_TC40 * ((theta - 25.0) / (theta - 40.0)) ** 0.5
    assert pred / _KOLL_TC25 == pytest.approx(1.0, abs=0.01)
    theta_cold = _koll_theta_absolute(m, derate=False)
    pred_cold = _KOLL_TC40 * ((theta_cold - 25.0) / (theta_cold - 40.0)) ** 0.5
    assert abs(pred_cold / _KOLL_TC25 - 1.0) > 0.02


def test_kollmorgen_implied_winding_temp_consistent_and_sane():
    # The R-free/Rth-free ratio estimate (154.1 C) and the absolute
    # watts estimate (~161 C) agree within 10 K over a ~117 K rise, and
    # both sit below the 170 C thermostat and the 180 C Class H limit —
    # consistent rating practice. Cold R_s is 36 K inconsistent.
    from phasesweep.machines.configs import load_motor
    m = load_motor(_KOLL_TOML)
    t_ratio = _koll_theta_ratio()
    t_abs = _koll_theta_absolute(m)
    for t in (t_ratio, t_abs):
        assert 145.0 < t < _KOLL_THERMOSTAT
    assert abs(t_abs - t_ratio) < 10.0
    assert abs(_koll_theta_absolute(m, derate=False) - t_ratio) > 25.0


def test_kollmorgen_thermal_variant_uses_budget_path():
    # Integration: pin winding_temp_limit at the ratio-implied theta and the
    # thermal_resistance path activates and reproduces the rated-current
    # budget within 5% on air-cooled hardware. The shipped TOML deliberately
    # leaves [thermal] unset so the nameplate anchors stay on rated_current.
    from dataclasses import replace

    from phasesweep.machines.configs import load_motor
    from phasesweep.models.thermal_duty import copper_loss
    m = load_motor(_KOLL_TOML)
    theta = _koll_theta_ratio()
    mt = replace(m, winding_temp_limit=theta, ambient_temp=40.0, r_th=_KOLL_RTH)
    p = prepare_thermal_duty(mt, ((_KOLL_TC40, 60.0),))
    assert p.budget_source == "thermal_resistance"
    p_cu = copper_loss(r_s_at_operating_temp(m.R_s, theta), m.I_rated)
    assert p_cu / p.p_s1_budget == pytest.approx(1.0, abs=0.05)


# --- Kollmorgen GOLDLINE two-ambient family (25 datasheet columns) ---
#
# Extends the B-104-B two-ambient analysis from n=1 to n=25 with
# datasheet-only values (data/kollmorgen_goldline/two_ambient_columns.toml):
# every GOLDLINE B-series column rates continuous stall torque at 25 C and
# 40 C ambient and prints Ic, Rm(25 C, line-line), Km and Rth. No electrical
# model (Ld/Lq/psi_f) is needed for either estimate below — these validate
# the thermal rating relation the thermal_resistance budget path models,
# not MTPA.

_GOLDLINE_TOML = REPO_ROOT / "data/kollmorgen_goldline/two_ambient_columns.toml"


def _goldline_columns():
    import tomllib
    with open(_GOLDLINE_TOML, "rb") as f:
        return tomllib.load(f)["columns"]


def _goldline_theta_ratio(col):
    # (Tc25/Tc40)^2 = (theta-25)/(theta-40): R_s and Rth cancel
    k = (col["tc25_Nm"] / col["tc40_Nm"]) ** 2
    return (25.0 - 40.0 * k) / (1.0 - k)


def _goldline_theta_absolute(col):
    # theta = 40 + Rth * P_cu(theta), P_cu = 3-phase copper loss at Ic with
    # per-phase R = Rm/2 derated from 25 C to theta. Fixed point (contraction,
    # Rth*P25*alpha << 1).
    p25 = 1.5 * col["rm_ll_Ohm"] * col["ic_Arms"] ** 2
    theta = 100.0
    for _ in range(100):
        theta = 40.0 + col["rth_K_per_W"] * p25 * (
            1.0 + COPPER_TEMP_COEFF * (theta - 25.0))
    return theta


def test_goldline_family_rated_to_one_winding_temperature():
    """All 25 columns imply the same absolute rating temperature, ~156 C.

    The datasheet-only absolute path (Ic + Rm + Rth + copper derating)
    lands in a <6 K band (154.6-158.4 C) across three frame sizes and
    25 windings: Kollmorgen rated the whole family to one winding design
    temperature through exactly the dT = Rth*P_cu(theta) relation the
    thermal_resistance budget path models — air-cooled r_th consistency
    at n=25. The band sits below the family thermostats (155/170 C) and
    the Class H limit (180 C), and brackets the B-104-B thesis-based
    estimates (154-161 C, tests above).
    """
    thetas = [_goldline_theta_absolute(c) for c in _goldline_columns()]
    assert len(thetas) == 25
    assert all(150.0 < t < 162.0 for t in thetas), thetas
    assert max(thetas) - min(thetas) < 6.0
    med = sorted(thetas)[len(thetas) // 2]
    assert 154.0 < med < 158.0


def test_goldline_ratio_path_agrees_at_the_median():
    """The R_s/Rth-cancelling ratio path centers on the same temperature.

    Per-column it is ill-conditioned against the 2-3 significant figures
    Tc is quoted to (a half-digit of rounding moves theta by tens of K,
    range 121-183 observed), so only the median and a wide per-column
    sanity band are asserted; the absolute path above is the precise
    per-column statement.
    """
    thetas = sorted(_goldline_theta_ratio(c) for c in _goldline_columns())
    assert all(115.0 < t < 195.0 for t in thetas), thetas
    med = thetas[len(thetas) // 2]
    assert 150.0 < med < 170.0


def test_goldline_kt_sqrt3_kb_consistency():
    """Kt = sqrt(3) * Kb(SI) per column within 2% — transcription guard.

    For a wye PMSM the stall torque constant (Nm/A_rms line) and the
    line-line back-EMF constant are the same number in SI units up to
    sqrt(3). Guards every transcribed Kt/Kb pair; this check is what
    caught the B-402-A datasheet misprint (N-m row says 2.51 where the
    printed lb-ft value converts to 2.25 — the TOML carries the corrected
    value with a note).
    """
    for col in _goldline_columns():
        kb_si = col["kb_Vrms_per_krpm"] / (1000.0 / 60.0 * 2.0 * 3.141592653589793)
        assert (3.0 ** 0.5) * kb_si / col["kt_Nm_per_Arms"] == pytest.approx(
            1.0, abs=0.02), col["model"]


def test_goldline_b104b_column_matches_anchor_toml():
    """The B-104-B family column stays consistent with the motor anchor."""
    from phasesweep.machines.configs import load_motor
    (col,) = [c for c in _goldline_columns() if c["model"] == "B-104-B"]
    assert col["tc40_Nm"] == _KOLL_TC40
    assert col["tc25_Nm"] == _KOLL_TC25
    assert col["rth_K_per_W"] == _KOLL_RTH
    m = load_motor(REPO_ROOT / "data/kollmorgen_b104b/kollmorgen_b104b_ipm.toml")
    assert m.I_rated == pytest.approx(2.0 ** 0.5 * col["ic_Arms"], rel=1e-3)
    # Column-implied rating temps agree with each other and the B-104-B window
    assert abs(_goldline_theta_absolute(col) - _goldline_theta_ratio(col)) < 5.0


# --- iron-loss coupling ---
# When all five [iron] fields are set, the lumped Bertotti p_fe at W_REF
# joins the duty consumption on both budget paths and the rated_current
# budget (nameplate S1 = copper at I_rated + iron at rated speed).
# Unset motors are bit-identical to the pre-iron model (verified
# before/after on Awan/ETEL/B-104-B during the change).

_IRON = dict(k_h=0.0153, k_e=6.35e-5, alpha_fe=1.62, m_core=2.0, B_core=1.3)


def _p_fe_handcalc(motor):
    from math import pi
    f_e = motor.n_p * motor.drive.W_REF / (2 * pi)
    return motor.m_core * (
        motor.k_h * f_e * motor.B_core**motor.alpha_fe
        + motor.k_e * f_e**2 * motor.B_core**2
    )


def test_iron_unset_is_copper_only():
    r = run_thermal_duty(_config(_motor()))
    assert r["p_fe"] == 0.0
    assert r["p_total_avg"] == r["p_cu_avg"]
    assert r["over_budget_ratio"] == r["p_cu_avg"] / r["p_s1_budget"]


def test_iron_partial_set_raises():
    with pytest.raises(ValueError, match="partially set"):
        run_thermal_duty(_config(_motor(k_h=0.0153, k_e=6.35e-5)))


def test_p_fe_matches_iron_loss_model():
    """thermal_duty's p_fe is the same number the iron_loss model reports."""
    from phasesweep.models.iron_loss import run_iron_loss
    m = _motor(**_IRON)
    r_td = run_thermal_duty(_config(m))
    r_fe = run_iron_loss(RunConfig(motor=m, model="iron_loss"))
    assert r_td["p_fe"] == pytest.approx(r_fe["p_fe"], rel=1e-12)
    assert r_td["p_fe"] == pytest.approx(_p_fe_handcalc(m), rel=1e-12)


def test_rated_current_nameplate_self_consistency():
    """A duty at exactly the nameplate point reads over_budget_ratio = 1
    with AND without iron — p_fe joins the rated_current budget precisely
    so that iron-aware consumption cannot condemn the motor at its own
    rating."""
    from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque
    for iron in ({}, _IRON):
        m = _motor(**iron)
        g = mtpa_gamma(m.psi_f, m.L_d, m.L_q, m.I_rated)
        tau_rated = mtpa_torque(m.n_p, m.psi_f, m.L_d, m.L_q, m.I_rated, g)
        r = run_thermal_duty(_config(m, ((tau_rated, 60.0),)))
        assert r["over_budget_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_iron_shrinks_reported_headroom():
    """Below the nameplate point, the iron-aware ratio sits closer to 1:
    (p_cu + p_fe)/(budget_cu + p_fe) > p_cu/budget_cu when p_cu < budget.
    This is the honest-er headroom the CREATOR finding motivated."""
    profile = ((2.0, 60.0),)
    r_cu = run_thermal_duty(_config(_motor(), profile))
    r_fe = run_thermal_duty(_config(_motor(**_IRON), profile))
    assert r_cu["over_budget_ratio"] < 1.0
    assert r_fe["over_budget_ratio"] > r_cu["over_budget_ratio"]
    assert r_fe["over_budget_ratio"] < 1.0


def test_thermal_resistance_budget_excludes_iron():
    """On the r_th path the budget is total-dissipation watts — p_fe joins
    the consumption only, raising the ratio by exactly p_fe/budget."""
    thermal = dict(r_th=0.5, winding_temp_limit=130.0, ambient_temp=20.0)
    r_cu = run_thermal_duty(_config(_motor(**thermal)))
    r_fe = run_thermal_duty(_config(_motor(**thermal, **_IRON)))
    assert r_fe["p_s1_budget"] == r_cu["p_s1_budget"]
    assert r_fe["budget_source"] == "thermal_resistance"
    expected = r_cu["over_budget_ratio"] + r_fe["p_fe"] / r_fe["p_s1_budget"]
    assert r_fe["over_budget_ratio"] == pytest.approx(expected, rel=1e-12)


def test_sustainable_duty_fraction_accounts_for_iron():
    """Iron loss spends budget at every duty fraction (speed sets it, not
    torque), so only the remaining budget scales with copper duty."""
    thermal = dict(r_th=0.5, winding_temp_limit=130.0, ambient_temp=20.0)
    heavy = ((60.0, 10.0), (5.0, 50.0))  # peak segment above the budget
    r_cu = run_thermal_duty(_config(_motor(**thermal), heavy))
    r_fe = run_thermal_duty(_config(_motor(**thermal, **_IRON), heavy))
    assert r_cu["sustainable_duty_fraction"] < 1.0
    expected = (r_fe["p_s1_budget"] - r_fe["p_fe"]) / r_fe["p_cu_peak"]
    assert r_fe["sustainable_duty_fraction"] == pytest.approx(
        expected, rel=1e-12)
    assert r_fe["sustainable_duty_fraction"] < r_cu["sustainable_duty_fraction"]
    # Iron alone above the budget -> nothing is sustainable
    r_hot = run_thermal_duty(
        _config(_motor(**thermal, **{**_IRON, "m_core": 500.0}), heavy))
    assert r_hot["sustainable_duty_fraction"] == 0.0


def test_w_ref_now_hashed_for_thermal_duty():
    """p_fe depends on W_REF, so W_REF must move the thermal_duty run ID."""
    from phasesweep.machines.motor import DriveParams
    m1 = _motor(**_IRON)
    m2 = _motor(**_IRON)
    m2 = Motor(**{**m1.__dict__, "drive": DriveParams(
        U_DC=m1.drive.U_DC, MAX_I_S=m1.drive.MAX_I_S,
        W_REF=2.0 * m1.drive.W_REF)})
    rc1 = _config(m1)
    rc2 = _config(m2)
    assert compute_run_id(rc1) != compute_run_id(rc2)


def test_params_reject_negative_p_fe():
    from phasesweep.solver_params import ThermalDutyParams
    with pytest.raises(ValueError, match="p_fe"):
        ThermalDutyParams(n_p=2, psi_f=0.1, R_s=0.5, p_s1_budget=10.0,
                          budget_source="rated_current",
                          duty_profile=((1.0, 1.0),), p_fe=-1.0)


# --- first-order thermal transient ---
# Optional Motor.thermal_time_constant enables an exact steady-periodic
# first-order march of the normalized winding temperature rise
# (x = rise / S1 rise; segment consumption / budget drives it). Converts
# the documented within_s1 short-segment caveat into a computed check.
# Closed-form periodic solution — no integration error; unset motors get
# no transient keys and stay bit-identical.


def test_transient_keys_absent_without_tau():
    r = run_thermal_duty(_config(_motor()))
    assert "transient_peak_ratio" not in r
    assert "within_s1_transient" not in r
    assert "winding_temp_peak" not in r


def test_transient_constant_profile_equals_average():
    """A constant duty is already at steady state: x = over_budget_ratio
    for any tau."""
    for tau in (1.0, 60.0, 3600.0):
        r = run_thermal_duty(
            _config(_motor(thermal_time_constant=tau), ((3.0, 30.0),)))
        assert r["transient_peak_ratio"] == pytest.approx(
            r["over_budget_ratio"], rel=1e-12)


def test_transient_fast_cycle_recovers_average_criterion():
    """Cycle time << tau -> the winding averages the loss thermally and
    the peak approaches over_budget_ratio (the pre-transient criterion)."""
    r = run_thermal_duty(
        _config(_motor(thermal_time_constant=3600.0), PROFILE))
    assert r["transient_peak_ratio"] == pytest.approx(
        r["over_budget_ratio"], rel=1e-2)
    assert r["transient_peak_ratio"] >= r["over_budget_ratio"]


def test_transient_long_segments_recover_peak_criterion():
    """Segments >> tau -> each segment reaches its own steady state and
    the peak approaches p_cu_peak-based loading."""
    r = run_thermal_duty(
        _config(_motor(thermal_time_constant=0.5), PROFILE))
    u_peak = r["p_cu_peak"] / r["p_s1_budget"]
    assert r["transient_peak_ratio"] == pytest.approx(u_peak, rel=1e-6)


def test_transient_peak_between_average_and_peak_loading():
    """For any tau the periodic peak sits between the cycle-average and
    the worst-segment loading."""
    for tau in (1.0, 10.0, 60.0, 600.0):
        r = run_thermal_duty(
            _config(_motor(thermal_time_constant=tau), PROFILE))
        u_peak = r["p_cu_peak"] / r["p_s1_budget"]
        assert (r["over_budget_ratio"] - 1e-12
                <= r["transient_peak_ratio"] <= u_peak + 1e-12)


def test_transient_hand_marched_two_segment():
    """Cross-check the closed-form periodic solution against a brute-force
    march to convergence."""
    from math import exp

    from phasesweep.models.thermal_duty import transient_peak
    u = [0.9, 0.2]
    dts = [20.0, 40.0]
    tau = 30.0
    x = 0.0
    for _ in range(2000):
        for u_k, dt in zip(u, dts):
            x = u_k + (x - u_k) * exp(-dt / tau)
    x0 = x
    xs = [x0]
    for u_k, dt in zip(u, dts):
        xs.append(u_k + (xs[-1] - u_k) * exp(-dt / tau))
    assert transient_peak(u, dts, tau) == pytest.approx(max(xs), rel=1e-9)


def test_transient_within_s1_can_fail_while_average_passes():
    """The scenario the caveat warned about: cycle-average within budget,
    but a long hot segment overheats mid-cycle."""
    profile = ((7.0, 200.0), (0.5, 800.0))
    m_avg = _motor()
    r_avg = run_thermal_duty(_config(m_avg, profile))
    assert r_avg["within_s1"]
    r = run_thermal_duty(
        _config(_motor(thermal_time_constant=30.0), profile))
    assert r["within_s1"]
    assert not r["within_s1_transient"]
    assert r["transient_peak_ratio"] > 1.0


def test_transient_absolute_temp_on_rth_path():
    """thermal_resistance path converts the normalized peak to deg C."""
    thermal = dict(r_th=0.5, winding_temp_limit=130.0, ambient_temp=20.0,
                   thermal_time_constant=60.0)
    r = run_thermal_duty(_config(_motor(**thermal)))
    expected = 20.0 + r["transient_peak_ratio"] * (130.0 - 20.0)
    assert r["winding_temp_peak"] == pytest.approx(expected, rel=1e-12)
    # rated_current path: normalized only, no absolute temperature
    r2 = run_thermal_duty(
        _config(_motor(thermal_time_constant=60.0)))
    assert "winding_temp_peak" not in r2
    assert "transient_peak_ratio" in r2


def test_transient_includes_iron_loss():
    """Iron loss raises the transient floor: at zero torque the winding
    still sits at p_fe/budget."""
    thermal = dict(r_th=0.5, winding_temp_limit=130.0, ambient_temp=20.0,
                   thermal_time_constant=60.0)
    r = run_thermal_duty(
        _config(_motor(**thermal, **_IRON), ((0.0, 100.0),)))
    assert r["transient_peak_ratio"] == pytest.approx(
        r["p_fe"] / r["p_s1_budget"], rel=1e-12)


def test_etel_0450_tau_loaded_and_continuous_consistency():
    """The ETEL TOMLs now carry the datasheet tau_th; at the continuous
    rating the transient peak equals the steady ratio (constant profile),
    preserving the one-sided budget band."""
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / "data/etel_tmb/etel_tmb0450_030_va.toml")
    assert m.thermal_time_constant == 135.0
    tc = 375.0
    r = run_thermal_duty(_config(m, ((tc, 60.0),)))
    assert r["transient_peak_ratio"] == pytest.approx(
        r["over_budget_ratio"], rel=1e-12)
    assert r["winding_temp_peak"] < m.winding_temp_limit


def test_motor_rejects_nonpositive_tau():
    with pytest.raises(ValueError, match="thermal_time_constant"):
        _motor(thermal_time_constant=0.0)
