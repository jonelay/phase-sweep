"""Tests for rated torque model.

Validation sources:
- CREATOR Case PMSM: Dhakal et al. (2025), COMPEL 44(4), DOI 10.1108/compel-11-2024-0462
  arXiv:2501.15921, dataset DOI 10.3217/sns1d-77m43. Table 10 circuit params.
  MTPA angles from paper text (§5.2): 100 deg @ 0.15 A_rms, 110 deg @ 0.30 A_rms.
- Awan 2.2-kW IPM: Awan (2019), Aalto doctoral dissertation 187/2019,
  ISBN 978-952-60-8765-8. Table 6.2 motor data. Also: Awan et al. (2018),
  IEEE TIA 54(6), pp. 6110-6120, DOI 10.1109/TIA.2018.2862410.
- Morimoto MTPA quadratic: Morimoto et al. (1994), IEEE TIA 30(4), pp. 920-926,
  DOI 10.1109/28.297908.
- Demo motors A/B/C: synthetic test fixtures (inline). No published source.
"""

from math import degrees, sqrt

import pytest

from phasesweep.geometry import default_inrunner
from phasesweep.motor import Motor
from phasesweep.rated_torque import mtpa_curve, mtpa_gamma, mtpa_torque
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solver_params import prepare_rated_torque
from tests.conftest import REPO_ROOT
from tests.conftest import make_motor as _make_motor


# --- Test 1: Non-salient hand-calc ---
def test_nonsalient_handcalc():
    m = _make_motor(I_rated=10.0)
    result = MODEL_REGISTRY["rated_torque"].fn(
        _make_config(m)
    )
    expected = 1.5 * 2 * 0.1 * 10.0  # 1.5 * n_p * psi_f * I_rated
    assert abs(result["tau_mtpa"] - expected) < 1e-12


# --- Test 2: Salient hand-calc (CREATOR params) ---
def test_salient_handcalc():
    m = _make_motor(
        psi_f=0.1144, L_d=0.2055, L_q=0.3320,
        I_rated=0.21 * sqrt(2),
    )
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    # Independent reference: computed from Morimoto MTPA quadratic with
    # CREATOR params (psi_f=0.1144, L_d=0.2055, L_q=0.3320, I_s=0.21*sqrt(2), n_p=2)
    # sin_gamma=0.2777341286, tau_mtpa=0.1068458108
    assert abs(result["tau_mtpa"] - 0.1068458108) < 1e-7
    assert result["tau_mtpa"] > 0


# --- Test 3: Non-salient fallback (L_d == L_q gives same as None) ---
def test_nonsalient_fallback():
    m_none = _make_motor(I_rated=5.0)
    m_equal = _make_motor(I_rated=5.0, L_d=4e-3, L_q=4e-3)
    r1 = MODEL_REGISTRY["rated_torque"].fn(_make_config(m_none))
    r2 = MODEL_REGISTRY["rated_torque"].fn(_make_config(m_equal))
    assert abs(r1["tau_mtpa"] - r2["tau_mtpa"]) < 1e-12


# --- Test 4: k_T_rms relation ---
def test_kt_rms_relation():
    m = _make_motor(I_rated=10.0)
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    assert abs(result["k_T_rms"] - result["k_T"] * sqrt(2)) < 1e-12


# --- Test 5: Missing I_rated raises ValueError ---
def test_missing_i_rated():
    m = _make_motor()  # no I_rated
    with pytest.raises(ValueError, match="I_rated"):
        prepare_rated_torque(m)


# --- Test 6: Missing psi_f (no B_rem either) raises ValueError ---
def test_missing_psi_f():
    m = Motor(
        name="bare", geometry=default_inrunner(), n_p=2,
        I_rated=5.0,
    )
    with pytest.raises(ValueError, match="psi_f"):
        prepare_rated_torque(m)


# --- Test 7: Registry integration ---
def test_registry_entry():
    info = MODEL_REGISTRY["rated_torque"]
    assert info.cost == "fast"
    assert info.source == "computed"
    assert {"tau_mtpa", "k_T", "k_T_rms", "k_T_effective", "gamma_opt_deg"}.issubset(info.produces)

    m = _make_motor(I_rated=5.0)
    info.validate(m)
    result = info.fn(_make_config(m))
    assert {"tau_mtpa", "k_T", "k_T_rms", "k_T_effective", "gamma_opt_deg"}.issubset(set(result.keys()))


# --- Test 8: TOML load populates I_rated ---
def test_toml_load():
    from phasesweep.configs import load_motor
    m = load_motor(REPO_ROOT / "motors/creator_case_pmsm.toml")
    assert m.I_rated is not None
    assert abs(m.I_rated - 0.21 * sqrt(2)) < 1e-10


# --- Test 9: Reverse saliency (L_d > L_q) takes the MTPA branch ---
def test_reverse_saliency():
    m = _make_motor(I_rated=5.0, L_d=6e-3, L_q=4e-3)
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    # Hand-calc: Morimoto + root with dL=-2e-3, psi_f=0.1, I=5, n_p=2:
    # gamma=-5.6284009065 deg (magnetizing i_d), tau=1.5074088669
    # vs 1.5 Nm at gamma=0 — brute-force verified global optimum
    assert result["tau_mtpa"] == pytest.approx(1.5074088669, rel=1e-9)
    assert result["tau_mtpa"] > 1.5 * 2 * 0.1 * 5.0  # beats k_T * I_rated
    assert result["gamma_opt_deg"] == pytest.approx(-5.6284009065, abs=1e-6)


def test_reverse_saliency_mirror_symmetry():
    """Swapping L_d and L_q negates gamma and preserves tau_mtpa."""
    psi_f, I_s = 0.1, 5.0
    g_fwd = mtpa_gamma(psi_f, 4e-3, 6e-3, I_s)
    g_rev = mtpa_gamma(psi_f, 6e-3, 4e-3, I_s)
    assert g_rev == pytest.approx(-g_fwd, rel=1e-12)
    tau_fwd = mtpa_torque(2, psi_f, 4e-3, 6e-3, I_s, g_fwd)
    tau_rev = mtpa_torque(2, psi_f, 6e-3, 4e-3, I_s, g_rev)
    assert tau_rev == pytest.approx(tau_fwd, rel=1e-12)


# ---------------------------------------------------------------------------
# Awan 2.2-kW IPM validation (Table 6.2, Awan 2019)
# ---------------------------------------------------------------------------

def test_awan_ipm_rated_torque():
    """Awan 2.2-kW IPM: salient MTPA hand-calc.

    Source: Awan (2019) Table 6.2 — n_p=3, psi_f=0.545 Wb, L_d=36 mH,
    L_q=51 mH, I_rated=4.3 A_rms. Rated torque 14 Nm at 0.80 p.u.
    """
    m = _make_motor(
        name="Awan 2.2-kW IPM",
        n_p=3,
        psi_f=0.545,
        L_d=0.036,
        L_q=0.051,
        I_rated=4.3 * sqrt(2),
    )
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    # Hand-calc: Morimoto MTPA quadratic with Awan Table 6.2 params
    # sin_gamma=0.1589165375, tau_mtpa=15.1160551204
    assert abs(result["tau_mtpa"] - 15.1160551204) < 1e-6
    assert result["k_T"] == pytest.approx(1.5 * 3 * 0.545, rel=1e-12)
    # Rated torque exceeds published rated torque (14 Nm) — expected: MTPA at peak
    # current gives more than continuous thermal rating
    assert result["tau_mtpa"] > 14.0


def test_awan_ipm_mtpa_angle():
    """Awan IPM MTPA current angle: ~9.1 deg from q-axis.

    At L_q/L_d = 1.42, the reluctance torque contribution is modest (~7%).
    """
    psi_f, L_d, L_q = 0.545, 0.036, 0.051
    I_s = 4.3 * sqrt(2)
    gamma = degrees(mtpa_gamma(psi_f, L_d, L_q, I_s))
    assert gamma == pytest.approx(9.144, abs=0.01)


# ---------------------------------------------------------------------------
# ETEL TMB reverse-salient anchors (manufacturer datasheets, L_d > L_q)
# ---------------------------------------------------------------------------
# First real machines exercising the reverse-salient MTPA branch (gamma < 0,
# magnetizing i_d): ETEL water-cooled direct-drive torque motors, saliency
# L_q/L_d 0.82-0.93, 22 to 88 poles across five frame sizes. psi_f is derived
# from the back-EMF constant Ku (magnet-only — no reluctance double-count)
# and cross-checked against two independent datasheet values: the
# short-circuit current (psi_f/L_d = sqrt(2)·Isc) and the torque constant Kt.
# kt_ratio (model k_T_effective / catalog Kt) is pinned per anchor: it drifts
# above 1.0 with the frame's reluctance share (gamma at Ic), reaching +3.2%
# on the strongly salient 0450 VA — direction is physics (the linear MTPA
# model adds reluctance torque the catalog small-signal Kt does not carry),
# the magnitude is a pinned observation. Sources in the TOMLs.

# (toml, gamma_opt_deg, k_T_effective_rms, catalog_Kt_Nm_per_Arms, kt_ratio,
#  Isc_Arms)
_ETEL_ANCHORS = [
    ("data/etel_tmb/etel_tmb0140_030_ra.toml", -9.1371, 4.4238, 4.35, 1.0170, 4.67),
    ("data/etel_tmb/etel_tmb0210_030_ta.toml", -5.7808, 7.0340, 7.05, 0.9977, 11.2),
    ("data/etel_tmb/etel_tmb0290_030_ra.toml", -3.3480, 19.6059, 19.6, 1.0003, 9.37),
    ("data/etel_tmb/etel_tmb0360_030_ta.toml", -8.1471, 32.9084, 32.6, 1.0095, 6.65),
    ("data/etel_tmb/etel_tmb0450_030_va.toml", -13.6786, 42.3966, 41.1, 1.0315, 8.02),
]


@pytest.mark.parametrize("toml,gamma_deg,kteff_rms,kt_cat,kt_ratio,isc",
                         _ETEL_ANCHORS)
def test_etel_reverse_salient_mtpa(toml, gamma_deg, kteff_rms, kt_cat,
                                   kt_ratio, isc):
    from phasesweep.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    assert m.L_d > m.L_q
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    assert result["gamma_opt_deg"] == pytest.approx(gamma_deg, abs=1e-3)
    assert result["tau_mtpa"] > result["k_T"] * m.I_rated  # reluctance adds
    k_eff_rms = result["k_T_effective"] * sqrt(2)
    assert k_eff_rms == pytest.approx(kteff_rms, abs=1e-3)
    # Catalog torque constant reproduced within ~1% on the mildly salient
    # frames; the pinned ratio tracks the reluctance-share drift (see block
    # comment).
    assert k_eff_rms / kt_cat == pytest.approx(kt_ratio, abs=0.002)


@pytest.mark.parametrize("toml,gamma_deg,kteff_rms,kt_cat,kt_ratio,isc",
                         _ETEL_ANCHORS)
def test_etel_psi_f_isc_crosscheck(toml, gamma_deg, kteff_rms, kt_cat,
                                   kt_ratio, isc):
    # psi_f/L_d equals the datasheet max short-circuit current (peak) —
    # independent confirmation of the Ku-derived psi_f AND the per-phase
    # (/2 terminal-to-terminal) convention. Guards future TOML edits.
    # Tolerance is the 3-significant-figure rounding budget of the inputs
    # (Ku alone contributes up to ~0.45%); observed family worst case is
    # -0.24% (0290 RA).
    from phasesweep.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    assert m.psi_f / m.L_d == pytest.approx(sqrt(2) * isc, rel=0.004)


# ---------------------------------------------------------------------------
# Du et al. 2021 CLF-RSPMSM — reverse-salient literature anchor (FEM-only)
# ---------------------------------------------------------------------------
# Du, Liu, Fu, Liang, Huang, CES Trans. Electr. Mach. Syst. 5(2):163-173,
# 2021, DOI 10.30941/cestems.2021.00020 (open access). 10 kW, 48-slot/8-pole
# CLF-RSPMSM deliberately designed reverse-salient (Table VI: Ld = 9.8 mH,
# phi_dq = Ld/Lq = 1.08, psi_pm = 0.245 Wb no-load; rated 35 A phase).
# Fig. 12(b) FEM torque vs current angle at rated amplitude 50 A: maximum
# 86.6 Nm at -5 deg with i_d = +4.4 A (magnetizing). Grounds the sign and
# operating point of the gamma < 0 branch; provenance in
# data/du2021_clf_rspmsm/du2021_clf_rspmsm.toml.

_DU2021 = dict(psi_f=0.245, L_d=9.8e-3, L_q=9.8e-3 / 1.08)


def test_du2021_mtpa_angle_sign_and_flat_optimum():
    from math import radians, sin
    g = mtpa_gamma(I_s=50.0, **_DU2021)
    # Linear-model optimum with Table VI params: same side as the paper's
    # FEM optimum (-5 deg, i_d = +4.4 A) — magnetizing i_d, gamma < 0.
    assert degrees(g) == pytest.approx(-8.173, abs=0.01)
    assert -50.0 * sin(g) == pytest.approx(7.108, abs=0.01)
    # The angle disagreement vs the FEM (-8.2 vs -5 deg) is low-stakes: the
    # MTPA optimum is flat. Evaluating the linear model AT the paper's angle
    # costs only 0.16% torque — and must cost > 0 (ours is the optimum).
    tau_ours = mtpa_torque(4, I_s=50.0, gamma=g, **_DU2021)
    tau_paper_angle = mtpa_torque(4, I_s=50.0, gamma=radians(-5.0), **_DU2021)
    assert tau_paper_angle < tau_ours
    assert tau_paper_angle / tau_ours == pytest.approx(0.9984, abs=1e-3)


def test_du2021_torque_with_load_dependent_flux():
    # VLF caveat: Table VI psi_f is no-load; the CLF rotor's d-axis flux
    # rises to 0.2904 Wb under heavy load (Fig. 10a). The linear model
    # under-predicts the FEM peak with the no-load value and reproduces it
    # within 1.4% with the loaded value — the discrepancy is the machine's
    # designed flux variability, not the MTPA math.
    n_p, i_pk = 4, 50.0
    g = mtpa_gamma(I_s=i_pk, **_DU2021)
    tau_noload = mtpa_torque(n_p, I_s=i_pk, gamma=g, **_DU2021)
    assert tau_noload == pytest.approx(74.29, abs=0.01)  # ~14% below 86.6
    loaded = dict(_DU2021, psi_f=0.2904)
    g2 = mtpa_gamma(I_s=i_pk, **loaded)
    tau_loaded = mtpa_torque(n_p, I_s=i_pk, gamma=g2, **loaded)
    assert tau_loaded == pytest.approx(86.6, rel=0.015)


def test_du2021_toml_rated_torque():
    from phasesweep.configs import load_motor
    m = load_motor(REPO_ROOT / "data/du2021_clf_rspmsm/du2021_clf_rspmsm.toml")
    assert m.L_d > m.L_q
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    assert result["gamma_opt_deg"] == pytest.approx(-8.0965, abs=1e-3)
    assert result["tau_mtpa"] > result["k_T"] * m.I_rated  # reluctance adds
    assert result["k_T_effective"] == pytest.approx(1.4854, abs=1e-3)




# ---------------------------------------------------------------------------
# CREATOR MTPA angle validation (Dhakal et al. 2025, COMPEL 44(4))
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("I_rms, published_angle_deg", [
    (0.15, 100.0),  # Table text / Fig. 12, Dhakal et al. 2025
    (0.30, 110.0),  # Table text / Fig. 12, Dhakal et al. 2025
])
def test_creator_mtpa_angle(I_rms, published_angle_deg):
    """CREATOR MTPA angle vs published FEM-derived values.

    Our linear-inductance Morimoto quadratic should agree within ~3 deg
    of the FEM-derived MTPA angles. Small discrepancy expected because
    the published values include saturation effects.
    """
    psi_f, L_d, L_q = 0.1144, 0.2055, 0.3320
    I_s = I_rms * sqrt(2)
    gamma = degrees(mtpa_gamma(psi_f, L_d, L_q, I_s))
    computed_angle = 90.0 + gamma  # electrical angle from d-axis
    assert abs(computed_angle - published_angle_deg) < 3.0, (
        f"MTPA angle {computed_angle:.1f} deg vs published {published_angle_deg} deg "
        f"(at {I_rms} A_rms)"
    )


def test_creator_mtpa_angle_monotonic():
    """CREATOR MTPA angle increases with current (more reluctance at higher I)."""
    psi_f, L_d, L_q = 0.1144, 0.2055, 0.3320
    angles = []
    for I_rms in [0.10, 0.15, 0.21, 0.30]:
        angles.append(mtpa_gamma(psi_f, L_d, L_q, I_rms * sqrt(2)))
    for i in range(len(angles) - 1):
        assert angles[i] < angles[i + 1]


def test_creator_rated_torque_vs_published():
    """CREATOR rated torque vs published rated torque.

    Published: 0.10 Nm at 0.21 A_rms (rated operating point, Table 6).
    Our model: 0.1068 Nm. The ~7% difference is because our model computes
    maximum torque at MTPA, while published 0.10 Nm is the continuous rating.
    """
    from phasesweep.configs import load_motor
    m = load_motor(REPO_ROOT / "motors/creator_case_pmsm.toml")
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    published_rated = 0.10  # Nm, Dhakal et al. Table 6
    assert result["tau_mtpa"] >= published_rated
    assert result["tau_mtpa"] < published_rated * 1.15  # within 15%


# ---------------------------------------------------------------------------
# Synthetic motor rated torque (inline test fixtures)
# ---------------------------------------------------------------------------

def _demo_motor_a():
    import dataclasses

    from phasesweep.geometry import default_inrunner
    return Motor(
        name="A: 4-pole SPMSM",
        geometry=dataclasses.replace(default_inrunner(), n_slots=12),
        n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
        N=50, k_w=0.966, L_stk=0.10, I_rated=10.0, coils_series=1,
    )

def _demo_motor_b():
    import dataclasses

    from phasesweep.geometry import default_inrunner
    return Motor(
        name="B: 4-pole IPMSM",
        geometry=dataclasses.replace(default_inrunner(), n_slots=12),
        n_p=2, R_s=0.2, L_d=4e-3, L_q=6e-3, psi_f=0.1, J=0.002,
        N=50, k_w=0.966, L_stk=0.10, I_rated=10.0, coils_series=1,
    )

def _demo_motor_c():
    import dataclasses

    from phasesweep.geometry import default_inrunner
    return Motor(
        name="C: 8-pole SPMSM",
        geometry=dataclasses.replace(default_inrunner(), n_slots=24),
        n_p=4, R_s=0.3, L_d=2e-3, L_q=2e-3, psi_f=0.08, J=0.002,
        N=50, k_w=0.966, L_stk=0.10, I_rated=8.0, coils_series=1,
    )

def test_demo_a_spm_rated_torque():
    """Demo A (4-pole SPMSM): non-salient, tau = k_T * I_rated."""
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(_demo_motor_a()))
    expected = 1.5 * 2 * 0.1 * 10.0  # 3.0 Nm
    assert result["tau_mtpa"] == pytest.approx(expected, rel=1e-12)
    assert result["k_T"] == pytest.approx(0.3, rel=1e-12)


def test_demo_b_ipm_rated_torque():
    """Demo B (4-pole IPMSM): salient (L_q/L_d=1.5), reluctance torque adds ~2%."""
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(_demo_motor_b()))
    assert result["tau_mtpa"] == pytest.approx(3.0573018190, rel=1e-8)
    assert result["tau_mtpa"] > 3.0


def test_demo_c_spm_rated_torque():
    """Demo C (8-pole SPMSM): non-salient, tau = k_T * I_rated."""
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(_demo_motor_c()))
    expected = 1.5 * 4 * 0.08 * 8.0  # 3.84 Nm
    assert result["tau_mtpa"] == pytest.approx(expected, rel=1e-12)
    assert result["k_T"] == pytest.approx(0.48, rel=1e-12)


def test_demo_b_reluctance_boost():
    """Demo B reluctance torque boost over Demo A (same psi_f, n_p, I_rated)."""
    r_a = MODEL_REGISTRY["rated_torque"].fn(_make_config(_demo_motor_a()))
    r_b = MODEL_REGISTRY["rated_torque"].fn(_make_config(_demo_motor_b()))
    boost_pct = (r_b["tau_mtpa"] - r_a["tau_mtpa"]) / r_a["tau_mtpa"] * 100
    assert 1.0 < boost_pct < 5.0


# ---------------------------------------------------------------------------
# MTPA curve functions
# ---------------------------------------------------------------------------

def test_mtpa_gamma_importable():
    g = mtpa_gamma(psi_f=0.1144, L_d=0.2055, L_q=0.3320, I_s=0.21 * sqrt(2))
    assert degrees(g) == pytest.approx(16.13, abs=0.1)


def test_mtpa_torque_matches_rated():
    m = _make_motor(psi_f=0.1144, L_d=0.2055, L_q=0.3320, I_rated=0.21 * sqrt(2))
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    g = mtpa_gamma(0.1144, 0.2055, 0.3320, 0.21 * sqrt(2))
    tau = mtpa_torque(2, 0.1144, 0.2055, 0.3320, 0.21 * sqrt(2), g)
    assert tau == pytest.approx(result["tau_mtpa"], rel=1e-10)


def test_curve_output_keys_and_length():
    c = mtpa_curve(n_p=2, psi_f=0.1, L_d=0.01, L_q=0.02, I_rated=5.0, n_pts=25)
    assert set(c.keys()) == {"I_curve", "gamma_curve_deg", "angle_d_curve_deg", "tau_curve"}
    assert all(len(c[k]) == 25 for k in c)


def test_gamma_opt_matches_curve_at_i_rated():
    m = _make_motor(psi_f=0.1144, L_d=0.2055, L_q=0.3320, I_rated=0.21 * sqrt(2))
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    import numpy as np
    gamma_at_rated = float(np.interp(
        m.I_rated, result["I_curve"], result["gamma_curve_deg"],
    ))
    assert gamma_at_rated == pytest.approx(result["gamma_opt_deg"], abs=0.5)


def test_angle_d_equals_90_plus_gamma():
    c = mtpa_curve(n_p=2, psi_f=0.1, L_d=0.01, L_q=0.02, I_rated=5.0)
    for g, d in zip(c["gamma_curve_deg"], c["angle_d_curve_deg"]):
        assert d == pytest.approx(90.0 + g, abs=1e-10)


def test_nonsalient_gamma_opt_zero():
    m = _make_motor(I_rated=5.0)  # no L_d/L_q → non-salient
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    assert result["gamma_opt_deg"] == 0.0


def test_salient_gamma_opt_positive():
    m = _make_motor(I_rated=5.0, L_d=0.01, L_q=0.02)
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    assert result["gamma_opt_deg"] > 0.0
    assert "I_curve" in result  # curves present for salient


# --- Helper ---
def _make_config(motor):
    from phasesweep.sweep_types import RunConfig
    return RunConfig(motor=motor, model="rated_torque")
