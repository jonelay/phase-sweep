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

from math import asin, degrees, sqrt

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
    assert {"tau_mtpa", "k_T", "k_T_rms", "gamma_opt_deg"}.issubset(info.produces)

    m = _make_motor(I_rated=5.0)
    info.validate(m)
    result = info.fn(_make_config(m))
    assert {"tau_mtpa", "k_T", "k_T_rms", "gamma_opt_deg"}.issubset(set(result.keys()))


# --- Test 8: TOML load populates I_rated ---
def test_toml_load():
    from phasesweep.configs import load_motor
    m = load_motor(REPO_ROOT / "motors/creator_case_pmsm.toml")
    assert m.I_rated is not None
    assert abs(m.I_rated - 0.21 * sqrt(2)) < 1e-10


# --- Test 9: Reverse saliency (L_d > L_q) falls through to non-salient ---
def test_reverse_saliency():
    m = _make_motor(I_rated=5.0, L_d=6e-3, L_q=4e-3)
    result = MODEL_REGISTRY["rated_torque"].fn(_make_config(m))
    expected = 1.5 * 2 * 0.1 * 5.0  # gamma=0: k_T * I_rated
    assert abs(result["tau_mtpa"] - expected) < 1e-12


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
    dL = L_q - L_d
    sin_g = (-psi_f + sqrt(psi_f**2 + 8 * dL**2 * I_s**2)) / (4 * dL * I_s)
    gamma = degrees(asin(sin_g))
    assert gamma == pytest.approx(9.144, abs=0.01)




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
    psi_f, L_d, L_q, n_p = 0.1144, 0.2055, 0.3320, 2
    I_s = I_rms * sqrt(2)
    dL = L_q - L_d
    sin_g = (-psi_f + sqrt(psi_f**2 + 8 * dL**2 * I_s**2)) / (4 * dL * I_s)
    gamma = degrees(asin(sin_g))
    computed_angle = 90.0 + gamma  # electrical angle from d-axis
    assert abs(computed_angle - published_angle_deg) < 3.0, (
        f"MTPA angle {computed_angle:.1f} deg vs published {published_angle_deg} deg "
        f"(at {I_rms} A_rms)"
    )


def test_creator_mtpa_angle_monotonic():
    """CREATOR MTPA angle increases with current (more reluctance at higher I)."""
    psi_f, L_d, L_q = 0.1144, 0.2055, 0.3320
    dL = L_q - L_d
    angles = []
    for I_rms in [0.10, 0.15, 0.21, 0.30]:
        I_s = I_rms * sqrt(2)
        sin_g = (-psi_f + sqrt(psi_f**2 + 8 * dL**2 * I_s**2)) / (4 * dL * I_s)
        angles.append(degrees(asin(sin_g)))
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
