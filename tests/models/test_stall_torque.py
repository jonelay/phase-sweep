"""Tests for stall torque model.

Stall torque emits two values:
- Current-limited: tau_stall at I_stall = I_LIMIT (or MAX_I_S).
- Electromagnetic: tau_stall_electromagnetic at I_stall_em = U_DC / (sqrt(3) * R_s).

Same Morimoto formula as rated_torque, evaluated at each current level.

Validation sources:
- CREATOR Case PMSM: Dhakal et al. (2025), COMPEL 44(4), Table 10 circuit params.
- Awan 2.2-kW IPM: Awan (2019), Aalto doctoral dissertation 187/2019, Table 6.2.
"""

from math import sqrt

import pytest

from phasesweep.machines.geometry import default_inrunner
from phasesweep.machines.motor import DriveParams, Motor
from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solver_params import prepare_stall_torque
from tests.conftest import REPO_ROOT
from tests.conftest import make_motor as _make_motor


def _make_config(motor):
    from phasesweep.sweep_types import RunConfig
    return RunConfig(motor=motor, model="stall_torque")


# --- I_stall calculation ---

def test_i_stall_uses_max_i_s():
    """I_stall = MAX_I_S when I_LIMIT not set."""
    m = _make_motor(R_s=100.0, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0))
    p = prepare_stall_torque(m)
    assert p.I_stall == pytest.approx(20.0, rel=1e-12)


def test_i_stall_uses_i_limit():
    """I_stall = I_LIMIT when set, overriding MAX_I_S."""
    m = _make_motor(R_s=0.1, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, I_LIMIT=5.0))
    p = prepare_stall_torque(m)
    assert p.I_stall == pytest.approx(5.0, rel=1e-12)


def test_i_stall_zero_resistance():
    """R_s = 0 (superconductor): I_stall = MAX_I_S, I_stall_em = None."""
    m = _make_motor(R_s=0.0, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0))
    p = prepare_stall_torque(m)
    assert p.I_stall == pytest.approx(20.0, rel=1e-12)
    assert p.I_stall_em is None


def test_i_stall_em_computed():
    """I_stall_em = U_DC / (sqrt(3) * R_s) — electromagnetic stall."""
    m = _make_motor(R_s=100.0, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0))
    p = prepare_stall_torque(m)
    expected_em = 540.0 / (sqrt(3) * 100.0)
    assert p.I_stall_em == pytest.approx(expected_em, rel=1e-12)


# --- Non-salient hand-calc ---

def test_nonsalient_handcalc():
    m = _make_motor(R_s=1.0, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0))
    p = prepare_stall_torque(m)
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    expected = 1.5 * 2 * 0.1 * p.I_stall
    assert abs(result["tau_stall"] - expected) < 1e-12


# --- Salient hand-calc ---

def test_salient_handcalc():
    m = _make_motor(
        psi_f=0.1144, L_d=0.2055, L_q=0.3320,
        R_s=100.0,
        drive=DriveParams(U_DC=540.0, MAX_I_S=20.0),
    )
    p = prepare_stall_torque(m)
    g = mtpa_gamma(0.1144, 0.2055, 0.3320, p.I_stall)
    tau_expected = mtpa_torque(2, 0.1144, 0.2055, 0.3320, p.I_stall, g)
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert abs(result["tau_stall"] - tau_expected) < 1e-10


# --- tau_stall >= tau_rated ---

def test_stall_ge_rated_nonsalient():
    """Non-salient motor where I_stall > I_rated → tau_stall > tau_rated."""
    m = _make_motor(
        I_rated=5.0, R_s=1.0,
        drive=DriveParams(U_DC=540.0, MAX_I_S=20.0),
    )
    r_rated = MODEL_REGISTRY["rated_torque"].fn(
        _make_config_for(m, "rated_torque")
    )
    r_stall = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert r_stall["tau_stall"] >= r_rated["tau_mtpa"]


def test_stall_ge_rated_salient():
    """Salient motor where I_stall > I_rated → tau_stall > tau_rated."""
    m = _make_motor(
        psi_f=0.545, L_d=0.036, L_q=0.051, n_p=3,
        I_rated=4.3 * sqrt(2), R_s=3.6,
        drive=DriveParams(U_DC=540.0, MAX_I_S=8.6),
    )
    r_rated = MODEL_REGISTRY["rated_torque"].fn(
        _make_config_for(m, "rated_torque")
    )
    r_stall = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert r_stall["tau_stall"] >= r_rated["tau_mtpa"]


# --- Error cases ---

def test_missing_r_s():
    m = _make_motor(R_s=None)
    with pytest.raises(ValueError, match="R_s"):
        prepare_stall_torque(m)


def test_missing_psi_f():
    m = Motor(
        name="bare", geometry=default_inrunner(), n_p=2,
        R_s=1.0,
    )
    with pytest.raises(ValueError, match="psi_f"):
        prepare_stall_torque(m)


# --- Saturation warning ---

def test_saturation_warning_high_ratio():
    """I_stall / I_rated > 3 → saturation_warning=True."""
    m = _make_motor(
        I_rated=1.0, R_s=0.1,
        drive=DriveParams(U_DC=540.0, MAX_I_S=20.0),
    )
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert result["I_stall"] / 1.0 > 3.0
    assert result["saturation_warning"] is True
    assert "linear_model_unreliable" not in result


def test_saturation_warning_low_ratio():
    """I_stall / I_rated <= 3 → saturation_warning=False."""
    m = _make_motor(
        I_rated=10.0, R_s=100.0,
        drive=DriveParams(U_DC=540.0, MAX_I_S=20.0),
    )
    p = prepare_stall_torque(m)
    assert p.I_stall / 10.0 <= 3.0
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert result["saturation_warning"] is False
    assert "linear_model_unreliable" not in result


def test_saturation_ratio_none_without_i_rated():
    """No I_rated on motor → saturation_ratio=None, warning=False."""
    m = _make_motor(R_s=1.0)  # no I_rated
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert result["saturation_ratio"] is None
    assert result["saturation_warning"] is False


# --- Electromagnetic stall metrics ---

def test_electromagnetic_stall_emitted():
    """tau_stall_electromagnetic and I_stall_electromagnetic in output."""
    m = _make_motor(R_s=1.0, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0))
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    expected_I_em = 540.0 / (sqrt(3) * 1.0)
    assert result["I_stall_electromagnetic"] == pytest.approx(expected_I_em, rel=1e-6)
    assert result["tau_stall_electromagnetic"] > 0
    assert result["tau_stall_electromagnetic"] > result["tau_stall"]


def test_electromagnetic_stall_absent_zero_rs():
    """R_s = 0 → no electromagnetic stall metrics."""
    m = _make_motor(R_s=0.0, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0))
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert "tau_stall_electromagnetic" not in result
    assert "I_stall_electromagnetic" not in result


# --- Registry integration ---

def test_registry_entry():
    info = MODEL_REGISTRY["stall_torque"]
    assert info.cost == "fast"
    assert info.source == "computed"
    assert {"tau_stall", "I_stall", "k_T", "k_T_effective", "gamma_opt_deg"}.issubset(info.produces)
    assert {"tau_stall_electromagnetic", "I_stall_electromagnetic"}.issubset(info.produces)
    assert "R_s" in info.needs

    m = _make_motor(R_s=1.0)
    info.validate(m)
    result = info.fn(_make_config(m))
    assert {"tau_stall", "I_stall", "k_T", "k_T_effective", "gamma_opt_deg"}.issubset(set(result.keys()))


# --- TOML motor integration ---

def test_creator_stall_torque():
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / "motors/creator_case_pmsm.toml")
    assert m.R_s is not None
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    # I_stall = MAX_I_S (no I_LIMIT set) = 0.42
    assert result["I_stall"] == pytest.approx(0.42, rel=1e-4)
    assert result["tau_stall"] > 0
    # Electromagnetic stall: U_DC/(sqrt(3)*R_s) >> MAX_I_S
    assert result["I_stall_electromagnetic"] > result["I_stall"]
    assert result["tau_stall_electromagnetic"] > result["tau_stall"]



# --- Curve output with caveat ---

def test_curve_present_with_saturation_warning():
    """Curve data is emitted even when saturation_warning=True."""
    m = _make_motor(
        I_rated=1.0, R_s=0.1, L_d=0.01, L_q=0.02,
        drive=DriveParams(U_DC=540.0, MAX_I_S=20.0),
    )
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert result["saturation_warning"] is True
    assert "I_curve" in result
    assert "tau_curve" in result
    assert "linear_model_unreliable" not in result


# --- Helper ---

def _make_config_for(motor, model):
    from phasesweep.sweep_types import RunConfig
    return RunConfig(motor=motor, model=model)


# --- Datasheet anchors (ETEL TMB reverse-salient torque motors) ---
#
# First stall_torque anchors against citable hardware, now spanning the
# five-frame TMB family. Tp for 0210/0360 is recovered from the Tp/Ip ratios
# recorded in the TOML notes (datasheet peak-duty column): TMB+0210
# Tp/Ip = 5.1 Nm/A_rms at Ip = 25.0 A_rms; TMB+0360 Tp/Ip = 17.4 at
# Ip = 25.3. 0140/0290/0450 quote Tp directly from the sheets. All five
# datasheets show heavy Kt collapse at Ip (vs the catalog Kt), so the
# linear-magnetics model must sit ABOVE Tp — these are bound anchors, not
# agreement anchors.

_ETEL_STALL_ANCHORS = [
    # (toml, Ip_Arms, Tp_Nm, tau_stall/Tp, saturation_ratio, warns)
    ("data/etel_tmb/etel_tmb0140_030_ra.toml", 15.8, 39.6, 1.881, 2.607, False),
    ("data/etel_tmb/etel_tmb0210_030_ta.toml", 25.0, 127.5, 1.408, 2.294, False),
    ("data/etel_tmb/etel_tmb0290_030_ra.toml", 20.6, 286.0, 1.428, 2.747, False),
    ("data/etel_tmb/etel_tmb0360_030_ta.toml", 25.3, 440.2, 2.016, 2.892, False),
    ("data/etel_tmb/etel_tmb0450_030_va.toml", 43.8, 719.0, 3.218, 3.650, True),
]


@pytest.mark.parametrize("toml,ip_rms,tp,over,sat,warns", _ETEL_STALL_ANCHORS)
def test_etel_linear_stall_bounds_datasheet_tp(toml, ip_rms, tp, over, sat,
                                               warns):
    """Linear-magnetics stall torque is an upper bound on datasheet Tp."""
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert result["I_stall"] == pytest.approx(sqrt(2) * ip_rms, rel=1e-3)
    assert result["tau_stall"] > tp
    # Reverse-salient MTPA branch is exercised at stall (magnetizing i_d)
    assert result["gamma_opt_deg"] < 0


@pytest.mark.parametrize("toml,ip_rms,tp,over,sat,warns", _ETEL_STALL_ANCHORS)
def test_etel_stall_overprediction_band_pinned(toml, ip_rms, tp, over, sat,
                                               warns):
    """Pin the over-prediction ratio so silent drift is visible.

    +41% (0210) to +222% (0450) over Tp — over-prediction grows with each
    frame's datasheet saturation level (0450 delivers only 1.9x Tc at 3.65x
    Ic). Direction and ordering are the physics; magnitudes are pinned
    observations.
    """
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert result["tau_stall"] / tp == pytest.approx(over, rel=0.02)


@pytest.mark.parametrize("toml,ip_rms,tp,over,sat,warns", _ETEL_STALL_ANCHORS)
def test_etel_saturation_heuristic_boundary(toml, ip_rms, tp, over, sat,
                                            warns):
    """Pin where the >3.0 saturation heuristic fires across the family.

    Four frames collapse Kt hard at Ip yet sit below the strict >3.0
    threshold, so saturation_warning stays False — the documented blind
    spot. The 0450 VA (Ip/Ic = 3.65) is the first anchor where the
    heuristic fires, and correctly so (tau_stall/Tp = 3.2 there). The bound
    framing (test above) is what covers all five machines; if the threshold
    is ever revisited, this pins the current behavior.
    """
    from phasesweep.machines.configs import load_motor
    m = load_motor(REPO_ROOT / toml)
    result = MODEL_REGISTRY["stall_torque"].fn(_make_config(m))
    assert result["saturation_ratio"] == pytest.approx(sat, rel=1e-3)
    assert result["saturation_warning"] is warns
