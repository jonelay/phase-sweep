"""Tests for the torque-speed envelope model."""

from itertools import pairwise
from math import pi, sqrt

import pytest

from phasesweep.machines.motor import DriveParams
from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque
from phasesweep.models.torque_speed import (
    base_speed,
    dq_voltage,
    envelope_at_speed,
    max_speed,
    run_torque_speed,
)
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solver_params import TorqueSpeedParams, prepare_torque_speed
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import make_motor

# Non-salient SPM reference case (R_s = 0 has textbook closed forms):
# n_p=4, psi_f=0.1 Wb, L=1e-3 H, I=50 A, U_max=100 V.
# Characteristic current psi_f/L_d = 100 A > I → bounded envelope.
_NP, _PSI, _L, _I, _U = 4, 0.1, 1e-3, 50.0, 100.0


def _spm_args(R_s=0.0):
    return (_NP, _PSI, R_s, _L, _L)


class TestParams:

    def test_missing_inductance_raises(self):
        motor = make_motor(L_d=None, L_q=None)
        with pytest.raises(ValueError, match="torque_speed needs L_d, L_q"):
            prepare_torque_speed(motor)

    def test_missing_r_s_raises(self):
        motor = make_motor(R_s=None)
        with pytest.raises(ValueError, match="torque_speed needs R_s"):
            prepare_torque_speed(motor)

    def test_i_peak_prefers_i_limit(self):
        motor = make_motor(drive=DriveParams(U_DC=540.0, MAX_I_S=20.0,
                                             W_REF=100.0, I_LIMIT=7.0))
        assert prepare_torque_speed(motor).I_peak == 7.0

    def test_i_peak_falls_back_to_max_i_s(self):
        motor = make_motor()
        assert prepare_torque_speed(motor).I_peak == motor.drive.MAX_I_S

    def test_u_max_is_svpwm_phase_peak(self):
        motor = make_motor()
        p = prepare_torque_speed(motor)
        assert p.U_max == pytest.approx(motor.drive.U_DC / sqrt(3))

    def test_i_cont_from_i_rated(self):
        assert prepare_torque_speed(make_motor(I_rated=5.0)).I_cont == 5.0
        assert prepare_torque_speed(make_motor()).I_cont is None

    def test_voltage_below_resistive_drop_raises(self):
        with pytest.raises(ValueError, match="resistive"):
            TorqueSpeedParams(n_p=4, psi_f=0.1, R_s=1.0, L_d=1e-3, L_q=1e-3,
                              I_peak=50.0, U_max=10.0)

    def test_registry_validate_wired(self):
        with pytest.raises(ValueError, match="torque_speed needs"):
            MODEL_REGISTRY["torque_speed"].validate(make_motor(L_d=None))


class TestSpmClosedForm:
    """R_s = 0 non-salient case against textbook field-weakening formulas."""

    def test_base_speed(self):
        # A * w_e^2 = U^2 with A = psi_f^2 + (L I)^2
        w_e = _U / sqrt(_PSI**2 + (_L * _I) ** 2)
        assert base_speed(*_spm_args(), _I, _U) == pytest.approx(w_e / _NP)

    def test_max_speed(self):
        w_e = _U / (_PSI - _L * _I)
        assert max_speed(*_spm_args(), _I, _U) == pytest.approx(w_e / _NP)

    def test_constant_torque_below_base(self):
        w_base = base_speed(*_spm_args(), _I, _U)
        for frac in (0.0, 0.5, 0.99):
            tau, i_d, i_q = envelope_at_speed(*_spm_args(), _I, _U,
                                              frac * w_base)
            assert tau == pytest.approx(1.5 * _NP * _PSI * _I)
            assert i_d == pytest.approx(0.0)
            assert i_q == pytest.approx(_I)

    def test_field_weakening_matches_closed_form(self):
        w_base = base_speed(*_spm_args(), _I, _U)
        w_max = max_speed(*_spm_args(), _I, _U)
        for frac in (1.2, 1.6, 2.0):  # CPSR here is 2.236
            w = frac * w_base
            assert w < w_max
            w_e = _NP * w
            i_d_ref = ((_U / w_e) ** 2 - _PSI**2 - (_L * _I) ** 2) / (2 * _PSI * _L)
            i_q_ref = sqrt(_I**2 - i_d_ref**2)
            tau_ref = 1.5 * _NP * _PSI * i_q_ref
            tau, i_d, i_q = envelope_at_speed(*_spm_args(), _I, _U, w)
            assert tau == pytest.approx(tau_ref, rel=1e-6)
            assert i_d == pytest.approx(i_d_ref, rel=1e-4)
            assert i_q == pytest.approx(i_q_ref, rel=1e-5)

    def test_zero_torque_at_max_speed(self):
        w_max = max_speed(*_spm_args(), _I, _U)
        # tau falls off as sqrt(speed margin): 0.05% below w_max → ~2% tau
        tau_near, _, _ = envelope_at_speed(*_spm_args(), _I, _U, 0.9995 * w_max)
        tau_past, _, _ = envelope_at_speed(*_spm_args(), _I, _U, 1.05 * w_max)
        assert 0 < tau_near < 0.05 * 1.5 * _NP * _PSI * _I
        assert tau_past == 0.0

    def test_envelope_monotone_nonincreasing(self):
        w_max = max_speed(*_spm_args(), _I, _U)
        taus = [envelope_at_speed(*_spm_args(), _I, _U, w_max * k / 40)[0]
                for k in range(41)]
        for a, b in pairwise(taus):
            assert b <= a * (1 + 1e-9)

    def test_solution_respects_both_limits(self):
        w_base = base_speed(*_spm_args(0.5), _I, _U)
        for frac in (0.5, 1.1, 1.8, 3.0):
            w = frac * w_base
            tau, i_d, i_q = envelope_at_speed(*_spm_args(0.5), _I, _U, w)
            if tau == 0.0:
                continue
            assert sqrt(i_d**2 + i_q**2) <= _I * (1 + 1e-9)
            assert dq_voltage(*_spm_args(0.5), i_d, i_q, w) <= _U * (1 + 1e-6)


class TestSalient:

    def test_below_base_matches_rated_torque_mtpa(self):
        # Forward-salient: exact MTPA branch, bit-identical to rated_torque
        n_p, psi_f, R_s, L_d, L_q, I_s = 3, 0.05, 0.1, 2e-3, 4e-3, 10.0
        g = mtpa_gamma(psi_f, L_d, L_q, I_s)
        tau_ref = mtpa_torque(n_p, psi_f, L_d, L_q, I_s, g)
        tau, _, _ = envelope_at_speed(n_p, psi_f, R_s, L_d, L_q, I_s, 200.0, 10.0)
        assert tau == tau_ref

    def test_continuous_across_base_speed(self):
        n_p, psi_f, R_s, L_d, L_q, I_s, U = 3, 0.05, 0.1, 2e-3, 4e-3, 10.0, 30.0
        w_b = base_speed(n_p, psi_f, R_s, L_d, L_q, I_s, U)
        tau_lo, _, _ = envelope_at_speed(n_p, psi_f, R_s, L_d, L_q, I_s, U,
                                         0.999 * w_b)
        tau_hi, _, _ = envelope_at_speed(n_p, psi_f, R_s, L_d, L_q, I_s, U,
                                         1.001 * w_b)
        assert tau_hi == pytest.approx(tau_lo, rel=1e-2)
        assert tau_hi <= tau_lo

    def test_reverse_salient_uses_magnetizing_id(self):
        # ETEL-like L_d > L_q: MTPA sits at i_d > 0 below base speed
        n_p, psi_f, R_s, L_d, L_q, I_s, U = 22, 0.2, 1.0, 25e-3, 20e-3, 15.0, 350.0
        tau, i_d, _ = envelope_at_speed(n_p, psi_f, R_s, L_d, L_q, I_s, U, 1.0)
        g = mtpa_gamma(psi_f, L_d, L_q, I_s)
        assert g < 0
        assert i_d > 0
        assert tau == mtpa_torque(n_p, psi_f, L_d, L_q, I_s, g)

    def test_unbounded_cpsr_when_char_current_within_limit(self):
        # psi_f/L_d = 20 A <= I = 50 A → MTPV branch, no zero crossing
        n_p, psi_f, R_s, L_d, L_q, I_s, U = 4, 0.02, 0.0, 1e-3, 2e-3, 50.0, 100.0
        assert max_speed(n_p, psi_f, R_s, L_d, L_q, I_s, U) is None
        w_b = base_speed(n_p, psi_f, R_s, L_d, L_q, I_s, U)
        tau, i_d, i_q = envelope_at_speed(n_p, psi_f, R_s, L_d, L_q, I_s, U,
                                          20.0 * w_b)
        assert tau > 0.0
        assert sqrt(i_d**2 + i_q**2) < I_s  # MTPV: interior to current circle


class TestRunner:

    def test_produces_peak_keys(self):
        rc = RunConfig(motor=make_motor(), model="torque_speed")
        r = run_torque_speed(rc)
        for key in ("speed_curve", "tau_curve_peak", "base_speed_peak",
                    "max_speed_peak", "p_max_peak", "u_max", "I_env_peak"):
            assert key in r
        assert len(r["speed_curve"]) == len(r["tau_curve_peak"])
        assert "tau_curve_cont" not in r  # no I_rated on make_motor()

    def test_cont_envelope_with_i_rated(self):
        rc = RunConfig(motor=make_motor(I_rated=10.0), model="torque_speed")
        r = run_torque_speed(rc)
        assert r["I_env_cont"] == 10.0
        assert r["I_env_peak"] == 20.0
        # Lower current → voltage limit reached later
        assert r["base_speed_cont"] > r["base_speed_peak"]
        for t_c, t_p in zip(r["tau_curve_cont"], r["tau_curve_peak"]):
            assert t_c <= t_p * (1 + 1e-9)
        assert r["p_max_cont"] <= r["p_max_peak"]

    def test_grid_spans_bounded_envelope(self):
        rc = RunConfig(motor=make_motor(I_rated=10.0), model="torque_speed")
        r = run_torque_speed(rc)
        w_ends = [w for w in (r["max_speed_peak"], r["max_speed_cont"])
                  if w is not None]
        assert r["speed_curve"][0] == 0.0
        assert r["speed_curve"][-1] == pytest.approx(max(w_ends))

    def test_registry_fn_dispatch(self):
        rc = RunConfig(motor=make_motor(), model="torque_speed")
        r = MODEL_REGISTRY["torque_speed"].fn(rc)
        assert r["base_speed_peak"] > 0

    def test_hash_sensitive_to_u_dc_not_fem_params(self):
        fields = MODEL_REGISTRY["torque_speed"].hash_fields
        m1 = make_motor()
        m2 = make_motor(drive=DriveParams(U_DC=48.0, MAX_I_S=20.0,
                                          W_REF=m1.drive.W_REF))
        rc1 = RunConfig(motor=m1, model="torque_speed")
        rc2 = RunConfig(motor=m2, model="torque_speed")
        assert compute_run_id(rc1, fields) != compute_run_id(rc2, fields)
        rc3 = RunConfig(motor=m1, model="torque_speed", maxh_fraction=0.10)
        assert compute_run_id(rc1, fields) == compute_run_id(rc3, fields)


# ---------------------------------------------------------------------------
# ETEL TMB torque-speed envelope anchors — first anchor class
# ---------------------------------------------------------------------------
# The TMB datasheets publish the voltage-limit corner points the model
# computes: nb "base speed" (at the rated current In), nm "maximum speed
# without flux weakening" (no-load), and the rated point (nn, Tn, In).
# U_max = 600/sqrt(3) V from the sheets' nominal DC bus.
#
# Two regimes, kept deliberately separate:
#  - nm and nb(In) are at/below rated current where the linear inductances
#    hold — agreement anchors.
#  - nb,i / nb,p (intermittent/peak current) sit at 1.6-3.7x Ic where the
#    sheets themselves show Kt collapse; the model misses by up to ~40% in
#    BOTH directions there, so they are documented in the TOMLs but not
#    anchored (same saturation blind spot as the stall Tp band).

_RPM = 60.0 / (2.0 * 3.141592653589793)

# (toml, In_Arms, nb_rpm, nm_rpm, Tn_Nm, nb_model/nb_datasheet)
_ETEL_ENVELOPE_ANCHORS = [
    ("data/etel_tmb/etel_tmb0140_030_ra.toml", 3.32, 1280.0, 1570.0, 12.2, 0.9739),
    ("data/etel_tmb/etel_tmb0210_030_ta.toml", 7.53, 817.0, 983.0, 46.0, 0.9593),
    ("data/etel_tmb/etel_tmb0290_030_ra.toml", 7.39, 238.0, 350.0, 124.0, 0.9886),
    ("data/etel_tmb/etel_tmb0360_030_ta.toml", 8.54, 119.0, 211.0, 216.0, 0.9492),
    ("data/etel_tmb/etel_tmb0450_030_va.toml", 11.4, 97.1, 167.0, 344.0, 0.8870),
]


def _etel_envelope_args(toml):
    from phasesweep.machines.configs import load_motor
    from tests.conftest import REPO_ROOT
    m = load_motor(REPO_ROOT / toml)
    return m, (m.n_p, m.psi_f, m.R_s, m.L_d, m.L_q), m.drive.U_DC / sqrt(3)


def test_etel_no_fw_max_speed_systematic_band():
    """nm reproduced +2.0..+2.4% high across all five frames.

    The zero-current voltage limit U_max/(n_p psi_f) overshoots the
    datasheet nm by a tight, uniform margin — ETEL reserves ~2% voltage/EMF
    headroom. Tight-band framing mirrors _NAMEPLATE_ANCHORS: an anchor
    outside 1.015..1.030, or a family spread above 0.005, breaks the
    systematic-margin attribution and must be investigated, not widened.
    """
    ratios = []
    for toml, in_rms, nb, nm, tn, nb_ratio in _ETEL_ENVELOPE_ANCHORS:
        _, args, u_max = _etel_envelope_args(toml)
        nm_model = base_speed(*args, 0.0, u_max) * _RPM
        ratios.append(nm_model / nm)
    assert all(1.015 < r < 1.030 for r in ratios), ratios
    assert max(ratios) - min(ratios) < 0.005, ratios


@pytest.mark.parametrize("toml,in_rms,nb,nm,tn,nb_ratio",
                         _ETEL_ENVELOPE_ANCHORS)
def test_etel_base_speed_at_rated_current(toml, in_rms, nb, nm, tn, nb_ratio):
    """Base speed at In under-predicts datasheet nb by 1-11%, pinned.

    Direction is physics: the linear (unsaturated) inductances overstate
    the w_e*L*i voltage drop, so the model's voltage-limit corner comes
    early; the miss grows with each frame's datasheet saturation level
    (largest on 0450 VA, Tc/(Kt*Ic) = 0.76). Magnitudes are pinned
    observations; the +2% ETEL voltage margin (see the nm band test) is
    folded into these ratios.
    """
    _, args, u_max = _etel_envelope_args(toml)
    nb_model = base_speed(*args, sqrt(2) * in_rms, u_max) * _RPM
    assert nb_model < nb  # one-sided at rated current, all five frames
    assert nb_model / nb == pytest.approx(nb_ratio, abs=0.005)


@pytest.mark.parametrize("toml,in_rms,nb,nm,tn,nb_ratio",
                         _ETEL_ENVELOPE_ANCHORS)
def test_etel_rated_point_is_voltage_feasible(toml, in_rms, nb, nm, tn,
                                              nb_ratio):
    """The datasheet rated point (nn, Tn, In) lies inside the model envelope.

    nn sits below the model's base speed at In on every frame (ETEL rates
    with speed margin), and the envelope torque at nn exceeds Tn — the
    feasibility statement this test exists to make. Tn itself is 15-40%
    below the MTPA torque at In (rating margin + saturation), so only the
    inequality is claimed.
    """
    m, args, u_max = _etel_envelope_args(toml)
    in_pk = sqrt(2) * in_rms
    nn_rad_s = m.drive.W_REF  # W_REF = rated speed nn in the ETEL TOMLs
    assert nn_rad_s < base_speed(*args, in_pk, u_max)
    tau_at_nn, _, _ = envelope_at_speed(*args, in_pk, u_max, nn_rad_s)
    assert tau_at_nn > tn


# ---------------------------------------------------------------------------
# Non-ETEL envelope checks: Kollmorgen B-104-B + CREATOR
# ---------------------------------------------------------------------------
# Negative finding: neither catalog rates its maximum speed at the
# voltage corner, so the ETEL nm agreement class does NOT generalize.
# Direct-drive torque motors (ETEL) are rated to the voltage limit;
# the servo catalog caps N Max at a mechanical/rating ceiling and the
# CREATOR benchmark at its inverter frequency limit. Claims here are
# one-sided feasibility plus pinned margins.


def _envelope_args(toml):
    from phasesweep.machines.configs import load_motor
    from tests.conftest import REPO_ROOT
    m = load_motor(REPO_ROOT / toml)
    return m, (m.n_p, m.psi_f, m.R_s, m.L_d, m.L_q), m.drive.U_DC / sqrt(3)


class TestKollmorgenEnvelope:
    TOML = "data/kollmorgen_b104b/kollmorgen_b104b_ipm.toml"

    def test_rated_point_voltage_feasible(self):
        """The datasheet rated point (7500 rpm, Tc = 1.57 Nm, Ic) is inside
        the model envelope: MTPA at Ic is still voltage-feasible at N Max
        (the corner sits 24% above), and the envelope torque there exceeds
        Tc. Extends the ETEL rated-point feasibility claim to the thesis
        Ld/Lq + VMax 250 V drive."""
        m, args, u_max = _envelope_args(self.TOML)
        nb_ratio = base_speed(*args, m.I_rated, u_max) * _RPM / 7500.0
        assert nb_ratio == pytest.approx(1.237, abs=0.005)
        tau, _, _ = envelope_at_speed(*args, m.I_rated, u_max, 7500.0 / _RPM)
        assert tau > 1.57

    def test_n_max_is_not_the_voltage_corner(self):
        """N Rated = N Max = 7500 on the sheet, but the no-load voltage
        limit sits 40% above — Kollmorgen's N Max is a mechanical/rating
        ceiling, NOT an ETEL-style voltage corner. One-sided: no nm
        agreement anchor exists in this catalog."""
        _, args, u_max = _envelope_args(self.TOML)
        nl_ratio = base_speed(*args, 0.0, u_max) * _RPM / 7500.0
        assert nl_ratio == pytest.approx(1.400, abs=0.005)


class TestCreatorEnvelope:
    TOML = "motors/creator_case_pmsm.toml"

    def test_max_speed_is_frequency_not_voltage_limited(self):
        """Catalog max speed 7050 rpm IS the 235 Hz inverter frequency
        limit exactly (2*pi*235/n_p); the no-load voltage limit at the
        326 V max DC link sits 11% above it (one-sided headroom)."""
        m, args, u_max = _envelope_args(self.TOML)
        assert 2.0 * pi * 235.0 / m.n_p * _RPM == pytest.approx(7050.0)
        nl_ratio = base_speed(*args, 0.0, u_max) * _RPM / 7050.0
        assert nl_ratio > 1.0
        assert nl_ratio == pytest.approx(1.114, abs=0.005)

    def test_catalog_points_inside_envelope(self):
        """Rated (2000 rpm, 0.10 Nm at I_nom), max-torque (0.15 Nm at
        I_max) and max-power (70 W) catalog points all lie inside the
        model envelope at the 326 V DC link. At max speed the envelope
        still carries 0.146 Nm (pinned) — characteristic current
        psi_f/L_d = 0.56 A exceeds I_max = 0.42 A, so the CPSR is wide
        and torque barely droops across the whole catalog speed range."""
        m, args, u_max = _envelope_args(self.TOML)
        w_rated = 2000.0 / _RPM
        w_top = 7050.0 / _RPM
        tau_nom, _, _ = envelope_at_speed(*args, m.I_rated, u_max, w_rated)
        tau_max, _, _ = envelope_at_speed(*args, m.drive.MAX_I_S, u_max,
                                          w_rated)
        tau_top, _, _ = envelope_at_speed(*args, m.drive.MAX_I_S, u_max,
                                          w_top)
        assert tau_nom > 0.10
        assert tau_max > 0.15
        assert tau_top == pytest.approx(0.1456, abs=0.002)
        assert tau_top * w_top > 70.0


# ---------------------------------------------------------------------------
# Kollmorgen AKM44H: multi-voltage rated points
# ---------------------------------------------------------------------------
# First anchor with the SAME winding's rated point published at four DC bus
# voltages (catalog AKM4x performance data). The Nrtd progression is
# voltage-set by construction — a mechanical or inverter-frequency ceiling
# could not scale with Vbus — which supplies the generalization evidence
# the ETEL-only corner class lacked. Kollmorgen rates Nrtd with a
# large drive/system margin under U_max = U_DC/sqrt(3) (voltage utilization
# at the published points 0.71-0.85, rising with bus), so claims are
# one-sided feasibility plus pinned margins, not ETEL-style agreement.
# Rated current at Nrtd is not printed; derived as Trtd / Kt with the hot
# catalog Kt (Trtd is a hot rating, note 1).

_AKM44H_KT_HOT = 1.06  # Nm/Arms, catalog Kt at dT = 100 C (note 1)

# (Vbus_Vdc, Trtd_Nm, Nrtd_rpm, nb_model/Nrtd)
_AKM44H_CORNERS = [
    (160.0, 5.44, 1000.0, 1.455),
    (320.0, 4.66, 2500.0, 1.239),
    (560.0, 3.48, 4500.0, 1.249),
    (640.0, 2.93, 5500.0, 1.180),
]


class TestAkm44hEnvelope:
    TOML = "data/kollmorgen_akm44h/kollmorgen_akm44h.toml"

    @pytest.mark.parametrize("vbus,trtd,nrtd,nb_ratio", _AKM44H_CORNERS)
    def test_rated_points_voltage_feasible(self, vbus, trtd, nrtd, nb_ratio):
        """Every published rated point lies inside the model envelope at
        its own bus voltage, with the model corner 18-46% above Nrtd
        (pinned). The margin shrinks as Vbus rises (fixed drive drops and
        current-loop headroom are a larger fraction of a small bus), so no
        single utilization factor exists — one-sided plus pins only."""
        _, args, _ = _envelope_args(self.TOML)
        u_max = vbus / sqrt(3)
        i_pk = sqrt(2) * trtd / _AKM44H_KT_HOT
        nb_model = base_speed(*args, i_pk, u_max) * _RPM
        assert nb_model > nrtd
        assert nb_model / nrtd == pytest.approx(nb_ratio, abs=0.005)
        tau, _, _ = envelope_at_speed(*args, i_pk, u_max, nrtd / _RPM)
        assert tau > trtd

    def test_corners_are_voltage_set_not_a_ceiling(self):
        """The published rated speeds cannot share a mechanical or
        inverter-frequency ceiling: they rise monotonically with Vbus,
        all sit below the 6000 rpm mechanical cap, and the model's
        NO-LOAD voltage limit at each of the two lower buses falls short
        of the next bus's published rated speed (1664 < 2500 at 160 Vdc,
        3328 < 4500 at 320 Vdc) — at the lower bus the motor could not
        reach the next corner even unloaded. This is the multi-voltage
        discriminator the single-voltage catalogs (B-104-B, CREATOR)
        structurally cannot provide."""
        _, args, _ = _envelope_args(self.TOML)
        speeds = [nrtd for _, _, nrtd, _ in _AKM44H_CORNERS]
        assert speeds == sorted(speeds) and speeds[-1] < 6000.0
        for (v_lo, _, _, _), (_, _, n_hi, _) in pairwise(_AKM44H_CORNERS[:3]):
            nl = base_speed(*args, 0.0, v_lo / sqrt(3)) * _RPM
            assert nl < n_hi


# ---------------------------------------------------------------------------
# ABB BSM50N-233: two-voltage rated speeds
# ---------------------------------------------------------------------------
# The brochure prints the same winding's rated speed at two voltages as
# table numbers ("Rated speed @ 300 volts: 7500 rpm" / "@ 160 volts:
# 4000 rpm") with 7500/4000 = 300/160 EXACTLY — rated speed proportional
# to voltage, the sharpest voltage-set evidence in the registry (a
# mechanical or frequency ceiling cannot scale with supply; max speed is
# a separate 10,000 rpm line). No torque/current is printed AT the rated
# speed, so claims are no-load margins plus stall-current feasibility.
# The series overview lists the same 7500 rpm at "320 Vdc" — the exact
# 1.875 ratio pins 300/160 as the effective pair (see the TOML header).

# (V, Nrtd_rpm, nb_model@Ics / Nrtd)
_ABB_BSM_CORNERS = [(160.0, 4000.0, 1.122), (300.0, 7500.0, 1.171)]


class TestAbbBsmEnvelope:
    TOML = "data/abb_bsm50n/abb_bsm50n233.toml"

    def test_rated_speeds_proportional_to_voltage(self):
        """The published speeds are exactly proportional to the published
        voltages, and the model's no-load voltage limit sits the IDENTICAL
        +25.0% above the rated speed at both buses — at no load the model
        corner is itself proportional to U, so exact proportionality in
        the data transfers to a zero-spread margin pair. ABB reserves a
        uniform ~20% no-load voltage/EMF headroom (pinned)."""
        assert 7500.0 / 4000.0 == 300.0 / 160.0
        _, args, _ = _envelope_args(self.TOML)
        ratios = [base_speed(*args, 0.0, v / sqrt(3)) * _RPM / nrtd
                  for v, nrtd, _ in _ABB_BSM_CORNERS]
        assert all(r == pytest.approx(1.250, abs=0.005) for r in ratios)
        assert max(ratios) - min(ratios) < 1e-9

    @pytest.mark.parametrize("vbus,nrtd,nb_ratio", _ABB_BSM_CORNERS)
    def test_rated_speed_feasible_at_full_stall_current(self, vbus, nrtd,
                                                        nb_ratio):
        """Even carrying the full hot continuous stall current (2.87 Arms,
        the sheet's only continuous current), the model base speed exceeds
        the published rated speed at both voltages (+12/+17%, pinned) —
        the rated point is voltage-feasible under any continuous load.
        Voltage utilization at (Nrtd, Ics) is 0.86-0.90, a tighter rating
        margin than Kollmorgen's 0.71-0.85."""
        m, args, _ = _envelope_args(self.TOML)
        nb_model = base_speed(*args, m.I_rated, vbus / sqrt(3)) * _RPM
        assert nb_model > nrtd
        assert nb_model / nrtd == pytest.approx(nb_ratio, abs=0.005)


# ---------------------------------------------------------------------------
# Tecnotion QTR-A-105-25-N: two-voltage LOADED corners
# ---------------------------------------------------------------------------
# The spec sheet tabulates "Maximum speed @ 48 Volt @ Tc" and "Maximum
# speed @ max. voltage @ Tc" (325 Vdc) for the same winding — the only
# anchor whose two published corners carry load (continuous torque), the
# direct two-voltage analog of ETEL's nb. The rating rows are hot-R
# quantities (coils @ 100 C): the printed continuous power loss closes
# with R(100 C) to +1.6% while cold R misses by -24%, so the corner
# reconstruction below uses R25 * (1 + 0.00393 * 75). The sheet's L is
# quoted valid for I < 0.6*Ip while Ic = 0.70*Ip — a mild saturation
# flag, largest at the reactance-dominated 325 V corner.

_QTR_R_HOT = 1.0 + 0.00393 * 75.0  # coils @ 100 C over the 25 C R quote

# (U_DC, max_speed_at_Tc_rpm, nb_model_hotR / published)
_QTR_CORNERS = [(48.0, 240.0, 1.090), (325.0, 3625.0, 1.110)]


class TestTecnotionQtrEnvelope:
    TOML = "data/tecnotion_qtr/tecnotion_qtr_a_105_25_n.toml"

    def _hot_args(self):
        m, (n_p, psi_f, r_s, l_d, l_q), _ = _envelope_args(self.TOML)
        return m, (n_p, psi_f, r_s * _QTR_R_HOT, l_d, l_q)

    def test_pc_pins_the_hot_r_reconstruction(self):
        """The printed continuous power loss Pc = 214 W (coils @ 100 C)
        equals 3*R(100 C)*Ic^2 to +1.6% — with the cold 25 C resistance it
        would read 163 W (-24%). This closes the sheet's own numbers only
        with hot R and licenses the hot-R corner tests below."""
        m, (_, _, r_hot, _, _) = self._hot_args()
        pc_model = 3.0 * r_hot * 5.3**2
        assert pc_model / 214.0 == pytest.approx(0.984, abs=0.005)

    @pytest.mark.parametrize("udc,nmax,nb_ratio", _QTR_CORNERS)
    def test_max_speed_at_tc_two_voltages(self, udc, nmax, nb_ratio):
        """Base speed at Ic with hot R sits a consistent +9.0/+11.0% above
        the published loaded corner at BOTH buses (pinned) — the tightest
        non-ETEL margins in the registry. The near-uniformity across a
        6.8x voltage span is itself evidence the published speeds are
        voltage corners with one vendor margin. Cold R would put the 48 V
        corner off by +52% (that corner is IR-dominated: EMF is only a
        quarter of the voltage budget) — the hot/cold trap this sheet's
        Pc row resolves."""
        m, args = self._hot_args()
        nb_model = base_speed(*args, m.I_rated, udc / sqrt(3)) * _RPM
        assert nb_model > nmax
        assert nb_model / nmax == pytest.approx(nb_ratio, abs=0.005)

    def test_corners_cannot_share_a_ceiling(self):
        """The model's no-load limit at 48 V (941 rpm) is far below the
        published 325 V loaded corner (3625 rpm), and both corners sit far
        below the series' 16,500 rpm mechanical cap — the two published
        speeds cannot be one mechanical or frequency ceiling."""
        m, args = self._hot_args()
        nl_48 = base_speed(*args, 0.0, 48.0 / sqrt(3)) * _RPM
        assert nl_48 == pytest.approx(941.0, abs=1.0)
        assert nl_48 < 3625.0 < 16500.0


# ---------------------------------------------------------------------------
# Beckhoff AM8051-E: four-voltage S1 nominal points
# ---------------------------------------------------------------------------
# The operation manual tabulates the same winding's S1 nominal point at
# four MAINS voltages (UN = 115/230/400/480 VAC -> 500/1400/2500/3000
# rpm), all far below the 9000 rpm mechanical cap. Corners are
# reconstructed under the documented rectifier assumption U_DC =
# sqrt(2)*UN (no DC bus value is printed in the motor manual). KT is hot
# by the manual's blanket statement (M0 = I0*KT closes to -0.4%), KE is
# 20 C — corner currents derive as Mn/KT except at 400 V where In = 2.55 A
# is printed (Mn/KT = 2.60, +1.9%).

_AM8051_KT_HOT = 1.77  # Nm/Arms, manual KTrms (hot by definition)

# (UN_VAC, Mn_Nm, Nn_rpm, In_printed_or_None, nb_model/Nn)
_AM8051_CORNERS = [
    (115.0, 4.80, 500.0, None, 1.354),
    (230.0, 4.70, 1400.0, None, 1.113),
    (400.0, 4.60, 2500.0, 2.55, 1.147),
    (480.0, 4.50, 3000.0, None, 1.160),
]


class TestBeckhoffAm8051Envelope:
    TOML = "data/beckhoff_am8051/beckhoff_am8051e.toml"

    @pytest.mark.parametrize("un,mn,nn,in_printed,nb_ratio",
                             _AM8051_CORNERS)
    def test_nominal_points_voltage_feasible(self, un, mn, nn, in_printed,
                                             nb_ratio):
        """Every S1 nominal point lies inside the model envelope at its
        rectified bus, with the model corner +11..+16% above Nn at the
        three upper voltages and +35% at the 115 V outlier (fixed drive
        drops loom largest on the smallest bus — the same low-voltage
        outlier shape as Kollmorgen's 160 Vdc point). Pinned; one-sided
        plus pins, as for the other three vendors."""
        _, args, _ = _envelope_args(self.TOML)
        u_max = sqrt(2) * un / sqrt(3)
        i_rms = in_printed if in_printed else mn / _AM8051_KT_HOT
        i_pk = sqrt(2) * i_rms
        nb_model = base_speed(*args, i_pk, u_max) * _RPM
        assert nb_model > nn
        assert nb_model / nn == pytest.approx(nb_ratio, abs=0.005)
        tau, _, _ = envelope_at_speed(*args, i_pk, u_max, nn / _RPM)
        assert tau > mn

    def test_nominal_speeds_are_voltage_set_not_a_ceiling(self):
        """Nn rises monotonically with UN, every Nn sits far below the
        9000 rpm mechanical cap, and the model's NO-LOAD limit at each of
        the two lower mains falls short of the next mains' published Nn
        (920 < 1400 rpm at 115 V; 1840 < 2500 rpm at 230 V) — the four
        published nominal speeds cannot share a mechanical or frequency
        ceiling. Same discriminator as the AKM44H anchor, here on mains
        voltages via the sqrt(2) rectifier assumption."""
        _, args, _ = _envelope_args(self.TOML)
        speeds = [nn for _, _, nn, _, _ in _AM8051_CORNERS]
        assert speeds == sorted(speeds) and speeds[-1] < 9000.0
        for (un_lo, _, _, _, _), (_, _, nn_hi, _, _) in \
                pairwise(_AM8051_CORNERS[:3]):
            nl = base_speed(*args, 0.0, sqrt(2) * un_lo / sqrt(3)) * _RPM
            assert nl < nn_hi


# ---------------------------------------------------------------------------
# UW/ORNL 6-kW FSCW SPM: measured field-weakening envelope
# ---------------------------------------------------------------------------
# ORNL/TM-2005/183 Table 7 publishes four MEASURED verification points
# (800/2000/3000/4000 rpm) with per-point phase voltage, d/q currents,
# and shaft torque — the only anchor exercising the field-weakening
# branch (all other envelope anchors test the voltage corner). The
# anchor L is 1.3 mH (ORNL SPEED value): the analytical AND measured FW
# i_d plateau sits at -psi/L = -26.8 Arms, which only L = 1.3 mH
# reproduces (the report's Table 4 1.03 mH would put the plateau at
# -33.8 and over-predict the measured envelope by +35..48%). Torque
# claims are one-sided: the model is the lossless electromagnetic
# optimum at the row's own measured voltage and the 43 Arms current
# limit; measured shaft torque includes iron/friction/harmonic losses
# (substantial in a 30-pole bonded-magnet machine at 1-2 kHz electrical).

# Table 7 experimental FW rows: (rpm, Vphase_rms, Tout_Nm, tau_model/Tout)
_ORNL_FW_POINTS = [
    (2000.0, 89.0, 29.0, 1.154),
    (3000.0, 91.0, 19.4, 1.177),
    (4000.0, 91.0, 14.7, 1.165),
]


class TestOrnlFscwEnvelope:
    TOML = "data/ornl_fscw/ornl_fscw_6kw_spm.toml"

    def test_corner_mtpa_vs_measured_shaft_torque(self):
        """At the 800 rpm corner the machine ran full rated current at
        i_d = 0 (measured); the model MTPA torque 1.5*n_p*psi_f*I sits
        +5.2% above the measured 64 Nm shaft torque (pinned) —
        electromagnetic optimum vs shaft, the expected sign and size."""
        m, _, _ = _envelope_args(self.TOML)
        tau = 1.5 * m.n_p * m.psi_f * m.I_rated
        assert tau / 64.0 == pytest.approx(1.052, abs=0.005)

    @pytest.mark.parametrize("rpm,v_rms,tout,tau_ratio", _ORNL_FW_POINTS)
    def test_fw_envelope_one_sided_above_measured(self, rpm, v_rms, tout,
                                                  tau_ratio):
        """Deep in field weakening (2.5-5x base speed) the model envelope
        torque at each row's own measured phase voltage exceeds the
        measured shaft torque by a consistent +15..18% band (pinned per
        point) — the lossless-optimum gap. First exercise of the FW
        solver branch against measured data."""
        m, args, _ = _envelope_args(self.TOML)
        u_max = sqrt(2) * v_rms
        tau, _, _ = envelope_at_speed(*args, m.I_rated, u_max, rpm / _RPM)
        assert tau > tout
        assert tau / tout == pytest.approx(tau_ratio, abs=0.005)

    def test_fw_id_plateau_agreement(self):
        """AGREEMENT anchor for the FW current placement: on the voltage
        boundary the model rides i_d = -psi_f/L = -26.8 Arms, and the
        measured 2000 rpm point sits at -26.9 Arms (<1%). This is what
        pins L = 1.3 mH as the physical synchronous inductance (1.03 mH
        would demand -33.8). The measured 3000/4000 rpm rows back off to
        i_d ~ -14.7 (controller choice, total current drops to
        ~23 Arms) and anchor torque feasibility only."""
        m, args, _ = _envelope_args(self.TOML)
        _, i_d, _ = envelope_at_speed(*args, m.I_rated, sqrt(2) * 89.0,
                                      2000.0 / _RPM)
        assert i_d / sqrt(2) == pytest.approx(-26.9, abs=0.5)
        assert m.psi_f / m.L_d / sqrt(2) == pytest.approx(26.8, abs=0.1)

    def test_cpsr_6_design_goal_reproduced(self):
        """The report states the CPSR = 6 design goal was met: at 6x the
        800 rpm corner and the 87.7 Vrms design voltage the model
        envelope still delivers 13.7 Nm = 6.9 kW > the 6 kW rating
        (characteristic current 26.8 Arms < I_max 43 -> wide CPSR)."""
        m, args, _ = _envelope_args(self.TOML)
        w = 4800.0 / _RPM
        tau, _, _ = envelope_at_speed(*args, m.I_rated, sqrt(2) * 87.7, w)
        assert tau == pytest.approx(13.74, abs=0.05)
        assert tau * w > 6000.0


# ---------------------------------------------------------------------------
# Rexroth MS2N04-D0BHN: precision voltage corner
# ---------------------------------------------------------------------------
# One numeric corner at one bus — but the manual STATES the mechanism in
# prose (section 6.3: "Rated speed is determined by the DC bus voltage
# UZK1"), where every other vendor anchor infers it from multi-voltage
# scaling, and prints the corner as precision data (non-round 2040 rpm vs
# the winding code's 2000 +/- 250 nominal; +/- 5% tolerance class). No
# numeric DC bus is printed for the "uncontrolled" 3 x AC 400 V infeed,
# so the corner is pinned under BOTH rectifier conventions: 1.35*U_LL =
# 540 V (loaded 6-pulse average) and sqrt(2)*U_LL = 565.7 V (no-load
# peak, the Beckhoff convention). The UZK2 controlled 400-480 V envelope
# is curves only (Fig. 41/42) — not numerically anchored.

# (U_DC assumption, nb_model/nN, voltage utilization at the printed corner)
_MS2N_BUSES = [
    (1.35 * 400.0, 1.046, 0.958),
    (sqrt(2) * 400.0, 1.099, 0.915),
]


class TestRexrothMs2nEnvelope:
    TOML = "data/rexroth_ms2n/rexroth_ms2n04_d0bhn.toml"

    def test_ke_km_consistency_pins_conventions(self):
        """psi_f from KE (159.1 V/krpm line-line rms, 20 C) reproduces the
        printed cold Km 2.62 Nm/Arms to -0.44% — KE and Km close mutually
        cold, pinning the line-line-rms reading of KE. The printed rated
        pair MN/IN = 2.301 sits -12.2% below cold Km: the 100 K hot/cold
        rating gap (why this anchor stays out of _NAMEPLATE_ANCHORS)."""
        m, _, _ = _envelope_args(self.TOML)
        k_t = 1.5 * m.n_p * m.psi_f * sqrt(2)
        assert 2.62 / k_t == pytest.approx(0.9956, abs=0.001)
        assert (3.75 / 1.63) / 2.62 == pytest.approx(0.878, abs=0.005)

    @pytest.mark.parametrize("udc,nb_ratio,util", _MS2N_BUSES)
    def test_rated_point_voltage_feasible_both_conventions(self, udc,
                                                           nb_ratio, util):
        """The precision corner (2040 rpm, MN = 3.75 Nm at the printed
        IN = 1.63 Arms) is voltage-feasible under both documented
        rectifier conventions, with the model corner +4.6% above nN at
        540 V and +9.9% at sqrt(2)*400 (pinned). Printed-corner voltage
        utilization 0.958/0.915 — the tightest catalog-vendor corner in
        the registry, approaching ETEL's agreement class but still
        one-sided + pinned."""
        m, args, _ = _envelope_args(self.TOML)
        u_max = udc / sqrt(3)
        nb_model = base_speed(*args, m.I_rated, u_max) * _RPM
        assert nb_model > 2040.0
        assert nb_model / 2040.0 == pytest.approx(nb_ratio, abs=0.005)
        util_model = dq_voltage(*args, 0.0, m.I_rated, 2040.0 / _RPM) / u_max
        assert util_model == pytest.approx(util, abs=0.005)
        tau, _, _ = envelope_at_speed(*args, m.I_rated, u_max, 2040.0 / _RPM)
        assert tau > 3.75

    def test_corner_is_voltage_set_not_a_ceiling(self):
        """Ceiling discriminator for a single-voltage sheet: the model
        NO-LOAD limit at UZK1 (2400 rpm at 540 V, 2514 at sqrt(2)*400)
        sits BELOW the printed nmax el = 4000 rpm — on the uncontrolled
        bus the motor cannot reach its own electrical ceiling even
        unloaded, so neither nmax el nor nmax mech (6000 rpm) can be
        what sets nN = 2040. Complements the manual's explicit section
        6.3 statement that rated speed is determined by the DC bus
        voltage."""
        _, args, _ = _envelope_args(self.TOML)
        for udc, _, _ in _MS2N_BUSES:
            nl = base_speed(*args, 0.0, udc / sqrt(3)) * _RPM
            assert 2040.0 < nl < 4000.0
        nl_540 = base_speed(*args, 0.0, 540.0 / sqrt(3)) * _RPM
        assert nl_540 == pytest.approx(2400.0, abs=1.0)


# ---------------------------------------------------------------------------
# Toyota Prius 2004: traction-motor IPM envelope
# ---------------------------------------------------------------------------
# The canonical IPM benchmark: V-shape interior magnets, 50 kW peak,
# strong saliency (L_q/L_d ~ 2.84). The characteristic current
# psi_f/L_d ~ 71 A is well below the peak current (250 A), so the motor
# enters deep flux reversal at peak current, giving unlimited max speed
# in the linear model (max_speed returns None).
#
# Circuit-only (no geometry) — validates the dq envelope model against
# the published system operating range (6000 rpm, 50 kW peak). The
# constant-inductance model over-predicts torque at peak current
# (~166% vs ORNL dynamometer); only low-current anchors are tight.
#
# Source: ORNL/TM-2004/185 rev. 2007 (OSTI 921782, locked-rotor torque);
#         system specs from all ORNL reports.


class TestPriusEnvelope:
    TOML = "motors/prius_2004.toml"

    def test_no_load_max_below_system_limit(self):
        """The no-load voltage limit (~4288 rpm) is below the 6000 rpm
        system maximum — the motor's back-EMF reaches the bus voltage
        before the mechanical speed limit. Under load (I_ch < I_peak),
        field weakening extends operation to 6000 rpm and beyond."""
        _, args, u_max = _envelope_args(self.TOML)
        nl_speed = base_speed(*args, 0.0, u_max) * _RPM
        assert nl_speed < 6000.0
        assert nl_speed == pytest.approx(4288, abs=10)

    def test_infinite_cpsr_at_peak_current(self):
        """At I_peak = 250 A, the characteristic current (71 A) is
        far exceeded — the motor enters deep flux reversal (total
        d-axis flux linkage goes negative) and max_speed is None
        (theoretically infinite). This is the design intent for a
        traction motor: no field-weakening speed limit within the
        drive's voltage capability."""
        m, args, u_max = _envelope_args(self.TOML)
        ms = max_speed(*args, m.drive.MAX_I_S, u_max)
        assert ms is None

    def test_base_speed_at_peak_current(self):
        """Base speed at MAX_I_S = 250 A peak: ~570 rpm. Below this,
        the motor delivers full MTPA torque (905 Nm with linear L);
        above, field weakening maintains constant power."""
        m, args, u_max = _envelope_args(self.TOML)
        nb = base_speed(*args, m.drive.MAX_I_S, u_max) * _RPM
        assert nb == pytest.approx(570, abs=5)

    def test_torque_at_6000_rpm(self):
        """At system max speed (6000 rpm) with peak current, the
        envelope torque is 51.9 Nm (32.6 kW). The ORNL-measured peak
        at 250 A is ~340 Nm, so the linear model's ~166% over-prediction
        at stall does not carry proportionally to the field-weakening
        region. Power at 6000 rpm exceeds 30 kW (published continuous
        system rating)."""
        m, args, u_max = _envelope_args(self.TOML)
        tau, _, _ = envelope_at_speed(*args, m.drive.MAX_I_S, u_max,
                                      6000.0 / _RPM)
        assert tau == pytest.approx(51.9, abs=1.0)
        power_kw = tau * 6000.0 / _RPM / 1000
        assert power_kw > 30.0

    def test_peak_power_exceeds_published(self):
        """Model peak power at I_peak exceeds the published 50 kW
        system rating — the linear model over-predicts (constant
        inductances overstate reluctance torque), so this is a one-sided
        bound, not a tight validation."""
        m, _, _ = _envelope_args(self.TOML)
        cfg = RunConfig(motor=m, model="torque_speed")
        ts = MODEL_REGISTRY["torque_speed"].fn(cfg)
        assert ts["p_max_peak"] / 1000 > 50.0
        assert ts["p_max_peak"] / 1000 == pytest.approx(62.3, abs=1.0)
