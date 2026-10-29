"""CREATOR Case induction motor — the first IM validation anchor.

Gates the im_equivalent_circuit / im_torque_slip models against the
CREATOR IM open-benchmark dataset (DOI 10.3217/kh3v7-dhn98): nameplate
rated point, breakdown bound, and the 45 measured locked-rotor points
(10-50 Hz, reduced voltage) — the deep-bar-sensitive end of the curve,
which is the only gate that discriminates a constant-parameter model
from one carrying R_2(f).

Also gates identify_from_tests end to end: CREATOR uniquely
publishes both the raw IEEE-112-style test structs AND the derived
circuit parameters, so identification is checked raw-data-in,
published-parameters-out.

Anchor TOML: data/creator_case_im/creator_case_im.toml (provenance and
convention notes there). Dataset-gated tests skip until
scripts/fetch_creator_im_dataset.py has run.
"""

import dataclasses as dc
from math import pi

import numpy as np
import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.models.im_circuit import prepare_im_circuit, solve_im_circuit
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig
from tests.conftest import CREATOR_IM_SKIP, REPO_ROOT, requires_fem

MOTOR_TOML = REPO_ROOT / "data/creator_case_im/creator_case_im.toml"
IM_DATA = REPO_ROOT / "data/creator_case_im"
EQ_CSV = (IM_DATA / "Induction_motor/Measurement_results/"
          "Equivalent_circuit_parameters")
BH_CSV = (IM_DATA / "Induction_motor/Design_parameters/Material_properties/"
          "Stator_and_rotor_iron/B_H.csv")

TAU_RATED = 24.7    # Nm nameplate (IM_general_specification.csv)
I_RATED_RMS = 6.9   # A rms nameplate
TAU_MAX = 31.0      # Nm system-limited maximum (≈ 4.6 kW / 1430 rpm =
                    # 30.7 Nm, the power limit) — a LOWER bound on
                    # circuit breakdown, not the breakdown itself


@pytest.fixture(scope="module")
def motor():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def params(motor):
    return prepare_im_circuit(motor)


@pytest.fixture(scope="module")
def sweep(motor):
    rc = RunConfig(motor=motor, model="im_torque_slip")
    return MODEL_REGISTRY["im_torque_slip"].fn(rc)


def _slip_for_torque(params, tau_target):
    from scipy.optimize import brentq
    return brentq(
        lambda s: solve_im_circuit(params, s).tau - tau_target, 0.005, 0.12)


class TestNameplateGates:
    """Gates that need only the anchor TOML (no dataset download)."""

    def test_rated_slip_torque_overprediction_pinned(self, params):
        """Torque at nameplate slip with 20 °C parameters: +34% high.

        PINNED FINDING, not agreement: the dataset quotes every
        resistance at 20 °C while the nameplate point is hot. In the
        low-slip region tau scales ~1/R_2, and a uniform resistance
        scale of ~1.37 — aluminum cage / copper winding at ~115 °C,
        plausible for a class-F TEWC machine — puts the circuit exactly
        on the nameplate. Direction is physics (cold parameters MUST
        over-predict); the magnitude is the pinned observation.
        """
        tau = solve_im_circuit(params, params.s_rated).tau
        assert tau / TAU_RATED == pytest.approx(1.34, abs=0.02)
        k_hot = 1.37
        p_hot = dc.replace(
            params, R_1=params.R_1 * k_hot, R_2=params.R_2 * k_hot,
            R_2_vs_f=tuple((f, r * k_hot) for f, r in params.R_2_vs_f),
        )
        tau_hot = solve_im_circuit(p_hot, params.s_rated).tau
        assert tau_hot == pytest.approx(TAU_RATED, rel=0.02)

    def test_current_at_rated_torque(self, params):
        """Stator current at the 24.7 Nm operating point: ~5% of nameplate.

        The temperature-robust half of the rated-point check: solving
        for the slip that delivers nameplate torque and comparing
        current sidesteps most of the R(T) sensitivity that dominates
        the fixed-slip comparison above.
        """
        s = _slip_for_torque(params, TAU_RATED)
        op = solve_im_circuit(params, s)
        assert op.I_1_rms == pytest.approx(I_RATED_RMS, rel=0.08)
        assert op.I_1_rms / I_RATED_RMS == pytest.approx(1.05, abs=0.02)

    def test_breakdown_exceeds_published_max(self, sweep):
        """One-sided bound: the machine demonstrably delivered 31 Nm
        (system-limited — the 4.6 kW max-power line at rated speed),
        so circuit breakdown must exceed it."""
        assert sweep["tau_breakdown"] > TAU_MAX
        assert sweep["tau_rated"] is not None

    def test_torque_slip_registry_run(self, sweep):
        assert sweep["slip_rated"] == pytest.approx(0.0466667)
        assert sweep["tau_starting"] > 0
        assert sweep["I_locked_rotor_rms"] > 4 * I_RATED_RMS


class TestDatasetTranscription:
    """The TOML is a transcription — hold it to the dataset verbatim."""

    @pytest.fixture(scope="class")
    def eq_params(self):
        path = EQ_CSV / "Equivalent_circuit_parameters.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        import csv
        with open(path, encoding="utf-8-sig") as f:
            return [row for row in csv.reader(f)][1:]

    def test_scalar_parameters_match_dataset(self, motor, eq_params):
        values = [float(r[1]) for r in eq_params]
        R_s, R_r, L_ss, L_sr, L_m = values
        assert motor.R_s == pytest.approx(R_s, rel=1e-6)
        ind = motor.induction
        assert ind.R_2 == pytest.approx(R_r, rel=1e-6)
        assert ind.L_sigma_s == pytest.approx(L_ss, rel=1e-6)
        assert ind.L_sigma_r == pytest.approx(L_sr, rel=1e-6)
        assert ind.L_m == pytest.approx(L_m, rel=1e-6)

    def test_deep_bar_table_matches_dataset(self, motor):
        path = EQ_CSV / "Rr.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
        assert motor.induction.R_2_vs_f == tuple(
            (float(f), float(r)) for f, r in d)

    def test_leakage_tables_match_dataset(self, motor):
        path = EQ_CSV / "L_sigmas.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
        assert motor.induction.L_sigma_s_vs_f == tuple(
            (float(f), float(v)) for f, v, _ in d)
        assert motor.induction.L_sigma_r_vs_f == tuple(
            (float(f), float(v)) for f, _, v in d)


class TestLockedRotorPoints:
    """The 45 measured locked-rotor points — the anchor's money gate.

    Reduced-voltage blocked-rotor tests at 10/20/30/40/50 Hz with
    measured shaft torque. At locked rotor f_rotor = f_supply, so this
    sweep exercises the deep-bar band directly. Vs in the dataset is
    per-phase rms (verified by power balance, see the fetch script).
    """

    @pytest.fixture(scope="class")
    def points(self):
        path = IM_DATA / "locked_rotor_tests.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        return np.genfromtxt(path, delimiter=",", skip_header=1)

    def test_point_count(self, points):
        assert len(points) == 45

    def test_current_all_points(self, params, points):
        """Stator current within 2.5% at every one of the 45 points."""
        for f, V, I_meas, _P, _T, _Rs in points:
            op = solve_im_circuit(params, 1.0, V_ph_rms=V, f_supply=f)
            assert op.I_1_rms == pytest.approx(I_meas, rel=0.025), (f, V)

    def test_torque_top_current_points(self, params, points):
        """At each frequency's highest-voltage point (the ~6.75 A rms
        test maximum): torque within 3.5%."""
        for f in (10, 20, 30, 40, 50):
            rows = points[np.abs(points[:, 0] - f) < 2]
            top = rows[np.argmax(rows[:, 1])]
            op = solve_im_circuit(params, 1.0, V_ph_rms=top[1],
                                  f_supply=top[0])
            assert op.tau == pytest.approx(abs(top[4]), rel=0.035), f

    def test_torque_all_points_combined_band(self, params, points):
        """All 45 points within ±(0.05 Nm + 2.5% of measured).

        Two error regimes, gated together: a near-CONSTANT per-frequency
        deficit (0.01 Nm at 10 Hz rising to ~0.03 Nm at 30-50 Hz) that
        dominates the smallest test voltages —
        the signature of the missing core-loss branch (iron-loss torque
        P_fe/omega_s), where relative error balloons to ~35% — plus a
        small relative error (< 2.5%) at the top of each sweep. Gate the
        combined band and pin the attribution here rather than widening
        a relative tolerance to hide the offset.
        """
        for f, V, _I, _P, T, _Rs in points:
            op = solve_im_circuit(params, 1.0, V_ph_rms=V, f_supply=f)
            t = abs(T)
            assert op.tau - t == pytest.approx(
                0.0, abs=0.05 + 0.025 * t), (f, V)

    def test_deep_bar_beats_constant_parameters(self, params, points):
        """The R_2(f) table earns its keep: the constant-parameter
        circuit under-predicts locked-rotor torque by ~30% mean (~55%
        at 50 Hz where the deep-bar rise is largest); the table model
        holds ~6% mean. Both pinned."""
        p_const = dc.replace(params, R_2_vs_f=None, L_sigma_s_vs_f=None,
                             L_sigma_r_vs_f=None)
        err_tab, err_const = [], []
        for f, V, _I, _P, T, _Rs in points:
            t = abs(T)
            err_tab.append(
                solve_im_circuit(params, 1.0, V_ph_rms=V, f_supply=f).tau
                / t - 1)
            err_const.append(
                solve_im_circuit(p_const, 1.0, V_ph_rms=V, f_supply=f).tau
                / t - 1)
        assert np.mean(np.abs(err_tab)) < 0.08
        assert np.mean(err_const) == pytest.approx(-0.30, abs=0.04)
        m50 = np.abs(points[:, 0] - 50) < 2
        assert np.mean(np.array(err_const)[m50]) < -0.45


class TestIdentification:
    """identify_from_tests end to end: the raw locked-rotor / no-load
    test points in, the independently published circuit parameters out.

    This is the anchor the deep dive flagged as unusually strong for
    identification — CREATOR publishes BOTH the raw IEEE-112-style test
    structs and the derived parameter set, so the identification route
    is gated against measurement rather than trusted from the textbook.
    """

    @pytest.fixture(scope="class")
    def lr_points(self):
        path = IM_DATA / "locked_rotor_tests.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        return np.genfromtxt(path, delimiter=",", skip_header=1)

    @pytest.fixture(scope="class")
    def nl_rated(self):
        """No-load point interpolated to the rated phase voltage on the
        50 Hz sweep: (V_ph, I, P), plus that block's DC R_s."""
        path = IM_DATA / "no_load_tests.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
        rows = d[np.abs(d[:, 0] - 50) < 2]
        rows = rows[np.argsort(rows[:, 1])]
        V_rated_ph = 400.0 / np.sqrt(3)
        return ((V_rated_ph,
                 float(np.interp(V_rated_ph, rows[:, 1], rows[:, 2])),
                 float(np.interp(V_rated_ph, rows[:, 1], rows[:, 3]))),
                float(rows[0, 5]), rows)

    @staticmethod
    def _top_point(lr_points, f_nom):
        rows = lr_points[np.abs(lr_points[:, 0] - f_nom) < 2]
        return rows[np.argmax(rows[:, 1])]

    def test_deep_bar_and_leakage_tables_reproduced(self, motor, params,
                                                    lr_points, nl_rated):
        """Identifying at each locked-rotor frequency's top-current
        point (~rated current) reproduces the published deep-bar
        R_2(f) table within 9% and both leakage tables within 5.5%
        (published stator/rotor split). R_2 is a consistent slight
        under-estimate — the magnetizing shunt the short-form route
        ignores — mean −4.3%, worst −8% at 10 Hz where X_m is
        smallest."""
        from phasesweep.models.im_circuit import identify_from_tests

        ind = motor.induction
        split = ind.L_sigma_s / (ind.L_sigma_s + ind.L_sigma_r)
        nl_point, _, _ = nl_rated
        errs = []
        for f_nom, R_2_pub in ind.R_2_vs_f:
            f, V, I_top, P, _T, Rs = self._top_point(lr_points, f_nom)
            ident = identify_from_tests(
                R_1=Rs, no_load=nl_point, locked_rotor=(V, I_top, P),
                f_no_load=50.0, f_locked_rotor=f, leakage_split=split)
            assert ident.R_2 == pytest.approx(R_2_pub, rel=0.09), f_nom
            assert ident.R_2 < R_2_pub
            errs.append(ident.R_2 / R_2_pub - 1)
            Lss_pub = dict(ind.L_sigma_s_vs_f)[f_nom]
            Lsr_pub = dict(ind.L_sigma_r_vs_f)[f_nom]
            assert ident.L_sigma_s == pytest.approx(Lss_pub, rel=0.055)
            assert ident.L_sigma_r == pytest.approx(Lsr_pub, rel=0.055)
        assert np.mean(errs) == pytest.approx(-0.043, abs=0.02)

    def test_no_load_lm_lands_on_saturation_curve(self, motor, lr_points,
                                                  nl_rated):
        """The identified L_m (0.205 H) sits ~9% below the published
        rated-point scalar (0.226 H) — but lands on the published
        L_m(I_m) saturation curve at the test's own magnetizing
        current within 1.5%. The gap is saturation state, not
        identification error."""
        from phasesweep.models.im_circuit import identify_from_tests

        lm_path = EQ_CSV / "Lm.csv"
        if not lm_path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        curve = np.genfromtxt(lm_path, delimiter=",", skip_header=1)
        ind = motor.induction
        split = ind.L_sigma_s / (ind.L_sigma_s + ind.L_sigma_r)
        nl_point, _, _ = nl_rated
        f, V, I_top, P, _T, Rs = self._top_point(lr_points, 50)
        ident = identify_from_tests(
            R_1=Rs, no_load=nl_point, locked_rotor=(V, I_top, P),
            f_no_load=50.0, f_locked_rotor=f, leakage_split=split)
        L_m_curve = float(np.interp(ident.I_m_rms, curve[:, 0], curve[:, 2]))
        assert ident.L_m == pytest.approx(L_m_curve, rel=0.015)
        assert ident.L_m / ind.L_m == pytest.approx(0.91, abs=0.02)

    def test_loss_separation_matches_published(self, nl_rated):
        """The V²-intercept friction+windage separation over the 50 Hz
        no-load sweep agrees with the dataset's own published P_fw at
        50 Hz (Pfw_flux_sq.csv, 161 W) within 8%. (The no-load SHAFT
        torque implies only ~92 W — a different quantity, on the
        dynamometer side of the coupling.)"""
        from phasesweep.models.im_circuit import separate_no_load_losses

        pfw_path = (IM_DATA / "Induction_motor/Measurement_results/"
                    "Locked_rotor_and_no_load_tests/Pfw_flux_sq.csv")
        if not pfw_path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        pfw_pub = float(np.genfromtxt(
            pfw_path, delimiter=",", skip_header=1)[0, 9])
        _, Rs, rows = nl_rated
        P_fw = separate_no_load_losses(
            [tuple(row) for row in rows[:, 1:4]], Rs)
        assert P_fw == pytest.approx(pfw_pub, rel=0.08)

    def test_identified_circuit_reproduces_published_at_rated(
            self, params, lr_points, nl_rated):
        """The full raw-tests-only route: DC R_s + the 10 Hz
        locked-rotor point (nearest the rated rotor frequency
        s·f = 2.3 Hz, per the IEEE-112 reduced-frequency guidance) +
        the rated-voltage no-load point, 50/50 split (no prior
        knowledge of the machine). The identified circuit matches the
        published-scalar-parameter circuit at the nameplate slip to
        +2.6% torque / +3.7% current (each pinned ±3 points) — and
        reproduces the pinned +34%-at-20 °C torque finding from raw
        data alone."""
        from phasesweep.models.im_circuit import identify_from_tests

        nl_point, _, _ = nl_rated
        f, V, I_top, P, _T, Rs = self._top_point(lr_points, 10)
        ident = identify_from_tests(
            R_1=Rs, no_load=nl_point, locked_rotor=(V, I_top, P),
            f_no_load=50.0, f_locked_rotor=f)
        p_pub = dc.replace(params, R_2_vs_f=None, L_sigma_s_vs_f=None,
                           L_sigma_r_vs_f=None)
        p_id = dc.replace(p_pub, R_1=ident.R_1, R_2=ident.R_2,
                          L_sigma_s=ident.L_sigma_s,
                          L_sigma_r=ident.L_sigma_r, L_m=ident.L_m)
        op_pub = solve_im_circuit(p_pub, params.s_rated)
        op_id = solve_im_circuit(p_id, params.s_rated)
        assert op_id.tau / op_pub.tau == pytest.approx(1.026, abs=0.03)
        assert op_id.I_1_rms / op_pub.I_1_rms == pytest.approx(1.037,
                                                               abs=0.03)
        assert op_id.tau / TAU_RATED == pytest.approx(1.36, abs=0.03)


class TestNoLoadDataset:
    """Dataset-integrity pin on the no-load points: friction + windage
    shaft torque is 0.41-0.59 Nm rising with frequency. Not a circuit
    gate (the circuit has no mechanical loss) — it documents the terms
    the eta_circuit output excludes."""

    def test_friction_windage_band(self):
        path = IM_DATA / "no_load_tests.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
        assert len(d) == 59
        t = np.abs(d[:, 4])
        assert t.min() == pytest.approx(0.41, abs=0.01)
        assert t.max() == pytest.approx(0.59, abs=0.01)
        t10 = t[np.abs(d[:, 0] - 10) < 2].mean()
        t50 = t[np.abs(d[:, 0] - 50) < 2].mean()
        assert t50 > t10


class TestCreatorGeometry:
    """Geometry_parameters.csv transcription.
    The TOML is in-repo, so these run without the dataset."""

    def test_transcription(self, motor):
        g = motor.geometry
        assert g.n_slots == 36 and g.n_bars == 28
        assert g.r_stator - g.r_rotor == pytest.approx(0.0004)   # 0.4 mm gap
        assert g.r_magnet == g.r_rotor                           # no PM annulus
        assert g.bar_area == pytest.approx(3.4e-3 * 21e-3)       # table average
        assert g.bar_radii[1] > g.r_inner
        assert g.end_ring_radial_height == pytest.approx((0.1242 - 0.081) / 2)
        assert motor.induction.n_bars == g.n_bars
        assert 24e6 <= motor.induction.sigma_bar <= 32e6          # dataset range

    def test_transcription_matches_dataset_table(self):
        path = (IM_DATA / "Induction_motor/Design_parameters/Motor_geometry/"
                "Geometry_parameters.csv")
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        import csv
        rows = {r["Parameter"]: float(r["Value"]) for r in csv.DictReader(path.open())}
        g = load_motor(MOTOR_TOML).geometry
        assert g.r_outer == pytest.approx(rows["Stator iron outer diameter"] / 2)
        assert g.r_stator == pytest.approx(rows["Stator iron inner diameter"] / 2)
        assert g.r_rotor == pytest.approx(rows["Rotor core outer diameter"] / 2)
        assert g.r_inner == pytest.approx(rows["Rotor core inner diameter"] / 2)
        assert g.slot_depth == pytest.approx(rows["Length of stator teeth"])
        assert g.slot_opening_width == pytest.approx(rows["Stator slot opening"])
        assert g.bar_width == pytest.approx(rows["Rotor slot average width"])
        assert g.bar_height == pytest.approx(rows["Rotor slot average height"])
        assert g.n_bars == int(rows["No. of rotor cage bars"])
        assert g.n_slots == int(rows["No. of stator slots"])
        assert g.end_ring_axial_length == pytest.approx(
            rows["Rotor end ring length (axial)"])
        assert g.end_ring_radial_height == pytest.approx(
            (rows["Rotor end ring outer diameter"]
             - rows["Rotor end ring inner diameter"]) / 2)

    @pytest.mark.slow
    def test_cage_mesh_builds(self, motor):
        pytest.importorskip("ngsolve")
        from ngsolve import CoefficientFunction, Integrate

        from phasesweep.solvers.fem_mesh import get_or_build_cage_mesh
        g = motor.geometry
        mesh = get_or_build_cage_mesh(g, 0.05)
        mats = set(mesh.GetMaterials())
        assert {f"bar{k}" for k in range(28)} <= mats
        bars = sum(Integrate(CoefficientFunction(1), mesh,
                             definedon=mesh.Materials(f"bar{k}")) for k in range(28))
        assert bars == pytest.approx(28 * g.bar_area, rel=1e-3)


@requires_fem
class TestCreatorBarImpedance:
    """The predicted deep-bar tables against the
    published R_r(f) / L_sigma_r(f) (Rr.csv / L_sigmas.csv, transcribed
    in the TOML) — REPORTED, never substituted into the circuit anchor.

    Findings pinned here: the DC rotor resistance lands within 3% from
    geometry, conductivity and the cage turns ratio alone, with the end
    ring carrying ~22% of it; the FEM bar + ring rise to 50 Hz is +27%
    against the published +76%, and the bar shape cannot close that gap
    (the F1 radial-sided bar is already top-heavy, and the real keyhole
    is more so, which LOWERS R_ac/R_dc) — the residual sits in the
    published table's provenance, a locked-rotor identification with no
    core-loss branch that books rotor-frequency iron and stray losses
    into R_r. Slot + tooth-top + ring leakage is ~29% below the published
    rotor leakage; adding the differential term σ_d·L_m closes the DC
    value to +7% while the published table's 16% fall to 50 Hz (a
    saturating leakage path at rising test current) is not a linear-iron
    effect.
    """

    @pytest.fixture(scope="class")
    def out(self, motor):
        return MODEL_REGISTRY["im_bar_impedance"].fn(
            RunConfig(motor=motor, model="im_bar_impedance"))

    @staticmethod
    def _at(out, key, f):
        return dict(zip(out["f_rotor_list"], out[key]))[f]

    def test_dc_resistance_from_geometry(self, out, motor):
        ind = motor.induction
        assert 0.99 < out["R_2_dc"] / ind.R_2 < 1.06            # 1.026
        assert out["referral_factor"] == pytest.approx(16268.36, rel=1e-4)
        assert 0.18 < out["ring_share_list"][0] < 0.26          # 0.216
        assert out["R_bar_dc"] == pytest.approx(0.1 / (28e6 * 3.4e-3 * 21e-3), rel=1e-9)

    def test_deep_bar_slope_reported_not_substituted(self, out, motor):
        ind = motor.induction
        ratios = {f: self._at(out, "R_2_list", f) / R for f, R in ind.R_2_vs_f}
        assert all(0.70 < r < 1.0 for r in ratios.values())
        assert ratios[10.0] > 0.95
        assert 0.70 < ratios[50.0] < 0.78                        # 0.741
        rise_fem = self._at(out, "R_2_list", 50.0) / out["R_2_dc"]
        rise_pub = dict(ind.R_2_vs_f)[50.0] / ind.R_2
        assert 1.22 < rise_fem < 1.32                            # 1.269
        assert rise_pub > 1.70                                   # 1.758
        assert prepare_im_circuit(motor).R_2_vs_f == ind.R_2_vs_f   # anchor untouched

    def test_leakage_decomposition(self, out, motor):
        ind = motor.induction
        # Non-negativity of every decomposed component
        assert all(v >= 0 for v in out["R_2_list"])
        assert all(v >= 0 for v in out["L_sigma_r_list"])
        assert out["R_bar_dc"] > 0
        assert out["R_ring_seg"] > 0
        assert out["L_ring_seg"] > 0
        assert out["sigma_diff"] >= 0
        assert out["L_sigma_r_diff"] >= 0
        pub = dict(ind.L_sigma_r_vs_f)
        for f, L in pub.items():
            assert 0.65 < self._at(out, "L_sigma_r_list", f) / L < 0.90   # slot + ring only
        assert out["sigma_diff"] == pytest.approx(0.01696, rel=1e-3)
        total_10 = self._at(out, "L_sigma_r_list", 10.0) + out["L_sigma_r_diff"]
        total_50 = self._at(out, "L_sigma_r_list", 50.0) + out["L_sigma_r_diff"]
        assert 1.0 < total_10 / pub[10.0] < 1.12                 # 1.071
        assert 1.15 < total_50 / pub[50.0] < 1.30                # 1.233
        assert 1.0 < (out["L_sigma_r_dc"] + out["L_sigma_r_diff"]) / ind.L_sigma_r < 1.12


@requires_fem
@pytest.mark.slow
class TestCreatorFemLockedRotor:
    """The full 36/28 cross-section with the 7/9
    double-layer winding, virtual blocked-rotor solves identified the
    IEEE-112 way, against the published tables and the 45 measured
    locked-rotor points — REPORTED, never substituted into the circuit
    anchor. Linear iron, 20 °C resistances, Boldea end-winding leakage
    L_ew on the stator side (v2).

    Findings pinned here. The identified R_2(f) lands 11% under the
    published table at 10 Hz and 32% under at 50 Hz, with a +26% rise
    across 10–50 Hz — the same slope im_bar_impedance found from the
    slot-pitch cell (+27%) against the published +66%: the whole-machine
    solve confirms the F3 reading that the published table's slope is a
    no-core-loss identification booking locked-rotor iron and stray loss
    into R_r, not a bar-shape or cell-model artifact. The identified
    R_2 sits ~9% below the F3 cell's at 10 Hz: the IEEE-112 step ignores
    the magnetizing shunt (identify_from_tests' documented bias), which
    the F3 referral does not. The FEM leakage total (L_sigma_s + L_sigma_r)
    carries a Boldea L_ew (~1.4 mH, semicircular) on the stator side;
    with it, the total is ~18.4 mH at 10 Hz (0.84× published) and
    ~17.1 mH at 50 Hz (0.92× published), closing ~50% of the gap at
    50 Hz. Against the 45 points the FEM over-predicts current by +3–5%
    (10 Hz) rising to +16–17% (50 Hz) — reduced from pre-L_ew (+4%/+17%)
    — while torque at each frequency's top-current point stays within
    ±10% and every point sits inside
    ±(0.05 Nm + 10%); the constant-offset signature of the missing core
    loss is the circuit's, larger. The three phases see the 28 bars
    differently at standstill (6% impedance spread), a real position
    dependence of blocked-rotor tests. The linear-iron no-load current
    at 230.9 V / 50 Hz is 3.10 A rms (L_ew in) — 5.5% under the measured
    3.20 A at 225.6 V once scaled — reported as a diagnostic, not as L_m.
    """

    @pytest.fixture(scope="class")
    def out(self, motor):
        return MODEL_REGISTRY["im_fem_locked_rotor"].fn(
            RunConfig(motor=motor, model="im_fem_locked_rotor"))

    @pytest.fixture(scope="class")
    def points(self):
        path = IM_DATA / "locked_rotor_tests.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        return np.genfromtxt(path, delimiter=",", skip_header=1)

    @staticmethod
    def _at(out, key, f):
        return dict(zip(out["f_locked_rotor_list"], out[key]))[f]

    @staticmethod
    def _predict(out, f, V):
        Z = complex(TestCreatorFemLockedRotor._at(out, "Z_re_list", f),
                    TestCreatorFemLockedRotor._at(out, "Z_im_list", f))
        I_rms = V / abs(Z)
        P = 3 * V * I_rms * Z.real / abs(Z)
        tau = TestCreatorFemLockedRotor._at(out, "tau_per_A2_list", f) * I_rms**2
        return I_rms, P, tau

    def test_setup_and_layout(self, out, motor):
        assert out["f_locked_rotor_list"] == [10.0, 20.0, 30.0, 40.0, 50.0]
        assert out["k_w_layout"] == pytest.approx(motor.k_w, rel=1e-3)   # 0.90191
        assert out["ring_model"] == "segment" and out["leakage_split_used"] == 0.5
        assert out["R_1"] == motor.R_s
        assert 0.04 < out["phase_asymmetry_max"] < 0.09                  # 0.063

    def test_identified_r2_reported_not_substituted(self, out, motor):
        ind = motor.induction
        ratios = {f: self._at(out, "R_2_list", f) / R for f, R in ind.R_2_vs_f}
        assert 0.86 < ratios[10.0] < 0.92                                # 0.892
        assert 0.65 < ratios[50.0] < 0.71                                # 0.680
        assert all(ratios[f] > ratios[g] for f, g in zip((10.0, 20.0, 30.0, 40.0),
                                                          (20.0, 30.0, 40.0, 50.0)))
        rise = self._at(out, "R_2_list", 50.0) / self._at(out, "R_2_list", 10.0)
        assert 1.22 < rise < 1.31                                        # 1.265
        assert prepare_im_circuit(motor).R_2_vs_f == ind.R_2_vs_f       # anchor untouched

    def test_same_slope_as_bar_cell(self, out, motor):
        """F3 and F4 agree on the deep-bar rise; F4's level sits below
        F3's by the magnetizing-shunt bias of the IEEE-112 step."""
        cell = MODEL_REGISTRY["im_bar_impedance"].fn(
            RunConfig(motor=motor, model="im_bar_impedance"))
        cell_at = dict(zip(cell["f_rotor_list"], cell["R_2_list"]))
        rise_cell = cell_at[50.0] / cell_at[10.0]                        # 1.253
        rise_fem = self._at(out, "R_2_list", 50.0) / self._at(out, "R_2_list", 10.0)
        assert rise_fem == pytest.approx(rise_cell, abs=0.03)
        assert 0.88 < self._at(out, "R_2_list", 10.0) / cell_at[10.0] < 0.94   # 0.908

    def test_leakage_below_published(self, out, motor):
        """With Boldea L_ew on the stator side, the total is closer to
        published: ~0.84 at 10 Hz, ~0.92 at 50 Hz (pre-L_ew: 0.775–0.848)."""
        ind = motor.induction
        pub_s, pub_r = dict(ind.L_sigma_s_vs_f), dict(ind.L_sigma_r_vs_f)
        for f in (10.0, 30.0, 50.0):
            total = self._at(out, "L_sigma_s_list", f) + self._at(out, "L_sigma_r_list", f)
            assert 0.80 < total / (pub_s[f] + pub_r[f]) < 0.96
        assert out["L_ew_H"] == pytest.approx(1.42e-3, rel=0.02)
        assert self._at(out, "X_lr_list", 50.0) / (2 * pi * 50) == pytest.approx(
            2 * self._at(out, "L_sigma_r_list", 50.0), rel=1e-9)

    def test_current_all_points_band(self, out, points):
        """Current over-predicted at every point: +3–5% at 10 Hz rising
        to +16–17% at 50 Hz (pre-L_ew: +4%/+17%)."""
        band = {10: (0.03, 0.05), 20: (0.09, 0.11), 30: (0.13, 0.15),
                40: (0.15, 0.17), 50: (0.15, 0.18)}
        for f, V, I_meas, _P, _T, _Rs in points:
            I_fem, _P_fem, _tau = self._predict(out, float(round(f)), V)
            lo, hi = band[round(f)]
            assert lo < I_fem / I_meas - 1 < hi, (f, V)

    def test_torque_top_current_points(self, out, points):
        """Torque within ±10% at each frequency's highest-voltage point."""
        for f in (10, 20, 30, 40, 50):
            rows = points[np.abs(points[:, 0] - f) < 2]
            top = rows[np.argmax(rows[:, 1])]
            _I, _P, tau = self._predict(out, float(f), top[1])
            assert tau == pytest.approx(abs(top[4]), rel=0.10), f

    def test_torque_all_points_combined_band(self, out, points):
        """All 45 points within ±(0.05 Nm + 10% of measured) — the same
        two-regime shape as the circuit anchor's gate, wider."""
        for f, V, _I, _P, T, _Rs in points:
            _I_fem, _P_fem, tau = self._predict(out, float(round(f)), V)
            t = abs(T)
            assert tau - t == pytest.approx(0.0, abs=0.05 + 0.10 * t), (f, V)

    def test_no_load_diagnostic(self, out, motor):
        """Linear iron at mu_r_fe = 1170 (the M800-50A initial slope)
        puts the no-load current 5.5% under the measured point — labeled,
        not an L_m identification (F5)."""
        assert out["f_no_load_Hz"] == 50.0
        assert out["V_phase_rms"] == pytest.approx(400 / 3**0.5)
        I_scaled = out["I_no_load_rms_linear_iron"] * 225.63 / out["V_phase_rms"]
        assert 0.93 < I_scaled / 3.20098 < 0.97                          # 0.945
        assert 0.22 < out["L_no_load_linear_iron"] < 0.25                # 0.237 (pub L_m 0.226)

    def test_rated_slip_agrees_with_circuit(self, motor):
        """At s_rated both models carry the 20 °C trap (+34% torque over
        nameplate); the FEM agrees with the published-parameter circuit
        to 3% torque / 2% current, Maxwell and power torques to 1%.
        L_ew (v2) brings FEM closer to the circuit (whose L_sigma_s
        already includes end-winding)."""
        s = motor.induction.s_rated
        fem = MODEL_REGISTRY["im_fem_harmonic"].fn(
            RunConfig(motor=motor, model="im_fem_harmonic", slip=s))
        circ = MODEL_REGISTRY["im_equivalent_circuit"].fn(
            RunConfig(motor=motor, model="im_equivalent_circuit", slip=s))
        assert fem["tau"] / circ["tau"] == pytest.approx(1.0, abs=0.04)
        assert fem["I_1_rms"] / circ["I_1_rms"] == pytest.approx(1.0, abs=0.03)
        assert fem["tau"] / fem["tau_power"] == pytest.approx(1.0, abs=0.015)
        assert fem["tau"] / TAU_RATED == pytest.approx(1.38, abs=0.04)
        assert fem["p_in"] == pytest.approx(fem["p_cu_s"] + fem["p_ag"], rel=1e-6)
        assert fem["p_ag"] == pytest.approx(fem["p_gap_poynting"], rel=0.01)
        assert fem["phase_asymmetry"] < 0.03
        assert fem["ring_model"] == "segment"
        grid = np.array(fem["B_mag_grid"], dtype=float)
        assert grid.shape == (120, 120) and np.isfinite(grid).sum() > 10000


@requires_fem
@pytest.mark.slow
@pytest.mark.timeout(3600)   # the saturation fixed point runs 10-70 solves per point
class TestCreatorFemSaturation:
    """The energy-equivalent saturation fixed point
    on the M800-50A curve ([materials] bh_curve) against the measured
    50 Hz no-load points and the published L_m(I_m) curve — REPORTED and
    pinned, the circuit anchor untouched.

    The FEM has no core-loss branch, so its no-load current is compared
    with the measured REACTIVE current I·sin φ (the quantity the
    published identification keys L_m on), interpolated at the sweep
    voltages. Findings: the saturated no-load current lands on the
    measured one at 0.5 and 1.1 of rated voltage (−0.5 %, −2 %) and runs
    under it between — −5.5 % at 0.7, −11 % at 0.85, −18 % at rated — so
    the catalog curve gives the FEM a sharper knee than the built machine
    has (build factor, stacking, punching: the [iron] finding's ~3.9×
    loss shortfall is the same story). A stacking-factor sensitivity puts
    a number on that: scaling the curve's B axis by k_fe (gross-section
    apparent B) moves the rated no-load current 2.751 A → 2.991 A at
    k_fe = 0.97 → 3.330 A at 0.94, reaching the measured 3.340 A at
    k_fe ≈ 0.94 — an ordinary stack for a punched lamination, so the
    whole shortfall is consistent with the catalog curve describing
    better iron than the machine was built from. Not pinned as a test:
    k_fe is not a measured quantity here, it is one free parameter fitted
    to one number. The identified L_m (IEEE-112
    no-load step with the FEM's own linear locked-rotor leakage) sits on
    the published curve at the low end (+0.2 % at 0.8 A), rises to +16 %
    at the knee and returns to +2 % at 4.1 A; the linear diagnostic at
    mu_r_fe = 1170 lands −5 % at rated only because that value sits near
    the rated-flux permeability. The locked-rotor identification is
    unchanged (linear).

    On the loaded machine the effect splits by slip: at rated slip it is a
    2 % correction, while at standstill on rated voltage it is first-order
    — the linear solve asks for 13.1 T, and relieving that collapses the
    leakage reactance 4.93 → 3.00 Ω, so current rises +30 % and torque
    +76 %. That standstill point sits far outside the measured envelope
    (the 50 Hz locked-rotor rows stop at 47.5 V phase of 230.9) and is
    pinned as model behavior only.

    `B_peak_iron_max` is bounded rather than pinned throughout: it is a
    maximum over iron elements, so it tracks the saturated tooth-tip
    fringe that the permeance-weighted stop rule deliberately gives no
    weight. At a fixed voltage the terminal quantities repeat to four
    digits across solver revisions while it moves ±13 %, and it is not
    monotonic in voltage.
    """

    @pytest.fixture(scope="class")
    def motor_bh(self, motor):
        """The anchor motor with the dataset's M800-50A curve attached.

        The curve is not transcribed into the tracked TOML: the CREATOR
        dataset is CC BY-NC 4.0 and this package is Apache-2.0, so it is
        fetch-gated like the measured CSVs. Without a fetched dataset the
        models fall back to the generic curve and these tests skip.
        """
        if not BH_CSV.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(BH_CSV, delimiter=",", skip_header=1)
        return dc.replace(motor, bh_curve=[[float(H), float(mT) / 1000] for H, mT in d])

    @pytest.fixture(scope="class")
    def out(self, motor_bh):
        return MODEL_REGISTRY["im_fem_locked_rotor"].fn(
            RunConfig(motor=motor_bh, model="im_fem_locked_rotor", nonlinear=True))

    @pytest.fixture(scope="class")
    def no_load_50(self):
        path = IM_DATA / "no_load_tests.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
        r = d[np.abs(d[:, 0] - 50) < 1]
        r = r[np.argsort(r[:, 1])]
        V, I_tot, P = r[:, 1], r[:, 2], r[:, 3]
        pf = P / (3 * V * I_tot)
        return V, I_tot, I_tot * np.sqrt(1 - pf**2)

    def test_setup(self, out, motor_bh):
        motor = motor_bh
        assert out["iron_model"] == "bh"
        assert motor.bh_curve is not None and len(motor.bh_curve) == 32
        assert out["V_no_load_list"] == pytest.approx(
            [f * 400 / 3**0.5 for f in (0.3, 0.5, 0.7, 0.85, 1.0, 1.1)])
        assert all(n >= 5 for n in out["bh_iterations_list"])
        # locked-rotor identification stays linear: the F4 pins hold
        ind = motor.induction
        r10 = dict(zip(out["f_locked_rotor_list"], out["R_2_list"]))[10.0] / dict(ind.R_2_vs_f)[10.0]
        assert 0.86 < r10 < 0.92
        assert 0.22 < out["L_no_load_linear_iron"] < 0.25

    def test_no_load_current_against_measured_reactive(self, out, no_load_50):
        V_m, _I_m, Im_m = no_load_50
        band = {0.5: (0.97, 1.03), 0.7: (0.92, 0.98), 0.85: (0.86, 0.92),
                1.0: (0.79, 0.85), 1.1: (0.95, 1.01)}          # 0.995 0.945 0.887 0.823 0.981
        for frac, V, I_fem in zip((0.3, 0.5, 0.7, 0.85, 1.0, 1.1), out["V_no_load_list"],
                                  out["I_no_load_rms_list"]):
            if frac not in band:
                continue                                        # 69 V: no 50 Hz measurement
            lo, hi = band[frac]
            assert lo < I_fem / np.interp(V, V_m, Im_m) < hi, frac
        # B_peak_iron_max is a fringe maximum, not a converged quantity: at a
        # fixed voltage the terminal values repeat to 4 digits across solver
        # revisions (I 1.805/1.806, L_nl 0.2849/0.2850 at 161.7 V) while this
        # swings 1.56/1.70/2.00 T, and it is not monotonic in voltage. The
        # permeance-weighted stop rule makes the saturated tooth-tip fringe
        # weightless by design, so it is the one output the rule does not
        # settle. Bounded, never pinned.
        B = out["B_peak_iron_no_load_list"]
        assert B[0] < 1.3, B                     # 0.3 V_rated sits below the knee
        assert all(1.3 < b < 2.6 for b in B[1:]), B
        assert max(B) > 1.9, B                   # the knee is reached in the sweep
        I_nl = out["I_no_load_rms_list"]
        assert I_nl[-1] / I_nl[-2] > 1.4                                       # 2.75 → 4.15 A past the knee

    def test_lm_against_published_curve(self, out):
        lm_path = EQ_CSV / "Lm.csv"
        if not lm_path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        curve = np.genfromtxt(lm_path, delimiter=",", skip_header=1)
        ratios = [L / np.interp(Im, curve[:, 0], curve[:, 2])
                  for L, Im in zip(out["L_m_list"], out["I_m_rms_list"])]
        band = [(0.97, 1.04), (1.02, 1.09), (1.08, 1.15), (1.12, 1.19), (1.11, 1.19), (0.97, 1.05)]
        for (lo, hi), r in zip(band, ratios):                   # 1.002 1.058 1.114 1.158 1.159 1.021
            assert lo < r < hi, ratios
        L = out["L_m_list"]
        assert L[-1] < 0.75 * max(L)                             # the knee is there
        assert L[2] > L[0]                                       # rising permeability first

    @pytest.fixture(scope="class")
    def op_points(self, motor_bh):
        """im_fem_harmonic at rated slip and at standstill, linear and saturated."""
        s_rated = motor_bh.induction.s_rated
        return {(s, nl): MODEL_REGISTRY["im_fem_harmonic"].fn(
                    RunConfig(motor=motor_bh, model="im_fem_harmonic", slip=s, nonlinear=nl))
                for s in (s_rated, 1.0) for nl in (False, True)}

    def test_rated_and_locked_rotor_nonlinear(self, op_points, motor_bh):
        motor = motor_bh
        """Saturation is a 2 % correction at rated slip and a first-order
        one at standstill: the linear standstill solve puts 13.9 T in the
        iron, and relieving it collapses the leakage reactance."""
        lin_r, sat_r = op_points[(motor.induction.s_rated, False)], \
            op_points[(motor.induction.s_rated, True)]
        lin_s, sat_s = op_points[(1.0, False)], op_points[(1.0, True)]

        assert sat_r["I_1_rms"] == pytest.approx(9.668, rel=0.03)
        assert sat_r["tau"] == pytest.approx(34.37, rel=0.03)
        assert sat_r["pf"] == pytest.approx(0.906, rel=0.03)
        assert 2.0 < sat_r["B_peak_iron_max"] < 3.2                     # 2.58, bounded
        assert abs(sat_r["I_1_rms"] / lin_r["I_1_rms"] - 1) < 0.03      # 0.992
        assert abs(sat_r["tau"] / lin_r["tau"] - 1) < 0.05              # 1.023

        # The linear standstill B_peak is a diagnostic, not a field.
        assert lin_s["B_peak_iron_max"] > 10                            # 13.10 T
        assert 2.5 < sat_s["B_peak_iron_max"] < 4.0                     # 3.18, bounded
        assert sat_s["I_1_rms"] == pytest.approx(46.49, rel=0.03)
        assert sat_s["tau"] == pytest.approx(52.15, rel=0.03)
        assert sat_s["pf"] == pytest.approx(0.720, rel=0.03)
        assert 0.59 < sat_s["Z_phase_im"] / lin_s["Z_phase_im"] < 0.69  # 3.45/5.37
        assert sat_s["Z_phase_re"] == pytest.approx(lin_s["Z_phase_re"], rel=0.03)
        assert sat_s["I_1_rms"] / lin_s["I_1_rms"] > 1.25               # 1.30
        assert sat_s["tau"] / lin_s["tau"] > 1.6                        # 1.76

        for o in (sat_r, sat_s):
            assert o["tau"] == pytest.approx(o["tau_power"], rel=0.02)
            assert o["iron_model"] == "bh" and o["bh_iterations"] >= 5

    def test_standstill_at_rated_voltage_is_extrapolation(self):
        """Guard on the claim boundary: the measured 50 Hz locked-rotor rows
        stop near a fifth of rated voltage, so the s=1 pins above are model
        behavior with no measurement behind them. If measured rows ever reach
        rated voltage, the pins become a validation claim and must be reworked."""
        path = IM_DATA / "locked_rotor_tests.csv"
        if not path.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
        r = d[np.abs(d[:, 0] - 50) < 1]
        assert r[:, 1].max() < 0.25 * (400 / 3**0.5)                    # 47.5 V of 230.9


class TestCreatorDesignAudit:
    """Regression pin for the Boldea design audit on the CREATOR IM."""

    @pytest.fixture(scope="class")
    def audit_results(self, motor, sweep):
        from phasesweep.audit.im_design_audit import im_design_audit
        return {r.name: r for r in im_design_audit(motor, sweep)}

    def test_rated_slip_in_band(self, audit_results):
        r = audit_results["S_n_pct"]
        assert r.value == pytest.approx(4.667, rel=1e-3)
        assert r.verdict == "pass"

    def test_breakdown_ratio_reasonable(self, audit_results):
        r = audit_results["T_ek_over_T_en"]
        assert 1.5 < r.value < 5.0
        assert r.verdict == "pass"

    def test_starting_current_ratio(self, audit_results):
        r = audit_results["I_start_over_I_n"]
        assert 3.0 < r.value < 10.0

    def test_no_load_current_ratio(self, audit_results):
        r = audit_results["I_0_over_I_n"]
        assert 0.1 < r.value < 0.9

    def test_airgap_flux(self, audit_results):
        r = audit_results["B_g_T"]
        assert 0.4 < r.value < 0.9

    def test_no_warnings(self, audit_results):
        warns = [r for r in audit_results.values() if r.verdict == "warn"]
        assert warns == [], f"unexpected warnings: {warns}"
