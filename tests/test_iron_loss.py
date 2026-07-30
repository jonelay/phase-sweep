"""Tests for the lumped Bertotti iron-loss model.

Material-tier validation fits (k_h, k_e, alpha) for M250-35A from the
CREATOR dataset's multi-frequency specific-loss tables; machine-tier
uses the measured no-load iron-loss curve (mass-free f / f² split).
"""

from math import pi

import numpy as np
import pytest

from phasesweep.iron_loss import (
    bertotti_loss_density,
    fit_bertotti,
    run_iron_loss,
)
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solver_params import IronLossParams, prepare_iron_loss
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import REPO_ROOT, STEEL_TABLES_SKIP, make_motor

STEEL_DIR = (REPO_ROOT / "data/creator_case_pmsm/PM_synchronous_motor/"
             "Design_parameters/Material_properties/Electrical_steel/Data")
NO_LOAD_CSV = REPO_ROOT / "data/creator_case_pmsm/iron_losses_measured.csv"

_IRON = dict(k_h=0.0275, k_e=0.00026, alpha_fe=1.7, m_core=0.5, B_core=1.0)


def _steel_table(b_max=1.5):
    csvs = sorted(STEEL_DIR.glob("*_Hz.csv"))
    if not csvs:
        pytest.skip(STEEL_TABLES_SKIP)
    rows = []
    for path in csvs:
        if path.stat().st_size == 0:
            continue  # 800_Hz.csv ships empty in the dataset
        f = float(path.stem.split("_")[0])
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        for B, _H, p, _mu in data:
            if np.isfinite(p) and 0 < B <= b_max:
                rows.append((f, B, p))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


class TestBertottiCore:

    def test_loss_density_handcalc(self):
        # M19 reference coefficients (docs/references.md) at 1 T, 60 Hz
        p = bertotti_loss_density(60.0, 1.0, 0.0275, 0.00026, 1.7)
        assert p == pytest.approx(0.0275 * 60 + 0.00026 * 3600, rel=1e-12)

    def test_fit_recovers_synthetic_coefficients(self):
        rng_f = np.array([50.0, 60, 100, 200, 400, 600, 1000] * 5)
        rng_B = np.repeat([0.3, 0.6, 0.9, 1.2, 1.5], 7)
        p = 0.02 * rng_f * rng_B**1.8 + 1e-4 * rng_f**2 * rng_B**2
        k_h, k_e, alpha = fit_bertotti(rng_f, rng_B, p)
        assert alpha == pytest.approx(1.8, abs=0.005)
        assert k_h == pytest.approx(0.02, rel=1e-3)
        assert k_e == pytest.approx(1e-4, rel=1e-3)

    def test_fit_rejects_nonfinite_table(self):
        f = np.array([50.0, 100.0, 200.0])
        B = np.array([1.0, np.nan, 1.5])
        p = np.array([1.0, 2.0, 4.0])
        with pytest.raises(ValueError, match="non-finite"):
            fit_bertotti(f, B, p)

    def test_fit_rejects_nonpositive_loss(self):
        f = np.array([50.0, 100.0, 200.0])
        B = np.array([1.0, 1.2, 1.5])
        p = np.array([1.0, 0.0, 4.0])
        with pytest.raises(ValueError, match="must be > 0"):
            fit_bertotti(f, B, p)


class TestM250Fit:
    """Global two-term fit over the in-repo M250-35A tables (50-1000 Hz)."""

    @pytest.fixture(scope="class")
    def fit(self):
        f, B, p = _steel_table()
        return fit_bertotti(f, B, p), (f, B, p)

    def test_alpha_and_quality(self, fit):
        (k_h, k_e, alpha), (f, B, p) = fit
        # Pinned fit (203 rows, B <= 1.5 T): a single global two-term
        # Bertotti tracks nine decades of the loss surface to ~7% median.
        assert alpha == pytest.approx(1.62, abs=0.03)
        assert k_h == pytest.approx(0.0153, rel=0.05)
        assert k_e == pytest.approx(6.35e-5, rel=0.08)
        rel = np.abs(np.array([
            bertotti_loss_density(fi, bi, k_h, k_e, alpha) for fi, bi in zip(f, B)
        ]) / p - 1)
        assert np.median(rel) < 0.10
        assert np.max(rel) < 0.30

    def test_matches_dataset_quick_reference_points(self, fit):
        # reference_scalars.json quotes 0.85 / 2.15 W/kg at (1 T, 1.5 T,
        # 50 Hz) from the same characterization; a global 50-1000 Hz fit
        # smooths single points — agreement within 20% is the claim.
        (k_h, k_e, alpha), _ = fit
        assert bertotti_loss_density(50, 1.0, k_h, k_e, alpha) == pytest.approx(
            0.85, rel=0.20)
        assert bertotti_loss_density(50, 1.5, k_h, k_e, alpha) == pytest.approx(
            2.15, rel=0.20)


@pytest.mark.skipif(not NO_LOAD_CSV.exists(), reason=(
    "CREATOR-derived CSVs not found — run "
    "python scripts/fetch_creator_dataset.py"
))
class TestCreatorNoLoadCurve:
    """Measured no-load iron loss: mass-free hysteresis/eddy split.

    At no-load the PM sets the core flux density independent of speed, so
    Bertotti predicts P(f) = A·f + C·f². The two-term fit is excellent
    (R² > 0.997) — but the machine is far more hysteresis-dominated than
    the catalog steel coefficients predict at any plausible core B: the
    measured eddy share at 233 Hz is ~11%, where the M250-35A table fit
    at B ~ 0.4-1 T would put it at 25-40%. Pinned as a FINDING, not an
    agreement anchor — consistent with the earlier observation that the
    curve follows f^1.05 (test_creator_model.py), and with low-B flux
    redistribution / build-factor effects the lumped model does not carry.
    """

    @pytest.fixture(scope="class")
    def split(self):
        d = np.genfromtxt(NO_LOAD_CSV, delimiter=",", skip_header=1)
        f, p = d[:, 0], d[:, 1]
        X = np.vstack([f, f**2]).T
        (A, C), *_ = np.linalg.lstsq(X, p, rcond=None)
        resid = X @ [A, C] - p
        r2 = 1 - np.sum(resid**2) / np.sum((p - p.mean()) ** 2)
        return f, A, C, r2

    def test_two_term_form_fits(self, split):
        _, A, C, r2 = split
        assert r2 > 0.997
        assert A > 0 and C > 0

    def test_eddy_share_pinned_below_catalog_prediction(self, split):
        f, A, C, _ = split
        f_max = f.max()
        eddy_share = C * f_max**2 / (A * f_max + C * f_max**2)
        assert eddy_share == pytest.approx(0.11, abs=0.02)
        # Catalog-coefficient share at the same frequency for any
        # B in [0.3, 1.5] T is well above the measured split.
        f_tab, B_tab, p_tab = _steel_table()
        k_h, k_e, alpha = fit_bertotti(f_tab, B_tab, p_tab)
        for B in (0.3, 0.6, 1.0, 1.5):
            share_cat = (k_e * f_max**2 * B**2) / bertotti_loss_density(
                f_max, B, k_h, k_e, alpha)
            assert share_cat > eddy_share + 0.05, (B, share_cat)


@pytest.mark.skipif(not NO_LOAD_CSV.exists(), reason=(
    "CREATOR-derived CSVs not found — run "
    "python scripts/fetch_creator_dataset.py"
))
class TestCreatorTomlIronBlock:
    """The shipped [iron] block on motors/creator_case_pmsm.toml.

    B_core there is CALIBRATED: it is back-solved so p_fe matches the
    measured no-load loss at the rated point, because the dataset gives
    no core flux density and the lump has no other free parameter. These
    tests hold that calibration to its stated terms — exact at rated,
    visibly drifting away from it — so the file cannot quietly turn into
    an agreement claim it has not earned. CREATOR does not validate
    iron_loss; it is fit to the same curve any such test would check.
    """

    @pytest.fixture(scope="class")
    def shipped(self):
        from phasesweep.configs import load_motor
        m = load_motor(REPO_ROOT / "motors/creator_case_pmsm.toml")
        d = np.genfromtxt(NO_LOAD_CSV, delimiter=",", skip_header=1)
        return m, d[:, 0], d[:, 1]

    def test_calibrated_to_measured_at_rated(self, shipped):
        m, f, p = shipped
        r = run_iron_loss(RunConfig(motor=m, model="iron_loss"))
        assert r["f_e_Hz"] == pytest.approx(66.654, abs=0.01)
        assert r["p_fe"] == pytest.approx(float(np.interp(r["f_e_Hz"], f, p)),
                                          rel=1e-3)

    def test_calibration_does_not_transport_across_frequency(self, shipped):
        # A single-B lump tied down at 66.7 Hz cannot also hold the ends:
        # it runs ~11% light at the bottom of the sweep and ~24% heavy at
        # the top, the same catalog-vs-machine frequency split that
        # TestCreatorNoLoadCurve pins as a finding.
        m, f, p = shipped
        pred = m.m_core * np.array([
            bertotti_loss_density(fi, m.B_core, m.k_h, m.k_e, m.alpha_fe)
            for fi in f])
        rel = pred / p - 1
        assert np.median(np.abs(rel)) == pytest.approx(0.035, abs=0.01)
        assert rel[np.argmin(f)] == pytest.approx(-0.115, abs=0.02)
        assert rel[np.argmax(f)] == pytest.approx(+0.236, abs=0.02)

    def test_coefficients_are_the_in_repo_m250_fit(self, shipped):
        m, _, _ = shipped
        assert (m.k_h, m.k_e, m.alpha_fe) == (0.0153, 6.35e-5, 1.62)

    def test_unfitted_geometry_flux_estimate_runs_light(self, shipped):
        # The alternative the TOML records: FEM per-pole flux over the yoke
        # (2 x 11.6 mm x L_stk) and 1.5 teeth per pole (14.8 mm x L_stk),
        # mass-and-alpha-weighted to B_core = 0.273 T. Kept live so the size
        # of the gap the calibration papers over stays a number in the suite,
        # not a comment. The derivation is shared with the calibration script
        # so the flux snapshot has ONE definition: it is a CREATOR fem solve,
        # not recomputed, and a fem version bump that moves the airgap field
        # must refresh scripts/calibrate_creator_b_core.py (and the matching
        # comment in the motor TOML) or this test pins a stale flux.
        from scripts.calibrate_creator_b_core import unfitted_b_core

        m, f, p = shipped
        B_eff = unfitted_b_core(m)
        assert B_eff == pytest.approx(0.273, abs=0.005)
        p_unfitted = m.m_core * bertotti_loss_density(
            66.654, B_eff, m.k_h, m.k_e, m.alpha_fe)
        assert p_unfitted / float(np.interp(66.654, f, p)) == pytest.approx(
            0.366, abs=0.02)


class TestIronLossModel:

    def test_runner_handcalc(self):
        m = make_motor(**_IRON)
        r = run_iron_loss(RunConfig(motor=m, model="iron_loss"))
        f_e = m.n_p * m.drive.W_REF / (2 * pi)
        assert r["f_e_Hz"] == pytest.approx(f_e)
        assert r["p_fe_hysteresis"] == pytest.approx(
            0.5 * 0.0275 * f_e * 1.0**1.7)
        assert r["p_fe_eddy"] == pytest.approx(0.5 * 0.00026 * f_e**2)
        assert r["p_fe"] == pytest.approx(
            r["p_fe_hysteresis"] + r["p_fe_eddy"])
        assert r["loss_density_W_per_kg"] == pytest.approx(r["p_fe"] / 0.5)

    def test_prepare_raises_on_partial_fields(self):
        m = make_motor(k_h=0.0275, k_e=0.00026)
        with pytest.raises(ValueError, match="iron_loss needs"):
            prepare_iron_loss(m)

    def test_registry_validate_wired(self):
        with pytest.raises(ValueError, match="iron_loss needs"):
            MODEL_REGISTRY["iron_loss"].validate(make_motor())
        MODEL_REGISTRY["iron_loss"].validate(make_motor(**_IRON))

    def test_params_validation(self):
        with pytest.raises(ValueError, match="B_core"):
            IronLossParams(n_p=2, k_h=0.03, k_e=1e-4, alpha_fe=1.7,
                           m_core=0.5, B_core=0.0, W_REF=100.0)

    def test_motor_field_bounds(self):
        with pytest.raises(ValueError, match="alpha_fe"):
            make_motor(alpha_fe=0.5)
        with pytest.raises(ValueError, match="m_core"):
            make_motor(m_core=-1.0)

    def test_iron_fields_flow_into_config_id(self):
        assert make_motor().config_id != make_motor(**_IRON).config_id
        # unset fields keep pre-change hashes (bit-compat rule)
        assert make_motor().config_id == make_motor(
            **{**_IRON, **dict.fromkeys(_IRON)}).config_id

    def test_hash_sensitive_to_w_ref_only(self):
        from phasesweep.motor import DriveParams
        fields = MODEL_REGISTRY["iron_loss"].hash_fields
        m1 = make_motor(**_IRON)
        m2 = make_motor(**_IRON, drive=DriveParams(W_REF=500.0))
        rc1 = RunConfig(motor=m1, model="iron_loss")
        rc2 = RunConfig(motor=m2, model="iron_loss")
        assert compute_run_id(rc1, fields) != compute_run_id(rc2, fields)
        rc3 = RunConfig(motor=m1, model="iron_loss", maxh_fraction=0.2)
        assert compute_run_id(rc1, fields) == compute_run_id(rc3, fields)
