"""M800-50A iron-loss validation from the CREATOR IM dataset.

Track E Phase 0: a second steel grade for the existing Bertotti model,
touching no induction-motor physics. The CREATOR induction-machine
dataset (DOI 10.3217/kh3v7-dhn98) publishes specific-loss tables at nine
frequencies (50-1000 Hz) for M800-50A — an order of magnitude richer
than the single-grade M250-35A fit the model shipped with — plus a
measured machine-level no-load iron-loss separation (Pfe vs stator flux
squared at five supply frequencies).

Files are gitignored dataset extracts. The fetcher and full induction anchor
are held on ``feature/induction-0.6``; in a standalone v0.5 checkout these
tests skip when the extracts are absent.
"""

import numpy as np
import pytest

from phasesweep.models.iron_loss import (
    bertotti_loss_density,
    fit_bertotti,
    run_iron_loss,
)
from phasesweep.sweep_types import RunConfig
from tests.conftest import CREATOR_IM_SKIP, REPO_ROOT

IM_DIR = REPO_ROOT / "data/creator_case_im/Induction_motor"
STEEL_DIR = (IM_DIR / "Design_parameters/Material_properties/"
             "Stator_and_rotor_iron/Stator_and_rotor_iron_data")
PFE_CSV = (IM_DIR / "Measurement_results/Locked_rotor_and_no_load_tests/"
           "Pfe_flux_sq.csv")


def _steel_table_m800(f_max=1e9, b_max=1.5):
    """(f, B, p) rows from the M800-50A loss tables, B in T."""
    csvs = sorted(STEEL_DIR.glob("*_Hz.csv"))
    if not csvs:
        pytest.skip(CREATOR_IM_SKIP)
    rows = []
    for path in csvs:
        f = float(path.stem.split("_")[0])
        if f > f_max:
            continue
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        for B_mT, _H, p, _mu in data:
            B = B_mT / 1000.0  # tables are in mT (the M250 tables are in T)
            if np.isfinite(p) and p > 0 and 0 < B <= b_max:
                rows.append((f, B, p))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


class TestM800Fit:
    """Global two-term fit over the M800-50A tables (50-1000 Hz, 9 freqs)."""

    @pytest.fixture(scope="class")
    def fit(self):
        f, B, p = _steel_table_m800()
        return fit_bertotti(f, B, p), (f, B, p)

    def test_pinned_fit_and_quality(self, fit):
        """Pinned global fit (240 rows, B <= 1.5 T).

        FINDING: the single global two-term Bertotti tracks M800-50A
        noticeably worse than M250-35A — median ~18% / max ~40% here
        against ~7% / <30% for the thinner grade (test_iron_loss.py
        TestM250Fit). A 0.50 mm high-loss lamination spans the same
        nine decades with more excess-loss content the two-term lump
        cannot absorb. The fit is still usable as a lumped screen; the
        quality gap is the documented price of the lump.
        """
        (k_h, k_e, alpha), (f, B, p) = fit
        assert alpha == pytest.approx(1.49, abs=0.03)
        assert k_h == pytest.approx(0.0238, rel=0.05)
        assert k_e == pytest.approx(1.86e-4, rel=0.08)
        rel = np.abs(np.array([
            bertotti_loss_density(fi, bi, k_h, k_e, alpha)
            for fi, bi in zip(f, B)
        ]) / p - 1)
        assert np.median(rel) < 0.20
        assert np.max(rel) < 0.45

    def test_mains_frequency_underprediction_pinned(self, fit):
        """The global fit under-predicts the 50 Hz rows by ~30%.

        Pinned as a caveat for anyone consuming the global-fit
        coefficients at mains frequency: the relative-residual weighting
        spreads the misfit across nine decades, and the 50 Hz decade
        lands on the low side. A machine that lives at 50 Hz (the
        CREATOR IM) is better served by a band-restricted fit — which
        test_low_band_eddy_not_identifiable shows has its own failure
        mode.
        """
        (k_h, k_e, alpha), _ = fit
        d50 = np.genfromtxt(STEEL_DIR / "50_Hz.csv", delimiter=",",
                            skip_header=1)
        for target_mT, table_p in ((1000, 2.40), (1500, 5.12)):
            row = d50[np.argmin(np.abs(d50[:, 0] - target_mT))]
            assert row[2] == pytest.approx(table_p, rel=0.01)  # table itself
            ratio = bertotti_loss_density(
                50.0, row[0] / 1000.0, k_h, k_e, alpha) / row[2]
            assert ratio == pytest.approx(0.66, abs=0.05)

    def test_low_band_eddy_not_identifiable(self):
        """Restricting the fit to <= 100 Hz returns a NEGATIVE k_e.

        LIMITATION PIN: over a narrow low-frequency band the f and f²
        regressors are nearly collinear and the unconstrained least
        squares in fit_bertotti is free to go unphysical. Downstream
        consumption is protected (IronLossParams rejects k_e <= 0), but
        the fitter itself does not guard — a band-restricted fit must be
        eyeballed before it is pasted into a TOML.
        """
        f, B, p = _steel_table_m800(f_max=100)
        _k_h, k_e, _alpha = fit_bertotti(f, B, p)
        assert k_e < 0


class TestImNoLoadIronLoss:
    """Machine-level no-load iron loss vs the M800-50A table fit.

    Pfe_flux_sq.csv gives measured iron loss vs stator flux linkage
    squared at five supply frequencies (10-50 Hz). Per frequency the loss
    is close to proportional to flux² (transformer-like), so the
    per-frequency slope a(f) = P/psi² carries the frequency dependence,
    and Bertotti predicts a(f) = A·f + C·f².

    CONTRAST WITH THE PMSM FINDING: the PMSM's measured no-load eddy
    share (~11% at 233 Hz) sat far below any catalog-table prediction
    (test_iron_loss.py TestCreatorNoLoadCurve). Here the same
    decomposition agrees with the steel tables: measured eddy share at
    50 Hz ~24%, and the M800-50A table fit puts the share at 21-32% for
    any core B in [0.5, 1.5] T. The machine-level lump is consistent
    with the material data for this machine — pinned as an agreement,
    not a discrepancy.
    """

    @pytest.fixture(scope="class")
    def split(self):
        if not PFE_CSV.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(PFE_CSV, delimiter=",", skip_header=1)
        freqs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        slopes = []
        for i in range(len(freqs)):
            x, y = d[:, 2 * i], d[:, 2 * i + 1]
            m = np.isfinite(x) & np.isfinite(y)
            s, *_ = np.linalg.lstsq(x[m][:, None], y[m], rcond=None)
            slopes.append(float(s[0]))
        slopes = np.array(slopes)
        X = np.vstack([freqs, freqs**2]).T
        (A, C), *_ = np.linalg.lstsq(X, slopes, rcond=None)
        resid = X @ [A, C] - slopes
        r2 = 1 - resid @ resid / np.sum((slopes - slopes.mean()) ** 2)
        return freqs, A, C, r2

    def test_two_term_form_fits(self, split):
        _, A, C, r2 = split
        assert r2 > 0.99
        assert A > 0 and C > 0

    def test_eddy_share_agrees_with_catalog_band(self, split):
        _, A, C, _ = split
        share = C * 50.0**2 / (A * 50.0 + C * 50.0**2)
        assert share == pytest.approx(0.24, abs=0.03)
        f_tab, B_tab, p_tab = _steel_table_m800()
        k_h, k_e, alpha = fit_bertotti(f_tab, B_tab, p_tab)
        shares = []
        for B in (0.5, 1.0, 1.5):
            shares.append(
                (k_e * 50.0**2 * B**2)
                / bertotti_loss_density(50.0, B, k_h, k_e, alpha))
        assert min(shares) - 0.05 < share < max(shares) + 0.05


class TestCreatorImIronBlock:
    """The shipped [iron] block on data/creator_case_im/creator_case_im.toml
    is DERIVED (geometry, winding, steel tables) — not calibrated — and
    runs through the slip-aware stator lump. These tests
    hold the derivation to its inputs and pin the shortfall against the
    measured no-load separation so the file cannot quietly become an
    agreement claim.
    """

    GEOM_CSV = IM_DIR / "Design_parameters/Motor_geometry/Geometry_parameters.csv"
    RHO_STEEL = 7800.0   # kg/m³, M800-50A (EN 10106)
    K_STACK = 0.97

    @pytest.fixture(scope="class")
    def motor(self):
        from phasesweep.machines.configs import load_motor
        return load_motor(REPO_ROOT / "data/creator_case_im/creator_case_im.toml")

    @pytest.fixture(scope="class")
    def geom(self):
        if not self.GEOM_CSV.exists():
            pytest.skip(CREATOR_IM_SKIP)
        rows = np.genfromtxt(self.GEOM_CSV, delimiter=",", skip_header=1,
                             dtype=str)
        return {r[0]: float(r[1]) for r in rows if r[1]}

    @pytest.fixture(scope="class")
    def measured_slopes(self):
        """Per-frequency P_fe / psi_rms² slopes (W/Wb²) at 10-50 Hz."""
        if not PFE_CSV.exists():
            pytest.skip(CREATOR_IM_SKIP)
        d = np.genfromtxt(PFE_CSV, delimiter=",", skip_header=1)
        slopes = {}
        for i, f in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
            x, y = d[:, 2 * i], d[:, 2 * i + 1]
            m = np.isfinite(x) & np.isfinite(y)
            s, *_ = np.linalg.lstsq(x[m][:, None], y[m], rcond=None)
            slopes[f] = float(s[0])
        return slopes

    def test_coefficients_are_the_in_repo_m800_fit(self, motor):
        f, B, p = _steel_table_m800()
        k_h, k_e, alpha = fit_bertotti(f, B, p)
        assert (motor.k_h, motor.k_e, motor.alpha_fe) == (0.0238, 1.86e-4, 1.49)
        assert k_h == pytest.approx(motor.k_h, rel=0.01)
        assert k_e == pytest.approx(motor.k_e, rel=0.01)
        assert alpha == pytest.approx(motor.alpha_fe, abs=0.005)

    def test_core_mass_from_lamination_geometry(self, motor, geom):
        od = geom["Stator iron outer diameter"]
        id_ = geom["Stator iron inner diameter"]
        L = geom["Length of stator iron"]
        t_s = geom["Length of stator teeth"]
        w_t = geom["Width of stator teeth (equivalent width)"]
        Q = geom["No. of stator slots"]
        r_i, r_s = id_ / 2, id_ / 2 + t_s
        v_teeth = Q * w_t * t_s * L
        v_yoke = np.pi * ((od / 2) ** 2 - r_s**2) * L
        m_core = (v_teeth + v_yoke) * self.RHO_STEEL * self.K_STACK
        assert m_core == pytest.approx(motor.m_core, rel=0.01)
        assert r_s > r_i  # slot region is inside the yoke

    def test_b_core_is_the_mass_weighted_flux_estimate(self, motor, geom):
        # 36 slots, 2 layers x 18 turns, 3 phases -> 216 turns/phase;
        # q = 3, 7/9 pitch -> k_w = k_p·k_d (Winding_properties_of_IM.csv)
        Q, p_pairs, q, turns = 36, motor.n_p, 3, 18
        N = Q * 2 * turns / (2 * 3)
        gamma = 2 * np.pi * p_pairs / Q
        k_w = np.sin(7 / 9 * np.pi / 2) * np.sin(q * gamma / 2) / (q * np.sin(gamma / 2))
        assert N == 216 and k_w == pytest.approx(0.902, abs=0.001)
        V_ph = motor.induction.V_LL_rated / np.sqrt(3)
        psi_pk = np.sqrt(2) * V_ph / (2 * np.pi * motor.induction.f_rated)
        phi_pole = psi_pk / (N * k_w)
        id_ = geom["Stator iron inner diameter"]
        od = geom["Stator iron outer diameter"]
        L = geom["Length of stator iron"]
        t_s = geom["Length of stator teeth"]
        w_t = geom["Width of stator teeth (equivalent width)"]
        tau_p = np.pi * id_ / (2 * p_pairs)
        tau_s = np.pi * id_ / Q
        h_y = (od - id_) / 2 - t_s
        B_gap = phi_pole * np.pi / (2 * tau_p * L)
        B_tooth = B_gap * tau_s / (w_t * self.K_STACK)
        B_yoke = phi_pole / 2 / (h_y * L * self.K_STACK)
        v_teeth = Q * w_t * t_s * L
        v_yoke = np.pi * ((od / 2) ** 2 - (id_ / 2 + t_s) ** 2) * L
        B_w = (B_tooth * v_teeth + B_yoke * v_yoke) / (v_teeth + v_yoke)
        assert B_tooth == pytest.approx(1.55, abs=0.01)
        assert B_yoke == pytest.approx(1.36, abs=0.01)
        assert B_w == pytest.approx(motor.B_core, abs=0.005)

    def test_runs_at_rated_supply(self, motor):
        r = run_iron_loss(RunConfig(motor=motor, model="iron_loss"))
        assert r["f_e_Hz"] == 50.0
        assert r["f_rotor_Hz"] == pytest.approx(0.0466667 * 50.0)
        assert r["p_fe"] == pytest.approx(33.4, abs=0.5)

    def test_shortfall_is_a_constant_factor_across_frequency(
            self, motor, measured_slopes):
        """FINDING: measured/model ≈ 3.9 at 50 Hz rated flux, and the
        same factor within ±7% at 10-40 Hz — the frequency dependence
        transports (contrast the PMSM, test_iron_loss.py
        TestCreatorTomlIronBlock), the level does not. The global
        M800-50A fit is ~30% light at 50 Hz (TestM800Fit); the rest is
        build factor plus rotor-surface / tooth-pulsation loss that the
        no-load separation books as iron and a stator lump cannot carry.
        Calibrating B_core to the level would need > 3 T."""
        V_ph = motor.induction.V_LL_rated / np.sqrt(3)
        psi_rms_sq = (V_ph / (2 * np.pi * 50.0)) ** 2
        ratio = {}
        for f, slope in measured_slopes.items():
            model = motor.m_core * bertotti_loss_density(
                f, motor.B_core, motor.k_h, motor.k_e, motor.alpha_fe)
            ratio[f] = slope * psi_rms_sq / model
        assert ratio[50.0] == pytest.approx(3.9, abs=0.15)
        for f in (10.0, 20.0, 30.0, 40.0):
            assert ratio[f] / ratio[50.0] == pytest.approx(1.0, abs=0.07)
        # the level cannot be absorbed by a physical B_core
        need = ratio[50.0] * motor.m_core * bertotti_loss_density(
            50.0, motor.B_core, motor.k_h, motor.k_e, motor.alpha_fe)
        assert motor.m_core * bertotti_loss_density(
            50.0, 3.0, motor.k_h, motor.k_e, motor.alpha_fe) < need
