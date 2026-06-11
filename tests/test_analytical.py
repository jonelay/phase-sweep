"""Structural verification of Zhu-Howe-Chan (2002) Eqs 16, 17, 18.

Pure-math checks (numpy only, no FEM/NGSolve) that catch transcription
errors: wrong signs, swapped exponents, missing terms.
"""

import numpy as np
import pytest

MU0 = 4e-7 * np.pi


# ---------------------------------------------------------------------------
# Helper: Eq 17 — inrunner (Rs > Rm > Rr), np != 1
# ---------------------------------------------------------------------------

def _eq17_KB(npp, Rs, Rm, Rr, mu_r, M_n):
    """KB coefficient for Eq 17 (inrunner, np!=1). M_n in A/m."""
    A3n = npp  # radial magnetization
    num = ((A3n - 1)
           + 2 * (Rr / Rm) ** (npp + 1)
           - (A3n + 1) * (Rr / Rm) ** (2 * npp))
    den = ((mu_r + 1) / mu_r * (1 - (Rr / Rs) ** (2 * npp))
           - (mu_r - 1) / mu_r * ((Rm / Rs) ** (2 * npp) - (Rr / Rm) ** (2 * npp)))
    return MU0 * M_n / mu_r * npp / (npp ** 2 - 1) * num / den


def _eq17_fBr(r, npp, Rs, Rm):
    return (r / Rs) ** (npp - 1) * (Rm / Rs) ** (npp + 1) + (Rm / r) ** (npp + 1)


def _eq17_fBt(r, npp, Rs, Rm):
    return -(r / Rs) ** (npp - 1) * (Rm / Rs) ** (npp + 1) + (Rm / r) ** (npp + 1)


def eq17_Br(r, theta, Rs, Rm, Rr, npp, mu_r, M_n):
    KB = _eq17_KB(npp, Rs, Rm, Rr, mu_r, M_n)
    return KB * _eq17_fBr(r, npp, Rs, Rm) * np.cos(npp * theta)


# ---------------------------------------------------------------------------
# Helper: Eq 18 — outrunner (Rs < Rm < Rr), np != 1
# ---------------------------------------------------------------------------

def _eq18_KB(npp, Rs, Rm, Rr, mu_r, M_n):
    A3n = npp
    num = ((A3n - 1) * (Rm / Rr) ** (2 * npp)
           + 2 * (Rm / Rr) ** (npp - 1)
           - (A3n + 1))
    den = ((mu_r + 1) / mu_r * (1 - (Rs / Rr) ** (2 * npp))
           - (mu_r - 1) / mu_r * ((Rs / Rm) ** (2 * npp) - (Rm / Rr) ** (2 * npp)))
    return -MU0 * M_n / mu_r * npp / (npp ** 2 - 1) * num / den


def _eq18_fBr(r, npp, Rs, Rm):
    return (r / Rm) ** (npp - 1) + (Rs / Rm) ** (npp - 1) * (Rs / r) ** (npp + 1)


def _eq18_fBt(r, npp, Rs, Rm):
    return -(r / Rm) ** (npp - 1) + (Rs / Rm) ** (npp - 1) * (Rs / r) ** (npp + 1)


def eq18_Br(r, theta, Rs, Rm, Rr, npp, mu_r, M_n):
    KB = _eq18_KB(npp, Rs, Rm, Rr, mu_r, M_n)
    return KB * _eq18_fBr(r, npp, Rs, Rm) * np.cos(npp * theta)


# ---------------------------------------------------------------------------
# Helper: Eq 16 — np = 1 (inrunner)
# ---------------------------------------------------------------------------

def _eq16_KB(Rs, Rm, Rr, mu_r, M_n):
    """KB for np=1 case.

    Paper's Eq 16 prints coefficient 1 on the ln term, but the L'Hopital
    limit of Eq 17 as np->1 requires coefficient 2.  Verified numerically:
    corrected Eq 16 matches Eq 17 limit to ~1e-5; uncorrected has ~17% error.
    """
    A3n = 1  # radial magnetization, np=1
    num = (A3n * (Rm / Rs) ** 2
           - A3n * (Rr / Rs) ** 2
           + 2 * (Rr / Rs) ** 2 * np.log(Rm / Rr))
    den = ((mu_r + 1) / mu_r * (1 - (Rr / Rs) ** 2)
           - (mu_r - 1) / mu_r * ((Rm / Rs) ** 2 - (Rr / Rm) ** 2))
    return MU0 * M_n / (2 * mu_r) * num / den


def _eq16_fBr(r, Rs):
    return 1 + (Rs / r) ** 2


def eq16_Br(r, theta, Rs, Rm, Rr, mu_r, M_n):
    KB = _eq16_KB(Rs, Rm, Rr, mu_r, M_n)
    return KB * _eq16_fBr(r, Rs) * np.cos(theta)


# ---------------------------------------------------------------------------
# Helper: A-formulation (topology-agnostic 2x2 system)
# ---------------------------------------------------------------------------

def aform_Br(r, theta, s, q, p, npp, mu_r, B_rem):
    """B_r via A-formulation. s=iron BC, q=PM/airgap, p=far iron BC."""
    n = npp
    mu = mu_r
    bp = n * B_rem / (n ** 2 - 1)

    a11 = q ** n + p ** (2 * n) * q ** (-n)
    a12 = -(q ** n + s ** (2 * n) * q ** (-n))
    b1 = -bp * (p ** (n + 1) * q ** (-n) / n + q)

    a21 = (1 / mu) * (q ** (n - 1) - p ** (2 * n) * q ** (-(n + 1)))
    a22 = -(q ** (n - 1) - s ** (2 * n) * q ** (-(n + 1)))
    b2 = -(bp / n) * (1 / mu) * (1 - p ** (n + 1) * q ** (-(n + 1)))

    det = a11 * a22 - a12 * a21
    C2 = (a11 * b2 - a21 * b1) / det

    K = C2 * n * (r ** (n - 1) + s ** (2 * n) * r ** (-(n + 1)))
    return K * np.cos(n * theta)


# ---------------------------------------------------------------------------
# Verification 1: div B = 0
# ---------------------------------------------------------------------------

class TestDivB:
    """d(r·f_Br)/dr + np·f_Bθ = 0 in source-free airgap."""

    EPS = 1e-7

    @pytest.mark.parametrize("npp", [2, 4, 8])
    def test_eq17_divB(self, npp):
        Rs, Rm, _Rr = 0.70, 0.64, 0.30
        r_vals = np.linspace(Rm + 0.002, Rs - 0.002, 8)
        for r in r_vals:
            rfBr_plus = (r + self.EPS) * _eq17_fBr(r + self.EPS, npp, Rs, Rm)
            rfBr_minus = (r - self.EPS) * _eq17_fBr(r - self.EPS, npp, Rs, Rm)
            d_rfBr_dr = (rfBr_plus - rfBr_minus) / (2 * self.EPS)
            fBt = _eq17_fBt(r, npp, Rs, Rm)
            assert d_rfBr_dr == pytest.approx(-npp * fBt, abs=1e-6), \
                f"div B != 0 for Eq 17 at r={r:.4f}, np={npp}"

    @pytest.mark.parametrize("npp", [2, 4, 8])
    def test_eq18_divB(self, npp):
        Rs, Rm, _Rr = 0.30, 0.64, 0.70
        r_vals = np.linspace(Rs + 0.002, Rm - 0.002, 8)
        for r in r_vals:
            rfBr_plus = (r + self.EPS) * _eq18_fBr(r + self.EPS, npp, Rs, Rm)
            rfBr_minus = (r - self.EPS) * _eq18_fBr(r - self.EPS, npp, Rs, Rm)
            d_rfBr_dr = (rfBr_plus - rfBr_minus) / (2 * self.EPS)
            fBt = _eq18_fBt(r, npp, Rs, Rm)
            assert d_rfBr_dr == pytest.approx(-npp * fBt, abs=1e-6), \
                f"div B != 0 for Eq 18 at r={r:.4f}, np={npp}"


# ---------------------------------------------------------------------------
# Verification 2: Limiting cases
# ---------------------------------------------------------------------------

class TestLimitingCases:

    def test_mu_r_unity_eq17(self):
        """mu_r=1 => denominator simplifies to 2·[1 - (Rr/Rs)^{2np}].
        Also cross-checks against the A-formulation at mu_r=1.
        """
        Rs, Rm, Rr, mu_r = 0.70, 0.64, 0.30, 1.0
        M_n = 1.0 / MU0
        B_rem = M_n * MU0
        r_mid = (Rs + Rm) / 2
        for npp in [2, 4, 8]:
            KB = _eq17_KB(npp, Rs, Rm, Rr, mu_r, M_n)
            Br_eq17 = KB * _eq17_fBr(r_mid, npp, Rs, Rm)
            assert np.isfinite(Br_eq17) and abs(Br_eq17) > 0.01
            Br_aform = aform_Br(r_mid, 0.0, Rs, Rm, Rr, npp, mu_r, B_rem)
            assert Br_eq17 == pytest.approx(Br_aform, rel=1e-10), \
                f"Eq17 vs A-form mismatch at mu_r=1, npp={npp}"

    def test_mu_r_unity_eq18(self):
        """mu_r=1 => second denominator term vanishes.
        Also cross-checks against the A-formulation at mu_r=1.
        """
        Rs, Rm, Rr, mu_r = 0.30, 0.64, 0.70, 1.0
        M_n = 1.0 / MU0
        B_rem = M_n * MU0
        r_mid = (Rs + Rm) / 2
        for npp in [2, 4, 8]:
            KB = _eq18_KB(npp, Rs, Rm, Rr, mu_r, M_n)
            Br_eq18 = KB * _eq18_fBr(r_mid, npp, Rs, Rm)
            assert np.isfinite(Br_eq18) and abs(Br_eq18) > 0.01
            Br_aform = aform_Br(r_mid, 0.0, Rs, Rm, Rr, npp, mu_r, B_rem)
            assert Br_eq18 == pytest.approx(Br_aform, rel=1e-10), \
                f"Eq18 vs A-form mismatch at mu_r=1, npp={npp}"

    def test_thin_magnet_eq17(self):
        """Rm -> Rr => B_r -> 0 (no magnet, no field)."""
        Rs, Rr, mu_r = 0.70, 0.30, 1.05
        Rm = Rr + 1e-6
        M_n = 1.2 / MU0
        r_mid = (Rm + Rs) / 2
        for npp in [2, 4, 8]:
            Br = eq17_Br(r_mid, 0.0, Rs, Rm, Rr, npp, mu_r, M_n)
            assert abs(Br) < 1e-3, f"Eq 17 thin magnet: Br={Br}"

    def test_thin_magnet_eq18(self):
        """Rm -> Rr (from below) => B_r -> 0."""
        Rs, Rr, mu_r = 0.30, 0.70, 1.05
        Rm = Rr - 1e-6
        M_n = 1.2 / MU0
        r_mid = (Rs + Rm) / 2
        for npp in [2, 4, 8]:
            Br = eq18_Br(r_mid, 0.0, Rs, Rm, Rr, npp, mu_r, M_n)
            assert abs(Br) < 1e-3, f"Eq 18 thin magnet: Br={Br}"

    @pytest.mark.parametrize("eq_fn,Rs,Rm,Rr", [
        (_eq17_fBr, 0.70, 0.64, 0.30),
        (_eq18_fBr, 0.30, 0.64, 0.70),
    ], ids=["eq17", "eq18"])
    def test_high_pole_count_decay(self, eq_fn, Rs, Rm, Rr):
        """Higher np => faster radial decay => smaller f_Br at midgap."""
        r_mid = (Rs + Rm) / 2 if Rs > Rm else (Rm + Rs) / 2
        # For outrunner, r_mid is between Rs and Rm (airgap)
        nps = [2, 4, 8, 16]
        vals = [abs(eq_fn(r_mid, npp, Rs, Rm)) for npp in nps]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1], \
                f"f_Br did not decay: np={nps[i]} -> {vals[i]:.6e}, np={nps[i+1]} -> {vals[i+1]:.6e}"


# ---------------------------------------------------------------------------
# Verification 3: A-formulation vs Eq 18 (outrunner)
# ---------------------------------------------------------------------------

class TestAFormVsEq18:
    """Two independent derivations of outrunner B_r must agree."""

    GEOMETRIES = [
        # (Rs, Rm, Rr) — outrunner: Rs < Rm < Rr
        (0.30, 0.64, 0.70),
        (0.20, 0.50, 0.55),
        (0.10, 0.40, 0.60),
        (0.25, 0.45, 0.50),
    ]

    @pytest.mark.parametrize("Rs,Rm,Rr", GEOMETRIES)
    @pytest.mark.parametrize("npp", [2, 4, 8])
    def test_aform_matches_eq18(self, Rs, Rm, Rr, npp):
        mu_r = 1.05
        B_rem = 1.2
        M_n = B_rem / MU0
        r_mid = (Rs + Rm) / 2
        theta = 0.0

        # A-formulation: s=Rs (inner iron), q=Rm, p=Rr (outer iron)
        Br_aform = aform_Br(r_mid, theta, s=Rs, q=Rm, p=Rr,
                            npp=npp, mu_r=mu_r, B_rem=B_rem)

        # Eq 18
        Br_eq18 = eq18_Br(r_mid, theta, Rs, Rm, Rr, npp, mu_r, M_n)

        assert Br_aform == pytest.approx(Br_eq18, rel=1e-12), \
            f"A-form={Br_aform:.12e}, Eq18={Br_eq18:.12e}"


# ---------------------------------------------------------------------------
# Also verify A-form vs Eq 17 (inrunner) — regression anchor
# ---------------------------------------------------------------------------

class TestAFormVsEq17:

    GEOMETRIES = [
        (0.70, 0.64, 0.30),
        (0.55, 0.50, 0.20),
        (0.60, 0.40, 0.10),
        (0.50, 0.45, 0.25),
    ]

    @pytest.mark.parametrize("Rs,Rm,Rr", GEOMETRIES)
    @pytest.mark.parametrize("npp", [2, 4, 8])
    def test_aform_matches_eq17(self, Rs, Rm, Rr, npp):
        mu_r = 1.05
        B_rem = 1.2
        M_n = B_rem / MU0
        r_mid = (Rm + Rs) / 2
        theta = 0.0

        # A-formulation: s=Rs (outer iron), q=Rm, p=Rr (inner iron)
        Br_aform = aform_Br(r_mid, theta, s=Rs, q=Rm, p=Rr,
                            npp=npp, mu_r=mu_r, B_rem=B_rem)

        Br_eq17 = eq17_Br(r_mid, theta, Rs, Rm, Rr, npp, mu_r, M_n)

        assert Br_aform == pytest.approx(Br_eq17, rel=1e-12), \
            f"A-form={Br_aform:.12e}, Eq17={Br_eq17:.12e}"


# ---------------------------------------------------------------------------
# Verification 4: Paper's Fig 5 — published motor results
# ---------------------------------------------------------------------------

class TestPaperFig5:
    """Reproduce Fig 5(a) from Zhu, Howe & Chan (2002)."""

    # Motor: 2p=8, Rs=48mm, Rm=40mm, Rr=30mm, B_r=1.2T, mu_r=1.05, alpha_p=1
    P = 4
    RS = 0.048
    RM = 0.040
    RR = 0.030
    B_rem = 1.2
    MU_R = 1.05

    def _M_rn(self, n):
        """Fourier coefficient for radial magnetization, alpha_p=1.
        M_rn = 2*B_r/mu_0 * sin(n*pi/2) / (n*pi/2)
        """
        return 2 * self.B_rem / MU0 * np.sin(n * np.pi / 2) / (n * np.pi / 2)

    def test_total_Br_peak(self):
        """Sum over harmonics n=1,3,5,...,19 at r=40.5mm, theta=0.

        1D estimate: B_r*h_m/(h_m + mu_r*g) = 1.2*10/(10+1.05*8) = 0.65T.
        2D cylindrical result is slightly lower due to radial spreading.
        """
        r_eval = 0.0405
        theta = 0.0
        Br_total = 0.0
        for n in range(1, 20, 2):
            npp = n * self.P
            M_n = self._M_rn(n)
            Br_total += eq17_Br(r_eval, theta, self.RS, self.RM, self.RR,
                                npp, self.MU_R, M_n)
        assert 0.45 < Br_total < 0.75, \
            f"Total Br peak = {Br_total:.4f} T, expected ~0.58 T"

    def test_fundamental_only(self):
        """Fundamental (n=1) alone overshoots the square-wave peak (Gibbs)."""
        r_eval = 0.0405
        npp = self.P
        M_n = self._M_rn(1)
        Br_fund = eq17_Br(r_eval, 0.0, self.RS, self.RM, self.RR,
                          npp, self.MU_R, M_n)
        assert 0.5 < Br_fund < 0.9, \
            f"Fundamental Br = {Br_fund:.4f} T, expected ~0.72 T"


# ---------------------------------------------------------------------------
# Verification 5: np -> 1 continuity (Eq 17 -> Eq 16)
# ---------------------------------------------------------------------------

class TestNpOneContinuity:
    """Eq 17 at np->1 must approach Eq 16 at np=1."""

    RS = 0.70
    RM = 0.64
    RR = 0.30
    MU_R = 1.05
    M_N = 1.2 / MU0

    def _eq17_Br_at_r(self, npp):
        r = (self.RM + self.RS) / 2
        return eq17_Br(r, 0.0, self.RS, self.RM, self.RR,
                       npp, self.MU_R, self.M_N)

    def _eq16_Br_at_r(self):
        r = (self.RM + self.RS) / 2
        return eq16_Br(r, 0.0, self.RS, self.RM, self.RR,
                       self.MU_R, self.M_N)

    def test_convergence_to_np1(self):
        Br_np1 = self._eq16_Br_at_r()
        assert np.isfinite(Br_np1) and abs(Br_np1) > 0

        # Approach np=1 from above
        errors = []
        for npp in [1.1, 1.01, 1.001]:
            Br_approx = self._eq17_Br_at_r(npp)
            rel_err = abs(Br_approx - Br_np1) / abs(Br_np1)
            errors.append(rel_err)

        # Must converge
        assert errors[-1] < 1e-3, \
            f"np=1.001 rel error = {errors[-1]:.6e}, expected < 1e-3"
        # Errors should decrease monotonically
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], \
                f"Convergence not monotonic: {errors}"


# ---------------------------------------------------------------------------
# Verification 6: Frozen regression anchors
# ---------------------------------------------------------------------------

class TestRegressionAnchors:
    """Hardcoded values that catch drift in constants or solver logic.

    If these fail, something fundamental changed — investigate before updating.
    Values computed once and frozen; they do NOT depend on another code path.
    """

    def test_paper_motor_fundamental(self):
        """Zhu & Howe Fig 5 motor: n_p=4, B_rem=1.2T at r=40.5mm, theta=0.

        Re-frozen S110 (= old anchor × 4/π) for the square-wave
        magnetization convention; matches the paper's Fig 5 fundamental
        (~0.72 T, see TestPaperFig5.test_fundamental_only).
        """
        from phasesweep.fem_field import zhu_howe_Br
        Br = zhu_howe_Br(
            np.array([0.0]), n_p=4, B_rem=1.2, r_eval=0.0405,
            r_stator=0.048, r_magnet=0.040, r_rotor=0.030, mu_r_pm=1.05,
        )[0]
        assert Br == pytest.approx(7.171969854301246e-01, rel=1e-10)

    def test_default_inrunner(self):
        """Default geometry (Rs=0.70, Rm=0.64, Rr=0.30), n_p=2, B_rem=1.0T.

        Re-frozen S110 (= old anchor × 4/π, square-wave convention).
        """
        from phasesweep.fem_field import zhu_howe_Br
        Br = zhu_howe_Br(np.array([0.0]), n_p=2, B_rem=1.0)[0]
        assert Br == pytest.approx(7.411415721291222e-01, rel=1e-10)


# ---------------------------------------------------------------------------
# Bridge: production code vs independent reference
# ---------------------------------------------------------------------------

ngsolve = pytest.importorskip("ngsolve")


class TestProductionVsReference:
    """Production zhu_howe_Br must match paper Eq 17 reference.

    Production uses the square-wave magnetization convention, so the
    reference M_n is the square wave's fundamental Fourier coefficient
    (4/π)·B_rem/μ0 (paper Eq 6a at n=1, alpha_p=1).
    """

    RS, RM, RR, MU_R = 0.70, 0.64, 0.30, 1.05

    @pytest.mark.parametrize("npp", [2, 4, 8])
    @pytest.mark.parametrize("r_eval", [0.65, 0.67, 0.69])
    def test_matches_eq17(self, npp, r_eval):
        B_rem = 1.2
        M_n = (4 / np.pi) * B_rem / MU0
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)

        from phasesweep.fem_field import zhu_howe_Br
        Br_prod = zhu_howe_Br(theta, npp, B_rem, r_eval=r_eval)

        Br_ref = np.array([
            eq17_Br(r_eval, t, self.RS, self.RM, self.RR,
                    npp, self.MU_R, M_n)
            for t in theta
        ])

        np.testing.assert_allclose(Br_prod, Br_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Verification 7: alpha_p (pole-arc ratio) scaling
# ---------------------------------------------------------------------------

class TestAlphaPScaling:
    """Partial-pitch magnets (alpha_p < 1) reduce fundamental B_r."""

    def test_alpha_p_reduces_B1(self):
        """alpha_p=0.75 gives lower B₁ than alpha_p=1.0."""
        from phasesweep.fem_field import zhu_howe_Br
        theta = np.array([0.0])
        Br_full = zhu_howe_Br(theta, n_p=4, B_rem=1.2, alpha_p=1.0)[0]
        Br_part = zhu_howe_Br(theta, n_p=4, B_rem=1.2, alpha_p=0.75)[0]
        assert Br_part < Br_full
        # sin(π*0.75/2) = sin(3π/8) ≈ 0.924
        expected_ratio = np.sin(np.pi * 0.75 / 2)
        assert Br_part / Br_full == pytest.approx(expected_ratio, rel=1e-10)

    def test_alpha_p_1_unchanged(self):
        """alpha_p=1.0 gives identical result to omitting it."""
        from phasesweep.fem_field import zhu_howe_Br
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        Br_default = zhu_howe_Br(theta, n_p=4, B_rem=1.2)
        Br_explicit = zhu_howe_Br(theta, n_p=4, B_rem=1.2, alpha_p=1.0)
        np.testing.assert_allclose(Br_explicit, Br_default, rtol=1e-15)
