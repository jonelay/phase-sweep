"""Creator Case PMSM validation — cross-checks against published reference data.

Tier 1: analytical cross-checks (no simulation).
Tier 2: FEM simulation vs measured waveforms from TU Graz dataset.

Source: arXiv:2501.15921, Dhakal et al., COMPEL 44(4), 2025.
Dataset: DOI 10.3217/sns1d-77m43
"""

import math
from pathlib import Path

import numpy as np
import pytest

from phasesweep.configs import load_motor
from phasesweep.fem_field import harmonics_1sided, solve_field_fem
from phasesweep.solver_params import prepare_fem

MOTOR_TOML = Path(__file__).parent.parent / "motors" / "creator_case_pmsm.toml"
DATA_DIR = Path(__file__).parent.parent / "data" / "creator_case_pmsm"
MEAS_DIR = DATA_DIR / "PM_synchronous_motor" / "Measurement_results" / "No_load_tests"

_has_meas = MEAS_DIR.is_dir()
needs_dataset = pytest.mark.skipif(
    not _has_meas,
    reason=(
        "CREATOR full dataset not found — download from "
        "https://doi.org/10.3217/sns1d-77m43 "
        "(see data/creator_case_pmsm/README.md)"
    ),
)


@pytest.fixture(scope="module")
def creator():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def back_emf_csv():
    return np.genfromtxt(MEAS_DIR / "Back_emf.csv", delimiter=",", skip_header=1)


@pytest.fixture(scope="module")
def cogging_csv():
    return np.genfromtxt(MEAS_DIR / "Cogging_torque.csv", delimiter=",", skip_header=1)


@pytest.fixture(scope="module")
def iron_loss_csv():
    path = DATA_DIR / "iron_losses_measured.csv"
    if not path.is_file():
        pytest.skip(
            "CREATOR-derived CSVs not found — run "
            "python scripts/fetch_creator_dataset.py"
        )
    return np.genfromtxt(path, delimiter=",", skip_header=1)


def _slot_width_ratio(motor):
    """Compute slot_width_ratio from real geometry (slot_opening / slot_pitch)."""
    # Read raw TOML geometry for dimensions not on Geometry dataclass
    import tomllib
    with open(MOTOR_TOML, "rb") as f:
        raw = tomllib.load(f)
    geom = raw["geometry"]
    return geom["slot_opening_width"] / (geom["stator_id"] * math.pi / motor.geometry.n_slots)


@pytest.fixture(scope="module")
def creator_slotted_fem(creator):
    """Slotted FEM result with real slot opening geometry and discrete
    magnet arcs (α_p=0.5), shared across tests 3-5."""
    params = prepare_fem(creator)
    geo = params.geometry
    return solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        n_theta=360, maxh_fraction=0.03 / geo.r_outer,
        n_slots=geo.n_slots, j_s=0.0,
        slot_width_ratio=_slot_width_ratio(creator),
        alpha_p=params.alpha_p,
    )


def test_back_emf_from_psi_f(creator):
    """Published E₀=47.37V @ 2000rpm vs computed from published ψ_f=0.1144Wb."""
    n_p = creator.n_p
    psi_f = creator.psi_f
    speed_rpm = 2000
    omega_e = 2 * math.pi * (speed_rpm / 60) * n_p
    E0 = omega_e * psi_f
    assert E0 == pytest.approx(47.37, rel=0.02)


def test_torque_constant_from_psi_f(creator):
    """Published τ=0.10Nm @ I=0.21A_rms vs computed from published ψ_f=0.1144Wb."""
    n_p = creator.n_p
    psi_f = creator.psi_f
    k_T = 1.5 * n_p * psi_f
    I_peak = 0.21 * math.sqrt(2)
    tau = k_T * I_peak
    assert tau == pytest.approx(0.10, rel=0.05)


def test_max_torque_from_max_current(creator):
    """Published τ_max=0.15Nm @ I=0.30A_rms, β=110° vs saliency-aware torque eq."""
    n_p = creator.n_p
    psi_f = creator.psi_f
    L_d, L_q = creator.L_d, creator.L_q
    I_peak = 0.30 * math.sqrt(2)
    beta = math.radians(110)
    i_d = I_peak * math.cos(beta)
    i_q = I_peak * math.sin(beta)
    tau = 1.5 * n_p * (psi_f * i_q + (L_d - L_q) * i_d * i_q)
    assert tau == pytest.approx(0.15, rel=0.08)


# ---------------------------------------------------------------------------
# Tier 2 — FEM simulation vs measured waveforms
# ---------------------------------------------------------------------------


@needs_dataset
def test_cogging_torque_harmonic_order(cogging_csv):
    """Dominant cogging harmonic is LCM(6,4)=12 per revolution for 6-slot/4-pole."""
    angle_deg = cogging_csv[:, 0]
    torque = cogging_csv[:, 1]

    # Resample to uniform grid (raw data may have duplicates at boundaries)
    n_pts = 4096
    angle_uniform = np.linspace(angle_deg.min(), angle_deg.max(), n_pts, endpoint=False)
    torque_uniform = np.interp(angle_uniform, angle_deg, torque)

    # FFT — orders per revolution
    amps = np.abs(np.fft.rfft(torque_uniform)) / n_pts
    amps[1:] *= 2
    # Order spacing = n_pts / n_pts = 1 cycle per full range (~360°)
    # So index k corresponds to order k per revolution
    dominant_order = np.argmax(amps[1:]) + 1  # skip DC

    assert dominant_order == 12, f"dominant order {dominant_order}, expected 12"
    median_other = np.median(np.delete(amps[1:], dominant_order - 1))
    assert amps[dominant_order] > 10 * median_other


def test_iron_loss_steinmetz_fit(iron_loss_csv):
    """Iron loss P = a * f^alpha: alpha physically reasonable for laminated steel.

    At no-load (constant B), α ≈ 1.0-1.1 is expected when eddy-current losses
    dominate (P_eddy ~ f²·B², but B decreases with f due to skin effect and
    flux redistribution). Classical range 1.3-1.8 applies at fixed B amplitude.
    """
    freq = iron_loss_csv[:, 0]
    loss = iron_loss_csv[:, 1]

    log_f = np.log(freq)
    log_p = np.log(loss)
    A = np.vstack([log_f, np.ones_like(log_f)]).T
    result = np.linalg.lstsq(A, log_p, rcond=None)
    alpha = result[0][0]
    residuals = log_p - A @ result[0]
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    assert 1.0 <= alpha <= 2.0, f"alpha={alpha:.3f} outside [1.0, 2.0]"
    assert r_squared > 0.99, f"R²={r_squared:.4f} < 0.99"


def test_fem_fundamental_range(creator):
    """FEM B₁ fundamental should be in 0.27–0.34 T for this geometry.

    Regression gate: CREATOR's 4-pole geometry with α_p=0.5 discrete
    magnet arcs and ferrite magnets. Smooth-bore FEM with arcs gives
    ~0.305 T fundamental under the square-wave convention (S110).
    The range anchors the FEM solver against published arXiv:2501.15921.
    """
    params = prepare_fem(creator)
    geo = params.geometry
    _, B_r = solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        n_theta=360, maxh_fraction=0.03 / geo.r_outer,
        n_slots=0, j_s=0.0,
        alpha_p=params.alpha_p,
    )
    amps = harmonics_1sided(B_r)
    fund = float(amps[params.n_p]) if params.n_p < len(amps) else 0.0
    assert 0.27 < fund < 0.34, f"FEM fundamental {fund:.4f} outside [0.27, 0.34] T"


def test_slotted_vs_smooth_bore_harmonics(creator, creator_slotted_fem):
    """Slotting introduces harmonics at Q ± n_p = 4 and 8."""
    n_p = creator.n_p

    # Smooth-bore FEM (same discrete arcs as the slotted fixture)
    params = prepare_fem(creator)
    geo = params.geometry
    theta_sm, Br_sm = solve_field_fem(
        geo=geo, n_p=n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        n_theta=360, maxh_fraction=0.03 / geo.r_outer,
        n_slots=0, j_s=0.0,
        alpha_p=params.alpha_p,
    )

    theta_sl, Br_sl = creator_slotted_fem

    h_smooth = harmonics_1sided(Br_sm)
    h_slotted = harmonics_1sided(Br_sl)

    # Slot harmonics at Q ± n_p = 6 ± 2 = 4 and 8. Magnet-arc harmonics
    # sit at odd multiples of n_p (2, 6, 10), so even orders stay near
    # the mesh-noise floor on the smooth bore; factor 1.5 keeps the
    # order-4 check robust against that floor (currently ~2×).
    for order in [4, 8]:
        if order < len(h_smooth) and order < len(h_slotted):
            assert h_slotted[order] > 1.5 * h_smooth[order], (
                f"order {order}: slotted {h_slotted[order]:.6f} "
                f"not > 1.5× smooth {h_smooth[order]:.6f}"
            )

    # Fundamental should stay within 15%
    assert h_slotted[n_p] == pytest.approx(h_smooth[n_p], rel=0.15)


@needs_dataset
def test_back_emf_waveform_shape(back_emf_csv, creator_slotted_fem):
    """Measured and FEM both show significant peaking (peak/fund > 1.25).

    Measured waveform has strong 5th/7th electrical harmonics from rectangular
    magnetization. FEM with discrete magnet arcs, square-wave magnetization
    (S110), and slot harmonics produces comparable peaking (currently
    1.44 vs measured 1.37). Both ratios should exceed 1.25 and agree
    within 10%.
    """
    angle_deg = back_emf_csv[:, 0]
    phase_u = back_emf_csv[:, 1]

    n_pts = 1024
    angle_uniform = np.linspace(angle_deg.min(), angle_deg.max(), n_pts, endpoint=False)
    phase_u_uniform = np.interp(angle_uniform, angle_deg, phase_u)

    meas_peak = np.max(np.abs(phase_u_uniform))
    amps = np.abs(np.fft.rfft(phase_u_uniform)) / n_pts
    amps[1:] *= 2
    meas_fund = amps[2]  # n_p=2 electrical cycles per revolution
    meas_ratio = meas_peak / meas_fund

    _, Br_sl = creator_slotted_fem
    fem_peak = np.max(np.abs(Br_sl))
    h_fem = harmonics_1sided(Br_sl)
    fem_fund = h_fem[2]
    fem_ratio = fem_peak / fem_fund

    assert meas_ratio > 1.25, f"measured peak/fund = {meas_ratio:.3f}, expected > 1.25"
    assert fem_ratio > 1.25, f"FEM peak/fund = {fem_ratio:.3f}, expected > 1.25"
    assert meas_ratio == pytest.approx(fem_ratio, rel=0.10), (
        f"measured ratio {meas_ratio:.3f} vs FEM {fem_ratio:.3f}"
    )


@needs_dataset
def test_back_emf_harmonic_spectrum(back_emf_csv, creator_slotted_fem):
    """FEM now captures the 5th electrical harmonic (trapezoidal signature).

    For 3-phase concentrated winding, the trapezoidal magnet shape produces
    strong 5th and 7th electrical harmonics (mechanical orders 10, 14).
    The square-wave magnetization model (S110) produces the 5th where the
    old sinusoidal source suppressed it; the ideal 2D rectangular model
    overshoots the measured ratio (currently 1.6×) because real
    magnetization profiles and 3D fringing round the edges.
    """
    angle_deg = back_emf_csv[:, 0]
    phase_u = back_emf_csv[:, 1]

    n_pts = 1024
    angle_uniform = np.linspace(angle_deg.min(), angle_deg.max(), n_pts, endpoint=False)
    phase_u_uniform = np.interp(angle_uniform, angle_deg, phase_u)

    meas_amps = np.abs(np.fft.rfft(phase_u_uniform)) / n_pts
    meas_amps[1:] *= 2
    n_p = 2
    meas_fund = meas_amps[n_p]
    # 5th electrical = mechanical order 5*n_p = 10
    meas_5th_ratio = meas_amps[5 * n_p] / meas_fund

    _, Br_sl = creator_slotted_fem
    h_fem = harmonics_1sided(Br_sl)
    fem_fund = h_fem[n_p]
    fem_5th_ratio = h_fem[5 * n_p] / fem_fund if 5 * n_p < len(h_fem) else 0.0

    # Measured 5th electrical > 10% of fundamental (trapezoidal signature)
    assert meas_5th_ratio > 0.10, (
        f"measured 5th/fund = {meas_5th_ratio:.4f}, expected > 0.10"
    )
    # FEM produces a significant 5th (square-wave source), overshooting
    # the measured ratio by less than 2× (ideal rectangular vs real edges)
    assert fem_5th_ratio > 0.10, (
        f"FEM 5th/fund = {fem_5th_ratio:.4f}, expected > 0.10"
    )
    assert fem_5th_ratio < 2.0 * meas_5th_ratio, (
        f"FEM 5th/fund = {fem_5th_ratio:.4f} >= 2× measured {meas_5th_ratio:.4f}"
    )


@needs_dataset
def test_back_emf_fundamental_vs_psi_f(back_emf_csv, creator):
    """FFT fundamental of Phase U matches psi_f-derived E₀ within 5%.

    Validates that the raw waveform data is consistent with the published
    flux linkage scalar (ψ_f = 0.1144 Wb → E₀ = ω_e × ψ_f ≈ 47.92 V).
    """
    angle_deg = back_emf_csv[:, 0]
    phase_u = back_emf_csv[:, 1]

    n_pts = 1024
    angle_uniform = np.linspace(angle_deg.min(), angle_deg.max(), n_pts, endpoint=False)
    phase_u_uniform = np.interp(angle_uniform, angle_deg, phase_u)

    meas_amps = np.abs(np.fft.rfft(phase_u_uniform)) / n_pts
    meas_amps[1:] *= 2
    n_p = creator.n_p
    meas_fund_V = meas_amps[n_p]

    omega_e = 2 * math.pi * (2000 / 60) * n_p
    expected_V = omega_e * creator.psi_f
    assert meas_fund_V == pytest.approx(expected_V, rel=0.05), (
        f"measured fundamental {meas_fund_V:.2f} V vs expected {expected_V:.2f} V"
    )


@needs_dataset
def test_back_emf_normalized_peak_ratio(back_emf_csv, creator_slotted_fem):
    """After fundamental-matched normalization, measured peak within 2× of FEM.

    Sanity gate for the normalization procedure used in report figures.
    Measured waveform has stronger higher harmonics (trapezoidal magnets)
    so its peak exceeds the FEM peak, but not by more than a factor of 2.
    """
    angle_deg = back_emf_csv[:, 0]
    phase_u = back_emf_csv[:, 1]

    n_pts = 1024
    n_p = 2
    angle_uniform = np.linspace(angle_deg.min(), angle_deg.max(), n_pts, endpoint=False)
    phase_u_uniform = np.interp(angle_uniform, angle_deg, phase_u)

    meas_amps = np.abs(np.fft.rfft(phase_u_uniform)) / n_pts
    meas_amps[1:] *= 2
    meas_fund = meas_amps[n_p]

    _, Br_sl = creator_slotted_fem
    fem_amps = harmonics_1sided(Br_sl)
    fem_fund = fem_amps[n_p]

    scale = fem_fund / meas_fund
    normalized_peak = np.max(np.abs(phase_u_uniform)) * scale
    fem_peak = np.max(np.abs(Br_sl))

    assert 0.5 < normalized_peak / fem_peak < 2.0, (
        f"normalized peak {normalized_peak:.4f} / FEM peak {fem_peak:.4f} "
        f"= {normalized_peak/fem_peak:.2f}, expected in [0.5, 2.0]"
    )
