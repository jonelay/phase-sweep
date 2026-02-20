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
from phasesweep.fem_field import solve_field_fem, harmonics_1sided

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
    return load_motor("motors/creator_case_pmsm.toml")


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
    geom = motor.metadata["geometry"]
    return geom["slot_opening_width"] / (geom["stator_id"] * math.pi / motor.config["n_slots"])


@pytest.fixture(scope="module")
def creator_slotted_fem(creator):
    """Slotted FEM result with real slot opening geometry, shared across tests 3-5."""
    cfg = creator.config
    return solve_field_fem(
        n_p=cfg["n_p"], psi_f=cfg["psi_f"], L_d=cfg["L_d"], L_q=cfg["L_q"],
        n_theta=360, maxh=0.03,
        n_slots=cfg["n_slots"], j_s=0.0,
        N=cfg["N"], k_w=cfg["k_w"], L_stk=cfg["L_stk"],
        slot_width_ratio=_slot_width_ratio(creator),
    )


def test_back_emf_from_psi_f(creator):
    """Published E₀=47.37V @ 2000rpm vs computed from published ψ_f=0.1144Wb."""
    n_p = creator.config["n_p"]
    psi_f = creator.config["psi_f"]
    speed_rpm = 2000
    omega_e = 2 * math.pi * (speed_rpm / 60) * n_p
    E0 = omega_e * psi_f
    assert E0 == pytest.approx(47.37, rel=0.02)


def test_torque_constant_from_psi_f(creator):
    """Published τ=0.10Nm @ I=0.21A_rms vs computed from published ψ_f=0.1144Wb."""
    n_p = creator.config["n_p"]
    psi_f = creator.config["psi_f"]
    k_T = 1.5 * n_p * psi_f
    I_peak = 0.21 * math.sqrt(2)
    tau = k_T * I_peak
    assert tau == pytest.approx(0.10, rel=0.05)


def test_max_torque_from_max_current(creator):
    """Published τ_max=0.15Nm @ I=0.30A_rms, β=110° vs saliency-aware torque eq."""
    cfg = creator.config
    n_p = cfg["n_p"]
    psi_f = cfg["psi_f"]
    L_d, L_q = cfg["L_d"], cfg["L_q"]
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


def test_slotted_vs_smooth_bore_harmonics(creator, creator_slotted_fem):
    """Slotting introduces harmonics at Q ± n_p = 4 and 8."""
    cfg = creator.config
    n_p = cfg["n_p"]

    # Smooth-bore FEM
    theta_sm, Br_sm = solve_field_fem(
        n_p=n_p, psi_f=cfg["psi_f"], L_d=cfg["L_d"], L_q=cfg["L_q"],
        n_theta=360, maxh=0.03,
        n_slots=0, j_s=0.0,
        N=cfg["N"], k_w=cfg["k_w"], L_stk=cfg["L_stk"],
    )

    theta_sl, Br_sl = creator_slotted_fem

    h_smooth = harmonics_1sided(Br_sm)
    h_slotted = harmonics_1sided(Br_sl)

    # Slot harmonics at Q ± n_p = 6 ± 2 = 4 and 8
    for order in [4, 8]:
        if order < len(h_smooth) and order < len(h_slotted):
            assert h_slotted[order] > 2 * h_smooth[order], (
                f"order {order}: slotted {h_slotted[order]:.6f} "
                f"not > 2× smooth {h_smooth[order]:.6f}"
            )

    # Fundamental should stay within 15%
    assert h_slotted[n_p] == pytest.approx(h_smooth[n_p], rel=0.15)


@needs_dataset
def test_back_emf_waveform_shape(back_emf_csv, creator_slotted_fem):
    """Measured back-EMF is more trapezoidal than FEM (higher peak/fund ratio).

    Measured waveform has strong 5th/7th electrical harmonics from rectangular
    magnetization, giving peak/fund > 1.25. FEM with sinusoidal magnetization
    has lower ratio (slot harmonics add some peaking, but less than real magnet
    shape). The key assertion: measured ratio > FEM ratio.
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
    assert meas_ratio > fem_ratio, (
        f"measured ratio {meas_ratio:.3f} should exceed FEM {fem_ratio:.3f}"
    )


@needs_dataset
def test_back_emf_harmonic_spectrum(back_emf_csv, creator_slotted_fem):
    """FEM underpredicts 5th electrical harmonic (trapezoidal signature).

    For 3-phase concentrated winding, the trapezoidal magnet shape produces
    strong 5th and 7th electrical harmonics (mechanical orders 10, 14).
    The 3rd electrical (order 6) is near-zero in phase voltage (zero-sequence).
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
    # FEM underpredicts 5th (sinusoidal magnetization model)
    assert fem_5th_ratio < meas_5th_ratio, (
        f"FEM 5th/fund = {fem_5th_ratio:.4f} >= measured {meas_5th_ratio:.4f}"
    )
