"""Cross-validation: Zhang 2021 9s/8p segmented Halbach PMSM (measured back-EMF).

Source: Zhang et al., IET EPA 15:717–727, 2021.
DOI: 10.1049/elp2.12069

First Halbach motor with full geometry AND measured back-EMF closure.
4-segment ideal Halbach array per pole (kappa_m = 4). Prototype
fabricated and tested at Shandong University.

Measured back-EMF at 400 r/min: ~7 V phase peak. Torque at rated
(10 A rms): 3.6–3.7 N·m. At 3x rated (30 A rms): nonlinear 10.0 N·m,
FEA 9.8–9.9 N·m, linear over-predicts 10.9 N·m.
"""

from math import pi

import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.solver_params import winding_transfer
from tests.conftest import requires_fem

TOML = "data/zhang_2021_halbach/zhang_2021_9s8p.toml"

N_P = 4
N_SLOTS = 9
B_REM = 1.20
MU_R_PM = 1.05
SPEED_RPM = 400
BACKEMF_PHASE_PEAK_V = 7.0
PSI_F = 0.04178
TORQUE_RATED_NM = 3.65
TORQUE_30A_FEA_NM = 9.85


class TestZhang2021Crossval:

    def test_toml_loads(self):
        m = load_motor(TOML)
        assert m.name == "Zhang 2021 Halbach 9s8p"
        assert m.n_p == N_P
        assert m.geometry is not None

    def test_geometry_airgap(self):
        m = load_motor(TOML)
        g = m.geometry
        airgap = g.r_stator - g.r_magnet
        assert airgap == pytest.approx(0.001, abs=1e-6), (
            f"airgap={airgap*1000:.2f} mm, expected 1.0 mm (Table 1)"
        )

    def test_geometry_pm_thickness(self):
        m = load_motor(TOML)
        g = m.geometry
        pm_thick = g.r_magnet - g.r_rotor
        assert pm_thick == pytest.approx(0.003, abs=1e-6), (
            f"PM thickness={pm_thick*1000:.2f} mm, expected 3.0 mm (Table 1)"
        )

    def test_geometry_slot_count(self):
        m = load_motor(TOML)
        assert m.geometry.n_slots == N_SLOTS

    def test_halbach_full_pole_coverage(self):
        """Halbach array spans full pole pitch (alpha_p = 1.0)."""
        m = load_motor(TOML)
        assert m.alpha_p == 1.0

    def test_psi_f_from_backemf(self):
        """psi_f must recover the measured phase back-EMF peak at 400 r/min.
        E_phase_peak = omega_e * psi_f."""
        omega_e = N_P * 2 * pi * SPEED_RPM / 60
        E_phase = omega_e * PSI_F
        assert E_phase == pytest.approx(BACKEMF_PHASE_PEAK_V, rel=0.02), (
            f"E_phase={E_phase:.2f} V, measured ~{BACKEMF_PHASE_PEAK_V} V"
        )

    def test_fractional_slot_ratio(self):
        """9 slots / 8 poles / 3 phases = 0.375 slot/pole/phase (fractional)."""
        m = load_motor(TOML)
        spp = m.geometry.n_slots / (2 * m.n_p * 3)
        assert spp == pytest.approx(0.375, rel=0.01)

    def test_slot_opening_arc(self):
        """slot_opening_width = R_s * b_oa_rad = 28.5 mm * 10 deg."""
        m = load_motor(TOML)
        g = m.geometry
        expected = g.r_stator * 10 * pi / 180
        assert g.slot_opening_width == pytest.approx(expected, rel=0.01)

    def test_winding_transfer_chain(self):
        """Verify the back-EMF closure path via the real winding_transfer.
        psi_f / transfer = B_1 implied, which must be plausible for a
        Halbach array with B_rem = 1.20 T."""
        m = load_motor(TOML)
        transfer = winding_transfer(m)
        B_1_implied = PSI_F / transfer
        assert 0.3 < B_1_implied < 1.5, (
            f"B_1_implied={B_1_implied:.3f} T — should be plausible "
            f"for a Halbach array with B_rem={B_REM} T"
        )

    def test_torque_constant_closure(self):
        """T = 1.5 * n_p * psi_f * I_peak at MTPA (gamma=0, SPM).
        m.I_rated is already peak (loader converts I_rated_rms * sqrt(2)).
        Published rated torque 3.6–3.7 N·m at 10 A rms."""
        m = load_motor(TOML)
        T = 1.5 * m.n_p * m.psi_f * m.I_rated
        assert T == pytest.approx(TORQUE_RATED_NM, rel=0.05), (
            f"T_calc={T:.2f} N·m, published ~{TORQUE_RATED_NM} N·m"
        )

    def test_halbach_magnetization_fields(self):
        """Halbach fields load correctly from TOML."""
        m = load_motor(TOML)
        assert m.magnetization == "halbach"
        assert m.halbach_segments == 4
        assert m.halbach_angles_deg is None  # ideal rule

    def test_toml_psi_f_matches_anchor(self):
        m = load_motor(TOML)
        assert m.psi_f == pytest.approx(PSI_F, rel=1e-6)

    def test_materials(self):
        m = load_motor(TOML)
        assert m.B_rem == pytest.approx(B_REM, rel=0.01)
        assert m.mu_r_pm == pytest.approx(MU_R_PM, rel=0.01)


@requires_fem
@pytest.mark.slow
class TestZhang2021Fem:
    """FEM-gated Zhang 2021 back-EMF closure (H1 from test audit)."""

    def test_backemf_closure(self):
        """FEM B1 -> winding_transfer -> psi_f -> E_phase_peak at 400 rpm.
        Measured: ~7.0 V phase peak (Fig. 7).
        Linear-iron 2D FEM over-predicts (+18%) because the stator
        saturates heavily (peak |B| ~2.5 T); the paper's nonlinear FEA
        matches the measurement.  Tolerance 0.20 covers this."""
        from phasesweep.solvers.fem_field import solve_field_fem
        from phasesweep.solvers.fem_runner import prepare_fem
        from phasesweep.solvers.harmonics import harmonics_1sided

        m = load_motor(TOML)
        p = prepare_fem(m)
        _, Br = solve_field_fem(
            geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
            mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
            n_theta=120, maxh_fraction=0.06,
            magnetization=p.magnetization,
            halbach_segments=p.halbach_segments,
            halbach_angles_rad=p.halbach_angles_rad,
        )
        amps = harmonics_1sided(Br)
        B1 = amps[p.n_p]
        transfer = winding_transfer(m)
        psi_f_fem = B1 * transfer
        omega_e = N_P * 2 * pi * SPEED_RPM / 60
        E_phase_peak = omega_e * psi_f_fem
        assert E_phase_peak == pytest.approx(BACKEMF_PHASE_PEAK_V, rel=0.20), (
            f"E_phase={E_phase_peak:.2f} V, measured ~{BACKEMF_PHASE_PEAK_V} V "
            f"(B1={B1:.4f} T, psi_f={psi_f_fem:.5f} Wb)"
        )

    def test_b1_pinned(self):
        """Pin FEM B1 — tighter than closure so mesh drift hits here first."""
        from phasesweep.solvers.fem_field import solve_field_fem
        from phasesweep.solvers.fem_runner import prepare_fem
        from phasesweep.solvers.harmonics import harmonics_1sided

        m = load_motor(TOML)
        p = prepare_fem(m)
        _, Br = solve_field_fem(
            geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
            mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
            n_theta=120, maxh_fraction=0.06,
            magnetization=p.magnetization,
            halbach_segments=p.halbach_segments,
            halbach_angles_rad=p.halbach_angles_rad,
        )
        B1 = harmonics_1sided(Br)[p.n_p]
        assert B1 == pytest.approx(0.943, rel=0.012)

    def test_halbach_b1_below_radial(self):
        """B1_halbach < B1_radial for this motor — thick rotor back-iron
        already provides the low-reluctance return path that Halbach
        arrays eliminate, so the Halbach advantage is diminished."""
        from phasesweep.solvers.fem_field import solve_field_fem
        from phasesweep.solvers.fem_runner import prepare_fem
        from phasesweep.solvers.harmonics import harmonics_1sided

        m = load_motor(TOML)
        p = prepare_fem(m)
        _, Br_h = solve_field_fem(
            geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
            mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
            n_theta=120, maxh_fraction=0.06,
            magnetization=p.magnetization,
            halbach_segments=p.halbach_segments,
            halbach_angles_rad=p.halbach_angles_rad,
        )
        _, Br_r = solve_field_fem(
            geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
            mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
            n_theta=120, maxh_fraction=0.06,
        )
        B1_h = harmonics_1sided(Br_h)[p.n_p]
        B1_r = harmonics_1sided(Br_r)[p.n_p]
        assert B1_h < B1_r

    def test_halbach_thd_lower_than_radial(self):
        """Halbach THD should be lower than radial (cleaner waveform)."""
        from phasesweep.solvers.fem_field import solve_field_fem
        from phasesweep.solvers.fem_runner import prepare_fem
        from phasesweep.solvers.harmonics import compute_thd, harmonics_1sided

        m = load_motor(TOML)
        p = prepare_fem(m)
        _, Br_h = solve_field_fem(
            geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
            mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
            n_theta=120, maxh_fraction=0.06,
            magnetization=p.magnetization,
            halbach_segments=p.halbach_segments,
            halbach_angles_rad=p.halbach_angles_rad,
        )
        _, Br_r = solve_field_fem(
            geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
            mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
            n_theta=120, maxh_fraction=0.06,
        )
        thd_h = compute_thd(harmonics_1sided(Br_h), p.n_p)
        thd_r = compute_thd(harmonics_1sided(Br_r), p.n_p)
        assert thd_h < thd_r
