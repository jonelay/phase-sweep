"""Cross-validation: NWPU 12 kW 3-segment Halbach PMSM (measured back-EMF).

Source: Liu, Shang, Gong, WEVJ 16(8):413, 2025.
DOI: 10.3390/wevj16080413

First Halbach-magnetization motor in the validation set. 20-pole/24-slot
inrunner SPM with 3-segment Halbach array per pole (main + two 45-deg
sub-poles). Prototype manufactured and tested at NWPU.

Measured line back-EMF at 1000 r/min: 75.2 V peak (FEM: 74.4, analytical: 76.4).
Rated: 5000 r/min, 23 Nm, 12 kW.
"""

from math import pi, sqrt

import pytest

from phasesweep.machines.configs import load_motor
from tests.conftest import requires_fem

TOML = "data/halbach_nwpu/halbach_nwpu_12kw.toml"

N_P = 10
PSI_F = 0.04147
SPEED_1000_RPM = 1000
SPEED_RATED_RPM = 5000
BACKEMF_LINE_PEAK_MEASURED = 75.2   # V, line-to-line peak at 1000 rpm
BACKEMF_PHASE_PEAK_FEM_5000 = 214.8  # V, FEM phase peak at 5000 rpm
TORQUE_RATED = 23.0                  # Nm
POWER_RATED = 12.0                   # kW


class TestHalbachNwpuCrossval:

    def test_toml_loads(self):
        m = load_motor(TOML)
        assert m.name == "NWPU Halbach 12 kW"
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
        assert m.geometry.n_slots == 24

    def test_halbach_full_pole_coverage(self):
        """Halbach array spans full pole pitch (alpha_p = 1.0)."""
        m = load_motor(TOML)
        assert m.alpha_p == 1.0

    def test_halbach_magnetization_fields(self):
        """Halbach fields load correctly from TOML."""
        m = load_motor(TOML)
        assert m.magnetization == "halbach"
        assert m.halbach_segments == 3
        assert m.halbach_angles_deg == (0.0, -45.0, -135.0)

    def test_toml_psi_f_matches_anchor(self):
        """TOML psi_f matches the measured-derived anchor constant."""
        m = load_motor(TOML)
        assert m.psi_f == pytest.approx(PSI_F, rel=1e-6)

    def test_toml_b_rem_matches_anchor(self):
        """TOML B_rem matches the anchor constant."""
        m = load_motor(TOML)
        assert m.B_rem == pytest.approx(1.35, rel=1e-6)

    def test_backemf_line_peak_at_1000rpm(self):
        """psi_f must recover the measured line back-EMF peak at 1000 r/min.
        E_line_peak = sqrt(3) * omega_e * psi_f."""
        omega_e = N_P * 2 * pi * SPEED_1000_RPM / 60
        E_line_peak = sqrt(3) * omega_e * PSI_F
        assert E_line_peak == pytest.approx(BACKEMF_LINE_PEAK_MEASURED, rel=0.02), (
            f"E_line_peak={E_line_peak:.1f} V, measured {BACKEMF_LINE_PEAK_MEASURED} V"
        )

    def test_backemf_phase_peak_at_5000rpm_fem(self):
        """psi_f cross-check against FEM phase peak at 5000 r/min.
        The measured-derived psi_f (from 1000 rpm) should predict the
        FEM back-EMF at 5000 rpm within ~5% (accounts for 1.1% FEM/measured
        difference plus speed extrapolation)."""
        omega_e = N_P * 2 * pi * SPEED_RATED_RPM / 60
        E_phase_peak = omega_e * PSI_F
        assert E_phase_peak == pytest.approx(BACKEMF_PHASE_PEAK_FEM_5000, rel=0.05), (
            f"E_phase_peak={E_phase_peak:.1f} V at 5000 rpm, "
            f"FEM {BACKEMF_PHASE_PEAK_FEM_5000} V"
        )

    def test_rated_power_consistency(self):
        """T * omega = P_rated."""
        omega = 2 * pi * SPEED_RATED_RPM / 60
        P = TORQUE_RATED * omega
        assert P == pytest.approx(POWER_RATED * 1000, rel=0.01)

    def test_fractional_slot_ratio(self):
        """24 slots / 20 poles / 3 phases = 0.4 slot/pole/phase (fractional)."""
        m = load_motor(TOML)
        spp = m.geometry.n_slots / (2 * m.n_p * 3)
        assert spp == pytest.approx(0.4, rel=0.01)

    def test_psi_f_consistency_between_speeds(self):
        """psi_f derived from measured (1000 rpm) vs FEM (5000 rpm)
        should agree within 2% — both are anchored in the same motor."""
        omega_1k = N_P * 2 * pi * 1000 / 60
        psi_measured = (BACKEMF_LINE_PEAK_MEASURED / sqrt(3)) / omega_1k

        omega_5k = N_P * 2 * pi * 5000 / 60
        psi_fem = BACKEMF_PHASE_PEAK_FEM_5000 / omega_5k

        assert psi_measured == pytest.approx(psi_fem, rel=0.02), (
            f"psi_measured={psi_measured:.5f}, psi_fem={psi_fem:.5f}"
        )


@requires_fem
@pytest.mark.slow
class TestHalbachNwpuFem:
    """FEM-gated NWPU Halbach crossval tests."""

    def test_halbach_b1_higher_than_radial(self):
        """Halbach FEM B1 exceeds radial at same B_rem."""
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
        h_h = harmonics_1sided(Br_h)
        h_r = harmonics_1sided(Br_r)
        assert h_h[p.n_p] > h_r[p.n_p] * 1.02

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


TOML_SLEEVED = "data/halbach_nwpu/halbach_nwpu_12kw_sleeved.toml"


class TestHalbachSleevedVariant:
    """What-if variant: NWPU Halbach with a 0.5 mm iron retaining sleeve.

    Exercises the surface-mount Halbach + bridge_thickness combination
    unlocked by decoupling bridge_thickness from magnet_mount='inset'.
    Not a crossval — no measured data exists for the sleeved rotor.
    """

    def test_toml_loads_with_bridge(self):
        m = load_motor(TOML_SLEEVED)
        assert m.geometry.magnet_mount == "surface"
        assert m.geometry.bridge_thickness == pytest.approx(0.5e-3)
        assert m.magnetization == "halbach"
        assert m.halbach_segments == 3

    def test_psi_f_not_carried_over(self):
        """The anchor's measured-derived psi_f must not leak into the
        variant — the sleeve shunts flux, so it does not apply."""
        m = load_motor(TOML_SLEEVED)
        assert m.psi_f is None

    def test_geometry_matches_anchor_except_bridge(self):
        m_a = load_motor(TOML)
        m_s = load_motor(TOML_SLEEVED)
        for attr in ("r_outer", "r_stator", "r_magnet", "r_rotor",
                     "r_inner", "n_slots"):
            assert getattr(m_s.geometry, attr) == getattr(m_a.geometry, attr)
        assert m_a.geometry.bridge_thickness is None
        assert m_s.geometry.config_id != m_a.geometry.config_id

    def test_prepare_fem_accepts_variant(self):
        from phasesweep.solvers.fem_runner import prepare_fem
        p = prepare_fem(load_motor(TOML_SLEEVED))
        assert p.geometry.bridge_thickness == pytest.approx(0.5e-3)

    def test_fem_gates_linear_solve(self):
        """The sleeve saturates by design — linear solve must be refused."""
        from phasesweep.solvers.fem_field import solve_field_fem
        from phasesweep.solvers.fem_runner import prepare_fem
        p = prepare_fem(load_motor(TOML_SLEEVED))
        with pytest.raises(ValueError, match="bridge_thickness requires nonlinear"):
            solve_field_fem(
                geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
                mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
                n_theta=120, maxh_fraction=0.06,
                magnetization=p.magnetization,
                halbach_segments=p.halbach_segments,
                halbach_angles_rad=p.halbach_angles_rad,
            )


@requires_fem
@pytest.mark.slow
class TestHalbachSleevedVariantFem:
    """First full FEM solve of the Halbach + bridge combination."""

    def test_iron_sleeve_shunts_fundamental(self):
        """The 0.5 mm steel sleeve must cost airgap fundamental vs the
        exposed-face anchor (nonlinear, same settings). Measured once at
        these settings: 0.769 vs 0.988 T, a ~22% penalty — the reason
        real high-speed Halbach sleeves are non-magnetic."""
        from phasesweep.solvers.fem_field import solve_field_fem
        from phasesweep.solvers.fem_runner import prepare_fem
        from phasesweep.solvers.harmonics import harmonics_1sided

        b1 = {}
        for key, toml in (("sleeved", TOML_SLEEVED), ("anchor", TOML)):
            p = prepare_fem(load_motor(toml))
            _, Br = solve_field_fem(
                geo=p.geometry, n_p=p.n_p, B_rem=p.B_rem,
                mu_r_pm=p.mu_r_pm, mu_r_fe=p.mu_r_fe,
                n_theta=120, maxh_fraction=0.06, nonlinear=True,
                magnetization=p.magnetization,
                halbach_segments=p.halbach_segments,
                halbach_angles_rad=p.halbach_angles_rad,
            )
            b1[key] = harmonics_1sided(Br)[p.n_p]
        assert b1["sleeved"] < 0.95 * b1["anchor"], (
            f"sleeved B1={b1['sleeved']:.4f} T not below "
            f"0.95 * anchor B1={b1['anchor']:.4f} T"
        )
        assert b1["sleeved"] > 0.5 * b1["anchor"], (
            f"sleeved B1={b1['sleeved']:.4f} T implausibly low vs "
            f"anchor {b1['anchor']:.4f} T — sleeve should shunt, not kill"
        )
