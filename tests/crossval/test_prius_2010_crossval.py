"""Cross-validation: Toyota Prius 2010 IPM (ORNL teardown geometry).

Source: Burress et al., ORNL/TM-2010/253, March 2011
        "Evaluation of the 2010 Toyota Prius Hybrid Synergy Drive System"

This motor has FULL GEOMETRY from the ORNL teardown (Table 2.7) but NO
published L_d/L_q/psi_f — only graphical back-EMF and locked-rotor data.
Validation is limited to geometry self-consistency and dimensional checks
against the ORNL measurements.

Eligible for: geometry/field solvers (with V-IPM radii caveat).
Not eligible for: rated_torque, stall_torque (no d-q model).
"""

import pytest

from phasesweep.machines.configs import load_motor

TOML = "data/prius_2010/prius_2010.toml"


@pytest.fixture(scope="module")
def motor():
    return load_motor(TOML)


class TestPrius2010Geometry:

    def test_toml_loads(self, motor):
        assert motor.name == "Toyota Prius 2010 IPM"
        assert motor.n_p == 4
        assert motor.geometry is not None

    def test_stator_od(self, motor):
        """Stator OD = 26.4 cm (Table 2.7)."""
        assert motor.geometry.r_outer * 2 == pytest.approx(0.264, abs=0.001)

    def test_stator_bore(self, motor):
        """Stator ID = 16.19 cm (Table 2.7)."""
        assert motor.geometry.r_stator * 2 == pytest.approx(0.1619, abs=0.001)

    def test_rotor_od(self, motor):
        """Rotor OD = 16.04 cm (Table 2.7)."""
        assert motor.geometry.r_magnet * 2 == pytest.approx(0.1604, abs=0.001)

    def test_airgap(self, motor):
        """Air gap = 0.73 mm (Table 2.7)."""
        gap = (motor.geometry.r_stator - motor.geometry.r_magnet) * 1000
        assert gap == pytest.approx(0.73, abs=0.05)

    def test_slot_count(self, motor):
        """48 stator slots (Table 2.7)."""
        assert motor.geometry.n_slots == 48

    def test_phase_resistance(self, motor):
        """R_s = 0.077 Ohm at 21°C (Table 2.7)."""
        assert motor.R_s == pytest.approx(0.077, abs=0.001)

    def test_same_stator_as_camry(self, motor):
        """The 2010 Prius shares stator/rotor diameters with the 2007
        Camry (ORNL report: "diameters ... are identical")."""
        assert motor.geometry.r_outer * 2 == pytest.approx(0.264, abs=0.001)
        assert motor.geometry.r_magnet * 2 == pytest.approx(0.1604, abs=0.001)

    def test_shorter_stack_than_2004(self, motor):
        """Stack length 50.8 mm vs 84 mm for 2004 Prius — a 40% reduction
        despite 20% higher power rating."""
        assert motor.L_stk == pytest.approx(0.0508, abs=0.001)
        assert motor.L_stk < 0.060

    def test_no_dq_model(self, motor):
        """L_d, L_q, psi_f NOT published in ORNL report — confirm None."""
        assert motor.L_d is None
        assert motor.L_q is None
        assert motor.psi_f is None

    def test_power_rating(self, motor):
        """60 kW peak (Table 2.1)."""
        assert motor.drive.U_DC == 650.0
