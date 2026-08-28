"""Cross-validation: Parviainen 5 kW / 300 rpm AFPM prototype (1R2S TORUS).

Validates the analytical_axial layer solver against the measured no-load
back-EMF from Parviainen's D.Sc. thesis (LUT 2005), Tables 3.1/3.3 —
the acceptance gate for axial-flux support: E_PM 211 V within
the 5% back-EMF band.

Waveform SHAPE is deliberately not compared: the prototype's magnets
are sinusoidally shaped (near-sinusoidal EMF, Fig. 3.6), while the
solver models an equivalent square-wave magnet (alpha_p = 0.575 matches
the fundamental only) — harmonic content and THD are not comparable for
this anchor.
"""

import warnings
from math import pi, sqrt

import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig
from tests.conftest import REPO_ROOT, requires_fem

MOTOR_TOML = REPO_ROOT / "data/parviainen_afpm/parviainen_5kw_afpm.toml"

E_PM_MEASURED_RMS = 211.0   # V, Table 3.1/3.3, at operating temperature
PSI_F_ANCHOR = 1.5831       # Wb peak, derived from E_PM (TOML header)
SPEED_RPM = 300.0


@pytest.fixture(scope="module")
def motor():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def result(motor):
    rc = RunConfig(motor=motor, model="analytical_axial", n_theta=3600)
    with warnings.catch_warnings():
        # slot_opening_width is not published; the solver warns on the
        # documented slot_width_ratio fallback
        warnings.simplefilter("ignore", UserWarning)
        return MODEL_REGISTRY["analytical_axial"].fn(rc)


class TestParviainenAxial:

    def test_motor_loads_with_axial_geometry(self, motor):
        from phasesweep.machines.geometry import AxialFluxGeometry
        geo = motor.geometry
        assert isinstance(geo, AxialFluxGeometry)
        assert geo.variant == "1R2S"
        assert motor.n_p == 6
        assert motor.B_rem == pytest.approx(1.05)
        assert motor.alpha_p == pytest.approx(0.575)

    def test_backemf_within_validation_band(self, result, motor):
        """Acceptance gate: model E_PM within 5% of measured 211 V.

        The model chain is fully geometric: layer-model B1 at r_mean,
        Carter on the physical gap (slot_width_ratio estimate — the
        opening is unpublished), Qu PM leakage with k_add = 0.95.
        Iron reluctance (k_sat) is not modelled; the open-slot Carter
        estimate over-corrects in the same direction that neglected
        saturation under-corrects, and the net lands within a fraction
        of a percent of measured (documented in the session notes, not
        a fit — no parameter was tuned against 211 V).
        """
        w_mech = 2 * pi * SPEED_RPM / 60
        E_rms = motor.n_p * w_mech * result["flux_linkage_peak"] / sqrt(2)
        assert E_rms == pytest.approx(E_PM_MEASURED_RMS, rel=0.05)

    def test_flux_linkage_vs_anchor(self, result):
        """Model psi_f vs the anchor value derived from measured E_PM."""
        assert result["flux_linkage_peak"] == pytest.approx(
            PSI_F_ANCHOR, rel=0.05)

    @requires_fem
    def test_fem_axial_brackets_measured_across_slot_uncertainty(self, motor):
        """fem_axial E_PM brackets 211 V over the plausible slot geometry.

        This anchor cannot gate the slotted FEM on equality. Parviainen
        publishes no slot opening, and slot_depth / slot_width_ratio in
        the TOML are estimates; the explicit-slot FEM is far more
        sensitive to them than the analytical Carter correction is
        (measured here: E_PM spans 192.6 V at ratio 0.50 to 234.3 V
        slotless — the 5% band is crossed twice inside the uncertainty).
        So the honest statement is a bracket: the measured value lies
        inside the range the model produces over that range of inputs.
        Fixing the ratio to the value that reproduces 211 V would be
        fitting to the answer, not validating against it.

        What IS gated for fem_axial: the slotless-plane agreement with
        the analytical layer model, the mirror-vs-periodic BC check and
        the 1D circuit limit (tests/solvers/test_fem_axial.py), plus
        cross-code agreement with Elmer
        (tests/crossval/test_elmer_axial_pinned.py).
        """
        import dataclasses

        from phasesweep.sweep_types import RunConfig

        w_mech = 2 * pi * SPEED_RPM / 60

        def e_pm(slot_width_ratio: float) -> float:
            geo = dataclasses.replace(
                motor.geometry, slot_width_ratio=slot_width_ratio)
            m = dataclasses.replace(motor, geometry=geo)
            rc = RunConfig(motor=m, model="fem_axial", n_theta=720,
                           maxh_fraction=0.05)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r = MODEL_REGISTRY["fem_axial"].fn(rc)
            return m.n_p * w_mech * r["flux_linkage_peak"] / sqrt(2)

        wide = e_pm(0.50)     # TOML estimate — widest plausible slot
        narrow = e_pm(0.25)   # semi-closed slot, still plausible
        assert wide < E_PM_MEASURED_RMS < narrow, (
            f"measured {E_PM_MEASURED_RMS} V outside the model bracket "
            f"[{wide:.1f}, {narrow:.1f}] V"
        )

    def test_regression_pin(self, result):
        """Pin the v1 model chain (fundamental, Carter, leakage).

        Recorded from the initial implementation: B1 = 0.7137 T,
        k_c = 1.1739, k_sigma = 0.9209, psi_f = 1.5819 Wb. A shift means
        the model changed — bump analytical_axial's version and justify
        here (physics-change checklist).
        """
        assert result["fundamental"] == pytest.approx(0.7137, abs=0.002)
        assert result["k_c"] == pytest.approx(1.174, abs=0.005)
        assert result["k_sigma_pm"] == pytest.approx(0.921, abs=0.005)
        assert result["flux_linkage_peak"] == pytest.approx(1.582, abs=0.01)
