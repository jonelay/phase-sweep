"""Unit tests for the quasi-3D Cartesian axial-flux layer solver."""

import warnings
from math import pi

import numpy as np
import pytest

from phasesweep.machines.geometry import axial_flux
from phasesweep.machines.motor import Motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solvers.analytical_axial import (
    ALPHA_P_SINUSOIDAL_EQUIV,
    GAP_LINKAGE,
    AnalyticalAxialParams,
    _layer_amplitude,
    axial_layer_Bz_series,
    carter_airgap_axial,
    pm_leakage_factor,
    prepare_analytical_axial,
    winding_transfer_axial,
)
from phasesweep.sweep_types import RunConfig


def slotless_1r2s(**over):
    kw = dict(
        variant="1R2S", r_out=0.164, r_in=0.0985,
        airgap=0.0015, magnet_thickness=0.004,
        rotor_yoke=0.020, stator_yoke=0.020,
    )
    kw.update(over)
    return axial_flux(**kw)


def slotted_1r2s(**over):
    return slotless_1r2s(
        n_slots=36, slot_depth=0.015, slot_width_ratio=0.5, **over
    )


class TestLayerAmplitude:

    def test_long_wavelength_limit_is_1d_circuit(self):
        """m -> 0 recovers B = b_src * l_pm / (l_pm + mu_r * g')."""
        l_pm, g, mu_r, b_src = 0.004, 0.0015, 1.05, 1.0
        A = _layer_amplitude(1e-6, b_src, g, l_pm, mu_r)
        assert A == pytest.approx(l_pm / (l_pm + mu_r * g), rel=1e-6)

    def test_amplitude_decreases_with_gap(self):
        a1 = _layer_amplitude(45.7, 1.0, 0.0015, 0.004, 1.05)
        a2 = _layer_amplitude(45.7, 1.0, 0.0030, 0.004, 1.05)
        assert a2 < a1

    def test_amplitude_increases_with_magnet_thickness(self):
        a1 = _layer_amplitude(45.7, 1.0, 0.0015, 0.004, 1.05)
        a2 = _layer_amplitude(45.7, 1.0, 0.0015, 0.008, 1.05)
        assert a2 > a1

    def test_short_wavelength_attenuates(self):
        """High orders decay — the series tail is negligible."""
        a1 = _layer_amplitude(45.7, 1.0, 0.0015, 0.004, 1.05)
        a9 = _layer_amplitude(9 * 45.7, 1.0, 0.0015, 0.004, 1.05)
        assert a9 < a1


class TestSeries:

    def test_fundamental_bin_matches_single_amplitude(self):
        n_p, tau_p = 6, pi * 0.13125 / 6
        theta = np.linspace(0, 2 * pi, 3600, endpoint=False)
        B = axial_layer_Bz_series(
            theta, n_p, 1.05, tau_p=tau_p, g_prime=0.0015, l_pm=0.004,
            alpha_p=0.8, max_order=1800,
        )
        A1 = _layer_amplitude(
            pi / tau_p, (4 * 1.05 / pi) * np.sin(0.8 * pi / 2),
            0.0015, 0.004, 1.05,
        )
        fft = np.fft.rfft(B) / len(B) * 2
        assert abs(fft[n_p]) == pytest.approx(A1, rel=1e-9)

    def test_alpha_two_thirds_kills_third_harmonic(self):
        n_p, tau_p = 6, pi * 0.13125 / 6
        theta = np.linspace(0, 2 * pi, 3600, endpoint=False)
        B = axial_layer_Bz_series(
            theta, n_p, 1.05, tau_p=tau_p, g_prime=0.0015, l_pm=0.004,
            alpha_p=2 / 3, max_order=1800,
        )
        fft = np.abs(np.fft.rfft(B) / len(B) * 2)
        assert fft[3 * n_p] < 1e-12 * fft[n_p]

    def test_no_low_pole_count_singularity(self):
        """Cartesian model has no (n^2 - 1) pole: n_p = 1 works."""
        theta = np.linspace(0, 2 * pi, 360, endpoint=False)
        B = axial_layer_Bz_series(
            theta, 1, 1.0, tau_p=0.4, g_prime=0.002, l_pm=0.005,
        )
        assert np.all(np.isfinite(B))
        assert np.max(np.abs(B)) > 0

    def test_max_order_caps_series(self):
        theta = np.linspace(0, 2 * pi, 3600, endpoint=False)
        B = axial_layer_Bz_series(
            theta, 6, 1.0, tau_p=0.0687, g_prime=0.0015, l_pm=0.004,
            max_order=7,  # only k=1 (order 6) survives
        )
        fft = np.abs(np.fft.rfft(B) / len(B) * 2)
        assert fft[18] < 1e-14

    def test_invalid_inputs_raise(self):
        theta = np.zeros(4)
        with pytest.raises(ValueError, match="n_p"):
            axial_layer_Bz_series(
                theta, 0, 1.0, tau_p=0.1, g_prime=0.001, l_pm=0.004)
        with pytest.raises(ValueError, match="must all be > 0"):
            axial_layer_Bz_series(
                theta, 6, 1.0, tau_p=0.1, g_prime=0.0, l_pm=0.004)


class TestCarter:

    def test_slotless_is_identity(self):
        g_prime, k_c = carter_airgap_axial(slotless_1r2s(), 1.05)
        assert k_c == 1.0
        assert g_prime == pytest.approx(0.0015)

    def test_slotted_widens_gap(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g_prime, k_c = carter_airgap_axial(slotted_1r2s(), 1.05)
        assert k_c > 1.0
        # Eq. 2.11: Carter multiplies the physical gap only
        assert g_prime == pytest.approx(k_c * 0.0015)

    def test_fallback_warns(self):
        with pytest.warns(UserWarning, match="slot_opening_ratio unset"):
            carter_airgap_axial(slotted_1r2s(), 1.05)

    def test_explicit_opening_no_warning(self):
        geo = slotted_1r2s(slot_opening_width=0.003)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g_prime, k_c = carter_airgap_axial(geo, 1.05)
        # narrower opening than the 0.5 body-width fallback -> smaller k_c
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, k_c_fallback = carter_airgap_axial(slotted_1r2s(), 1.05)
        assert 1.0 < k_c < k_c_fallback


class TestPmLeakage:

    def test_in_unit_interval(self):
        geo = slotted_1r2s()
        tau_p = pi * geo.r_mean / 6
        k = pm_leakage_factor(geo, 1.05, 0.575, 0.0018, tau_p)
        assert 0.8 < k < 1.0

    def test_full_pitch_magnet_drops_m2m_term(self):
        """alpha_p = 1: no interpole gap — must not divide by zero."""
        geo = slotless_1r2s()
        tau_p = pi * geo.r_mean / 6
        k = pm_leakage_factor(geo, 1.05, 1.0, 0.0015, tau_p)
        assert 0.9 < k <= 1.0

    def test_k_add_scales(self):
        geo = slotless_1r2s()
        tau_p = pi * geo.r_mean / 6
        k1 = pm_leakage_factor(geo, 1.05, 0.575, 0.0015, tau_p, k_add=1.0)
        k2 = pm_leakage_factor(geo, 1.05, 0.575, 0.0015, tau_p, k_add=0.95)
        assert k2 == pytest.approx(0.95 * k1)

    def test_nonpositive_zigzag_factor_raises(self):
        """Slot pitch far above the magnet width drives Eq. (2.16) below
        zero; that is outside the model, not a negative flux linkage."""
        geo = slotted_1r2s()
        tau_slot = 2 * pi * geo.r_mean / geo.n_slots
        with pytest.raises(ValueError, match=r"k_zz=.* <= 0.*n_slots="):
            pm_leakage_factor(geo, 1.05, 0.1, 0.0015, tau_slot / 2)


class TestWindingTransfer:

    def test_matches_annulus_integral(self):
        """2·N·k_w·r_mean·L/n_p == N·k_w·(r_out^2 - r_in^2)/n_p."""
        geo = slotless_1r2s()
        wt = winding_transfer_axial(geo, 840, 1.0, 6)
        assert wt == pytest.approx(
            840 * 1.0 * (geo.r_out**2 - geo.r_in**2) / 6
        )

    def test_gap_linkage_covers_iron_variants(self):
        assert GAP_LINKAGE == {
            "1R2S": 1, "2R1S-NN": 2, "2R1S-NS": 2, "DSSR": 1,
        }


class TestPrepare:

    def _motor(self, geo, **over):
        kw = dict(name="ax", geometry=geo, n_p=6, B_rem=1.05)
        kw.update(over)
        return Motor(**kw)

    def test_valid(self):
        p = prepare_analytical_axial(self._motor(slotless_1r2s()))
        assert isinstance(p, AnalyticalAxialParams)
        assert p.B_rem == 1.05

    def test_radial_geometry_rejected(self):
        from phasesweep.machines.geometry import outrunner
        geo = outrunner(
            r_outer=0.0158, r_rotor=0.0148, r_magnet=0.0128,
            r_stator=0.0113, r_inner=0.005,
        )
        with pytest.raises(ValueError, match="requires AxialFluxGeometry"):
            prepare_analytical_axial(self._motor(geo))

    def test_missing_b_rem_rejected(self):
        with pytest.raises(ValueError, match=r"explicit .* B_rem"):
            prepare_analytical_axial(
                self._motor(slotless_1r2s(), B_rem=None))

    def test_coreless_rejected(self):
        geo = axial_flux(
            variant="coreless", r_out=0.120, r_in=0.070,
            airgap=0.010, magnet_thickness=0.008, rotor_yoke=0.010,
        )
        with pytest.raises(ValueError, match="coreless"):
            prepare_analytical_axial(self._motor(geo))


class TestRunner:

    def _config(self, geo=None, n_theta=720, **motor_over):
        motor_kw = dict(
            name="ax", geometry=geo or slotless_1r2s(), n_p=6, B_rem=1.05,
        )
        motor_kw.update(motor_over)
        return RunConfig(
            motor=Motor(**motor_kw),
            model="analytical_axial", n_theta=n_theta,
        )

    def test_registry_entry(self):
        info = MODEL_REGISTRY["analytical_axial"]
        assert info.version == 1
        assert info.cost == "fast"
        assert "flux_linkage_peak" in info.produces
        assert "B_rem" in info.needs

    def test_runs_without_winding_data(self):
        r = MODEL_REGISTRY["analytical_axial"].fn(self._config())
        assert r["flux_linkage_peak"] is None
        assert r["fundamental"] > 0
        assert len(r["B_r_list"]) == 720

    def test_n_theta_guard(self):
        with pytest.raises(ValueError, match="cannot resolve"):
            MODEL_REGISTRY["analytical_axial"].fn(self._config(n_theta=13))

    def test_torus_links_both_gaps(self):
        """2R1S flux linkage doubles the per-gap value of 1R2S."""
        wind = dict(N=140, coils_series=6, k_w=1.0)
        base = dict(
            r_out=0.164, r_in=0.0985, airgap=0.0015,
            magnet_thickness=0.004, rotor_yoke=0.020, stator_yoke=0.020,
        )
        r1 = MODEL_REGISTRY["analytical_axial"].fn(
            self._config(axial_flux(variant="1R2S", **base), **wind))
        r2 = MODEL_REGISTRY["analytical_axial"].fn(
            self._config(axial_flux(variant="2R1S-NN", **base), **wind))
        assert r2["flux_linkage_peak"] == pytest.approx(
            2 * r1["flux_linkage_peak"])

    def test_sinusoidal_equiv_constant(self):
        assert ALPHA_P_SINUSOIDAL_EQUIV == pytest.approx(0.5750, abs=1e-4)
