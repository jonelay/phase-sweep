"""BMW i3 rated torque: numeric MTPA on the flux-linkage surface vs the
constant-parameter (secant L_d/L_q) Morimoto MTPA.

Source: Gierczynski & Grzesiak, Energies 2021, 14(9), 2343 (CC BY 4.0),
FEA-extracted 7x7 maps in data/bmw_i3/flux_maps_7x7.json (mWb) and the
same grid in Wb under [flux_map] in motors/bmw_i3.toml.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from phasesweep.machines.configs import load_motor
from phasesweep.machines.motor import FluxMap
from phasesweep.models.flux_map import flux_map_psi_f
from phasesweep.models.rated_torque import run_rated_torque
from phasesweep.sweep_types import RunConfig

ROOT = Path(__file__).resolve().parent.parent.parent
MAPS_FILE = ROOT / "data" / "bmw_i3" / "flux_maps_7x7.json"
MOTOR_FILE = ROOT / "motors" / "bmw_i3.toml"
MAX_I_S = 565.7



@pytest.fixture(scope="module")
def maps():
    return json.loads(MAPS_FILE.read_text())


@pytest.fixture(scope="module")
def flux_map(maps):
    return FluxMap(
        i_d_A=maps["axes"]["i_d_A"],
        i_q_A=maps["axes"]["i_q_A"],
        psi_d_Wb=[[x * 1e-3 for x in row] for row in maps["psi_d_mWb"]],
        psi_q_Wb=[[x * 1e-3 for x in row] for row in maps["psi_q_mWb"]],
    )


@pytest.fixture(scope="module")
def motor_toml():
    return load_motor(MOTOR_FILE)


@pytest.fixture(scope="module")
def motor_analytical(motor_toml):
    return dataclasses.replace(motor_toml, flux_map=None, I_rated=MAX_I_S)


@pytest.fixture(scope="module")
def motor_flux_map(motor_toml, flux_map):
    return dataclasses.replace(motor_toml, flux_map=flux_map, I_rated=MAX_I_S)


def _rated(motor):
    return run_rated_torque(RunConfig(motor=motor, model="rated_torque"))


def test_toml_flux_map_matches_json(motor_toml, flux_map):
    toml_fm = motor_toml.flux_map
    assert toml_fm.i_d_A == flux_map.i_d_A
    assert toml_fm.i_q_A == flux_map.i_q_A
    np.testing.assert_allclose(toml_fm.psi_d_Wb, flux_map.psi_d_Wb, rtol=1e-12)
    np.testing.assert_allclose(toml_fm.psi_q_Wb, flux_map.psi_q_Wb, rtol=1e-12)


def test_map_psi_f_matches_toml(motor_toml):
    assert flux_map_psi_f(motor_toml.flux_map) == pytest.approx(
        motor_toml.psi_f, rel=0.01)


def test_flux_map_mtpa_vs_torque_sheet(maps, motor_flux_map, flux_map):
    r = _rated(motor_flux_map)
    assert r["mtpa_source"] == "flux_map"
    gamma = np.radians(r["gamma_opt_deg"])
    i_d, i_q = -MAX_I_S * np.sin(gamma), MAX_I_S * np.cos(gamma)
    sheet = RegularGridInterpolator(
        (flux_map.i_d_A, flux_map.i_q_A), np.array(maps["torque_Nm"]).T,
        method="cubic",
    )
    tau_sheet = float(sheet([[i_d, i_q]])[0])
    assert r["tau_mtpa"] == pytest.approx(tau_sheet, rel=0.05)


def test_flux_map_mtpa_near_published_peak_point(motor_flux_map):
    """Gierczynski's peak-torque point is (i_d, i_q) = (-401, 399) A."""
    r = _rated(motor_flux_map)
    gamma = np.radians(r["gamma_opt_deg"])
    assert -MAX_I_S * np.sin(gamma) == pytest.approx(-401, abs=15)
    assert MAX_I_S * np.cos(gamma) == pytest.approx(399, abs=15)


def test_analytical_and_flux_map_disagree(motor_analytical, motor_flux_map):
    """The secant-L constant-parameter model is off for this saturated
    machine; the surface model is the correction."""
    ra = _rated(motor_analytical)
    rf = _rated(motor_flux_map)
    assert ra["mtpa_source"] == "analytical"
    assert ra["k_T"] == pytest.approx(rf["k_T"])
    assert abs(ra["tau_mtpa"] - rf["tau_mtpa"]) / rf["tau_mtpa"] > 0.05


def test_flux_map_changes_config_id(motor_analytical, motor_flux_map):
    assert motor_analytical.config_id != motor_flux_map.config_id


def test_flux_map_round_trip(flux_map):
    """FluxMap to_dict → from_dict preserves data and config_id."""
    from phasesweep.machines.motor import FluxMap

    rt = FluxMap.from_dict(flux_map.to_dict())
    assert rt.i_d_A == flux_map.i_d_A
    assert rt.i_q_A == flux_map.i_q_A
    np.testing.assert_allclose(rt.psi_d_Wb, flux_map.psi_d_Wb, rtol=1e-15)
    np.testing.assert_allclose(rt.psi_q_Wb, flux_map.psi_q_Wb, rtol=1e-15)


def test_motor_round_trip_with_flux_map(motor_flux_map):
    """Motor to_dict → from_dict preserves config_id with a flux map."""
    from phasesweep.machines.motor import Motor

    rt = Motor.from_dict(motor_flux_map.to_dict())
    assert rt.config_id == motor_flux_map.config_id
    assert rt.flux_map is not None
    assert rt.flux_map.i_d_A == motor_flux_map.flux_map.i_d_A


def test_stall_falls_back_beyond_grid(motor_flux_map):
    """The drive-limit stall point is on the map; the electromagnetic
    stall current lies far beyond the grid and falls back to analytical."""
    from phasesweep.models.rated_torque import run_stall_torque

    r = run_stall_torque(RunConfig(motor=motor_flux_map, model="stall_torque"))
    assert r["mtpa_source"] == "flux_map"
    assert r["mtpa_source_electromagnetic"] == "analytical"
