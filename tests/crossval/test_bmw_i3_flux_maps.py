"""Data integrity tests for BMW i3 flux linkage maps.

Source: Gierczynski & Grzesiak, Energies 2021, 14(9), 2343 (CC BY 4.0).
Angle-averaged from FEMResults_1000rpm.xlsx supplementary data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
MAPS_FILE = ROOT / "data" / "bmw_i3" / "flux_maps_7x7.json"



@pytest.fixture(scope="module")
def maps():
    return json.loads(MAPS_FILE.read_text())


def test_file_exists():
    assert MAPS_FILE.exists()


def test_grid_shape(maps):
    assert len(maps["axes"]["i_d_A"]) == 7
    assert len(maps["axes"]["i_q_A"]) == 7
    assert len(maps["psi_d_mWb"]) == 7
    assert all(len(row) == 7 for row in maps["psi_d_mWb"])
    assert len(maps["psi_q_mWb"]) == 7
    assert all(len(row) == 7 for row in maps["psi_q_mWb"])


def test_axes_monotonic(maps):
    i_d = maps["axes"]["i_d_A"]
    i_q = maps["axes"]["i_q_A"]
    assert i_d == sorted(i_d)
    assert i_q == sorted(i_q)
    assert i_d[-1] == 0
    assert i_q[0] == 0


def test_psi_f_consistency(maps):
    """ψ_d(i_d=0, i_q=0) should match the TOML's psi_f = 0.0436 Wb."""
    psi_d_00 = maps["psi_d_mWb"][0][-1]
    assert psi_d_00 == pytest.approx(43.6, abs=0.5)


def test_psi_q_near_zero_at_origin(maps):
    """ψ_q(i_d=0, i_q=0) should be near zero (PM flux is on d-axis)."""
    psi_q_00 = maps["psi_q_mWb"][0][-1]
    assert abs(psi_q_00) < 1.0


def test_psi_d_decreases_with_negative_id(maps):
    """More negative i_d → more demagnetization → lower ψ_d."""
    for iq_row in maps["psi_d_mWb"]:
        for j in range(len(iq_row) - 1):
            assert iq_row[j] < iq_row[j + 1]


def test_psi_q_increases_with_iq(maps):
    """More positive i_q → more q-axis flux, at every i_d."""
    for id_idx in range(7):
        psi_q_col = [maps["psi_q_mWb"][iq_idx][id_idx]
                     for iq_idx in range(7)]
        for j in range(len(psi_q_col) - 1):
            assert psi_q_col[j] < psi_q_col[j + 1]


def test_torque_cross_product(maps):
    """τ = 1.5·n_p·(ψ_d·i_q − ψ_q·i_d) should match the Torque sheet
    within 3% at loaded operating points (|τ| > 10 Nm)."""
    n_p = maps["n_p"]
    i_d_vals = maps["axes"]["i_d_A"]
    i_q_vals = maps["axes"]["i_q_A"]
    max_err = 0.0
    for iq_idx, i_q in enumerate(i_q_vals):
        for id_idx, i_d in enumerate(i_d_vals):
            pd = maps["psi_d_mWb"][iq_idx][id_idx] * 1e-3
            pq = maps["psi_q_mWb"][iq_idx][id_idx] * 1e-3
            tau_calc = 1.5 * n_p * (pd * i_q - pq * i_d)
            tau_sheet = maps["torque_Nm"][iq_idx][id_idx]
            if abs(tau_sheet) > 10.0:
                err = abs(tau_calc - tau_sheet) / abs(tau_sheet)
                max_err = max(max_err, err)
    assert max_err < 0.03


def test_layout_not_transposed(maps):
    """Discriminate [i_q][i_d] from a transposed [i_d][i_q] layout."""
    assert maps["layout"] == "[i_q_idx][i_d_idx]"
    assert maps["psi_q_mWb"][0][3] == pytest.approx(0.0, abs=1.0)
    assert maps["psi_q_mWb"][3][0] == pytest.approx(47.4, abs=1.0)


def test_secant_inductance_at_peak_torque(maps):
    """Secant L_d/L_q at (i_d=-400, i_q=400) should match the TOML."""
    psi_d_00 = maps["psi_d_mWb"][0][-1]
    psi_d_pt = maps["psi_d_mWb"][4][2]  # i_q=400, i_d=-400
    psi_q_pt = maps["psi_q_mWb"][4][2]
    L_d_uH = (psi_d_pt - psi_d_00) / (-400) * 1e3
    L_q_uH = psi_q_pt / 400 * 1e3
    assert L_d_uH == pytest.approx(71.2, abs=0.5)
    assert L_q_uH == pytest.approx(141.3, abs=0.5)


def test_zero_iq_torque_near_zero(maps):
    """At i_q=0, torque should be near zero (no alignment torque)."""
    for tau in maps["torque_Nm"][0]:
        assert abs(tau) < 2.0


def test_convention_fields(maps):
    assert maps["convention"]["current"] == "peak"
    assert maps["convention"]["flux_linkage"] == "peak"
    assert maps["speed_rpm"] == 1000
