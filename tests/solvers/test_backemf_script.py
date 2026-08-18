"""Tests for backemf_validation.py script helpers (EMF synthesis, scalar guard)."""

from importlib.util import find_spec

import pytest

if not find_spec("matplotlib"):
    pytest.skip("requires phasesweep[viz] (matplotlib not installed)", allow_module_level=True)

import sys
from pathlib import Path

import numpy as np

from phasesweep.solvers.harmonics import harmonics_1sided

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import backemf_validation as bv


def test_measured_scalars_consistent():
    bv.check_measured_scalars()


def test_kw_harmonic_ratio_9s12p():
    assert bv._kw_harmonic_ratio(1, 6, 9) == pytest.approx(1.0)
    assert bv._kw_harmonic_ratio(3, 6, 9) == pytest.approx(0.0, abs=1e-12)
    assert bv._kw_harmonic_ratio(5, 6, 9) == pytest.approx(1.0)
    assert bv._kw_harmonic_ratio(7, 6, 9) == pytest.approx(1.0)


def test_emf_synthesis_pure_fundamental():
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    B = 0.5 * np.cos(6 * th + 0.3)
    deg, e = bv._emf_from_rotor_harmonics(B, 6, 9, 2.0)
    assert np.allclose(e, 1.0 * np.sin(np.radians(deg)), atol=1e-9)


def test_emf_synthesis_drops_stator_locked_orders():
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    B = 0.5 * np.cos(6 * th)
    _, e_ref = bv._emf_from_rotor_harmonics(B, 6, 9, 2.0)
    # slot-ripple order (3 = n_slots - n_p, not a multiple of n_p)
    _, e_slot = bv._emf_from_rotor_harmonics(B + 0.2 * np.cos(3 * th), 6, 9, 2.0)
    assert np.allclose(e_slot, e_ref, atol=1e-9)
    # triplen rotor harmonic (order 18) pitched out exactly
    _, e_trip = bv._emf_from_rotor_harmonics(B + 0.1 * np.cos(18 * th), 6, 9, 2.0)
    assert np.allclose(e_trip, e_ref, atol=1e-9)


def test_emf_synthesis_nontriplen_harmonic_survives():
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    B = 0.5 * np.cos(6 * th)
    _, e_ref = bv._emf_from_rotor_harmonics(B, 6, 9, 2.0)
    _, e_5th = bv._emf_from_rotor_harmonics(B + 0.05 * np.cos(30 * th), 6, 9, 2.0)
    assert not np.allclose(e_5th, e_ref, atol=1e-6)
    amp_5th = harmonics_1sided(e_5th)[5]
    assert amp_5th == pytest.approx(0.05 * 2.0, rel=1e-6)
