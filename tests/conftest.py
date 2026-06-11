"""Shared test fixtures."""

from pathlib import Path

import pytest

from phasesweep.geometry import default_inrunner, inrunner, outrunner
from phasesweep.motor import Motor

# Anchor for repo-relative test data — keeps tests cwd-independent
REPO_ROOT = Path(__file__).resolve().parent.parent

CREATOR_CSV_SKIP = (
    "CREATOR-derived CSVs not found — run "
    "python scripts/fetch_creator_dataset.py"
)


def make_motor(**overrides):
    """Build a Motor with sensible defaults for testing.

    No B_rem in defaults — supply explicitly when testing the explicit-B_rem path.
    """
    defaults = dict(
        name="test", geometry=default_inrunner(), n_p=2,
        R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
        N=50, k_w=0.966, L_stk=0.10, coils_series=1,
    )
    defaults.update(overrides)
    return Motor(**defaults)


@pytest.fixture
def default_geometry():
    return default_inrunner()


@pytest.fixture
def paper_geometry_8pole():
    """Zhu & Howe 2002 paper geometry: 2p=8, Rs=48mm, Rm=40mm, Rr=30mm."""
    return inrunner(
        r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
    )


@pytest.fixture
def default_motor(default_geometry):
    return Motor(
        name="test_default",
        geometry=default_geometry,
        n_p=2,
        R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
        N=50, k_w=0.966, L_stk=0.10, coils_series=1,
    )


@pytest.fixture
def paper_geometry_8pole_outrunner():
    """Outrunner with rearranged Zhu & Howe 2002 paper dims.

    Same radii as inrunner (30/40/48mm) rearranged for outrunner topology:
    r_stator=30mm (bore), r_magnet=40mm, r_rotor=48mm, r_outer=60mm.
    """
    return outrunner(
        r_outer=0.060, r_rotor=0.048, r_magnet=0.040,
        r_stator=0.030, r_inner=0.020,
    )


@pytest.fixture
def paper_motor_8pole(paper_geometry_8pole):
    return Motor(
        name="paper_8pole",
        geometry=paper_geometry_8pole,
        n_p=4,
        B_rem=1.2, mu_r_pm=1.05,
    )


@pytest.fixture
def paper_motor_8pole_outrunner(paper_geometry_8pole_outrunner):
    return Motor(
        name="paper_8pole_outrunner",
        geometry=paper_geometry_8pole_outrunner,
        n_p=4,
        B_rem=1.2, mu_r_pm=1.05,
    )
