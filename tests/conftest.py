"""Shared test fixtures."""

import os
from importlib.util import find_spec
from pathlib import Path

import pytest

from phasesweep.machines.geometry import default_inrunner, inrunner, outrunner
from phasesweep.machines.motor import DriveParams, Motor

# Anchor for repo-relative test data — keeps tests cwd-independent
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Optional-dep detection + skip markers
# ---------------------------------------------------------------------------

FEM_SKIP = "requires phasesweep[fem] (ngsolve not installed)"
SIM_SKIP = "requires phasesweep[sim] (motulator not installed)"
VIZ_SKIP = "requires phasesweep[viz] (matplotlib not installed)"
SERVER_SKIP = "requires phasesweep[server] (fastapi not installed)"

def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

_has_fem = _try_import("ngsolve")
_has_sim = find_spec("motulator") is not None
_has_viz = find_spec("matplotlib") is not None
_has_server = find_spec("fastapi") is not None

# `fem` is a selection marker AND the ngsolve gate: the skip is attached in
# pytest_collection_modifyitems below rather than composed here, so the two
# cannot drift apart the way `slow` drifted from its FEM sites. One symbol,
# usable as a decorator, in `pytestmark`, or in `pytest.param(marks=...)`.
requires_fem = pytest.mark.fem
requires_sim = pytest.mark.skipif(not _has_sim, reason=SIM_SKIP)
requires_viz = pytest.mark.skipif(not _has_viz, reason=VIZ_SKIP)
requires_server = pytest.mark.skipif(not _has_server, reason=SERVER_SKIP)

# Skips that are legitimate on a clean checkout: the datasets behind them are
# gitignored third-party downloads, not repo content. Every other skip means a
# dependency or a repo file went missing, which must not read as green. The
# gate is calibrated for a repo checkout — an sdist run lacks data/*/captures/
# and would trip it. Import these at the skip sites so the text cannot drift.
CREATOR_DATASET_SKIP = (
    "CREATOR full dataset not found — download from "
    "https://doi.org/10.3217/sns1d-77m43 "
    "(see data/creator_case_pmsm/README.md)"
)
CREATOR_CSV_SKIP = (
    "CREATOR-derived CSVs not found — run "
    "python scripts/fetch_creator_dataset.py"
)
CREATOR_JSON_SKIP = (
    "CREATOR-derived JSONs not found — run "
    "python scripts/fetch_creator_dataset.py"
)
STEEL_TABLES_SKIP = "M250-35A steel loss tables absent (gitignored dataset)"
CREATOR_IM_SKIP = (
    "CREATOR IM dataset not found — run "
    "python scripts/fetch_creator_im_dataset.py"
)
# Lab capture CSVs are tracked on main but excluded from the public release
# branch, where their absence is by design rather than a broken checkout.
CAPTURES_SKIP = "lab capture CSVs absent (not on the public release branch)"
# Deliberately NOT in ALLOWED_SKIP_SUBSTRINGS: the pinned Elmer JSONs are
# committed repo files (scripts/elmer_*/pinned/), so their absence means a
# broken checkout and must fail a strict-skip run.
ELMER_PINNED_SKIP = "pinned Elmer comparison JSON not present"

_CORE_LANE = bool(os.environ.get("PHASESWEEP_CORE_LANE"))

ALLOWED_SKIP_SUBSTRINGS = (
    CREATOR_DATASET_SKIP,
    CREATOR_CSV_SKIP,
    CREATOR_JSON_SKIP,
    STEEL_TABLES_SKIP,
    CREATOR_IM_SKIP,
    CAPTURES_SKIP,
    FEM_SKIP,
)
if _CORE_LANE:
    ALLOWED_SKIP_SUBSTRINGS = (
        *ALLOWED_SKIP_SUBSTRINGS,
        SIM_SKIP, VIZ_SKIP, SERVER_SKIP,
    )


def pytest_collection_modifyitems(config, items):
    """Attach the ngsolve gate to every `fem`-marked test."""
    if _has_fem:
        return
    skip = pytest.mark.skip(reason=FEM_SKIP)
    for item in items:
        if item.get_closest_marker("fem"):
            item.add_marker(skip)


def _skip_label(reason: str) -> str:
    """Bucket a skip reason to the allowlist constant that permits it."""
    for allowed in ALLOWED_SKIP_SUBSTRINGS:
        if allowed in reason:
            return allowed
    return "UNEXPECTED"


def pytest_sessionfinish(session, exitstatus):
    """Fail the session on any unexpected skip when PHASESWEEP_STRICT_SKIP=1.

    CI ran without the `server` extra for months, so `importorskip
    ("fastapi")` silently dropped all 41 API tests and the suite still
    reported success.

    PHASESWEEP_EXPECTED_SKIPS=<n> additionally pins the *total* skip count.
    An allowlisted reason only says a skip is a legitimate kind, not that it
    was legitimate to grow: a newly added dataset-gated anchor skips under a
    reason already on the allowlist and would otherwise never be noticed.
    The per-reason tally below is printed on every strict run so the number
    to pin for a lane is readable straight out of its log.
    """
    if not os.environ.get("PHASESWEEP_STRICT_SKIP"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    unexpected = []
    tally: dict[str, int] = {}
    skipped = reporter.stats.get("skipped", [])
    for report in skipped:
        reason = str(getattr(report, "longrepr", ""))
        label = _skip_label(reason)
        tally[label] = tally.get(label, 0) + 1
        if label == "UNEXPECTED":
            unexpected.append(f"{report.nodeid}: {reason}")

    reporter.write_sep("=", f"skip tally: {len(skipped)} total (PHASESWEEP_STRICT_SKIP)")
    for label, count in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        reporter.write_line(f"  {count:>4}  {label}")

    failures = []
    if unexpected:
        failures.append("unexpected skips")
        reporter.write_sep("=", "unexpected skips (PHASESWEEP_STRICT_SKIP)", red=True)
        for line in unexpected:
            reporter.write_line(line)

    expected = os.environ.get("PHASESWEEP_EXPECTED_SKIPS")
    if expected is not None and len(skipped) != int(expected):
        failures.append("skip count")
        reporter.write_sep("=", "skip count off manifest", red=True)
        reporter.write_line(
            f"expected {int(expected)} skips (PHASESWEEP_EXPECTED_SKIPS), got {len(skipped)}"
        )

    if failures:
        session.exitstatus = session.exitstatus or 1


def make_motor(**overrides):
    """Build a Motor with sensible defaults for testing.

    No B_rem in defaults — supply explicitly when testing the explicit-B_rem path.
    """
    defaults = dict(
        name="test", geometry=default_inrunner(), n_p=2,
        R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
        N=50, k_w=0.966, L_stk=0.10, coils_series=1,
        drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.159265358979),
    )
    defaults.update(overrides)
    return Motor(**defaults)


@pytest.fixture
def paper_geometry_8pole():
    """Zhu & Howe 2002 paper geometry: 2p=8, Rs=48mm, Rm=40mm, Rr=30mm."""
    return inrunner(
        r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
    )


@pytest.fixture
def default_motor():
    return make_motor(name="test_default", geometry=default_inrunner())


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


