"""Agent cases (tests/agent_cases/*.toml) replayed without an LLM.

Each case is the single definition of something an agent should find
through MCP. Here its tool sequence runs against a server loaded with the
case's motor, and every doc that quotes the case is checked against the
same values, so the docs, the model output, and the trust caveats cannot
drift apart.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import SERVER_SKIP, requires_server

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES = sorted((REPO_ROOT / "tests" / "agent_cases").glob("*.toml"))


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _measured(case: dict[str, Any]) -> float:
    ref = case["reference"]
    raw = json.loads((REPO_ROOT / ref["file"]).read_text())
    return float(raw["quantities"][ref["quantity"]])


# Sections other than id/motor/prompt/applicability are optional: [result]
# + [reference] + [docs] for a fast model's value, [plan] for a dry run.
DOC_CASES = [p for p in CASES if "docs" in _load(p)]


@pytest.fixture(params=CASES, ids=[p.stem for p in CASES])
def case(request) -> dict[str, Any]:
    return _load(request.param)


@pytest.fixture(params=DOC_CASES, ids=[p.stem for p in DOC_CASES])
def doc_case(request) -> dict[str, Any]:
    return _load(request.param)


def test_cases_exist():
    assert CASES, "no agent cases found"


def test_docs_quote_case_values(doc_case):
    """Every doc listed in the case states the pinned prediction and the
    reference measurement, rounded half up, as '<n> N·m'."""
    case = doc_case
    unit = case["result"]["unit"]
    quoted = {
        "prediction": f"{math.floor(case['result']['value'] + 0.5)} {unit}",
        "measurement": f"{math.floor(_measured(case) + 0.5)} {unit}",
    }
    for rel in case["docs"]["files"]:
        text = (REPO_ROOT / rel).read_text()
        for what, s in quoted.items():
            # "905 N·m" or "905\nN·m" across a line wrap
            pat = re.escape(s).replace(r"\ ", r"\s+")
            assert re.search(pat, text), f"{rel} does not quote the {what} {s!r}"


@requires_server
def test_case_replays_over_mcp(case, tmp_path):
    pytest.importorskip("mcp", reason=SERVER_SKIP)
    from tests.test_server import make_client
    from tests.test_server_mcp import Mcp

    motors = tmp_path / "motors"
    motors.mkdir()
    shutil.copy(REPO_ROOT / "motors" / f"{case['motor']}.toml", motors)
    workspace = {"motors": motors, "user": tmp_path / "user_configs",
                 "output": tmp_path / "output"}

    with make_client(workspace) as c:
        # The reference measurement is in the store, as on a server that has
        # imported its anchors; results must then point at it.
        if "reference" in case:
            ref = json.loads((REPO_ROOT / case["reference"]["file"]).read_text())
            r = c.post(f"/api/measured/{case['motor']}", json=ref)
            assert r.status_code == 201, r.text
        mcp = Mcp(c)

        models = {m["key"]: m for m in mcp.call("list_models", motor=case["motor"])}
        app = case["applicability"]
        for key in app["runnable"]:
            m = models[key]
            assert m["applicable"] and not m["missing_params"], m
        for key, reason in app["blocked"].items():
            m = models[key]
            assert not m["applicable"], m
            assert reason in m["reason"], m

        if "plan" in case:
            plan = case["plan"]
            out = mcp.call("submit_job", motor=case["motor"], model=plan["model"],
                           params=plan["params"], dry_run=True)
            assert out["dry_run"] and out["subtasks"] == plan["subtasks"], out
            assert not mcp.call("list_jobs"), "a dry run must queue nothing"

        if "result" not in case:
            return
        res = case["result"]
        job = mcp.call("submit_job", motor=case["motor"], model=res["model"])
        deadline = time.monotonic() + 60
        while job["status"] not in ("completed", "failed", "cancelled"):
            assert time.monotonic() < deadline, job
            time.sleep(0.05)
            job = mcp.call("get_job", job_id=job["id"])
        assert job["status"] == "completed", job

        out = mcp.call("get_result", result_id=job["result_ids"][0])
        head = {h["key"]: h for h in out["headline"]}[res["headline"]]
        assert head["unit"] == res["unit"]
        assert head["value"] == pytest.approx(res["value"], abs=res["abs"])
        cav = {c["code"]: c for c in out["caveats"]}
        assert cav[res["caveat_code"]]["severity"] == res["caveat_severity"]
        pointer = cav["measured_data_available"]
        assert f"motor={case['motor']!r}" in pointer["text"]

