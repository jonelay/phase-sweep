"""API fuzzing for the phase-sweep server (QC option #12).

Hammers every REST route with malformed / hostile payloads and reports any
response that is a 500 (unhandled server exception) or an unexpected crash.
A 4xx is a PASS — it means the input was rejected deliberately.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from phasesweep.server.app import ServerSettings, create_app

MOTOR_TOML = """
[motor]
name = "Fuzz Inrunner"
topology = "inrunner"
[circuit]
n_p = 2
R_s = 0.2
L_d = 4e-3
L_q = 4e-3
psi_f = 0.1
J = 0.002
I_rated = 5.0
[winding]
N = 50
k_w = 0.966
coils_series = 1
[geometry]
r_outer = 1.0
r_stator = 0.70
r_magnet = 0.64
r_rotor = 0.30
r_inner = 0.0
L_stk = 0.10
[materials]
mu_r_fe = 1000.0
mu_r_pm = 1.05
[drive]
U_DC = 48.0
MAX_I_S = 10.0
W_REF = 314.159
"""

# Hostile scalars reused across string/number slots.
NASTY = [
    None, True, 0, -1, 10**30, -(10**30), 1e308 * 10, float("nan"), float("inf"),
    "", " ", "\x00", "\n\r", "../../etc/passwd", "..%2f..%2fetc%2fpasswd",
    "%00", "'; DROP TABLE x; --", "<script>alert(1)</script>",
    "${jndi:ldap://x}", "{{7*7}}", "😀" * 10, "A" * 10000,
    [], {}, [1, 2, 3], {"a": 1}, [[["deep"]]],
]

findings: list[tuple[str, str, int, str]] = []
checked = 0


def probe(client, method: str, url: str, label: str, **kw) -> None:
    """Issue one request; record 500s and transport-level explosions."""
    global checked
    checked += 1
    try:
        r = client.request(method, url, **kw)
    except Exception:
        findings.append((label, f"{method} {url}", -1,
                         traceback.format_exc(limit=3).strip().splitlines()[-1]))
        return
    if r.status_code >= 500:
        findings.append((label, f"{method} {url}", r.status_code, r.text[:300]))


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    motors = tmp / "motors"
    motors.mkdir()
    (motors / "fuzz_inrunner.toml").write_text(MOTOR_TOML)
    app = create_app(ServerSettings(
        motors_dir=motors, user_configs_dir=tmp / "user",
        output_dir=tmp / "output", workers=1, subtask_timeout_s=30.0,
    ))
    with TestClient(app) as c:
        # --- POST /api/jobs: body shape ---
        for v in NASTY:
            probe(c, "POST", "/api/jobs", "jobs.motor", json={"motor": v, "model": "analytical"})
            probe(c, "POST", "/api/jobs", "jobs.model", json={"motor": "fuzz_inrunner", "model": v})
            probe(c, "POST", "/api/jobs", "jobs.params",
                  json={"motor": "fuzz_inrunner", "model": "analytical", "params": v})
        probe(c, "POST", "/api/jobs", "jobs.body-list", json=[1, 2, 3])
        probe(c, "POST", "/api/jobs", "jobs.body-str", json="hello")
        probe(c, "POST", "/api/jobs", "jobs.body-null", json=None)
        probe(c, "POST", "/api/jobs", "jobs.raw-garbage",
              content=b"\x00\x01\x02not json", headers={"content-type": "application/json"})

        # param values inside a valid submit
        for v in NASTY:
            for key in ("n_theta", "maxh_fraction", "nonlinear", "j_s",
                        "i_fault", "duty_profile", "sim_plan", "dataset_id"):
                probe(c, "POST", "/api/jobs", f"jobs.params.{key}",
                      json={"motor": "fuzz_inrunner", "model": "analytical",
                            "params": {key: v}})

        # --- path params ---
        for v in NASTY:
            s = str(v)
            probe(c, "GET", f"/api/jobs/{s}", "jobs.id")
            probe(c, "DELETE", f"/api/jobs/{s}", "jobs.cancel")
            probe(c, "GET", f"/api/results/{s}", "results.id")
            probe(c, "GET", f"/api/validation/{s}", "validation.motor")
            probe(c, "GET", f"/api/configs/{s}", "configs.get")
            probe(c, "GET", f"/api/configs/{s}/raw", "configs.raw")
            probe(c, "GET", f"/api/model-defaults/{s}/analytical", "defaults.motor")
            probe(c, "GET", f"/api/model-defaults/fuzz_inrunner/{s}", "defaults.model")

        # --- query params on /api/results ---
        for v in NASTY:
            s = str(v)
            for q in ("motor", "motor_config_id", "model", "source", "limit"):
                probe(c, "GET", f"/api/results?{q}={s}", f"results.q.{q}")
        for lim in ("-1", "0", "-999999", "99999999999999999999", "abc", "1e5", "", "nan"):
            probe(c, "GET", f"/api/results?limit={lim}", "results.limit")
        probe(c, "GET", "/api/jobs?status=" + "A" * 5000, "jobs.status")

        # --- PUT /api/configs/{name} ---
        for v in NASTY:
            s = str(v)
            probe(c, "PUT", f"/api/configs/{s}", "config.name", json={"motor": {"name": "x"}})
        for v in NASTY:
            probe(c, "PUT", "/api/configs/fuzzcfg", "config.body", json=v)
            probe(c, "PUT", "/api/configs/fuzzcfg", "config.motor", json={"motor": v})
            probe(c, "PUT", "/api/configs/fuzzcfg", "config.circuit",
                  json={"motor": {"name": "x", "topology": "inrunner"}, "circuit": v})
            probe(c, "PUT", "/api/configs/fuzzcfg", "config.geometry",
                  json={"motor": {"name": "x", "topology": "inrunner"},
                        "circuit": {"n_p": 2}, "geometry": v})
        # deeply nested / self-referential-ish payloads
        deep: object = {"a": 1}
        for _ in range(200):
            deep = {"n": deep}
            probe(c, "PUT", "/api/configs/fuzzdeep", "config.deep", json={"motor": deep})
            break
        probe(c, "PUT", "/api/configs/fuzzcfg", "config.hugekeys",
              json={"motor": {"name": "x"}, **{f"s{i}": {"k": i} for i in range(2000)}})

        # --- POST /api/measured/{motor} ---
        for v in NASTY:
            probe(c, "POST", "/api/measured/fuzz_inrunner", "measured.body", json=v)
            probe(c, "POST", "/api/measured/fuzz_inrunner", "measured.test_type",
                  json={"test_type": v, "quantities": {"R_s": 0.2},
                        "source_file": "f.json"})
            probe(c, "POST", "/api/measured/fuzz_inrunner", "measured.quantities",
                  json={"test_type": "resistance_test", "quantities": v,
                        "source_file": "f.json"})
            probe(c, "POST", "/api/measured/fuzz_inrunner", "measured.conditions",
                  json={"test_type": "resistance_test", "quantities": {"R_s": 0.2},
                        "conditions": v, "source_file": "f.json"})
            s = str(v)
            probe(c, "POST", f"/api/measured/fuzz_inrunner?dataset_id={s}",
                  "measured.dataset_id",
                  json={"test_type": "resistance_test", "quantities": {"R_s": 0.2},
                        "source_file": "f.json"})
            probe(c, "POST", f"/api/measured/{s}", "measured.motor",
                  json={"test_type": "resistance_test", "quantities": {"R_s": 0.2},
                        "source_file": "f.json"})

        # --- static / misc ---
        for p in ("/api/models", "/api/configs", "/api/jobs", "/api/results"):
            probe(c, "GET", p, "smoke")
        probe(c, "GET", "/api/../../etc/passwd", "traversal")
        probe(c, "POST", "/api/models", "wrong-method")

    print(f"requests issued: {checked}")
    print(f"findings (5xx / transport errors): {len(findings)}\n")
    seen = set()
    for label, url, code, body in findings:
        key = (label, code, body[:120])
        if key in seen:
            continue
        seen.add(key)
        print(f"[{code}] {label}  {url}\n    {body[:280]}\n")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
