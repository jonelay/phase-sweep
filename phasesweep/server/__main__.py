"""uvicorn entry point: `uv run phasesweep-server` or `python -m phasesweep.server`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phasesweep._extras import SERVER_MODULES, install_hint, missing_modules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="phase-sweep local server (single-user, localhost)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; also whitelisted as a literal Host value. "
                             "The Host/Origin check guards a local browser against DNS "
                             "rebinding; it is not access control. Binding a "
                             "non-loopback address such as 0.0.0.0 exposes the "
                             "unauthenticated server to every client that can reach it")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--motors-dir", type=Path, default=Path("motors"))
    parser.add_argument("--user-configs-dir", type=Path, default=Path("user_configs"),
                        help="editable configs saved from the dashboard editor")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--workers", type=int, default=4,
                        help="worker processes for job execution")
    parser.add_argument("--subtask-timeout", type=float, default=600.0,
                        help="per-sub-task timeout budget in seconds")
    parser.add_argument("--mesh-cache-dir", default=None,
                        help="FEM mesh disk cache directory")
    parser.add_argument("--log-json", action="store_true",
                        help="JSON log output (default: colored console)")
    args = parser.parse_args()

    missing = missing_modules(SERVER_MODULES)
    if missing:
        sys.exit(f"phasesweep-server: needs {', '.join(missing)}; {install_hint('server')}")

    import uvicorn

    from phasesweep.server.app import ServerSettings, create_app

    app = create_app(ServerSettings(
        motors_dir=args.motors_dir,
        user_configs_dir=args.user_configs_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        subtask_timeout_s=args.subtask_timeout,
        mesh_cache_dir=args.mesh_cache_dir,
        log_json=args.log_json,
        extra_allowed_hosts=(args.host,),
    ))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
