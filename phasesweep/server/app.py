"""FastAPI app factory: settings, logging, lifespan wiring."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from phasesweep.machines.configs import load_motor
from phasesweep.machines.motor import Motor
from phasesweep.result_store import ResultStore
from phasesweep.server.jobs import JobManager
from phasesweep.server.routes import ConnectionManager, api, ws_router

log = structlog.get_logger()


@dataclass(frozen=True)
class ServerSettings:
    motors_dir: Path = Path("motors")
    user_configs_dir: Path = Path("user_configs")
    output_dir: Path = Path("output")
    workers: int = 4
    subtask_timeout_s: float = 600.0
    mesh_cache_dir: str | None = None
    log_json: bool = False
    extra_allowed_hosts: tuple[str, ...] = ()


# "testserver" is starlette's TestClient default host (Django precedent);
# not publicly resolvable, so it adds no rebinding surface.
_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})


def _host_hostname(host: str) -> str:
    if host.startswith("["):
        return host.partition("]")[0].lstrip("[")
    return host.rsplit(":", 1)[0]


class LocalOnlyMiddleware:
    """Reject requests whose Host or Origin is not local — closes the
    DNS-rebinding / cross-site hole in the localhost server.
    Non-browser clients without an Origin header pass through."""

    def __init__(self, app: ASGIApp, extra_hosts: frozenset[str] = frozenset()) -> None:
        self.app = app
        self.allowed = _LOCAL_HOSTNAMES | extra_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "")
        if _host_hostname(host.strip()).lower() not in self.allowed:
            log.warning("host_rejected", host=host)
            await self._reject(scope, receive, send, 400, "invalid Host header")
            return
        origin = headers.get("origin")
        if origin is not None and urlsplit(origin).hostname not in self.allowed:
            log.warning("origin_rejected", origin=origin)
            await self._reject(scope, receive, send, 403,
                               "cross-origin request rejected")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: Scope, receive: Receive, send: Send, status: int, detail: str,
    ) -> None:
        if scope["type"] == "websocket":
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 1008})
            return
        await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)


def configure_logging(log_json: bool = False) -> None:
    """Human-readable console output for development, JSON for production —
    same log calls, different renderer."""
    renderer = (structlog.processors.JSONRenderer() if log_json
                else structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


def load_motor_configs(motors_dir: Path) -> dict[str, Motor]:
    """Motors keyed by TOML filename stem; invalid files
    are skipped with a warning, matching configs.load_motors."""
    out: dict[str, Motor] = {}
    for p in sorted(Path(motors_dir).glob("*.toml")):
        try:
            out[p.stem] = load_motor(p)
        except (KeyError, ValueError) as e:
            log.warning("config_skipped", file=p.name, error=str(e))
    return out


def load_user_configs(user_dir: Path, motors: dict[str, Motor]) -> set[str]:
    """Merge editable user configs into `motors`, returning
    their names. A stem colliding with an anchor config is skipped — the
    provenance-carrying fleet in motors_dir cannot be shadowed."""
    user_names: set[str] = set()
    for stem, m in load_motor_configs(user_dir).items():
        if stem in motors:
            log.warning("user_config_shadows_anchor", file=f"{stem}.toml")
            continue
        motors[stem] = m
        user_names.add(stem)
    return user_names


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    configure_logging(settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        motors = load_motor_configs(settings.motors_dir)
        user_names = load_user_configs(settings.user_configs_dir, motors)
        store = ResultStore(settings.output_dir)
        ws_manager = ConnectionManager()
        manager = JobManager(
            motors, store,
            workers=settings.workers,
            subtask_timeout_s=settings.subtask_timeout_s,
            mesh_cache_dir=settings.mesh_cache_dir,
        )
        app.state.manager = manager
        app.state.ws_manager = ws_manager
        app.state.settings = settings
        app.state.user_config_names = user_names
        await manager.start(ws_manager.broadcast)
        log.info("server_started", motors=len(motors), user_configs=len(user_names),
                 output_dir=str(settings.output_dir), workers=settings.workers)
        try:
            yield
        finally:
            await manager.stop()
            log.info("server_stopped")

    app = FastAPI(title="phase-sweep server", lifespan=lifespan)
    app.add_middleware(
        LocalOnlyMiddleware,
        extra_hosts=frozenset(h.lower() for h in settings.extra_allowed_hosts))
    app.include_router(api)
    app.include_router(ws_router)

    # Dashboard static mount — lands with Phase 3b
    dashboard = Path(__file__).resolve().parent.parent / "dashboard"
    if dashboard.is_dir():
        app.mount("/", StaticFiles(directory=dashboard, html=True), name="dashboard")
    return app
