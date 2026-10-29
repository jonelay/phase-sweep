"""Dependency-injected FastAPI app factory.

``create_app(AppConfig(...))`` produces an application with:

- a ``ConnectionManager`` at ``app.state.ws_manager`` and the WS endpoint
  mounted (``ws_path``), created before the consumer's lifespan runs so
  the consumer can hand ``ws_manager.broadcast`` to its job runner;
- the consumer's routers registered;
- local-only Host/Origin enforcement (DNS-rebinding and cross-site guard)
  with Tauri origins allowed by default;
- view-sweep assets (``css/``, ``js/``, ``spec/`` only) served under
  ``view_sweep_mount`` and the consumer's static tree at ``/``
  (``html=True``), mounted last so API routes win.

The factory imports nothing from any domain; everything project-specific
arrives through ``AppConfig``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import structlog
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from phasesweep._view_sweep_server.ws import ConnectionManager, make_ws_router

log = structlog.get_logger()

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


# -- JSON --------------------------------------------------------------------------


def json_finite(obj: Any) -> Any:
    """Recursively map non-finite floats to None (JSON null)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, list):
        return [json_finite(v) for v in obj]
    if isinstance(obj, dict):
        return {k: json_finite(v) for k, v in obj.items()}
    return obj


class FiniteJSONResponse(JSONResponse):
    """NaN/inf -> null. Starlette's JSONResponse refuses non-finite floats,
    and numeric models legitimately produce them. Use as a router's
    ``default_response_class`` so every endpoint is covered."""

    def render(self, content: Any) -> bytes:
        return super().render(json_finite(content))


# -- host / origin guard -----------------------------------------------------------

# "testserver" is starlette's TestClient default host (Django precedent);
# not publicly resolvable, so it adds no rebinding surface.
LOCAL_HOSTNAMES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})

# Tauri webviews present ``tauri://localhost`` (macOS/Linux) or
# ``http://tauri.localhost`` (Windows) as the page origin.
TAURI_ORIGIN_HOSTNAMES: frozenset[str] = frozenset({"localhost", "tauri.localhost"})


def host_hostname(host: str) -> str:
    """Hostname part of a Host header value (handles bracketed IPv6)."""
    if host.startswith("["):
        return host.partition("]")[0].lstrip("[")
    return host.rsplit(":", 1)[0]


class LocalOnlyMiddleware:
    """Reject requests whose Host or Origin is not local.

    Non-browser clients that send no Origin header pass through. Applies
    to both HTTP and WebSocket scopes; a rejected WebSocket handshake is
    closed with policy-violation code 1008.
    """

    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: frozenset[str] = frozenset(),
        allowed_origin_hosts: frozenset[str] = frozenset(),
    ) -> None:
        self.app = app
        self.allowed_hosts = LOCAL_HOSTNAMES | allowed_hosts
        self.allowed_origin_hosts = self.allowed_hosts | allowed_origin_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "")
        if host_hostname(host.strip()).lower() not in self.allowed_hosts:
            log.warning("host_rejected", host=host)
            await self._reject(scope, receive, send, 400, "invalid Host header")
            return
        origin = headers.get("origin")
        if origin is not None and urlsplit(origin).hostname not in self.allowed_origin_hosts:
            log.warning("origin_rejected", origin=origin)
            await self._reject(scope, receive, send, 403, "cross-origin request rejected")
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


# -- logging -----------------------------------------------------------------------


def configure_logging(log_json: bool = False, level: int = logging.INFO) -> None:
    """Console renderer for development, JSON for production — same log
    calls, different renderer."""
    renderer = (structlog.processors.JSONRenderer() if log_json
                else structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


# -- assets ------------------------------------------------------------------------


ASSET_DIRS: tuple[str, ...] = ("css", "js", "spec")
"""Subdirectories of ``view_sweep_dir`` that are served; nothing else is."""


def default_view_sweep_dir() -> Path | None:
    """Directory holding ``css/``, ``js/`` and ``spec/``: the repository
    root under an editable install (``py/view_sweep_server/app.py`` ->
    repo root), else the assets packaged into the wheel, else None.
    Consumers with a vendored copy pass their own path."""
    root = Path(__file__).resolve().parents[2]
    if (root / "css" / "viewsweep.css").is_file():
        return root
    packaged = Path(__file__).resolve().parent / "assets"
    return packaged if (packaged / "css" / "viewsweep.css").is_file() else None


# -- factory -----------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    title: str = "view-sweep server"
    routers: Sequence[APIRouter] = ()
    lifespan: Lifespan | None = None
    static_dir: Path | None = None
    view_sweep_dir: Path | None = field(default_factory=default_view_sweep_dir)
    view_sweep_mount: str = "/view-sweep"
    ws_path: str = "/ws"
    on_cancel: Callable[[str], Any] | None = None
    allowed_hosts: tuple[str, ...] = ()
    allowed_origin_hosts: tuple[str, ...] = tuple(sorted(TAURI_ORIGIN_HOSTNAMES))
    log_json: bool | None = None
    """None leaves logging untouched; True/False calls ``configure_logging``."""


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    if config.log_json is not None:
        configure_logging(config.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ws_manager = ConnectionManager(on_cancel=config.on_cancel)
        app.state.app_config = config
        if config.lifespan is None:
            yield
            return
        async with config.lifespan(app):
            yield

    app = FastAPI(title=config.title, lifespan=lifespan)
    app.add_middleware(
        LocalOnlyMiddleware,
        allowed_hosts=frozenset(h.lower() for h in config.allowed_hosts),
        allowed_origin_hosts=frozenset(h.lower() for h in config.allowed_origin_hosts),
    )
    for router in config.routers:
        app.include_router(router)
    app.include_router(make_ws_router(config.ws_path))

    # Only the asset subtrees are mounted: under an editable install the
    # parent is a full repository checkout (.git, tests, docs, uv.lock).
    if config.view_sweep_dir is not None and Path(config.view_sweep_dir).is_dir():
        for name in ASSET_DIRS:
            sub = Path(config.view_sweep_dir) / name
            if sub.is_dir():
                app.mount(f"{config.view_sweep_mount.rstrip('/')}/{name}",
                          StaticFiles(directory=sub), name=f"view-sweep-{name}")
    elif config.static_dir is not None:
        log.warning("view_sweep_assets_missing",
                    hint="no css/js/spec found; pass AppConfig.view_sweep_dir or "
                         "install a wheel that packages the assets")
    if config.static_dir is not None and Path(config.static_dir).is_dir():
        app.mount("/", StaticFiles(directory=config.static_dir, html=True), name="static")
    return app
