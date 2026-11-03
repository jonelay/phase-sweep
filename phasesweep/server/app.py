"""FastAPI app factory: settings and domain lifespan on top of the
view-sweep app factory.

view-sweep owns the generic layer — WebSocket connection manager, local-
only Host/Origin guard, logging, static mounts. This module owns what is
phase-sweep's: motor configs, the result store, the job manager, and the
dashboard tree.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from phasesweep._extras import MissingExtra

try:
    import structlog
    from fastapi import FastAPI
except ImportError as e:
    raise MissingExtra(f"phasesweep.server needs {e.name}", "server") from None

from phasesweep._view_sweep_server import AppConfig
from phasesweep._view_sweep_server import create_app as create_view_sweep_app
from phasesweep.machines.configs import load_motor
from phasesweep.machines.motor import Motor
from phasesweep.result_store import ResultStore
from phasesweep.server.jobs import JobManager
from phasesweep.server.mcp import McpActivity, build_mcp, mcp_routes
from phasesweep.server.routes import api

__all__ = [
    "ServerSettings",
    "create_app",
    "load_motor_configs",
    "load_user_configs",
]

log = structlog.get_logger()


@dataclass(frozen=True)
class ServerSettings:
    motors_dir: Path = Path("motors")
    user_configs_dir: Path = Path("user_configs")
    output_dir: Path = Path("output")
    workers: int = 4
    subtask_timeout_s: float = 600.0
    max_sweep_points: int = 500
    mesh_cache_dir: str | None = None
    log_json: bool = False
    extra_allowed_hosts: tuple[str, ...] = ()


def load_motor_configs(motors_dir: Path) -> dict[str, Motor]:
    """Motors keyed by TOML filename stem; invalid files
    are skipped with a warning, matching configs.load_motors."""
    out: dict[str, Motor] = {}
    for p in sorted(Path(motors_dir).glob("*.toml")):
        try:
            out[p.stem] = load_motor(p)
        except (KeyError, ValueError, TypeError) as e:
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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        motors = load_motor_configs(settings.motors_dir)
        user_names = load_user_configs(settings.user_configs_dir, motors)
        store = ResultStore(settings.output_dir)
        manager = JobManager(
            motors, store,
            workers=settings.workers,
            subtask_timeout_s=settings.subtask_timeout_s,
            mesh_cache_dir=settings.mesh_cache_dir,
            max_sweep_points=settings.max_sweep_points,
        )
        app.state.manager = manager
        manager.on_terminal(app.state.mcp_activity.record_job)
        app.state.settings = settings
        app.state.user_config_names = user_names
        ws_manager = app.state.ws_manager  # created by view-sweep before us
        ws_manager.on_cancel = _cancel_hook(manager)
        await manager.start(ws_manager.broadcast)
        log.info("server_started", motors=len(motors), user_configs=len(user_names),
                 output_dir=str(settings.output_dir), workers=settings.workers)
        try:
            yield
        finally:
            await manager.stop()
            log.info("server_stopped")

    @asynccontextmanager
    async def lifespan_with_mcp(app: FastAPI) -> AsyncIterator[None]:
        async with lifespan(app), mcp_server.session_manager.run():
            yield

    # Dashboard static mount
    dashboard = Path(__file__).resolve().parent.parent / "dashboard"
    app = create_view_sweep_app(AppConfig(
        title="phase-sweep server",
        routers=[api],
        lifespan=lifespan_with_mcp,
        static_dir=dashboard,
        allowed_hosts=settings.extra_allowed_hosts,
        log_json=settings.log_json,
    ))
    # view-sweep mounts the dashboard at "/" last; /mcp must precede it
    # or the catch-all static mount shadows it.
    app.state.mcp_activity = McpActivity()
    mcp_server = build_mcp(app, app.state.mcp_activity)
    app.router.routes[0:0] = mcp_routes(mcp_server)
    return app


def _cancel_hook(manager: JobManager):
    """WS `cancel` frames map to JobManager.cancel; unknown ids are a
    client error, not a server one, so they are logged and dropped."""
    def cancel(job_id: str) -> None:
        try:
            manager.cancel(job_id)
        except KeyError:
            log.info("ws_cancel_unknown_job", job_id=job_id)
    return cancel
