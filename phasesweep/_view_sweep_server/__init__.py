"""Vendored copy of view-sweep-server 0.3.0 (Apache-2.0, same author).

view-sweep-server is not yet on a public package index, so the server
toolkit and its browser assets (`assets/`, served at `/view-sweep/`) ship
inside phasesweep until it is. Keep the modules identical to the upstream
tag apart from the package name in imports.
"""

from phasesweep._view_sweep_server.app import (
    AppConfig,
    FiniteJSONResponse,
    LocalOnlyMiddleware,
    configure_logging,
    create_app,
    default_view_sweep_dir,
    json_finite,
)
from phasesweep._view_sweep_server.protocol import (
    CLIENT_MSG_TYPES,
    JOB_STATUSES,
    SERVER_MSG_TYPES,
    ClientMsg,
    ConnectionState,
    JobStatus,
    ServerMsg,
    parse_client_msg,
)
from phasesweep._view_sweep_server.ws import ConnectionManager, make_ws_router, ws_router

__version__ = "0.3.0"

__all__ = [
    "CLIENT_MSG_TYPES",
    "JOB_STATUSES",
    "SERVER_MSG_TYPES",
    "AppConfig",
    "ClientMsg",
    "ConnectionManager",
    "ConnectionState",
    "FiniteJSONResponse",
    "JobStatus",
    "LocalOnlyMiddleware",
    "ServerMsg",
    "__version__",
    "configure_logging",
    "create_app",
    "default_view_sweep_dir",
    "json_finite",
    "make_ws_router",
    "parse_client_msg",
    "ws_router",
]
