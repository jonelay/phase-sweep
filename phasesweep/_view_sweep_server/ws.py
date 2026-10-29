"""WebSocket connection manager and endpoint.

``ConnectionManager`` tracks connected clients and their job subscriptions
and fans server messages out to the subscribers. It is stateless across
reconnects: a client that drops re-subscribes and re-fetches over REST.

The manager owns no job state. Consumers hand it a ``broadcast`` target
(``manager.broadcast``) and, optionally, an ``on_cancel`` hook that the
``cancel`` client message dispatches to.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from phasesweep._view_sweep_server.protocol import ClientMsg, ServerMsg, parse_client_msg

log = structlog.get_logger()


class _Socket(Protocol):
    async def accept(self) -> None: ...
    async def send_json(self, data: Any) -> None: ...


class ConnectionManager:
    """Per-client subscription sets plus filtered broadcast.

    ``on_cancel(job_id)`` must be a plain synchronous callable; it runs
    inline for each ``cancel`` frame. Exceptions are logged and swallowed
    so one bad request cannot take down the socket loop. An ``async def``
    hook is rejected at call time (its coroutine is closed and a warning
    logged) rather than silently never running.
    """

    def __init__(self, on_cancel: Callable[[str], object] | None = None) -> None:
        self._subs: dict[Any, set[str]] = {}
        self.on_cancel = on_cancel

    # -- lifecycle -----------------------------------------------------------------

    async def connect(self, ws: _Socket) -> None:
        await ws.accept()
        self._subs[ws] = set()

    def disconnect(self, ws: Any) -> None:
        self._subs.pop(ws, None)

    # -- inspection ----------------------------------------------------------------

    @property
    def client_count(self) -> int:
        return len(self._subs)

    def clients(self) -> Iterable[Any]:
        return list(self._subs)

    def subscriptions(self, ws: Any) -> set[str]:
        return set(self._subs.get(ws, ()))

    # -- inbound -------------------------------------------------------------------

    def handle(self, ws: Any, raw: object) -> ClientMsg | None:
        """Apply one inbound frame. Returns the parsed message, or None if
        the frame was malformed or the client is no longer registered."""
        msg = parse_client_msg(raw)
        if msg is None:
            return None
        subs = self._subs.get(ws)
        if subs is None:
            return None
        kind = msg["type"]
        if kind == "subscribe":
            subs.update(msg["job_ids"])
        elif kind == "unsubscribe":
            subs.difference_update(msg["job_ids"])
        elif kind == "cancel" and self.on_cancel is not None:
            try:
                result = self.on_cancel(msg["job_id"])
            except Exception as e:  # noqa: BLE001 - hook failures must not kill the loop
                log.warning("ws_cancel_failed", job_id=msg["job_id"], error=str(e))
            else:
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    log.warning("ws_cancel_hook_async", job_id=msg["job_id"],
                                hint="on_cancel must be synchronous; the hook did not run")
        return msg

    # -- outbound ------------------------------------------------------------------

    async def broadcast(self, msg: ServerMsg) -> None:
        """Send ``msg`` to every client subscribed to its ``job_id``; clients
        whose send fails are dropped."""
        job_id = msg["job_id"]
        dead = []
        for ws, subs in list(self._subs.items()):
            if job_id not in subs:
                continue
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001 - any transport failure means "gone"
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def make_ws_router(path: str = "/ws", state_attr: str = "ws_manager") -> APIRouter:
    """Router exposing one WebSocket endpoint that drives the
    ``ConnectionManager`` found at ``app.state.<state_attr>``."""
    router = APIRouter()

    @router.websocket(path)
    async def websocket_endpoint(ws: WebSocket) -> None:
        cm: ConnectionManager = getattr(ws.app.state, state_attr)
        await cm.connect(ws)
        try:
            while True:
                try:
                    raw = await ws.receive_json()
                except (ValueError, KeyError):
                    # not JSON, or a binary frame (starlette raises KeyError('text'))
                    await ws.close(code=1003)  # unsupported data
                    return
                cm.handle(ws, raw)
        except WebSocketDisconnect:
            pass
        finally:
            cm.disconnect(ws)

    return router


ws_router = make_ws_router()
"""Default router: ``/ws`` bound to ``app.state.ws_manager``."""
