"""WebSocket message shapes shared by servers and the WebSocketProvider.

Server -> client messages carry a job lifecycle: submit -> progress ->
complete | failed | cancelled. Client -> server messages manage the
per-connection subscription set and request cancellation. Every message
is a JSON object whose ``type`` field selects the shape.

``spec/events.toml`` mirrors the server -> client shapes as ``vs:job-*``
DOM events so panels see one vocabulary regardless of provider.
"""

from __future__ import annotations

from typing import Literal, TypedDict, get_args

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
JOB_STATUSES: tuple[str, ...] = get_args(JobStatus)

ConnectionState = Literal["connected", "connecting", "disconnected"]


# -- server -> client -----------------------------------------------------------


class JobProgressMsg(TypedDict):
    type: Literal["job_progress"]
    job_id: str
    completed: int
    total: int
    latest_result_id: str | None


class JobCompleteMsg(TypedDict):
    type: Literal["job_complete"]
    job_id: str
    result_ids: list[str]


class JobFailedMsg(TypedDict):
    type: Literal["job_failed"]
    job_id: str
    error: str


class JobCancelledMsg(TypedDict):
    type: Literal["job_cancelled"]
    job_id: str


ServerMsg = JobProgressMsg | JobCompleteMsg | JobFailedMsg | JobCancelledMsg
SERVER_MSG_TYPES: tuple[str, ...] = (
    "job_progress", "job_complete", "job_failed", "job_cancelled",
)


# -- client -> server -----------------------------------------------------------


class SubscribeMsg(TypedDict):
    type: Literal["subscribe"]
    job_ids: list[str]


class UnsubscribeMsg(TypedDict):
    type: Literal["unsubscribe"]
    job_ids: list[str]


class CancelMsg(TypedDict):
    type: Literal["cancel"]
    job_id: str


ClientMsg = SubscribeMsg | UnsubscribeMsg | CancelMsg
CLIENT_MSG_TYPES: tuple[str, ...] = ("subscribe", "unsubscribe", "cancel")


# -- parsing ----------------------------------------------------------------------


def parse_client_msg(raw: object) -> ClientMsg | None:
    """Validate an inbound JSON object; return None for anything malformed.

    Unknown types and wrong field shapes are dropped rather than raised so
    a misbehaving client cannot take the socket down.
    """
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    if kind in ("subscribe", "unsubscribe"):
        ids = raw.get("job_ids")
        if not isinstance(ids, list) or not all(isinstance(j, str) for j in ids):
            return None
        return {"type": kind, "job_ids": ids}  # type: ignore[return-value]
    if kind == "cancel":
        job_id = raw.get("job_id")
        if not isinstance(job_id, str):
            return None
        return {"type": "cancel", "job_id": job_id}
    return None
