"""WebSocket message shapes and job status vocabulary."""

from __future__ import annotations

from typing import Literal, TypedDict

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


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
