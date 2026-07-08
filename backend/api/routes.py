"""FastAPI routes for backend control plane."""

import asyncio
import logging
import time

from fastapi import APIRouter, Request

from backend.common.config import get_settings
from backend.common.types import SessionConfigRequest, SessionStatusResponse
from backend.core.orchestrator import LiveStreamOrchestrator

logger = logging.getLogger("backend.routes")
router = APIRouter()

_start_time = time.time()


def _is_running(task: asyncio.Task | None) -> bool:
    return task is not None and not task.done()


async def _stop_task(task: asyncio.Task | None) -> None:
    if task is None:
        return

    task_loop = task.get_loop()
    if task_loop.is_closed():
        return

    current_loop = asyncio.get_running_loop()
    if task_loop is not current_loop:
        if not task.done():
            task_loop.call_soon_threadsafe(task.cancel)
        return

    if task.done():
        await asyncio.gather(task, return_exceptions=True)
        return

    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "stream-02-backend",
        "uptime": time.time() - _start_time,
    }


@router.post("/session/start")
async def start_session(req: SessionConfigRequest, request: Request):
    task: asyncio.Task | None = getattr(request.app.state, "orchestrator_task", None)
    orchestrator: LiveStreamOrchestrator | None = getattr(request.app.state, "orchestrator", None)

    if _is_running(task) and orchestrator is not None:
        return {
            "status": "already_started",
            "session_id": orchestrator.session_id,
            "config": req.model_dump(),
        }

    orchestrator = LiveStreamOrchestrator(get_settings())
    task = asyncio.create_task(
        orchestrator.start(req),
        name="live_stream_orchestrator",
    )
    request.app.state.orchestrator = orchestrator
    request.app.state.orchestrator_task = task

    logger.info(f"Session started: {req.model_dump()}")
    return {"status": "started", "config": req.model_dump()}


@router.post("/session/stop")
async def stop_session(request: Request):
    task: asyncio.Task | None = getattr(request.app.state, "orchestrator_task", None)
    orchestrator: LiveStreamOrchestrator | None = getattr(request.app.state, "orchestrator", None)

    if orchestrator is not None:
        await orchestrator.stop()

    await _stop_task(task)

    request.app.state.orchestrator = None
    request.app.state.orchestrator_task = None

    logger.info("Session stopped")
    return {"status": "stopped"}


@router.get("/status")
async def get_status(request: Request) -> SessionStatusResponse:
    task: asyncio.Task | None = getattr(request.app.state, "orchestrator_task", None)
    orchestrator: LiveStreamOrchestrator | None = getattr(request.app.state, "orchestrator", None)

    if orchestrator is None:
        return SessionStatusResponse(
            active=False,
            uptime_seconds=time.time() - _start_time,
        )

    return SessionStatusResponse(
        session_id=orchestrator.session_id or "",
        active=_is_running(task) and orchestrator.active,
        llm_connected=orchestrator.llm_connected,
        livekit_connected=orchestrator.livekit_connected,
        bili_connected=orchestrator.bili_connected,
        uptime_seconds=orchestrator.uptime_seconds,
        event_count=orchestrator.pipeline.event_count,
        last_inference_time=orchestrator.last_inference_time,
    )
