"""FastAPI routes for backend control plane."""

import time
import logging

from fastapi import APIRouter, HTTPException

from common.types import SessionConfigRequest, SessionStatusResponse

logger = logging.getLogger("backend.routes")
router = APIRouter()

_start_time = time.time()
_session_active = False
_session_config: SessionConfigRequest | None = None


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "stream-02-backend",
        "uptime": time.time() - _start_time,
    }


@router.post("/session/start")
async def start_session(req: SessionConfigRequest):
    global _session_active, _session_config
    _session_active = True
    _session_config = req
    logger.info(f"Session started: {req.model_dump()}")
    return {"status": "started", "config": req.model_dump()}


@router.post("/session/stop")
async def stop_session():
    global _session_active, _session_config
    _session_active = False
    _session_config = None
    logger.info("Session stopped")
    return {"status": "stopped"}


@router.get("/status")
async def get_status() -> SessionStatusResponse:
    return SessionStatusResponse(
        active=_session_active,
        uptime_seconds=time.time() - _start_time,
    )
