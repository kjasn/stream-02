"""FastAPI application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.common.config import get_settings
from backend.core.orchestrator import LiveStreamOrchestrator
from backend.io.mongo import mongo

settings = get_settings()
logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongo.connect()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    app.state.orchestrator = None
    app.state.orchestrator_task = None
    try:
        yield
    finally:
        await mongo.close()
        logger.info("Shutting down")
        orchestrator: LiveStreamOrchestrator | None = app.state.orchestrator
        task: asyncio.Task | None = app.state.orchestrator_task

        if orchestrator is not None:
            await orchestrator.stop()

        if task is not None:
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=5)
                except asyncio.TimeoutError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            elif not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    logger.error(
                        "Orchestrator task failed",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )

        app.state.orchestrator = None
        app.state.orchestrator_task = None


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .api.routes import router

    app.include_router(router)

    @app.get("/")
    async def root():
        return {"service": settings.app_name, "version": settings.app_version}

    return app


app = create_app()
