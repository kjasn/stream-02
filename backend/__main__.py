"""Entry point: python -m backend"""

import logging

import uvicorn

from backend.common.config import get_settings


def main():
    settings = get_settings()
    log_level = settings.server.log_level.lower()
    valid_levels = {"critical", "error", "warning", "info", "debug", "trace"}
    if log_level not in valid_levels:
        log_level = "info"

    logging.basicConfig(level=log_level.upper())

    uvicorn.run(
        "backend.main:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
