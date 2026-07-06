"""Mixed-mode inference trigger: time-window + event-driven."""

import asyncio
import logging
import time
from typing import Optional

from common.types import DanmakuEvent

logger = logging.getLogger("backend.core.trigger")


class InferenceTrigger:
    """Decides WHEN to trigger inference.

    - Timer mode: fires when ``time_window_seconds`` has elapsed since last inference
      AND there are pending events.
    - Event-driven mode: fires immediately for high-value events (SuperChat, Guard, etc.).
    """

    def __init__(
        self,
        time_window_seconds: float = 15.0,
        event_driven_types: Optional[set[str]] = None,
    ):
        self._window = time_window_seconds
        self._event_types = event_driven_types or set()
        self._last_inference: float = time.time()
        self._pending_count: int = 0
        self._lock = asyncio.Lock()

    async def report_event(self, event: DanmakuEvent) -> bool:
        """Record an event. Returns True if this should trigger immediate inference."""
        async with self._lock:
            self._pending_count += 1
            if event.cmd in self._event_types:
                self._last_inference = time.time()
                self._pending_count = 0
                logger.info(
                    f"Event-driven trigger: {event.cmd} (priority={event.priority})"
                )
                return True
        return False

    async def should_infer(self) -> bool:
        """Return True if the time window has elapsed and events are pending."""
        async with self._lock:
            if self._pending_count == 0:
                return False
            elapsed = time.time() - self._last_inference
            if elapsed >= self._window:
                self._last_inference = time.time()
                self._pending_count = 0
                logger.info(f"Timer trigger: {elapsed:.1f}s elapsed")
                return True
        return False

    def reset(self) -> None:
        self._last_inference = time.time()
        self._pending_count = 0
