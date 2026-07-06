"""Event buffer that collects and formats live-stream events as LLM context text."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from backend.pipeline.coordinator import DanmakuPipeline

# 优先级
_EVENT_PRIORITY = {
    "LIVE_OPEN_PLATFORM_SUPER_CHAT": 0,
    "LIVE_OPEN_PLATFORM_SUPER_CHAT_DEL": 0,
    "LIVE_OPEN_PLATFORM_GUARD": 1,
    "LIVE_OPEN_PLATFORM_SEND_GIFT": 2,
    "LIVE_OPEN_PLATFORM_DM": 3,
}


class DanmakuCollector:
    """Collects Bilibili live events and formats them for LLM prompt injection.

    If an optional DanmakuPipeline is provided, each event is run through
    the pipeline stages (filter → priority → semantic dedup → context)
    before being stored in the local buffer.
    """

    def __init__(
        self,
        max_events: int = 20,
        ttl_seconds: float = 60.0,
        pipeline: DanmakuPipeline | None = None,
    ):
        self._max_events = max_events
        self._ttl_seconds = ttl_seconds
        self._buffer: deque[tuple[float, str, str]] = deque()
        self._pipeline = pipeline
        # Each entry: (timestamp, cmd, formatted_line)

    def add_event(self, data: dict) -> None:
        """Ingest a raw event from BiliLiveClient.

        When a pipeline is configured, the event is run through all pipeline
        stages; if filtered out, nothing is stored.
        """
        if self._pipeline is not None:
            event = self._pipeline.process_raw(data)
            if event is None:
                return
            # Use the (possibly merged) formatted output from the pipeline
            cmd = event.cmd
            line = self._format_event(cmd, data)
        else:
            cmd = data.get("cmd", "")
            line = self._format_event(cmd, data)

        if line is None:
            return

        now = time.time()
        self._buffer.append((now, cmd, line))
        self._expire(now)

        while len(self._buffer) > self._max_events:
            self._buffer.popleft()

    def get_context_text(self) -> str:
        """Return the current event buffer as a formatted context string."""
        self._expire(time.time())
        if not self._buffer:
            return ""

        entries = sorted(
            list(self._buffer),
            key=lambda x: (_EVENT_PRIORITY.get(x[1], 99), x[0]),
        )
        return "【直播间动态】\n" + "\n".join(e[2] for e in entries)

    @property
    def event_count(self) -> int:
        self._expire(time.time())
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    # ── private ──

    def _expire(self, now: float) -> None:
        """Remove entries older than ttl_seconds."""
        cutoff = now - self._ttl_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    @staticmethod
    def _format_event(cmd: str, data: dict) -> Optional[str]:
        """Format a single event into a one-line description. Returns None if not supported."""
        uname = (data.get("user_info") or data).get("uname", "匿名用户")

        if cmd == "LIVE_OPEN_PLATFORM_DM":
            msg = data.get("msg", "")
            return f"[弹幕] {uname}: {msg}"

        elif cmd == "LIVE_OPEN_PLATFORM_SEND_GIFT":
            gift_name = data.get("gift_name", "礼物")
            gift_num = data.get("gift_num", 1)
            return f"[礼物] {uname} 送出 {gift_name} x{gift_num}"

        elif cmd == "LIVE_OPEN_PLATFORM_SUPER_CHAT":
            rmb = data.get("rmb", 0)
            msg = data.get("message", "")
            return f"[SC ¥{rmb}] {uname}: {msg}"

        elif cmd == "LIVE_OPEN_PLATFORM_SUPER_CHAT_DEL":
            return f"[SC删除] {uname} 的醒目留言已被撤回"

        elif cmd == "LIVE_OPEN_PLATFORM_GUARD":
            guard_level = data.get("guard_level", 0)
            level_name = {1: "总督", 2: "提督", 3: "舰长"}.get(guard_level, "?")
            return f"[大航海] {uname} 开通了{level_name}"

        elif cmd == "LIVE_OPEN_PLATFORM_LIKE":
            return None  # Too noisy to include

        elif cmd == "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER":
            return None  # Too noisy

        else:
            return None
