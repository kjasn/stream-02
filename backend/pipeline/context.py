"""Stage 4: Context assembly — prompt template with token budget and priority ordering."""

import time
from collections import deque
from typing import Optional

from backend.common.types import DanmakuEvent
from .priority import compute_priority

PROMPT_TEMPLATE = """你是一个直播间AI助手。以下是最近{seconds}秒的弹幕动态：
{events}

请基于以上弹幕和当前音视频画面，用{language}给出合适的回复。"""


class Stage4ContextAssembler:
    """Assembles context text for LLM from buffered DanmakuEvents."""

    def __init__(self, ttl_seconds: float = 60.0, max_events: int = 50, token_budget: int = 2048):
        self._ttl = ttl_seconds
        self._token_budget = token_budget
        self._buffer: deque[DanmakuEvent] = deque(maxlen=max_events)

    def add_event(self, event: DanmakuEvent) -> None:
        self._expire()
        self._buffer.append(event)

    def assemble_prompt(self, language: str = "zh") -> str:
        """Build a full prompt string from buffered events, respecting token budget."""
        self._expire()
        events_text = self._format_events()
        return PROMPT_TEMPLATE.format(
            seconds=int(self._ttl),
            events=events_text,
            language=language,
        )

    def assemble_context(self) -> str:
        """Return just the formatted events block (without the template wrapper)."""
        self._expire()
        return self._format_events()

    @property
    def event_count(self) -> int:
        self._expire()
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    def _expire(self) -> None:
        cutoff = time.time() - self._ttl
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def _format_events(self) -> str:
        if not self._buffer:
            return "暂无弹幕"

        ordered = sorted(
            self._buffer,
            key=lambda e: compute_priority(e.cmd, e.value),
        )
        lines: list[str] = []
        char_budget = self._token_budget * 2  # ~2 chars per token for CJK

        for ev in ordered:
            line = self._format_one(ev)
            if not line:
                continue
            if len("\n".join(lines + [line])) > char_budget:
                break
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _format_one(event: DanmakuEvent) -> Optional[str]:
        cmd = event.cmd
        uname = event.user_name or "匿名用户"

        if cmd == "LIVE_OPEN_PLATFORM_DM":
            return f"[弹幕] {uname}: {event.msg}"
        if cmd == "LIVE_OPEN_PLATFORM_SEND_GIFT":
            return f"[礼物] {uname} 送出礼物 (¥{event.value:.0f})"
        if cmd == "LIVE_OPEN_PLATFORM_SUPER_CHAT":
            return f"[SC ¥{event.value:.0f}] {uname}: {event.msg}"
        if cmd == "LIVE_OPEN_PLATFORM_SUPER_CHAT_DEL":
            return f"[SC删除] {uname}"
        if cmd == "LIVE_OPEN_PLATFORM_GUARD":
            return f"[大航海] {uname} 开通大航海"
        if cmd == "LIVE_OPEN_PLATFORM_LIVE_START":
            return "[直播] 开播了"
        if cmd == "LIVE_OPEN_PLATFORM_LIVE_END":
            return "[直播] 下播了"
        return None
