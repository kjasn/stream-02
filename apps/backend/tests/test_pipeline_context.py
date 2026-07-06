"""Tests for Stage 4: Context assembly."""

import time

from backend.common.types import DanmakuEvent
from backend.pipeline.context import Stage4ContextAssembler


def _event(cmd="LIVE_OPEN_PLATFORM_DM", uname="用户", msg="test", value=0, ts=None):
    if ts is None:
        ts = time.time()
    return DanmakuEvent(cmd=cmd, raw={}, timestamp=ts, user_id="1", user_name=uname, msg=msg, value=value)


def test_empty_buffer():
    ctx = Stage4ContextAssembler()
    text = ctx.assemble_context()
    assert "暂无弹幕" in text


def test_formatted_output():
    ctx = Stage4ContextAssembler()
    ctx.add_event(_event(cmd="LIVE_OPEN_PLATFORM_SUPER_CHAT", uname="大佬", msg="666", value=100))
    ctx.add_event(_event(msg="日常弹幕"))
    prompt = ctx.assemble_prompt()
    assert "大佬" in prompt
    assert "日常弹幕" in prompt
    assert "SC" in prompt


def test_token_budget_truncation():
    ctx = Stage4ContextAssembler(token_budget=10)
    for i in range(10):
        ctx.add_event(_event(msg=f"这是一条很长的弹幕消息{i}" * 5))
    prompt = ctx.assemble_prompt()
    assert len(prompt) > 0


def test_reset():
    ctx = Stage4ContextAssembler()
    ctx.add_event(_event(msg="msg1"))
    assert ctx.event_count == 1
    ctx.reset()
    assert ctx.event_count == 0


def test_event_count():
    ctx = Stage4ContextAssembler(max_events=5)
    for i in range(10):
        ctx.add_event(_event(msg=f"msg{i}"))
    # deque is limited to max_events
    assert ctx.event_count <= 5
