"""Tests for Stage 3: Semantic dedup."""

import time

from backend.common.types import DanmakuEvent
from backend.pipeline.semantic import Stage3SemanticDedup


def _event(msg="test", uname="user1"):
    return DanmakuEvent(
        cmd="LIVE_OPEN_PLATFORM_DM",
        raw={},
        timestamp=time.time(),
        user_id="1",
        user_name=uname,
        msg=msg,
        value=0,
    )


def test_first_message_accepted():
    sd = Stage3SemanticDedup()
    assert sd.merge_or_accept(_event(msg="hello")) is None


def test_identical_message_merged():
    sd = Stage3SemanticDedup(similarity_threshold=0.85)
    sd.merge_or_accept(_event(msg="来了来了", uname="user1"))
    result = sd.merge_or_accept(_event(msg="来了来了", uname="user2"))
    assert result is not None
    assert "user1" in result and "user2" in result


def test_similar_message_merged():
    sd = Stage3SemanticDedup(similarity_threshold=0.8)
    sd.merge_or_accept(_event(msg="主播好厉害", uname="A"))
    result = sd.merge_or_accept(_event(msg="主播好厉害啊", uname="B"))
    assert result is not None


def test_different_message_not_merged():
    sd = Stage3SemanticDedup(similarity_threshold=0.85)
    sd.merge_or_accept(_event(msg="hello world"))
    result = sd.merge_or_accept(_event(msg="完全不同的内容"))
    assert result is None


def test_reset():
    sd = Stage3SemanticDedup()
    sd.merge_or_accept(_event(msg="msg1"))
    sd.reset()
    assert sd.merge_or_accept(_event(msg="msg1")) is None
