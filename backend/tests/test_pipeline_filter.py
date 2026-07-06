"""Tests for Stage 1: Filter."""

import time

from backend.common.types import DanmakuEvent
from backend.pipeline.filter import Stage1Filter


def _event(cmd="LIVE_OPEN_PLATFORM_DM", uid="123", uname="测试", msg="test"):
    return DanmakuEvent(cmd=cmd, raw={}, timestamp=time.time(), user_id=uid, user_name=uname, msg=msg, value=0)


def test_normal_event_passes():
    f = Stage1Filter()
    assert f.accept(_event(msg="hello"))


def test_spam_keyword_blocked():
    f = Stage1Filter(spam_keywords=["广告", "加群"])
    assert not f.accept(_event(msg="加群12345"))


def test_blacklist_uid_blocked():
    f = Stage1Filter(blacklist_uids=["bad_user"])
    assert not f.accept(_event(uid="bad_user", msg="hello"))


def test_duplicate_message_blocked():
    f = Stage1Filter(dedup_window_seconds=60)
    e = _event(msg="哈哈哈哈")
    assert f.accept(e)
    # Same user, same message
    assert not f.accept(e)


def test_different_message_passes():
    f = Stage1Filter(dedup_window_seconds=60)
    assert f.accept(_event(uid="a", msg="msg1"))
    assert f.accept(_event(uid="a", msg="msg2"))


def test_rate_limit():
    f = Stage1Filter(rate_limit_per_user=3, rate_limit_window_seconds=60)
    uid = "user123"
    for i in range(3):
        assert f.accept(_event(uid=uid, msg=f"msg{i}"))
    # 4th message blocked
    assert not f.accept(_event(uid=uid, msg="msg4"))


def test_reset_clears_dedup_and_rate_limit():
    f = Stage1Filter(rate_limit_per_user=1)
    assert f.accept(_event(msg="hello"))
    # Rate limit kicks in
    assert not f.accept(_event(msg="hello2"))
    f.reset()
    # After reset, rate limit buffer cleared — can accept again
    assert f.accept(_event(msg="hello3"))
