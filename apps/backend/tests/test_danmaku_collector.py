"""Tests for DanmakuCollector — event buffering and formatting."""

import time

from backend.services.live.danmaku_collector import DanmakuCollector


class TestFormatEvent:
    def test_danmaku(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_DM",
                "uname": "用户A",
                "msg": "哈哈哈",
            }
        )
        text = collector.get_context_text()
        assert "[弹幕] 用户A: 哈哈哈" in text

    def test_gift(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
                "uname": "用户B",
                "gift_name": "小花",
                "gift_num": 3,
            }
        )
        text = collector.get_context_text()
        assert "[礼物] 用户B 送出 小花 x3" in text

    def test_super_chat(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_SUPER_CHAT",
                "uname": "用户C",
                "rmb": 30,
                "message": "太强了",
            }
        )
        text = collector.get_context_text()
        assert "[SC ¥30] 用户C: 太强了" in text

    def test_guard(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_GUARD",
                "guard_level": 1,
                "user_info": {"uname": "用户D"},
            }
        )
        text = collector.get_context_text()
        assert "[大航海] 用户D 开通了总督" in text

    def test_like_is_ignored(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_LIKE",
                "uname": "用户E",
                "like_text": "点赞",
            }
        )
        assert collector.event_count == 0

    def test_room_enter_is_ignored(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER",
                "uname": "用户F",
            }
        )
        assert collector.event_count == 0

    def test_super_chat_del(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_SUPER_CHAT_DEL",
                "user_info": {"uname": "用户G"},
            }
        )
        text = collector.get_context_text()
        assert "用户G 的醒目留言已被撤回" in text


class TestEventCollection:
    def test_empty_collector_returns_empty(self):
        collector = DanmakuCollector()
        assert collector.get_context_text() == ""
        assert collector.event_count == 0

    def test_multiple_events_sorted_by_priority(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_DM",
                "uname": "用户A",
                "msg": "弹幕1",
            }
        )
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_SUPER_CHAT",
                "uname": "用户B",
                "rmb": 50,
                "message": "SC消息",
            }
        )
        text = collector.get_context_text()
        sc_pos = text.find("SC")
        dm_pos = text.find("[弹幕]")
        assert sc_pos < dm_pos  # SC sorted before danmaku

    def test_event_count(self):
        collector = DanmakuCollector()
        for i in range(5):
            collector.add_event(
                {
                    "cmd": "LIVE_OPEN_PLATFORM_DM",
                    "uname": f"用户{i}",
                    "msg": f"弹幕{i}",
                }
            )
        assert collector.event_count == 5

    def test_max_events_eviction(self):
        collector = DanmakuCollector(max_events=3)
        for i in range(5):
            collector.add_event(
                {
                    "cmd": "LIVE_OPEN_PLATFORM_DM",
                    "uname": f"用户{i}",
                    "msg": f"弹幕{i}",
                }
            )
        assert collector.event_count <= 3

    def test_reset_clears_all(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_DM",
                "uname": "用户A",
                "msg": "弹幕",
            }
        )
        collector.reset()
        assert collector.event_count == 0
        assert collector.get_context_text() == ""

    def test_context_header(self):
        collector = DanmakuCollector()
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_DM",
                "uname": "用户A",
                "msg": "弹幕",
            }
        )
        text = collector.get_context_text()
        assert text.startswith("【直播间动态】")


class TestTimeDecay:
    def test_expired_events_removed(self, monkeypatch):
        collector = DanmakuCollector(ttl_seconds=1.0)
        collector.add_event(
            {
                "cmd": "LIVE_OPEN_PLATFORM_DM",
                "uname": "用户A",
                "msg": "旧弹幕",
            }
        )

        # Fast-forward time by 2 seconds
        fake_now = time.time() + 2.0
        monkeypatch.setattr(time, "time", lambda: fake_now)

        assert collector.get_context_text() == ""
        assert collector.event_count == 0
