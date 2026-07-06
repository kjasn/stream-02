"""Tests for Stage 2: Priority."""

from backend.pipeline.priority import compute_priority


def test_sc_highest_priority():
    p, _ = compute_priority("LIVE_OPEN_PLATFORM_SUPER_CHAT")
    assert p == 0


def test_guard_mid_priority():
    p, _ = compute_priority("LIVE_OPEN_PLATFORM_GUARD")
    assert p == 1


def test_gift_mid_priority():
    p, _ = compute_priority("LIVE_OPEN_PLATFORM_SEND_GIFT")
    assert p == 2


def test_dm_lowest():
    p, _ = compute_priority("LIVE_OPEN_PLATFORM_DM")
    assert p == 3


def test_unknown_default():
    p, _ = compute_priority("UNKNOWN_CMD")
    assert p == 99


def test_sort_order():
    """Lower priority sorts first; higher value sorts before lower value."""
    items = [
        ("LIVE_OPEN_PLATFORM_DM", 0),
        ("LIVE_OPEN_PLATFORM_SUPER_CHAT", 10),
        ("LIVE_OPEN_PLATFORM_SUPER_CHAT", 100),
        ("LIVE_OPEN_PLATFORM_GUARD", 0),
    ]
    sorted_items = sorted(items, key=lambda x: compute_priority(x[0], x[1]))
    # SC ¥100 first, then SC ¥10, then Guard, then DM
    assert sorted_items[0] == ("LIVE_OPEN_PLATFORM_SUPER_CHAT", 100)
    assert sorted_items[1] == ("LIVE_OPEN_PLATFORM_SUPER_CHAT", 10)
    assert sorted_items[2] == ("LIVE_OPEN_PLATFORM_GUARD", 0)
    assert sorted_items[3] == ("LIVE_OPEN_PLATFORM_DM", 0)
