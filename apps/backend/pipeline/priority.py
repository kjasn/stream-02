"""Stage 2: Priority sort — SC > Guard > Gift > DM, with value sub-ordering."""

_PRIORITY_TABLE: dict[str, int] = {
    "LIVE_OPEN_PLATFORM_SUPER_CHAT": 0,
    "LIVE_OPEN_PLATFORM_SUPER_CHAT_DEL": 0,
    "LIVE_OPEN_PLATFORM_GUARD": 1,
    "LIVE_OPEN_PLATFORM_SEND_GIFT": 2,
    "LIVE_OPEN_PLATFORM_DM": 3,
}
DEFAULT_PRIORITY = 99


def compute_priority(cmd: str, value: float = 0.0) -> tuple[int, float]:
    """Return (base_priority, sort_value) — lower base = higher priority.

    sort_value is negative so that higher-value events (e.g. expensive SC)
    sort before lower-value events of the same type.
    """
    base = _PRIORITY_TABLE.get(cmd, DEFAULT_PRIORITY)
    return (base, -value)
