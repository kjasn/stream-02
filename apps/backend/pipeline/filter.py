"""Stage 1: Filter — dedup, spam keywords, blacklist, per-user rate limiting."""

import hashlib
import time
from typing import Optional

from common.types import DanmakuEvent


class Stage1Filter:
    """Filters events: blacklist → spam keywords → exact dedup → rate limit."""

    def __init__(
        self,
        dedup_window_seconds: float = 30.0,
        rate_limit_per_user: int = 5,
        rate_limit_window_seconds: float = 60.0,
        spam_keywords: Optional[list[str]] = None,
        blacklist_uids: Optional[list[str]] = None,
    ):
        self._dedup_window = dedup_window_seconds
        self._rate_limit = rate_limit_per_user
        self._rate_window = rate_limit_window_seconds
        self._spam_keywords = set(spam_keywords or [])
        self._blacklist = set(blacklist_uids or [])

        # (user_id, msg_hash) → timestamp
        self._recent_messages: dict[tuple[str, str], float] = {}
        # user_id → list of timestamps within rate window
        self._user_timestamps: dict[str, list[float]] = {}

    def accept(self, event: DanmakuEvent) -> bool:
        """Return True if the event passes all filters. Side-effect: records event."""
        now = time.time()
        self._expire(now)

        # 1. Blacklist
        if event.user_id in self._blacklist:
            return False

        # 2. Spam keywords
        if any(kw in event.msg for kw in self._spam_keywords):
            return False

        # 3. Exact dedup (same user + same message hash within window)
        msg_hash = hashlib.md5(event.msg.encode()).hexdigest()[:16]
        key = (event.user_id, msg_hash)
        if key in self._recent_messages:
            return False

        # 4. Rate limit per user
        timestamps = self._user_timestamps.get(event.user_id, [])
        if len(timestamps) >= self._rate_limit:
            return False

        # Record
        self._recent_messages[key] = now
        timestamps.append(now)
        self._user_timestamps[event.user_id] = timestamps
        return True

    def reset(self) -> None:
        self._recent_messages.clear()
        self._user_timestamps.clear()

    def _expire(self, now: float) -> None:
        """Purge stale dedup entries and rate-limit timestamps."""
        dedup_cutoff = now - self._dedup_window
        self._recent_messages = {
            k: ts for k, ts in self._recent_messages.items() if ts >= dedup_cutoff
        }
        rate_cutoff = now - self._rate_window
        for uid in list(self._user_timestamps):
            self._user_timestamps[uid] = [
                ts for ts in self._user_timestamps[uid] if ts >= rate_cutoff
            ]
            if not self._user_timestamps[uid]:
                del self._user_timestamps[uid]
