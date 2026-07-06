"""Stage 3: Semantic dedup — merge similar messages within a time window."""

import difflib
import time
from typing import Optional

from backend.common.types import DanmakuEvent


class Stage3SemanticDedup:
    """Merge similar messages (similarity > threshold) into aggregate entries."""

    def __init__(self, similarity_threshold: float = 0.85, merge_window: float = 10.0):
        self._threshold = similarity_threshold
        self._merge_window = merge_window
        # (merged_msg_hash, user_names_str) → expiry_time
        self._recent: dict[tuple[str, str], float] = {}

    def merge_or_accept(self, event: DanmakuEvent) -> Optional[str]:
        """Return merged text if this event was merged into an existing one,
        or None if it should be kept as-is.
        """
        now = time.time()
        self._expire(now)

        for (existing_msg, existing_users), expiry in list(self._recent.items()):
            ratio = difflib.SequenceMatcher(None, event.msg, existing_msg).ratio()
            if ratio >= self._threshold:
                # Merge: append user name
                merged_users = f"{existing_users}、{event.user_name}"
                merged_msg = f"[{merged_users}]: {existing_msg}"
                self._recent[(existing_msg, merged_users)] = now
                return merged_msg

        # Accept as new — record
        self._recent[(event.msg, event.user_name)] = now
        return None

    def reset(self) -> None:
        self._recent.clear()

    def _expire(self, now: float) -> None:
        cutoff = now - self._merge_window
        self._recent = {k: ts for k, ts in self._recent.items() if ts >= cutoff}
