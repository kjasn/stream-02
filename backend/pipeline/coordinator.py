"""Pipeline coordinator — chains filter → priority → semantic dedup → context."""

import time
from typing import Optional

from backend.common.config import PipelineSettings
from backend.common.types import DanmakuEvent

from .context import Stage4ContextAssembler
from .filter import Stage1Filter
from .priority import compute_priority
from .semantic import Stage3SemanticDedup


class DanmakuPipeline:
    """Multi-stage danmaku/superchat filtering and reranking pipeline.

    Usage:
        pipeline = DanmakuPipeline(config)
        event = pipeline.process(raw_event)  # Returns None if filtered out
        context = pipeline.assemble_prompt()
    """

    def __init__(self, config: PipelineSettings):
        self.stage1 = Stage1Filter(
            dedup_window_seconds=config.dedup_window_seconds,
            rate_limit_per_user=config.rate_limit_per_user,
            rate_limit_window_seconds=config.rate_limit_window_seconds,
            spam_keywords=list(config.spam_keywords),
            blacklist_uids=list(config.blacklist_uids),
        )
        self.stage3 = Stage3SemanticDedup(
            similarity_threshold=config.similarity_threshold,
            merge_window=config.merge_window_seconds,
        )
        self.stage4 = Stage4ContextAssembler(
            ttl_seconds=config.ttl_seconds,
            max_events=config.max_events,
            token_budget=config.token_budget,
        )

    def process(self, event: DanmakuEvent) -> Optional[DanmakuEvent]:
        """Run event through stages 1-4. Returns the (possibly merged) event, or None if filtered."""
        # Stage 1: Filter
        if not self.stage1.accept(event):
            return None

        # Stage 2: Priority
        event.priority, _ = compute_priority(event.cmd, event.value)

        # Stage 3: Semantic dedup
        merged_text = self.stage3.merge_or_accept(event)
        if merged_text is not None:
            event.msg = merged_text

        # Stage 4: Add to context buffer
        self.stage4.add_event(event)
        return event

    def process_raw(self, raw: dict) -> Optional[DanmakuEvent]:
        """Parse raw bili event dict into DanmakuEvent and run through pipeline."""
        cmd = raw.get("cmd", "")
        data = raw.get("data", raw)
        user_info = data.get("user_info") or data

        event = DanmakuEvent(
            cmd=cmd,
            raw=raw,
            timestamp=time.time(),
            user_id=str(user_info.get("uid", "")),
            user_name=user_info.get("uname", "匿名用户"),
            msg=data.get("msg", data.get("message", "")),
            value=data.get("rmb", data.get("price", 0.0)),
        )
        return self.process(event)

    def assemble_prompt(self, language: str = "zh") -> str:
        return self.stage4.assemble_prompt(language)

    def assemble_context(self) -> str:
        return self.stage4.assemble_context()

    @property
    def event_count(self) -> int:
        return self.stage4.event_count

    def reset(self) -> None:
        self.stage1.reset()
        self.stage3.reset()
        self.stage4.reset()
