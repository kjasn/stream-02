"""Shared Pydantic models used across backend modules."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    OMNI = "omni"


class DanmakuEvent(BaseModel):
    cmd: str = ""
    raw: dict = Field(default_factory=dict)
    timestamp: float = 0.0
    user_id: str = ""
    user_name: str = "匿名用户"
    msg: str = ""
    priority: int = 99
    value: float = 0.0


class AvSegment(BaseModel):
    audio_base64: Optional[str] = None
    audio_sample_rate: int = 16000
    audio_duration: float = 0.0
    image_base64: Optional[str] = None


class SessionConfig(BaseModel):
    media_type: MediaType = MediaType.OMNI
    duplex_mode: bool = False
    language: str = "zh"
    bili_room_code: Optional[str] = None
