from typing import Optional

from pydantic import BaseModel

from backend.common.types import MediaType


class SessionConfigRequest(BaseModel):
    media_type: MediaType = MediaType.OMNI
    duplex_mode: bool = False
    language: str = "zh"
    bili_room_code: Optional[str] = None


class SessionStatusResponse(BaseModel):
    session_id: str = ""
    active: bool = False
    llm_connected: bool = False
    livekit_connected: bool = False
    bili_connected: bool = False
    uptime_seconds: float = 0.0
    event_count: int = 0
    last_inference_time: Optional[float] = None
