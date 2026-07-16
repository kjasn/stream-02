import enum

from pydantic import Field

from backend.db.models.base import MongoBaseModel


class UserStatus(enum.Enum):
    ACTIVE = 1
    INACTIVE = 2


class Tenant(MongoBaseModel):
    uid: int = Field(default=0)
    name: str = Field(..., min_length=1, max_length=20)
    local_call: int = Field(default=0)  # 本地模型调用次数
    status: UserStatus = Field(default=UserStatus.INACTIVE)
