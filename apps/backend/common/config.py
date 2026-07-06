"""Configuration management — Pydantic BaseSettings + YAML with ENV override.

Pattern adopted from omini_backend_code/config/settings.py.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveKitSettings(BaseSettings):
    url: str = Field(default="ws://localhost:7880", description="LiveKit server URL")
    api_key: str = Field(default="devkey", description="LiveKit API Key")
    api_secret: str = Field(default="", description="LiveKit API Secret")
    room: str = Field(default="stream-room", description="Default room name")
    identity_prefix: str = Field(default="ai-backend", description="Bot identity prefix")

    model_config = SettingsConfigDict(env_prefix="LIVEKIT_", case_sensitive=False)


class BilibiliSettings(BaseSettings):
    enabled: bool = Field(default=True)
    id_code: str = Field(default="")
    app_id: int = Field(default=0)
    key: str = Field(default="")
    secret: str = Field(default="")
    reconnect_delay: float = Field(default=5.0)
    max_reconnect_attempts: int = Field(default=0)

    model_config = SettingsConfigDict(env_prefix="BILIBILI_", case_sensitive=False)


class LLMServerSettings(BaseSettings):
    base_url: str = Field(default="http://127.0.0.1:8060")
    default_duplex_mode: bool = Field(default=False)
    language: str = Field(default="zh")
    request_timeout: float = Field(default=300.0)

    model_config = SettingsConfigDict(env_prefix="LLM_SERVER_", case_sensitive=False)


class PipelineSettings(BaseSettings):
    ttl_seconds: float = Field(default=60.0)
    max_events: int = Field(default=50)
    dedup_window_seconds: float = Field(default=30.0)
    rate_limit_per_user: int = Field(default=5)
    rate_limit_window_seconds: float = Field(default=60.0)
    spam_keywords: list[str] = Field(default_factory=lambda: ["广告", "加群", "私信"])
    blacklist_uids: list[str] = Field(default_factory=list)
    token_budget: int = Field(default=2048)
    similarity_threshold: float = Field(default=0.85)
    merge_window_seconds: float = Field(default=10.0)

    model_config = SettingsConfigDict(env_prefix="PIPELINE_", case_sensitive=False)


class InferenceSettings(BaseSettings):
    time_window_seconds: float = Field(default=15.0)
    event_driven_types: list[str] = Field(
        default_factory=lambda: [
            "LIVE_OPEN_PLATFORM_SUPER_CHAT",
            "LIVE_OPEN_PLATFORM_GUARD",
        ]
    )

    model_config = SettingsConfigDict(env_prefix="INFERENCE_", case_sensitive=False)


class ServerSettings(BaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8070)
    log_level: str = Field(default="info")

    model_config = SettingsConfigDict(env_prefix="SERVER_", case_sensitive=False)


class Settings(BaseSettings):
    app_name: str = Field(default="stream-02 Backend")
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="local")

    server: ServerSettings = Field(default_factory=ServerSettings)
    livekit: LiveKitSettings = Field(default_factory=LiveKitSettings)
    bilibili: BilibiliSettings = Field(default_factory=BilibiliSettings)
    llm_server: LLMServerSettings = Field(default_factory=LLMServerSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load from YAML file, with env-var overrides.

        Priority: env vars > {app_env}.yaml > base.yaml > defaults
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "default.yaml"

        app_env = os.getenv("APP_ENV", "local").lower()
        config_dir = config_path.parent

        config_data: dict = {}

        # Load base config
        base_path = config_dir / "base.yaml"
        if base_path.exists():
            with open(base_path, encoding="utf-8") as f:
                config_data = _deep_merge(config_data, yaml.safe_load(f) or {})

        # Load default config
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config_data = _deep_merge(config_data, yaml.safe_load(f) or {})

        # Load env-specific config
        env_path = config_dir / f"{app_env}.yaml"
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                config_data = _deep_merge(config_data, yaml.safe_load(f) or {})

        config_data["app_env"] = app_env
        return cls(**config_data)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache()
def get_settings() -> Settings:
    config_path = os.getenv("BACKEND_CONFIG")
    if config_path:
        return Settings.from_yaml(Path(config_path))
    return Settings.from_yaml()
