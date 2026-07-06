"""Test configuration loading."""

from backend.common.config import Settings, get_settings


def test_default_settings():
    s = get_settings()
    assert s.app_name == "stream-02 Backend"
    assert s.server.port == 8070
    assert s.livekit.url == "ws://localhost:7880"
    assert s.llm_server.base_url == "http://127.0.0.1:8060"


def test_pipeline_config():
    s = get_settings()
    assert s.pipeline.max_events == 50
    assert s.pipeline.token_budget == 2048
    assert "广告" in s.pipeline.spam_keywords


def test_inference_config():
    s = get_settings()
    assert s.inference.time_window_seconds == 15.0
    assert "LIVE_OPEN_PLATFORM_SUPER_CHAT" in s.inference.event_driven_types
