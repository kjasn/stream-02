"""Test LiveKit token generation."""

from backend.services.livekit.token import LiveKitTokenProvider


def test_token_creation():
    provider = LiveKitTokenProvider(api_key="test_key", api_secret="test_secret")
    token = provider.create_token(identity="test-bot", room="test-room")
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    # JWT has 3 dot-separated parts
    assert token.count(".") == 2


def test_token_no_publish():
    provider = LiveKitTokenProvider(api_key="k", api_secret="s")
    token = provider.create_token(identity="listener", room="room", can_publish=False, can_subscribe=True)
    assert isinstance(token, str)
