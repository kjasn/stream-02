"""LiveKit AccessToken self-issuance using PyJWT.

Pattern adopted from the login endpoint in omini_backend_code routes.py.
"""

import logging
import time

import jwt

logger = logging.getLogger("backend.livekit.token")

TOKEN_TTL_SECONDS = 3600  # 1 hour


class LiveKitTokenProvider:
    """Issues LiveKit access tokens for bots and participants."""

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret

    def create_token(self, identity: str, room: str, can_publish: bool = True, can_subscribe: bool = True) -> str:
        """Generate a JWT access token for a LiveKit participant.

        Creates a token compatible with LiveKit's access token format:
        https://docs.livekit.io/home/get-started/authentication/
        """
        now = int(time.time())
        payload = {
            "iss": self._api_key,
            "sub": identity,
            "nbf": now,
            "exp": now + TOKEN_TTL_SECONDS,
            "video": {
                "roomJoin": True,
                "room": room,
                "canPublish": can_publish,
                "canSubscribe": can_subscribe,
            },
        }
        return jwt.encode(payload, self._api_secret, algorithm="HS256")
