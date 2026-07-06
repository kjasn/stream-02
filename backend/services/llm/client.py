"""HTTP client for llm_server API — backed by aiohttp.

Pattern adopted from MiniCpmModel in model_call.py.
"""

import base64
import json
import logging
from collections.abc import AsyncGenerator

import aiohttp
import numpy as np

from backend.common.types import SessionConfigRequest

logger = logging.getLogger("backend.llm.client")


class LLMClient:
    """Async HTTP client for llm_server inference endpoints."""

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "LLMClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("LLMClient not opened — use 'async with'")
        return self._session

    async def init_session(self, config: SessionConfigRequest) -> dict:
        """POST /omni/init_sys_prompt — initialise or reconfigure the model session."""
        body = {
            "media_type": config.media_type.value,
            "duplex_mode": config.duplex_mode,
            "language": config.language,
        }
        async with self.session.post(
            f"{self.base_url}/omni/init_sys_prompt",
            json=body,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def prefill(
        self,
        session_id: str,
        audio_base64: str | None = None,
        image_base64: str | None = None,
        prompt_text: str | None = None,
        **kwargs,
    ) -> dict:
        """POST /omni/streaming_prefill — send prefill data (audio, image, text)."""
        body: dict = {"session_id": session_id}
        if audio_base64 is not None:
            body["audio"] = audio_base64
        if image_base64 is not None:
            body["image"] = image_base64
        if prompt_text is not None:
            body["prompt_text"] = prompt_text
        body.update(kwargs)

        async with self.session.post(
            f"{self.base_url}/omni/streaming_prefill",
            json=body,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def generate(self, session_id: str) -> AsyncGenerator[dict, None]:
        """POST /omni/streaming_generate — SSE stream yielding {chunk_idx, chunk_data: {wav, sample_rate, text?}}.

        SSE parsing mirrors MiniCpmModel._parse_stream_chunk().
        """
        async with self.session.post(
            f"{self.base_url}/omni/streaming_generate",
            json={"session_id": session_id},
            timeout=self._timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in _iter_sse_lines(resp):
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:]
                else:
                    payload = line

                if "[DONE]" in payload:
                    yield {"type": "done"}
                    return

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    yield {"type": "done"}
                    return

                # Decode base64 wav → numpy int16
                if "chunk_data" in data and "wav" in data["chunk_data"]:
                    wav_b64 = data["chunk_data"]["wav"]
                    if isinstance(wav_b64, str):
                        wav_bytes = base64.b64decode(wav_b64)
                        data["chunk_data"]["wav"] = np.frombuffer(wav_bytes, dtype=np.int16)

                yield data

    async def break_generation(self) -> dict:
        """POST /omni/break — interrupt current generation round."""
        async with self.session.post(
            f"{self.base_url}/omni/break",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def stop(self) -> dict:
        """POST /omni/stop — stop session (preserve KV cache)."""
        async with self.session.post(
            f"{self.base_url}/omni/stop",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


async def _iter_sse_lines(resp: aiohttp.ClientResponse):
    """Yield lines from an SSE response, handling chunked transfer encoding."""
    buffer = ""
    async for chunk, _ in resp.content.iter_chunks():
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.strip()
    if buffer.strip():
        yield buffer.strip()
