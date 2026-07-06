"""Central orchestrator — coordinates Bili client, LiveKit, LLM, and pipeline.

Pattern adopted from room_start_monitor() in omini_backend_code.
"""

import asyncio
import logging
import signal
import time
from typing import Optional

from backend.common.config import Settings, get_settings
from backend.common.types import MediaType, SessionConfigRequest
from backend.pipeline.coordinator import DanmakuPipeline
from backend.services.live.danmaku_collector import DanmakuCollector
from backend.services.livekit.lk_room import LiveKitRoom
from backend.services.livekit.token import LiveKitTokenProvider
from backend.services.llm.client import LLMClient

from .trigger import InferenceTrigger

logger = logging.getLogger("backend.orchestrator")


class LiveStreamOrchestrator:
    """Owns all services and coordinates the inference lifecycle."""

    def __init__(self, config: Optional[Settings] = None):
        self._config = config or get_settings()
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._start_time: float = 0.0

        # Pipeline
        self.pipeline = DanmakuPipeline(self._config.pipeline)
        self.collector = DanmakuCollector(
            max_events=self._config.pipeline.max_events,
            ttl_seconds=self._config.pipeline.ttl_seconds,
            pipeline=self.pipeline,
        )
        self.trigger = InferenceTrigger(
            time_window_seconds=self._config.inference.time_window_seconds,
            event_driven_types=set(self._config.inference.event_driven_types),
        )

        # LiveKit
        token_provider = LiveKitTokenProvider(
            api_key=self._config.livekit.api_key,
            api_secret=self._config.livekit.api_secret,
        )
        self.livekit = LiveKitRoom(self._config.livekit, token_provider)

        # LLM
        self.llm: Optional[LLMClient] = None
        self._session_id: Optional[str] = None
        self._llm_connected: bool = False

        # Bili
        self._bili_connected: bool = False

    # ── Public API ──

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def active(self) -> bool:
        return not self._shutdown_event.is_set()

    @property
    def uptime_seconds(self) -> float:
        if self._start_time == 0:
            return 0
        return time.time() - self._start_time

    @property
    def llm_connected(self) -> bool:
        return self._llm_connected

    @property
    def livekit_connected(self) -> bool:
        return self.livekit.is_connected

    @property
    def bili_connected(self) -> bool:
        return self._bili_connected

    async def start(self, session_config: Optional[SessionConfigRequest] = None) -> None:
        """Start all services and begin the inference loop."""
        if session_config is None:
            session_config = SessionConfigRequest(
                media_type=MediaType.OMNI,
                duplex_mode=self._config.llm_server.default_duplex_mode,
                language=self._config.llm_server.language,
            )

        self._start_time = time.time()
        logger.info("Orchestrator starting")

        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop = asyncio.get_event_loop()
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass

        # Init LLM session
        try:
            async with LLMClient(
                base_url=self._config.llm_server.base_url,
                timeout=self._config.llm_server.request_timeout,
            ) as llm:
                init_resp = await llm.init_session(session_config)
                self._session_id = init_resp.get("session_id", "default")
                self._llm_connected = True
                logger.info(f"LLM session initialized: {self._session_id}")
        except Exception as e:
            logger.warning(f"LLM init failed — inference will be unavailable: {e}")
            self._llm_connected = False

        # Launch background tasks
        self._tasks.append(asyncio.create_task(self._inference_loop(), name="inference_loop"))

        if self._config.bilibili.enabled:
            self._tasks.append(asyncio.create_task(self._run_bili(), name="bili_client"))

        # LiveKit connection (optional — fails gracefully if no server)
        self._tasks.append(asyncio.create_task(self._run_livekit(), name="livekit"))

        logger.info("Orchestrator started — waiting for shutdown signal")
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Orchestrator shutting down")
        self._shutdown_event.set()

        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        await self.livekit.disconnect()

        try:
            async with LLMClient(
                base_url=self._config.llm_server.base_url,
                timeout=self._config.llm_server.request_timeout,
            ) as llm:
                await llm.stop()
        except Exception:
            pass

        self._llm_connected = False
        self._bili_connected = False
        logger.info("Orchestrator stopped")

    # ── Event handlers ──

    async def on_bili_event(self, data: dict) -> None:
        """Handle a raw Bilibili event. Routes through pipeline → collector → trigger."""
        event = self.pipeline.process_raw(data)
        if event is None:
            return

        # Also feed the raw data through the collector (backward compat)
        self.collector.add_event(data)

        # Check for immediate inference trigger
        if await self.trigger.report_event(event):
            await self._run_inference()

    # ── Background tasks ──

    async def _inference_loop(self) -> None:
        """Periodic timer-based inference trigger."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1.0)
            if self._llm_connected and await self.trigger.should_infer():
                await self._run_inference()

    async def _run_bili(self) -> None:
        """Run Bilibili client with reconnection."""
        from ..services.live.bili_client import BiliLiveClient

        while not self._shutdown_event.is_set():
            try:
                cfg = self._config.bilibili
                client = BiliLiveClient(
                    id_code=cfg.id_code,
                    app_id=cfg.app_id,
                    key=cfg.key,
                    secret=cfg.secret,
                )

                @client.on("*")
                async def on_raw_event(data):
                    await self.on_bili_event(data)

                self._bili_connected = True
                logger.info("Bilibili client connected")
                await client.run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bilibili client error: {e}")
                self._bili_connected = False
                await asyncio.sleep(self._config.bilibili.reconnect_delay)

    async def _run_livekit(self) -> None:
        """Connect to LiveKit room."""
        try:
            identity = f"{self._config.livekit.identity_prefix}-{id(self)}"
            await self.livekit.connect(identity=identity)
            logger.info("LiveKit connected")

            # Wait until shutdown
            while not self._shutdown_event.is_set():
                await asyncio.sleep(1.0)
            await self.livekit.disconnect()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"LiveKit connection failed: {e}")

    async def _run_inference(self) -> None:
        """Full inference cycle: context + AV → prefill → generate → publish TTS."""
        if not self._llm_connected or self._session_id is None:
            return

        logger.info("Starting inference cycle")
        try:
            context_text = self.pipeline.assemble_context()
            av = self.livekit.get_av_segment(audio_duration=5.0)

            async with LLMClient(
                base_url=self._config.llm_server.base_url,
                timeout=self._config.llm_server.request_timeout,
            ) as llm:
                # Prefill with context text + AV data
                await llm.prefill(
                    session_id=self._session_id,
                    audio_base64=av.audio_base64,
                    image_base64=av.image_base64,
                    prompt_text=context_text,
                )

                # Generate — stream audio back
                async for chunk in llm.generate(self._session_id):
                    if self._shutdown_event.is_set():
                        await llm.break_generation()
                        break

                    if chunk.get("type") == "done":
                        break

                    cd = chunk.get("chunk_data", {})
                    wav_np = cd.get("wav")
                    if wav_np is not None and self.livekit.is_connected:
                        import io

                        import soundfile as sf

                        buf = io.BytesIO()
                        sf.write(
                            buf,
                            wav_np,
                            cd.get("sample_rate", 24000),
                            format="WAV",
                            subtype="PCM_16",
                        )
                        await self.livekit.publish_audio(buf.getvalue())

                    text = cd.get("text", "")
                    if text:
                        logger.info(f"[LLM text] {text}")

            logger.info("Inference cycle complete")
        except Exception as e:
            logger.error(f"Inference cycle failed: {e}")
