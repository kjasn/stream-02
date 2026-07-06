import asyncio
import io
import logging
import time
from typing import Optional

import numpy as np
from common.types import AvSegment
from livekit import rtc

from .avbuf import AudioFrameBuffer, VideoFrameBuffer
from .token import LiveKitTokenProvider

logger = logging.getLogger("backend.livekit.client")


# LiveKit audio: 48kHz mono int16
WEBRTC_SAMPLE_RATE = 48000
NUM_CHANNELS = 1


class LiveKitRoom:
    """Manages LiveKit room connection, track subscriptions, and TTS output."""

    def __init__(self, config, token_provider: LiveKitTokenProvider):
        self._config = config
        self._token_provider = token_provider
        self._room: Optional[rtc.Room] = None
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_buffer = AudioFrameBuffer()
        self._video_buffer = VideoFrameBuffer()
        self._connected = False
        self._tasks: set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()

    async def connect(self, identity: str = "ai-backend") -> None:
        """Connect to LiveKit room and subscribe to tracks."""
        token = self._token_provider.create_token(
            identity=identity,
            room=self._config.room,
            can_publish=True,
            can_subscribe=True,
        )

        loop = asyncio.get_event_loop()
        self._room = rtc.Room(loop=loop)

        @self._room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, *_):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Subscribed to audio track: {track.name}")
                audio_stream = rtc.AudioStream(track)
                task = asyncio.create_task(self._consume_audio(audio_stream))
                self._tasks.add(task)
                task.add_done_callback(lambda t: self._tasks.discard(t))
            elif track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"Subscribed to video track: {track.name}")
                video_stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
                task = asyncio.create_task(self._consume_video(video_stream))
                self._tasks.add(task)
                task.add_done_callback(lambda t: self._tasks.discard(t))

        @self._room.on("track_unsubscribed")
        def on_track_unsubscribed(track: rtc.Track, *_):
            logger.info(f"Track unsubscribed: {track.name}")

        @self._room.on("disconnected")
        def on_disconnected(reason: str):
            logger.info(f"LiveKit disconnected: {reason}")
            self._connected = False

        await self._room.connect(self._config.url, token)
        self._connected = True

        # Create audio source for TTS output
        self._audio_source = rtc.AudioSource(WEBRTC_SAMPLE_RATE, NUM_CHANNELS, 960)
        track = rtc.LocalAudioTrack.create_audio_track("tts-output", self._audio_source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await self._room.local_participant.publish_track(track, options)
        logger.info(f"Published TTS output track: {publication.sid}")

    async def disconnect(self) -> None:
        """Disconnect from the LiveKit room."""
        self._stop_event.set()
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        self._tasks.clear()
        if self._room:
            await self._room.disconnect()
            self._room = None
        self._connected = False
        self._audio_source = None

    async def publish_audio(self, wav_bytes: bytes) -> None:
        """Publish TTS audio (WAV bytes) to the LiveKit room.

        Resamples to 48kHz int16 and pushes frames via AudioSource.
        """
        import soundfile as sf
        from scipy.signal import resample_poly

        if self._audio_source is None:
            logger.warning("No audio source — cannot publish")
            return

        buf = io.BytesIO(wav_bytes)
        data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to WebRTC 48kHz
        if sr != WEBRTC_SAMPLE_RATE:
            data = resample_poly(data, WEBRTC_SAMPLE_RATE, sr)

        # Convert to int16
        data_int16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)

        # Push in 20ms frames (960 samples at 48kHz)
        frame_size = 960
        for i in range(0, len(data_int16), frame_size):
            chunk = data_int16[i : i + frame_size]
            if len(chunk) < frame_size:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=WEBRTC_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=frame_size,
            )
            await self._audio_source.capture_frame(frame)
            await asyncio.sleep(0)  # Yield to event loop

    def get_av_segment(self, audio_duration: float = 5.0) -> AvSegment:
        """Get current audio/video segment for LLM prefill."""
        return AvSegment(
            audio_base64=self._audio_buffer.get_segment_base64(audio_duration),
            audio_sample_rate=self._audio_buffer.sample_rate,
            audio_duration=audio_duration,
            image_base64=self._video_buffer.get_frame_base64(),
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_audio(self) -> bool:
        return self._audio_buffer.has_data

    async def _consume_audio(self, audio_stream: rtc.AudioStream) -> None:
        """Background task: consume audio frames into the buffer."""
        frame_count = 0
        last_ts = time.time()
        async for frame_event in audio_stream:
            if self._stop_event.is_set():
                break
            if time.time() - last_ts > 2.0:
                last_ts = time.time()
                continue
            self._audio_buffer.add_frame(frame_event.frame)
            frame_count += 1
            last_ts = time.time()
        logger.info(f"Audio stream ended after {frame_count} frames")

    async def _consume_video(self, video_stream: rtc.VideoStream) -> None:
        """Background task: consume video frames into the buffer."""
        frame_count = 0
        last_process = 0.0
        async for frame_event in video_stream:
            if self._stop_event.is_set():
                break
            # Throttle to ~1 FPS
            if time.time() - last_process < 0.85:
                continue
            self._video_buffer.add_frame(frame_event.frame)
            frame_count += 1
            last_process = time.time()
        logger.info(f"Video stream ended after {frame_count} frames")
