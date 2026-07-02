"""LiveKit room connection, audio/video subscription, and TTS publishing.

Pattern adopted from LiveKitRoom in omini_backend_code/voice_chat/livekit_room.py.
"""

import io
import logging
from typing import Optional

import numpy as np
from livekit import rtc

logger = logging.getLogger("backend.livekit.client")


class AudioFrameBuffer:
    """Circular buffer of float32 PCM audio at 16kHz.

    Ingests LiveKit AudioFrames (48kHz int16), resamples to 16kHz float32,
    and produces WAV byte segments on demand.
    """

    def __init__(self, max_duration: float = 10.0, sample_rate: int = 16000):
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self._buffer: np.ndarray = np.array([], dtype=np.float32)

    def add_frame(self, frame: rtc.AudioFrame) -> None:
        """Ingest a LiveKit AudioFrame."""
        from scipy.signal import resample_poly

        arr = (
            np.frombuffer(frame.data.tobytes(), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        if arr.ndim > 1 and arr.shape[1] > 1:
            arr = arr.mean(axis=1)
        if frame.sample_rate != self.sample_rate:
            arr = resample_poly(arr, self.sample_rate, frame.sample_rate)
        self._buffer = np.concatenate([self._buffer, arr.astype(np.float32)])
        if len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[-self.max_samples :]

    def get_segment(self, duration: float) -> Optional[bytes]:
        """Return last N seconds as WAV bytes, or None if too little data."""
        import soundfile as sf

        needed = int(duration * self.sample_rate)
        if len(self._buffer) < min(needed, self.sample_rate // 2):
            return None
        segment = (
            self._buffer[-needed:] if len(self._buffer) >= needed else self._buffer
        )
        buf = io.BytesIO()
        sf.write(buf, segment, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def get_segment_base64(self, duration: float) -> Optional[str]:
        """Return WAV as base64 string."""
        wav_bytes = self.get_segment(duration)
        if wav_bytes is None:
            return None
        import base64

        return base64.b64encode(wav_bytes).decode("utf-8")

    @property
    def has_data(self) -> bool:
        return len(self._buffer) >= self.sample_rate // 4

    def reset(self) -> None:
        self._buffer = np.array([], dtype=np.float32)


class VideoFrameBuffer:
    """Keep the most recent video frame as JPEG/PNG bytes."""

    def __init__(self):
        self._latest: Optional[bytes] = None
        self._latest_base64: Optional[str] = None
        self._has_frame: bool = False

    def add_frame(self, frame: rtc.VideoFrame) -> None:
        """Convert a LiveKit VideoFrame to PNG bytes + base64."""
        import base64

        from PIL import Image

        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(
            (frame.height, frame.width, 3)
            if frame.num_planes == 1
            else frame.num_planes
        )
        # Handle RGB24 format from LiveKit VideoStream
        try:
            img = Image.fromarray(arr, mode="RGB")
        except (ValueError, TypeError):
            # Try reinterpreting if shape doesn't match
            if arr.size == frame.height * frame.width * 3:
                arr = arr.reshape((frame.height, frame.width, 3))
                img = Image.fromarray(arr, mode="RGB")
            else:
                logger.warning(
                    f"Cannot interpret video frame: shape={arr.shape}, size={arr.size}"
                )
                return

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        self._latest = buf.getvalue()
        self._latest_base64 = base64.b64encode(self._latest).decode("utf-8")
        self._has_frame = True

    def get_frame_base64(self) -> Optional[str]:
        return self._latest_base64

    @property
    def has_frame(self) -> bool:
        return self._has_frame
