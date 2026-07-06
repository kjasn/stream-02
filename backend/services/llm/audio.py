"""Audio preprocessing: numpy int16 → 16kHz WAV → base64."""

import base64
import io
import logging

import numpy as np

logger = logging.getLogger("backend.llm.audio")


def encode_audio_to_base64(
    audio_data: np.ndarray,
    input_sample_rate: int = 48000,
    output_sample_rate: int = 16000,
) -> str:
    """Convert numpy audio array to base64-encoded WAV (16kHz mono PCM16).

    Mirrors MiniCpmModel._encode_audio_to_base64() in model_call.py.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    if input_sample_rate != output_sample_rate:
        audio_data = resample_poly(audio_data, output_sample_rate, input_sample_rate)

    audio_data = np.clip(audio_data, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, audio_data, output_sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def wav_bytes_to_float32(wav_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode WAV bytes into float32 numpy array, resampled to target_sr."""
    import soundfile as sf
    from scipy.signal import resample_poly

    buf = io.BytesIO(wav_bytes)
    data, sr = sf.read(buf, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample_poly(data, target_sr, sr)
    return data.astype(np.float32)
