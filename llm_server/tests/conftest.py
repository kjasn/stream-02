"""Pytest fixtures for model access layer tests."""

import os
import tempfile
from argparse import Namespace
from unittest import mock

import pytest


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory structure mimicking a GGUF model dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_args():
    """Return a Namespace simulating CLI args."""
    return Namespace(
        host="0.0.0.0",
        port=8060,
        llamacpp_root="/fake/llamacpp",
        model_dir="/fake/models",
        llm_model="test_q4.gguf",
        gpu_devices="",
        duplex=False,
        simplex=True,
        output_dir=None,
        vision_backend="metal",
    )


@pytest.fixture(autouse=True)
def clean_config_globals():
    """Reset config globals before each test to prevent state leakage."""
    import llm_server.inference_server.config as cfg

    saved = {}
    for name in dir(cfg):
        if name.startswith("_") or name.startswith("__"):
            continue
        saved[name] = getattr(cfg, name)

    yield

    for name, value in saved.items():
        try:
            setattr(cfg, name, value)
        except (AttributeError, TypeError):
            pass


@pytest.fixture
def sample_image():
    """Create a minimal synthetic PIL image."""
    from PIL import Image

    return Image.new("RGB", (64, 64), color=(255, 0, 0))


@pytest.fixture
def audio_wav_bytes():
    """Return a minimal valid WAV byte sequence."""
    import struct

    sample_rate = 24000
    num_samples = 240
    data_size = num_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    samples = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
    return header + samples
