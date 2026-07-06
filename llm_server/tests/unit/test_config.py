"""Tests for config.py — model auto-detection, init_from_args."""

import os
import tempfile
from argparse import Namespace
from unittest import mock

import pytest

from llm_server.inference_server import config as cfg


class TestAutoDetectLlmModel:
    """Tests for auto_detect_llm_model()."""

    def test_empty_or_none_dir_returns_empty(self):
        assert cfg.auto_detect_llm_model("") == ""
        assert cfg.auto_detect_llm_model(None) == ""

    def test_nonexistent_dir_returns_empty(self):
        assert cfg.auto_detect_llm_model("/nonexistent/path") == ""

    def test_priority_q4_k_m_wins(self, temp_model_dir, monkeypatch):
        candidates = [
            "model-F16.gguf",
            "model-Q8_0.gguf",
            "model-Q4_K_M.gguf",
            "model-Q4_K_S.gguf",
        ]
        for f in candidates:
            Path = __import__("pathlib").Path
            (Path(temp_model_dir) / f).touch()

        monkeypatch.setattr(os.path, "isdir", lambda path: path == temp_model_dir)

        result = cfg.auto_detect_llm_model(temp_model_dir)
        assert result == "model-Q4_K_M.gguf"

    def test_priority_q8_0_when_no_q4(self, temp_model_dir, monkeypatch):
        candidates = ["model-F16.gguf", "model-Q8_0.gguf"]
        for f in candidates:
            (__import__("pathlib").Path(temp_model_dir) / f).touch()

        monkeypatch.setattr(os.path, "isdir", lambda path: path == temp_model_dir)

        result = cfg.auto_detect_llm_model(temp_model_dir)
        assert result == "model-Q8_0.gguf"

    def test_priority_f16_when_no_q_series(self, temp_model_dir, monkeypatch):
        candidates = ["model-F16.gguf", "model-random.gguf"]
        for f in candidates:
            (__import__("pathlib").Path(temp_model_dir) / f).touch()

        monkeypatch.setattr(os.path, "isdir", lambda path: path == temp_model_dir)

        result = cfg.auto_detect_llm_model(temp_model_dir)
        assert result == "model-F16.gguf"

    def test_subdir_gguf_files_ignored(self, temp_model_dir, monkeypatch):
        """GGUF files in subdirectories should be excluded from priority match."""
        (__import__("pathlib").Path(temp_model_dir) / "model-Q4_K_M.gguf").touch()
        subdir = __import__("pathlib").Path(temp_model_dir) / "subdir"
        subdir.mkdir()
        (subdir / "better-model-Q4_K_M.gguf").touch()

        monkeypatch.setattr(os.path, "isdir", lambda path: path == temp_model_dir)

        result = cfg.auto_detect_llm_model(temp_model_dir)
        assert result == "model-Q4_K_M.gguf"

    def test_filters_audio_vision_tts_files(self, temp_model_dir, monkeypatch):
        """GGUF files containing 'audio', 'vision', 'tts', 'projector' are excluded from fallback."""
        candidates = [
            "llm-audio.gguf",
            "llm-vision.gguf",
            "llm-tts.gguf",
            "projector.gguf",
            "model-F16.gguf",
        ]
        for f in candidates:
            (__import__("pathlib").Path(temp_model_dir) / f).touch()

        monkeypatch.setattr(os.path, "isdir", lambda path: path == temp_model_dir)

        result = cfg.auto_detect_llm_model(temp_model_dir)
        assert result == "model-F16.gguf"

    def test_all_filtered_returns_empty(self, temp_model_dir, monkeypatch):
        candidates = ["llm-audio.gguf", "llm-vision.gguf"]
        for f in candidates:
            (__import__("pathlib").Path(temp_model_dir) / f).touch()

        monkeypatch.setattr(os.path, "isdir", lambda path: path == temp_model_dir)

        result = cfg.auto_detect_llm_model(temp_model_dir)
        assert result == ""

    def test_no_gguf_files_returns_empty(self, temp_model_dir, monkeypatch):
        monkeypatch.setattr(os.path, "isdir", lambda path: path == temp_model_dir)
        result = cfg.auto_detect_llm_model(temp_model_dir)
        assert result == ""


class TestInitFromArgs:
    """Tests for init_from_args()."""

    def test_sets_port_and_url(self):
        args = Namespace(port=8060)
        cfg.init_from_args(args)
        assert cfg.CPP_SERVER_PORT == 18060
        assert cfg.CPP_SERVER_URL == f"http://{cfg.CPP_SERVER_HOST}:18060"

    def test_sets_output_dir(self):
        args = Namespace(port=8060, output_dir="/tmp/test_output")
        cfg.init_from_args(args)
        assert cfg.CPP_OUTPUT_DIR == "/tmp/test_output"

    def test_sets_model_dir(self):
        args = Namespace(port=8060, model_dir="/tmp/models")
        cfg.init_from_args(args)
        assert cfg.DEFAULT_MODEL_DIR == "/tmp/models"

    def test_sets_duplex_mode(self):
        args = Namespace(port=8060, duplex=True)
        cfg.init_from_args(args)
        assert cfg.current_duplex_mode is True

    def test_missing_attributes_dont_error(self):
        """Attributes that don't exist on args are ignored."""
        args = Namespace(port=8060)
        cfg.init_from_args(args)
        # current_duplex_mode should stay unchanged (whatever it was)
        # Just verify no exception raised


class TestConfigDefaults:
    """Tests for config module-level defaults."""

    def test_llamacpp_root_default(self):
        import llm_server.inference_server.config as fresh_cfg

        assert "llama.cpp-omni" in str(fresh_cfg.LLAMACPP_ROOT)

    def test_vision_backend_default(self):
        assert cfg.VISION_BACKEND in ("metal", "coreml")

    def test_token2wav_device_default(self):
        assert cfg.TOKEN2WAV_DEVICE in ("gpu:0", "cpu")

    def test_fixed_timbre_path_exists_or_default(self):
        path = cfg.FIXED_TIMBRE_PATH
        if path and os.path.exists(path):
            assert path.endswith(".wav")
        else:
            assert "default_ref_audio.wav" in path

    def test_get_default_register_url_fallback(self):
        """Fallback to localhost when network discovery fails."""
        result = cfg._get_default_register_url()
        assert result.startswith("http://")
        assert ":8025" in result
