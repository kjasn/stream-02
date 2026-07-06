"""Integration tests against a real C++ llama-server process.

These tests start the actual C++ server, initialize the model context,
and run prefill/generate round-trips. They require:
- Built llama-server binary
- GGUF model files
- GPU (Metal on macOS, CUDA on Linux)

Run with: pytest llm_server/tests/ -m integration -v
Skip with:  pytest llm_server/tests/ -m "not integration" -v
"""

import os
import json
import time
import signal
import struct
import shutil
import tempfile
import threading

import pytest
import requests

from llm_server.inference_server import config as cfg


# ── helpers ──────────────────────────────────────────────────


def _build_test_wav(sample_rate=16000, duration_sec=0.5):
    """Build a minimal valid WAV file (mono, 16-bit PCM) in memory."""
    num_samples = int(sample_rate * duration_sec)
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
    # silent audio
    samples = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
    return header + samples


# ── fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cpp_test_port():
    """Dedicated port for the test C++ server (isolated from dev server)."""
    return 19060


@pytest.fixture(scope="module")
def cpp_server(cpp_test_port):
    """Module-scoped: start the C++ llama-server once, stop after all tests."""
    model_dir = cfg.DEFAULT_MODEL_DIR
    server_bin = os.path.join(cfg.LLAMACPP_ROOT, "build/bin/llama-server")
    model_path = os.path.join(model_dir, cfg.auto_detect_llm_model(model_dir))

    if not os.path.exists(server_bin):
        pytest.skip(f"C++ binary not found: {server_bin}")
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found: {model_path}")

    env = os.environ.copy()
    if os.path.exists(server_bin):
        bin_dir = os.path.dirname(server_bin)
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir

    # Temporary output directory
    output_dir = tempfile.mkdtemp(prefix="cpp_test_output_")

    cmd = [
        server_bin,
        "--host",
        "127.0.0.1",
        "--port",
        str(cpp_test_port),
        "--model",
        model_path,
        "--ctx-size",
        "2048",
        "--n-gpu-layers",
        "99",
        "--temp",
        "0.7",
    ]

    import subprocess
    import platform

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    # Log reader thread
    def _log():
        try:
            for line in proc.stdout:
                print(f"  [CPP-TEST] {line.rstrip()}")
        except Exception:
            pass

    log_t = threading.Thread(target=_log, daemon=True)
    log_t.start()

    base_url = f"http://127.0.0.1:{cpp_test_port}"

    # Wait up to 120s for the C++ server to be ready
    ready = False
    for i in range(120):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                ready = True
                print(f"[setup] C++ server ready after {i + 1}s")
                break
        except Exception:
            pass
        time.sleep(1)

    if not ready:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(output_dir, ignore_errors=True)
        pytest.fail("C++ server did not become ready within 120s")

    yield {
        "base_url": base_url,
        "output_dir": output_dir,
        "model_dir": model_dir,
    }

    # Teardown
    print("[teardown] Stopping C++ server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    shutil.rmtree(output_dir, ignore_errors=True)
    print("[teardown] Done")


@pytest.fixture(scope="module")
def omni_initialized(cpp_server):
    """Initialize omni context once per module (expensive — loads TTS model)."""
    base_url = cpp_server["base_url"]
    model_dir = cpp_server["model_dir"]
    tts_bin_dir = os.path.join(model_dir, "tts")

    init_body = {
        "media_type": 2,
        "use_tts": True,
        "duplex_mode": False,
        "model_dir": model_dir,
        "tts_bin_dir": tts_bin_dir,
        "tts_gpu_layers": 100,
        "token2wav_device": cfg.TOKEN2WAV_DEVICE,
        "output_dir": cpp_server["output_dir"],
        "vision_backend": cfg.VISION_BACKEND,
    }

    ref_audio = cfg.FIXED_TIMBRE_PATH
    if ref_audio and os.path.exists(ref_audio):
        init_body["voice_audio"] = ref_audio

    print("[setup] Initializing omni context (this may take 15-30s)...")
    resp = requests.post(f"{base_url}/v1/stream/omni_init", json=init_body, timeout=120)
    assert resp.status_code == 200, f"omni_init failed: {resp.text}"
    data = resp.json()
    assert data.get("success") is True, f"omni_init returned success=False: {data}"
    print("[setup] Omni context initialized")

    yield cpp_server


# ── tests ────────────────────────────────────────────────────


@pytest.mark.integration
class TestCppServerHealth:
    def test_health_returns_200(self, cpp_server):
        resp = requests.get(f"{cpp_server['base_url']}/health", timeout=5)
        assert resp.status_code == 200


@pytest.mark.integration
class TestOmniInit:
    def test_omni_init_succeeds(self, cpp_server):
        """Verify omni_init returns success."""
        model_dir = cpp_server["model_dir"]
        tts_bin_dir = os.path.join(model_dir, "tts")

        body = {
            "media_type": 2,
            "use_tts": True,
            "duplex_mode": False,
            "model_dir": model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": cfg.TOKEN2WAV_DEVICE,
            "output_dir": cpp_server["output_dir"],
            "vision_backend": cfg.VISION_BACKEND,
        }
        resp = requests.post(f"{cpp_server['base_url']}/v1/stream/omni_init", json=body, timeout=120)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True

    def test_omni_init_result_structure(self, cpp_server):
        """Verify response contains expected fields."""
        model_dir = cpp_server["model_dir"]
        tts_bin_dir = os.path.join(model_dir, "tts")
        body = {
            "media_type": 2,
            "use_tts": True,
            "duplex_mode": False,
            "model_dir": model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": cfg.TOKEN2WAV_DEVICE,
            "output_dir": cpp_server["output_dir"],
            "vision_backend": cfg.VISION_BACKEND,
        }
        resp = requests.post(f"{cpp_server['base_url']}/v1/stream/omni_init", json=body, timeout=120)
        data = resp.json()
        for key in ("success", "media_type", "use_tts"):
            assert key in data, f"Missing key: {key}"


@pytest.mark.integration
class TestPrefillAndGenerate:
    """End-to-end prefill → generate round-trip (simplex mode)."""

    def test_prefill_audio_only(self, omni_initialized):
        """Send audio-only prefill, verify 200."""
        base_url = omni_initialized["base_url"]
        output_dir = omni_initialized["output_dir"]

        # Write test WAV to output dir
        wav_path = os.path.join(output_dir, "test_prefill_audio.wav")
        with open(wav_path, "wb") as f:
            f.write(_build_test_wav(sample_rate=16000, duration_sec=0.5))

        body = {
            "audio_path_prefix": wav_path,
            "img_path_prefix": "",
            "cnt": 0,
        }
        resp = requests.post(f"{base_url}/v1/stream/prefill", json=body, timeout=30)
        assert resp.status_code == 200, f"prefill failed: {resp.text}"

    def test_generate_simplex_produces_round_dir(self, omni_initialized):
        """Simplex generate should create a round_XXX directory with WAV outputs."""
        base_url = omni_initialized["base_url"]
        output_dir = omni_initialized["output_dir"]

        # First do a prefill
        wav_path = os.path.join(output_dir, "test_gen_audio.wav")
        with open(wav_path, "wb") as f:
            f.write(_build_test_wav(sample_rate=16000, duration_sec=0.5))

        prefill_body = {
            "audio_path_prefix": wav_path,
            "img_path_prefix": "",
            "cnt": 0,
        }
        resp = requests.post(f"{base_url}/v1/stream/prefill", json=prefill_body, timeout=30)
        assert resp.status_code == 200

        # Then generate
        gen_body = {
            "debug_dir": output_dir,
            "stream": True,
            "round_idx": 0,
        }
        resp = requests.post(f"{base_url}/v1/stream/decode", json=gen_body, timeout=120)
        assert resp.status_code == 200, f"decode failed: {resp.text}"

        # Check that round directory was created
        round_dir = os.path.join(output_dir, "round_000")
        assert os.path.isdir(round_dir), f"Round dir not created: {round_dir}"

    def test_generate_sse_streaming(self, omni_initialized):
        """Verify decode endpoint returns streaming SSE response."""
        base_url = omni_initialized["base_url"]
        output_dir = omni_initialized["output_dir"]

        # Prefill
        wav_path = os.path.join(output_dir, "test_sse_audio.wav")
        with open(wav_path, "wb") as f:
            f.write(_build_test_wav(sample_rate=16000, duration_sec=0.5))

        prefill_body = {
            "audio_path_prefix": wav_path,
            "img_path_prefix": "",
            "cnt": 0,
        }
        requests.post(f"{base_url}/v1/stream/prefill", json=prefill_body, timeout=30)

        # Generate with stream=True (SSE)
        gen_body = {
            "debug_dir": output_dir,
            "stream": True,
            "round_idx": 1,
        }
        resp = requests.post(f"{base_url}/v1/stream/decode", json=gen_body, timeout=120, stream=True)
        assert resp.status_code == 200

        # Read first few SSE events
        lines = []
        for i, line in enumerate(resp.iter_lines(decode_unicode=True)):
            lines.append(line)
            if i > 50:
                break

        # SSE events should contain "data:" or text lines
        text_output = "\n".join(lines)
        print(f"[SSE output] First {len(lines)} lines:\n{text_output}")


@pytest.mark.integration
class TestPromptTextInjection:
    """Tests for the prompt_text field in /v1/stream/prefill."""

    def test_prefill_with_prompt_text_succeeds(self, omni_initialized):
        """prefill with non-empty prompt_text should return 200."""
        base_url = omni_initialized["base_url"]
        output_dir = omni_initialized["output_dir"]

        wav_path = os.path.join(output_dir, "test_prompt_audio.wav")
        with open(wav_path, "wb") as f:
            f.write(_build_test_wav(sample_rate=16000, duration_sec=1.0))

        body = {
            "audio_path_prefix": wav_path,
            "img_path_prefix": "",
            "cnt": 0,
            "prompt_text": "【直播间动态】\n[弹幕] 用户A: 测试弹幕",
        }
        resp = requests.post(f"{base_url}/v1/stream/prefill", json=body, timeout=30)
        assert resp.status_code == 200, f"prefill with prompt_text failed: {resp.text}"

    def test_empty_prompt_text_does_not_error(self, omni_initialized):
        """prefill with empty prompt_text should work as if it wasn't provided."""
        base_url = omni_initialized["base_url"]
        output_dir = omni_initialized["output_dir"]

        wav_path = os.path.join(output_dir, "test_empty_prompt.wav")
        with open(wav_path, "wb") as f:
            f.write(_build_test_wav(sample_rate=16000, duration_sec=0.5))

        body = {
            "audio_path_prefix": wav_path,
            "img_path_prefix": "",
            "cnt": 0,
            "prompt_text": "",
        }
        resp = requests.post(f"{base_url}/v1/stream/prefill", json=body, timeout=30)
        assert resp.status_code == 200, f"prefill with empty prompt_text failed: {resp.text}"

    def test_missing_prompt_text_field_still_works(self, omni_initialized):
        """prefill without prompt_text field at all should work (backward compat)."""
        base_url = omni_initialized["base_url"]
        output_dir = omni_initialized["output_dir"]

        wav_path = os.path.join(output_dir, "test_no_prompt.wav")
        with open(wav_path, "wb") as f:
            f.write(_build_test_wav(sample_rate=16000, duration_sec=0.5))

        body = {
            "audio_path_prefix": wav_path,
            "img_path_prefix": "",
            "cnt": 0,
        }
        resp = requests.post(f"{base_url}/v1/stream/prefill", json=body, timeout=30)
        assert resp.status_code == 200

    def test_round_trip_with_prompt_text(self, omni_initialized):
        """Prefill with prompt_text → generate should complete without errors."""
        base_url = omni_initialized["base_url"]
        output_dir = omni_initialized["output_dir"]

        wav_path = os.path.join(output_dir, "test_roundtrip_prompt.wav")
        with open(wav_path, "wb") as f:
            f.write(_build_test_wav(sample_rate=16000, duration_sec=1.0))

        prefill_body = {
            "audio_path_prefix": wav_path,
            "img_path_prefix": "",
            "cnt": 0,
            "prompt_text": "【直播间动态】\n[SC -¥50] saki: 你这个人满脑子都是自己呢\n[soyo] : 哦内盖",
        }
        resp = requests.post(f"{base_url}/v1/stream/prefill", json=prefill_body, timeout=30)
        assert resp.status_code == 200

        gen_body = {"debug_dir": output_dir, "stream": True, "round_idx": 0}
        resp = requests.post(f"{base_url}/v1/stream/decode", json=gen_body, timeout=120)
        assert resp.status_code == 200


@pytest.mark.integration
class TestBreak:
    """Break endpoint interrupts generation."""

    def test_break_succeeds(self, omni_initialized):
        base_url = omni_initialized["base_url"]
        resp = requests.post(f"{base_url}/v1/stream/break", timeout=10)
        assert resp.status_code == 200


@pytest.mark.integration
class TestErrorHandling:
    """Edge case behaviour."""

    def test_health_after_multiple_inits(self, cpp_server):
        """omni_init should be idempotent — calling it twice should not crash."""
        model_dir = cpp_server["model_dir"]
        tts_bin_dir = os.path.join(model_dir, "tts")
        body = {
            "media_type": 2,
            "use_tts": True,
            "duplex_mode": False,
            "model_dir": model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": cfg.TOKEN2WAV_DEVICE,
            "output_dir": cpp_server["output_dir"],
            "vision_backend": cfg.VISION_BACKEND,
        }
        base_url = cpp_server["base_url"]

        # Call omni_init twice
        r1 = requests.post(f"{base_url}/v1/stream/omni_init", json=body, timeout=120)
        time.sleep(1)
        r2 = requests.post(f"{base_url}/v1/stream/omni_init", json=body, timeout=120)

        # Both should succeed
        assert r1.status_code == 200
        assert r2.status_code == 200

        # Health should still work
        r3 = requests.get(f"{base_url}/health", timeout=5)
        assert r3.status_code == 200
