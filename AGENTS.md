# AGENTS.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two independent subsystems under `apps/`:
- **`llm_server/`** — FastAPI server wrapping a C++ `llama.cpp-omni` backend for MiniCPM-o multimodal inference (audio/video/omni modes, streaming SSE, TTS with voice cloning)
- **`backend/`** — Bilibili Live open-platform WebSocket client receiving live-stream events (danmaku, gifts, superchats)

The two subsystems are **not yet connected**. The Bilibili client only prints events to stdout.

## Build

### llama.cpp-omni (C++ backend)
```bash
cd llm_server/llama.cpp-omni
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON
cmake --build build -j $(sysctl -n hw.ncpu)
```
The inference server expects the `llama-server` binary under this directory.

### Python dependencies
Uses `uv` for dependency management:
```bash
cd apps && uv sync
```

Note: `pyproject.toml` only lists production dependencies. The inference server also needs `fastapi`, `uvicorn`, `pydantic`, `librosa`, `soundfile`, `Pillow`, `numpy`, `requests`, `httpx`. These are installed in the shared `.venv/`.

## Run

### LLM inference server
```bash
python llm_server/inference_server/server.py \
  --llamacpp-root llm_server/llama.cpp-omni \
  --model-dir llm_server/models/openbmb/MiniCPM-o-4_5-gguf \
  --port 8060 \
  [--duplex|--simplex]
```
Port layout: Python FastAPI on port N, health/break server on N+1, C++ backend on N+10000.

### Bilibili client (standalone)
```bash
python backend/services/bili_client.py
```
Requires `.env` with `BILI_ID_CODE`, `BILI_APP_ID`, `BILI_KEY`, `BILI_SECRET`.

## Architecture

```
External Client (HTTP/SSE)
        |
        v
FastAPI server.py (port N)  ── health+break thread (port N+1)
        |  httpx
        v
C++ llama-server (port N+10000)  ── writes WAV files, images to disk
        |
        v
llama.cpp-omni (GGUF model inference)
```

**Key pattern — Python as proxy, not model runner:** The FastAPI server never touches model tensors. It proxies client requests to the C++ backend, streams results back, and manages the C++ process lifecycle (start/stop/restart/health-check).

**Two streaming modes:**
- **Simplex** (single-turn): Caches prefill chunks, sends on final chunk → polls filesystem for WAV outputs → returns to client
- **Duplex** (full-duplex): Forwards each chunk to C++ → reads SSE from C++ decode endpoint → streams text+audio events back to client in real-time

**Separate health thread** (`cpp_manager.py:HealthCheckHandler`): Runs on port N+1 using `http.server.HTTPServer` in a daemon thread, not FastAPI. This ensures break/stop commands are processed even when the main asyncio event loop is blocked.

**GPU memory monitoring** (`cpp_manager.py`): Periodically checks `nvidia-smi`, auto-restarts the C++ server if free GPU memory drops below 2 GB.

**Global mutable state** (`config.py`): Central module-level variables for session mode, C++ process handle, HTTP client, prefill cache. Accessed directly by both `server.py` and `cpp_manager.py` via `from . import config`.

## Configuration

| Env Var | Purpose | Default |
|---------|---------|---------|
| `LLAMACPP_ROOT` | Path to llama.cpp-omni | `llm_server/llama.cpp-omni` |
| `MODEL_DIR` | Path to GGUF models | `llm_server/models/openbmb/MiniCPM-o-4_5-gguf` |
| `LLM_MODEL` | LLM GGUF filename | auto-detected (prefers Q4_K_M > Q8_0 > F16) |
| `VISION_BACKEND` | `"metal"` (GPU) or `"coreml"` (ANE) | `"metal"` |
| `TOKEN2WAV_DEVICE` | TTS device | `"gpu:0"` |
| `REF_AUDIO` | Voice cloning reference audio | `assets/default_ref_audio.wav` |
| `CTX_SIZE` | Model context size | `8192` |
| `BILI_ID_CODE` / `BILI_APP_ID` / `BILI_KEY` / `BILI_SECRET` | Bilibili open platform credentials | — |

## Important Conventions

- Python 3.11 (`.python-version`)
- Big-endian binary protocol for Bilibili WebSocket (`struct.pack(">i", ...)`)
- Import style within `inference_server/`: `from . import config as cfg`, `from . import cpp_manager as cppmgr`
- Bilibili proto uses zlib compression for ver=2 packets, JSON for ver=0
- C++ server stdout is consumed by a daemon thread to prevent pipe buffer deadlock

## Testing

No formal test suite exists. The project has no lint/typecheck configuration at the `apps/` level.
