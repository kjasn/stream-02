# stream-02

`stream-02` is an open-source live-stream interaction assistant, inspired by the
idea of a "二号机" for streamers.

> [!WARNING]
> The project is still in the early design stage.
>
> Most of the documentation is written by AI and needs to be rewritten.

## Core Use Case

During a live stream, `stream-02` should be able to:

1. Read audience comments and danmaku from the live-stream room.
2. Capture and transcribe the anchor's voice.
3. Send useful context to an LLM.
4. Generate a reply for the audience or the anchor.
5. Convert the reply to speech.
6. Play the voice reply back into the stream workflow.

## Architecture

The project is split into a Go core and a Python model service.

The Go core owns the long-running application logic:

- platform danmaku adapters
- OCR adapter orchestration
- event bus and message routing
- conversation state
- LLM provider adapters
- TTS provider adapters
- configuration and plugin-style extension points

The Python service owns local model inference:

- OCR with PyTorch
- optional local STT
- model loading and caching
- CPU/GPU device selection
- structured inference responses for the Go core

The first implementation should keep these parts loosely coupled. The Go process
talks to the Python model service through a local HTTP API, likely implemented
with FastAPI.

## Important Tech Stack

- **Go**: main application runtime, platform adapters, event bus, provider
  clients, configuration, and orchestration.
- **Python**: local AI model runtime for OCR and optional STT.
- **PyTorch**: OCR model inference and future local model experiments.
- **FastAPI**: local Python service API exposed to the Go core.
- **LLM provider adapters**: pluggable clients for hosted or local LLMs.
- **TTS provider adapters**: pluggable clients for voice reply generation.
- **Platform API/WebSocket adapters**: high-quality danmaku ingestion when a
  platform integration is available.
- **OCR screen reader adapter**: universal fallback for platforms that are hard
  to integrate directly.

## Project Flow

Audience message flow:

```text
Live Stream Screen / Platform API
        |
        v
Danmaku Sources
  - Platform API/WebSocket adapter
  - OCR screen reader adapter
        |
        v
Go Core Event Bus
        |
        +--> DanmakuEvent
        +--> AnchorVoiceTranscriptEvent
        |
        v
Conversation Engine
        |
        v
LLM Provider
        |
        v
TTS Provider
        |
        v
Voice Reply Output
```

Python model service flow:

```text
Go Core
  |
  | local HTTP
  v
Python Model Service
  - OCR with PyTorch
  - optional local STT
  - model loading and device selection
  |
  v
Structured inference result
```

Voice interaction flow:

```text
Anchor Microphone
        |
        v
Voice Activity Detection
        |
        v
Speech-to-Text
        |
        v
Text Context for LLM
        |
        v
LLM Reply
        |
        v
Text-to-Speech
        |
        v
Voice Reply Output
```

## Danmaku Input Strategy

`stream-02` treats every danmaku source as an adapter that emits normalized
events.

Platform API or WebSocket adapters should be used when they are available,
because they can provide structured data such as username, timestamp, badges,
gift events, membership status, and message metadata.

OCR is an important fallback path. It allows `stream-02` to support platforms
where direct integration is expensive, unstable, private, or unavailable. The
OCR adapter reads visible comments from the live-stream screen and converts them
into normalized danmaku events with confidence information.

This means v1 can support both paths:

```text
Platform API/WebSocket -> DanmakuEvent
Screen OCR             -> DanmakuEvent
```

## Voice Strategy

The default v1 voice path should be text-centered:

```text
audio -> STT -> text -> LLM -> text -> TTS -> audio
```

This is easier to debug, moderate, customize, and swap between providers than a
direct audio-to-audio pipeline. Direct realtime voice APIs can be added later as
an advanced mode, but they are not the default design target for the first
version.

## Why Go + Python?

Go is a good fit for the main application because it is simple to deploy,
concurrent by default, and reliable for long-running services. Most platform
connectors, provider clients, and event routing logic should live in Go.

Python is the practical choice for local model inference. OCR and STT ecosystems
are strongest in Python, and PyTorch model support is mature. Keeping Python in a
separate local service avoids tightly coupling Go application logic to Python
runtime and package management details.

This split lets contributors work on different parts of the system independently:

- Go contributors can improve platform adapters, event flow, LLM/TTS providers,
  and app behavior.
- Python contributors can improve OCR, STT, model loading, and inference
  performance.

## Current Direction

- Build a local-first open-source tool.
- Use Go for the core application.
- Use Python + FastAPI + PyTorch for local OCR and optional STT.
- Treat OCR as a first-class danmaku source, not a temporary hack.
- Keep platform APIs and WebSocket integrations as higher-quality adapters.
- Use STT -> LLM -> TTS as the default voice interaction pipeline.
- Keep UI technology undecided until the core flow is clearer.

## Roadmap

- Define the normalized event types for danmaku, voice transcripts, LLM replies,
  and TTS output.
- Build the Go event bus and adapter interfaces.
- Build the Python OCR service API.
- Add the first OCR-based danmaku source.
- Add one platform API/WebSocket danmaku adapter.
- Add provider adapters for at least one LLM and one TTS service.
- Create a minimal local configuration format.
- Add a simple operator UI after the core pipeline is usable.
