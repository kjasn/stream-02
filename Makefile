.PHONY: build-cpp test test-integration test-all build clean

# ── Paths ─────────────────────────────────────────────────────

APPS_DIR := $(shell pwd)
LLAMACPP_ROOT := $(APPS_DIR)/llm_server/llama.cpp-omni
LLAMACPP_BUILD := $(LLAMACPP_ROOT)/build
SERVER_BIN := $(LLAMACPP_BUILD)/bin/llama-server
MODEL_DIR := $(APPS_DIR)/llm_server/models/openbmb/MiniCPM-o-4_5-gguf

# ── Build ─────────────────────────────────────────────────────

build-cpp: $(SERVER_BIN)

$(SERVER_BIN):
	cd $(LLAMACPP_ROOT) && \
		cmake -B build -DCMAKE_BUILD_TYPE=Release && \
		cmake --build build --target llama-server -j $$(sysctl -n hw.ncpu 2>/dev/null || nproc)

build: build-cpp
	uv sync

# ── Test ──────────────────────────────────────────────────────

test:
	uv run pytest llm_server/tests/ backend/tests/ -v -m "not integration"

test-integration:
	uv run pytest llm_server/tests/ -v -m integration --tb=long -s

test-all:
	uv run pytest llm_server/tests/ backend/tests/ -v --tb=long

# ── Clean ─────────────────────────────────────────────────────

clean:
	rm -rf $(LLAMACPP_BUILD)

# ── Run ───────────────────────────────────────────────────────

run-llm-server:
	MODEL_DIR=$(MODEL_DIR) \
	LLAMACPP_ROOT=$(LLAMACPP_ROOT) \
	uv run -m llm_server \
		--llamacpp-root $(LLAMACPP_ROOT) \
		--model-dir $(MODEL_DIR) \
		--port 8060

run-backend:
	uv run python -m backend
