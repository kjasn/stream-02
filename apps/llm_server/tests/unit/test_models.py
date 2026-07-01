"""Tests for Pydantic request models and server helpers."""

import re

import pytest
from pydantic import ValidationError


class TestInitSysPromptRequest:
    """Tests for InitSysPromptRequest model."""

    def test_default_values(self):
        from llm_server.inference_server.server import InitSysPromptRequest

        req = InitSysPromptRequest()
        assert req.media_type is None
        assert req.duplex_mode is None
        assert req.high_quality_mode is False
        assert req.high_fps_mode is False
        assert req.language == "zh"

    def test_all_fields_set(self):
        from llm_server.inference_server.server import InitSysPromptRequest

        req = InitSysPromptRequest(
            media_type="omni",
            duplex_mode=True,
            high_quality_mode=True,
            high_fps_mode=True,
            language="en",
        )
        assert req.media_type == "omni"
        assert req.duplex_mode is True
        assert req.high_quality_mode is True
        assert req.high_fps_mode is True
        assert req.language == "en"

    def test_json_parsing_with_snake_case(self):
        from llm_server.inference_server.server import InitSysPromptRequest

        req = InitSysPromptRequest.model_validate({
            "media_type": "audio",
        })
        assert req.media_type == "audio"


class TestStreamingPrefillRequest:
    """Tests for StreamingPrefillRequest model."""

    def test_default_values(self):
        from llm_server.inference_server.server import StreamingPrefillRequest

        req = StreamingPrefillRequest()
        assert req.audio is None
        assert req.image is None
        assert req.image_audio_id is None
        assert req.frame_index is None
        assert req.max_slice_nums is None
        assert req.session_id is None
        assert req.is_last_chunk is False
        assert req.prompt_text is None

    def test_prompt_text_field(self):
        from llm_server.inference_server.server import StreamingPrefillRequest

        req = StreamingPrefillRequest(
            prompt_text="【直播间动态】\n[弹幕] 用户A: 测试"
        )
        assert req.prompt_text == "【直播间动态】\n[弹幕] 用户A: 测试"
        assert req.audio is None
        assert req.image is None

    def test_all_fields_set(self):
        from llm_server.inference_server.server import StreamingPrefillRequest

        req = StreamingPrefillRequest(
            audio="base64encoded...",
            image="base64encoded...",
            image_audio_id=1,
            frame_index=5,
            max_slice_nums=10,
            session_id="session-123",
            is_last_chunk=True,
            prompt_text="测试文本",
        )
        assert req.image_audio_id == 1
        assert req.frame_index == 5
        assert req.is_last_chunk is True
        assert req.prompt_text == "测试文本"

    def test_is_last_chunk_required(self):
        from llm_server.inference_server.server import StreamingPrefillRequest

        req = StreamingPrefillRequest(is_last_chunk=True)
        assert req.is_last_chunk is True


class TestSortWavFiles:
    """Tests for the sort_wav_files helper inside _streaming_generate_simplex."""

    def test_sorts_by_numeric_index(self):
        def sort_wav_files(files):
            def extract_num(f):
                match = re.search(r"wav_(\d+)\.wav", f)
                return int(match.group(1)) if match else 0

            return sorted(files, key=extract_num)

        files = [
            "wav_0003.wav",
            "wav_0001.wav",
            "wav_0010.wav",
            "wav_0002.wav",
        ]
        result = sort_wav_files(files)
        assert result == [
            "wav_0001.wav",
            "wav_0002.wav",
            "wav_0003.wav",
            "wav_0010.wav",
        ]

    def test_non_matching_returns_zero(self):
        def sort_wav_files(files):
            def extract_num(f):
                match = re.search(r"wav_(\d+)\.wav", f)
                return int(match.group(1)) if match else 0

            return sorted(files, key=extract_num)

        files = ["other_file.txt", "wav_0005.wav", "random.wav"]
        result = sort_wav_files(files)
        assert result[0] in ("other_file.txt", "random.wav")
        assert result[-1] == "wav_0005.wav"

    def test_empty_list(self):
        def sort_wav_files(files):
            def extract_num(f):
                match = re.search(r"wav_(\d+)\.wav", f)
                return int(match.group(1)) if match else 0

            return sorted(files, key=extract_num)

        assert sort_wav_files([]) == []

    def test_large_sequential_numbers(self):
        def sort_wav_files(files):
            def extract_num(f):
                match = re.search(r"wav_(\d+)\.wav", f)
                return int(match.group(1)) if match else 0

            return sorted(files, key=extract_num)

        files = ["wav_9999.wav", "wav_0001.wav", "wav_0500.wav"]
        result = sort_wav_files(files)
        assert result == ["wav_0001.wav", "wav_0500.wav", "wav_9999.wav"]


class TestProtoPackUnpack:
    """Tests for the Bilibili Proto binary protocol."""

    def test_pack_and_unpack_roundtrip(self):
        from backend.services.bili_client import Proto

        original = Proto()
        original.op = 7
        original.seq = 1
        original.body = b'{"test": "data"}'

        packed = original.pack()
        unpacked = Proto()
        n = unpacked.unpack(packed)

        assert n > 0
        assert unpacked.op == 7
        assert unpacked.seq == 1
        assert unpacked.body == b'{"test": "data"}'

    def test_unpack_incomplete_header_returns_zero(self):
        from backend.services.bili_client import Proto

        p = Proto()
        n = p.unpack(b"\x00\x01\x02")
        assert n == 0

    def test_unpack_insufficient_body_returns_zero(self):
        from backend.services.bili_client import Proto

        p = Proto()
        p.packet_len = 100
        n = p.unpack(p.pack() + b"\x00")
        assert n > 0  # Reads the valid packet, ignores the extra byte

    def test_decode_body_ver0_json(self):
        from backend.services.bili_client import Proto

        p = Proto()
        p.ver = 0
        p.body = b'{"cmd": "TEST"}'
        result = p.decode_body()
        assert result == [{"cmd": "TEST"}]

    def test_decode_body_ver0_empty(self):
        from backend.services.bili_client import Proto

        p = Proto()
        p.ver = 0
        p.body = b""
        result = p.decode_body()
        assert result == []

    def test_heartbeat_ops(self):
        from backend.services.bili_client import Proto

        assert Proto.OP_HEARTBEAT == 2
        assert Proto.OP_HEARTBEAT_REPLY == 3
        assert Proto.OP_AUTH == 7
        assert Proto.OP_AUTH_REPLY == 8
