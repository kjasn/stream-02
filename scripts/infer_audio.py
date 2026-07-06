"""Quick inference script: prefill + generate → play TTS output.

Usage:
    python scripts/infer_audio.py [--port 18060] [--text "弹幕上下文"]
"""

import argparse
import os
import struct
import sys
import tempfile
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_server.inference_server import config as cfg


def build_wav(sample_rate=16000, duration_sec=1.0):
    """Generate a short silent WAV (mono 16-bit PCM)."""
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
    samples = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
    return header + samples


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o quick audio inference")
    parser.add_argument("--port", type=int, default=19060, help="C++ server port")
    parser.add_argument("--text", type=str, default="请用中文简单介绍一下你自己。", help="Prompt text")
    parser.add_argument("--play", action="store_true", default=True, help="Play audio after generation")
    parser.add_argument("--no-play", dest="play", action="store_false")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    output_dir = tempfile.mkdtemp(prefix="infer_audio_")
    model_dir = cfg.DEFAULT_MODEL_DIR

    print(f"C++ server: {base_url}")
    print(f"Model dir:  {model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt:     {args.text}")

    # ── Health check ──
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health: {r.status_code}")
    except Exception as e:
        sys.exit(f"Cannot reach C++ server at {base_url}: {e}")

    # ── Omni init ──
    tts_bin_dir = os.path.join(model_dir, "tts")
    init_body = {
        "media_type": 1,  # audio mode
        "use_tts": True,
        "duplex_mode": False,
        "model_dir": model_dir,
        "tts_bin_dir": tts_bin_dir,
        "tts_gpu_layers": 100,
        "token2wav_device": cfg.TOKEN2WAV_DEVICE,
        "output_dir": output_dir,
        "vision_backend": cfg.VISION_BACKEND,
    }
    ref_audio = cfg.FIXED_TIMBRE_PATH
    if ref_audio and os.path.exists(ref_audio):
        init_body["voice_audio"] = ref_audio
        print(f"Ref audio:  {ref_audio}")

    print("\n[1/3] Initializing omni context (TTS model loading, ~15-30s)...")
    t0 = time.time()
    resp = requests.post(f"{base_url}/v1/stream/omni_init", json=init_body, timeout=120)
    assert resp.status_code == 200, f"omni_init failed: {resp.text}"
    data = resp.json()
    assert data.get("success") is True, f"omni_init: {data}"
    print(f"  done ({time.time() - t0:.1f}s)")

    # ── Prefill ──
    wav_path = os.path.join(output_dir, "test_audio.wav")
    with open(wav_path, "wb") as f:
        f.write(build_wav(16000, 0.3))

    prefill_body = {
        "audio_path_prefix": wav_path,
        "img_path_prefix": "",
        "cnt": 0,
        "prompt_text": args.text,
    }

    print(f"\n[2/3] Prefilling (prompt_text={len(args.text)} chars)...")
    t0 = time.time()
    resp = requests.post(f"{base_url}/v1/stream/prefill", json=prefill_body, timeout=30)
    assert resp.status_code == 200, f"prefill failed: {resp.text}"
    print(f"  done ({time.time() - t0:.1f}s)")

    # ── Generate ──
    gen_body = {"debug_dir": output_dir, "stream": False, "round_idx": 0}
    print("\n[3/3] Generating (streaming)...")
    t0 = time.time()
    resp = requests.post(f"{base_url}/v1/stream/decode", json=gen_body, timeout=300)
    print(f"  status={resp.status_code}, time={time.time() - t0:.1f}s")

    # ── Collect output WAVs ──
    round_dir = os.path.join(output_dir, "round_000")
    tts_wav_dir = os.path.join(round_dir, "tts_wav")
    if os.path.isdir(tts_wav_dir):
        wav_files = sorted(
            [f for f in os.listdir(tts_wav_dir) if f.endswith(".wav")],
            key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 0,
        )
        print(f"\nOutput WAVs ({len(wav_files)} files):")
        for f in wav_files:
            path = os.path.join(tts_wav_dir, f)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {f}  ({size_kb:.1f} KB)")

        if args.play and wav_files:
            out_path = "/tmp/minicpmo_output.wav"
            # Concatenate all WAVs into one file
            import wave

            all_frames = []
            params = None
            for fname in wav_files:
                fpath = os.path.join(tts_wav_dir, fname)
                with wave.open(fpath, "rb") as wf:
                    if params is None:
                        params = wf.getparams()
                    all_frames.append(wf.readframes(wf.getnframes()))

            with wave.open(out_path, "wb") as wf_out:
                wf_out.setparams(params)
                for frames in all_frames:
                    wf_out.writeframes(frames)

            print(f"\nSaved combined audio to: {out_path}")
            print("Playing...")
            os.system(f"afplay {out_path}")
    else:
        print("\nNo TTS output found. round_dir contents:")
        if os.path.isdir(round_dir):
            for item in os.listdir(round_dir):
                print(f"  {item}")
        else:
            print(f"  round_dir does not exist: {round_dir}")

    # Keep output around
    print(f"\nOutput kept at: {output_dir}")


if __name__ == "__main__":
    main()
