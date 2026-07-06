"""
MiniCPMO C++ HTTP Server - FastAPI 应用入口
管理 HTTP 路由、请求处理和 SSE 流式生成
"""

import asyncio
import base64
import io
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import httpx
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from . import config as cfg
from . import cpp_manager as cppmgr

# ====================== FastAPI 应用 ======================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 动态计算 C++ 端口
    cfg.CPP_SERVER_PORT = app.state.port + 10000
    cfg.CPP_SERVER_URL = f"http://{cfg.CPP_SERVER_HOST}:{cfg.CPP_SERVER_PORT}"
    print(
        f"C++ 服务器端口: {cfg.CPP_SERVER_PORT} (Python 端口 {app.state.port} + 10000)",
        flush=True,
    )
    print(f"显存监控: {'启用' if cfg.GPU_MEMORY_CHECK else '禁用'}", flush=True)

    # 启动健康检查服务器
    cfg.health_server_thread = threading.Thread(
        target=cppmgr.start_health_server, args=(app.state.port,), daemon=True
    )
    cfg.health_server_thread.start()

    # 创建临时目录
    if os.path.exists(cfg.TEMP_DIR):
        shutil.rmtree(cfg.TEMP_DIR, ignore_errors=True)
    os.makedirs(cfg.TEMP_DIR, exist_ok=True)

    # 启动时清理 output 目录
    cppmgr.reset_output_dir()

    # 启动 C++ 服务器
    print("正在启动 C++ llama-server...", flush=True)
    try:
        cppmgr.start_cpp_server(
            model_dir=app.state.model_dir,
            gpu_devices=app.state.gpu_devices,
            port=cfg.CPP_SERVER_PORT,
        )
    except Exception as e:
        print(f"C++ 服务器启动失败: {e}", flush=True)
        raise

    # 创建 HTTP 客户端
    cfg.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    # 预初始化 omni context
    print("正在预初始化 omni context（TTS + APM + Python T2W）...", flush=True)
    try:
        model_dir = app.state.model_dir
        tts_bin_dir = os.path.join(model_dir, "tts")

        pre_init_request = {
            "media_type": 2,
            "use_tts": True,
            "duplex_mode": app.state.default_duplex_mode,
            "model_dir": model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": cfg.TOKEN2WAV_DEVICE,
            "output_dir": cfg.CPP_OUTPUT_DIR,
            "vision_backend": cfg.VISION_BACKEND,
        }

        if os.path.exists(cfg.FIXED_TIMBRE_PATH):
            pre_init_request["voice_audio"] = cfg.FIXED_TIMBRE_PATH

        pre_init_resp = await cfg.http_client.post(
            f"{cfg.CPP_SERVER_URL}/v1/stream/omni_init",
            json=pre_init_request,
            timeout=120.0,
        )

        if pre_init_resp.status_code == 200:
            cfg.model_state_initialized = True
            cfg.current_duplex_mode = app.state.default_duplex_mode
            cfg.current_msg_type = 2
            print(f"预初始化成功: {pre_init_resp.json()}", flush=True)
        else:
            print(f"预初始化失败（不影响后续使用）: {pre_init_resp.text}", flush=True)
    except Exception as e:
        print(f"预初始化异常（不影响后续使用）: {e}", flush=True)

    print("MiniCPMO C++ HTTP Server 初始化完成", flush=True)

    # 注册服务节点
    try:
        cppmgr.register_service_node(
            port=app.state.port, duplex_mode=app.state.default_duplex_mode
        )
    except Exception as e:
        print(f"服务节点注册失败: {e}", flush=True)

    try:
        yield
    finally:
        if cfg.http_client:
            await cfg.http_client.aclose()
        cppmgr.stop_cpp_server()


app = FastAPI(title="MiniCPMO C++ HTTP Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== 请求模型 ======================


class InitSysPromptRequest(BaseModel):
    media_type: Optional[str] = None
    duplex_mode: Optional[bool] = None
    high_quality_mode: Optional[bool] = False
    high_fps_mode: Optional[bool] = False
    language: Optional[str] = "zh"


class StreamingPrefillRequest(BaseModel):
    audio: Optional[str] = None
    image: Optional[str] = None
    image_audio_id: Optional[int] = None
    frame_index: Optional[int] = None
    max_slice_nums: Optional[int] = None
    session_id: Optional[str] = None
    is_last_chunk: bool = False
    prompt_text: Optional[str] = None


# ====================== API 端点 ======================


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "服务正常 (C++ backend)",
        "backend": "cpp",
        "duplex_mode": cfg.current_duplex_mode,
    }


@app.post("/omni/stop")
async def omni_stop(session_id: Optional[str] = None):
    print("======= 收到会话停止指令 =======", flush=True)
    stopped_session_id = cfg.current_active_session_id

    try:
        assert cfg.http_client
        break_resp = await cfg.http_client.post(
            f"{cfg.CPP_SERVER_URL}/v1/stream/break", json={}
        )
        if break_resp.status_code == 200:
            print("[omni_stop] C++ 生成已中止", flush=True)
    except Exception as e:
        print(f"[omni_stop] C++ break 调用异常: {e}", flush=True)

    cfg.is_breaking = True

    if cfg.wav_timing_log_file:
        try:
            cfg.wav_timing_log_file.write(f"{'-' * 120}\n")
            cfg.wav_timing_log_file.write(
                f"[会话停止] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
            )
            cfg.wav_timing_log_file.close()
        except Exception:
            pass
        cfg.wav_timing_log_file = None
    cfg.last_wav_send_time = None

    with cfg.session_lock:
        cfg.current_active_session_id = None
        cfg.current_request_counter = 0
        cfg.current_round_number = 0
        cfg.pending_prefill_data = None
        cfg.global_sent_wav_count = 0
        cfg.global_parsed_line_count = 0
        cfg.global_parsed_texts = []
        cfg.global_text_send_idx = 0
        cfg.global_sent_wav_files = set()

    print(f"会话已暂停: {stopped_session_id}", flush=True)
    return {
        "success": True,
        "message": "生成已中止，会话保留，可直接继续对话",
        "state": "generation_stopped",
        "session_id": stopped_session_id,
        "kv_cache_preserved": True,
    }


@app.post("/omni/break")
async def omni_break():
    if not cfg.model_state_initialized:
        raise HTTPException(status_code=503, detail="模型未初始化")

    try:
        print("======= 收到单轮打断指令 =======", flush=True)
        cfg.is_breaking = True

        try:
            assert cfg.http_client
            break_resp = await cfg.http_client.post(
                f"{cfg.CPP_SERVER_URL}/v1/stream/break", json={}
            )
            if break_resp.status_code == 200:
                print("[omni_break] C++ 生成已中止", flush=True)
            else:
                print(
                    f"[omni_break] C++ break 调用失败: {break_resp.status_code}",
                    flush=True,
                )
        except Exception as e:
            print(f"[omni_break] C++ break 调用异常: {e}", flush=True)

        print("======= 当前轮对话已打断 =======", flush=True)
        return {"success": True, "message": "当前轮对话已打断", "state": "break"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"打断失败: {str(e)}")


@app.post("/omni/init_sys_prompt")
async def init_sys_prompt(request: InitSysPromptRequest):
    """初始化系统提示"""
    if cfg.is_breaking:
        print(
            "[init_sys_prompt] 检测到残留的 is_breaking=True，重置为 False", flush=True
        )
        cfg.is_breaking = False

    if cfg.cpp_restarting:
        print("[init_sys_prompt] 服务正在重启中，请稍后重试", flush=True)
        raise HTTPException(status_code=503, detail="服务正在重启中，请稍后重试")

    try:
        assert cfg.http_client
        cppmgr.clear_output_subfolders()

        if request.duplex_mode is not None:
            duplex_mode = request.duplex_mode
        else:
            duplex_mode = app.state.default_duplex_mode

        new_session_id = str(uuid.uuid4())[:8]

        if request.media_type:
            if request.media_type.lower() == "audio":
                msg_type = 1
            elif request.media_type.lower() in ["video", "omni"]:
                msg_type = 2
            else:
                raise HTTPException(
                    status_code=400, detail=f"不支持的media_type: {request.media_type}"
                )
        else:
            msg_type = 2

        high_quality_mode = (
            request.high_quality_mode
            if request.high_quality_mode is not None
            else False
        )
        high_fps_mode = (
            request.high_fps_mode if request.high_fps_mode is not None else False
        )
        language = request.language if request.language is not None else "zh"

        is_audio_mode = msg_type == 1
        mode_name = "audio" if is_audio_mode else "omni"
        duplex_name = "双工" if duplex_mode else "单工"
        quality_name = "高清" if high_quality_mode else "普通"
        fps_name = "高刷" if high_fps_mode else "标准帧率"

        duplex_mode_changed = cfg.model_state_initialized and (
            cfg.current_duplex_mode != duplex_mode
        )
        if duplex_mode_changed:
            print(
                f"[警告] duplex_mode 从 {cfg.current_duplex_mode} 变为 {duplex_mode}，将被忽略",
                flush=True,
            )
            duplex_mode = cfg.current_duplex_mode

        media_type_changed = cfg.model_state_initialized and (
            cfg.current_msg_type != msg_type
        )
        if media_type_changed:
            print(
                f"[模式切换] media_type 从 {cfg.current_msg_type} 变为 {msg_type}",
                flush=True,
            )

        cfg.current_msg_type = msg_type
        cfg.current_duplex_mode = duplex_mode

        if not cfg.model_state_initialized:
            model_dir = app.state.model_dir
            tts_bin_dir = os.path.join(model_dir, "tts")

            cpp_request = {
                "media_type": msg_type,
                "use_tts": True,
                "duplex_mode": duplex_mode,
                "model_dir": model_dir,
                "tts_bin_dir": tts_bin_dir,
                "tts_gpu_layers": 100,
                "token2wav_device": cfg.TOKEN2WAV_DEVICE,
                "output_dir": cfg.CPP_OUTPUT_DIR,
                "language": language,
                "vision_backend": cfg.VISION_BACKEND,
            }
            if high_quality_mode:
                cpp_request["max_slice_nums"] = 2
                print("[高清模式] 启用图片切片 max_slice_nums=2", flush=True)

            cfg.current_high_quality_mode = high_quality_mode
            cfg.current_high_fps_mode = high_fps_mode

            print(
                f"[模式设置] 双工={duplex_mode}, 高清={high_quality_mode}, 高刷={high_fps_mode}",
                flush=True,
            )

            if os.path.exists(cfg.FIXED_TIMBRE_PATH):
                cpp_request["voice_audio"] = cfg.FIXED_TIMBRE_PATH
                print(f"使用音色文件: {cfg.FIXED_TIMBRE_PATH}", flush=True)

            print(
                f"初始化，调用 C++ omni_init: {json.dumps(cpp_request, ensure_ascii=False)}",
                flush=True,
            )

            resp = await cfg.http_client.post(
                f"{cfg.CPP_SERVER_URL}/v1/stream/omni_init", json=cpp_request
            )

            if resp.status_code != 200:
                error_text = resp.text
                print(f"C++ omni_init 失败: {error_text}", flush=True)
                raise HTTPException(
                    status_code=500, detail=f"C++ omni_init 失败: {error_text}"
                )

            cpp_result = resp.json()
            print(f"C++ omni_init 成功: {cpp_result}", flush=True)
            cfg.model_state_initialized = True
            fast_resume = False
            init_message = f"初始化完成（{mode_name}模式，{duplex_name}，{quality_name}画质，{fps_name}）"
        elif media_type_changed:
            cfg.current_high_quality_mode = high_quality_mode
            cfg.current_high_fps_mode = high_fps_mode

            update_request = {
                "media_type": msg_type,
                "duplex_mode": duplex_mode,
                "language": language,
            }
            if os.path.exists(cfg.FIXED_TIMBRE_PATH):
                update_request["voice_audio"] = cfg.FIXED_TIMBRE_PATH

            print(
                f"[模式切换] 调用 C++ update_session_config: {json.dumps(update_request, ensure_ascii=False)}",
                flush=True,
            )
            resp = await cfg.http_client.post(
                f"{cfg.CPP_SERVER_URL}/v1/stream/update_session_config",
                json=update_request,
                timeout=30.0,
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"C++ update_session_config 失败: {resp.text}",
                )

            cpp_result = resp.json()
            print(f"C++ update_session_config 成功: {cpp_result}", flush=True)
            fast_resume = False
            init_message = f"模式切换完成（{mode_name}模式，{duplex_name}，{quality_name}画质，{fps_name}）"
        else:
            cfg.current_high_quality_mode = high_quality_mode
            cfg.current_high_fps_mode = high_fps_mode

            update_request = {
                "media_type": msg_type,
                "duplex_mode": duplex_mode,
                "language": language,
            }
            if os.path.exists(cfg.FIXED_TIMBRE_PATH):
                update_request["voice_audio"] = cfg.FIXED_TIMBRE_PATH

            print("[极速恢复] 调用 C++ update_session_config 重置状态", flush=True)
            resp = await cfg.http_client.post(
                f"{cfg.CPP_SERVER_URL}/v1/stream/update_session_config",
                json=update_request,
                timeout=30.0,
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"C++ update_session_config 失败: {resp.text}",
                )

            cpp_result = resp.json()
            print(f"快速恢复成功: {cpp_result}", flush=True)
            fast_resume = True
            init_message = f"初始化成功（{mode_name}模式，{duplex_name}，{quality_name}画质，{fps_name}，快速恢复）"

        if cfg.wav_timing_log_file:
            try:
                cfg.wav_timing_log_file.write(f"{'-' * 120}\n")
                cfg.wav_timing_log_file.write(
                    f"[新会话初始化，关闭旧日志] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
                )
                cfg.wav_timing_log_file.close()
            except Exception:
                pass
            cfg.wav_timing_log_file = None
        cfg.last_wav_send_time = None

        with cfg.session_lock:
            cfg.current_active_session_id = new_session_id
            cfg.current_request_counter = 0
            cfg.current_round_number = 0
            cfg.pending_prefill_data = None
            cfg.global_sent_wav_count = 0
            cfg.global_parsed_line_count = 0
            cfg.global_parsed_texts = []
            cfg.global_text_send_idx = 0
            cfg.global_sent_wav_files = set()

        with cfg.high_fps_cache_lock:
            cfg.high_fps_subimage_cache.clear()
            print("[init_sys_prompt] 已清理高刷模式图片缓存", flush=True)

        return {
            "success": True,
            "message": init_message,
            "msg_type": msg_type,
            "duplex_mode": duplex_mode,
            "session_id": new_session_id,
            "fast_resume": fast_resume,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


@app.post("/omni/streaming_prefill")
async def streaming_prefill(request: StreamingPrefillRequest):
    """流式预填充"""
    if cfg.cpp_restarting:
        raise HTTPException(status_code=503, detail="服务正在重启中，请稍后重试")

    if not cfg.current_active_session_id:
        raise HTTPException(
            status_code=400,
            detail="未找到活跃会话，请先调用 /omni/init_sys_prompt 初始化会话",
        )

    prefill_start_time = time.time()
    timing_stats = {}

    try:
        # 解码音频
        t0 = time.time()
        audio_np = None
        sr = 16000
        if request.audio:
            try:
                audio_bytes = base64.b64decode(request.audio)
                audio_np, file_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.mean(axis=1)
                if file_sr != 16000:
                    audio_np = librosa.resample(
                        audio_np, orig_sr=file_sr, target_sr=16000
                    )
                audio_np = audio_np.astype(np.float32)
                sr = 16000
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"音频数据解码失败: {str(e)}"
                )
        timing_stats["audio_decode"] = (time.time() - t0) * 1000

        # 解码图片
        t0 = time.time()
        pil_image = None
        if request.image:
            try:
                image_bytes = base64.b64decode(request.image)
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"图片数据解码失败: {str(e)}"
                )
        timing_stats["image_decode"] = (time.time() - t0) * 1000

        # 高刷模式处理
        is_main_image = False
        if cfg.current_high_fps_mode and request.image_audio_id is not None:
            frame_idx = request.frame_index if request.frame_index is not None else 0

            if pil_image is not None and audio_np is None:
                if frame_idx == 0:
                    print(
                        f"[高刷模式] 主图到达 image_audio_id={request.image_audio_id}，立即 prefill",
                        flush=True,
                    )
                    pil_images = [pil_image]
                    audio_np = None
                    is_main_image = True
                else:
                    with cfg.high_fps_cache_lock:
                        if request.image_audio_id not in cfg.high_fps_subimage_cache:
                            cfg.high_fps_subimage_cache[request.image_audio_id] = {}
                        cfg.high_fps_subimage_cache[request.image_audio_id][
                            frame_idx
                        ] = pil_image
                        cached_count = len(
                            cfg.high_fps_subimage_cache[request.image_audio_id]
                        )
                        all_subframes_ready = all(
                            i in cfg.high_fps_subimage_cache[request.image_audio_id]
                            for i in [1, 2, 3, 4]
                        )

                    print(
                        f"[高刷模式] 子图缓存 image_audio_id={request.image_audio_id}, frame={frame_idx}, 已缓存{cached_count}帧",
                        flush=True,
                    )

                    if all_subframes_ready:
                        pending_audio = None
                        with cfg.high_fps_audio_lock:
                            if request.image_audio_id in cfg.high_fps_pending_audio:
                                pending_audio = cfg.high_fps_pending_audio.pop(
                                    request.image_audio_id
                                )

                        if pending_audio is not None:
                            audio_np, sr, _ = pending_audio
                            with cfg.high_fps_cache_lock:
                                cached_frames = cfg.high_fps_subimage_cache.pop(
                                    request.image_audio_id, {}
                                )
                            sorted_frames = sorted(
                                cached_frames.items(), key=lambda x: x[0]
                            )
                            subimages = [img for _, img in sorted_frames]
                            stacked_image = cppmgr.stack_images(subimages)
                            pil_images = [stacked_image]
                            print(
                                f"[高刷模式] 子图收齐+待处理音频，stack {len(subimages)} 帧，prefill",
                                flush=True,
                            )
                        else:
                            return {
                                "success": True,
                                "message": f"子图已缓存完毕，等待音频 (image_audio_id={request.image_audio_id})",
                                "cached_frames": cached_count,
                                "mode": "high_fps_cache_ready",
                            }
                    else:
                        return {
                            "success": True,
                            "message": f"子图已缓存 (image_audio_id={request.image_audio_id}, frame={frame_idx})",
                            "cached_frames": cached_count,
                            "mode": "high_fps_cache",
                        }

            elif audio_np is not None:
                with cfg.high_fps_cache_lock:
                    cached_frames = cfg.high_fps_subimage_cache.pop(
                        request.image_audio_id, {}
                    )

                if len(cached_frames) > 0:
                    sorted_frames = sorted(cached_frames.items(), key=lambda x: x[0])
                    subimages = [img for _, img in sorted_frames]
                    stacked_image = cppmgr.stack_images(subimages)
                    pil_images = [stacked_image]
                    print(
                        f"[高刷模式] 音频到达，取出 {len(subimages)} 帧子图 stack，prefill",
                        flush=True,
                    )
                    if pil_image is not None:
                        pil_images.append(pil_image)
                else:
                    with cfg.high_fps_audio_lock:
                        cfg.high_fps_pending_audio[request.image_audio_id] = (
                            audio_np,
                            sr,
                            None,
                        )
                    print(
                        "[高刷模式] 音频到达但无子图缓存，暂存音频等待子图", flush=True
                    )
                    return {
                        "success": True,
                        "message": f"音频已暂存，等待子图 (image_audio_id={request.image_audio_id})",
                        "mode": "high_fps_audio_pending",
                    }
        else:
            pil_images = [pil_image] if pil_image is not None else []

        if audio_np is None and len(pil_images) == 0:
            raise HTTPException(status_code=400, detail="必须提供音频或图片至少一项")

        audio_duration = len(audio_np) / sr if audio_np is not None else 0.0
        omni_mode = cfg.current_msg_type == 2

        if cfg.current_duplex_mode:
            return await _streaming_prefill_duplex(
                request,
                audio_np,
                pil_images,
                sr,
                audio_duration,
                omni_mode,
                timing_stats,
                prefill_start_time,
            )
        elif cfg.current_high_fps_mode and cfg.current_msg_type == 2:
            return await _streaming_prefill_highfps_direct(
                request,
                audio_np,
                pil_images,
                sr,
                audio_duration,
                omni_mode,
                timing_stats,
                prefill_start_time,
                is_main_image=is_main_image,
            )
        else:
            return await _streaming_prefill_simplex(
                request,
                audio_np,
                pil_images,
                sr,
                audio_duration,
                omni_mode,
                timing_stats,
                prefill_start_time,
            )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预填充失败: {str(e)}")


@app.post("/omni/streaming_generate")
async def streaming_generate():
    """流式生成"""
    if cfg.cpp_restarting:
        raise HTTPException(status_code=503, detail="服务正在重启中，请稍后重试")

    if not cfg.current_active_session_id:
        raise HTTPException(status_code=400, detail="未找到活跃会话")

    cfg.is_breaking = False
    generate_request_time = time.time()
    print(
        f"[Generate] 开始生成 (Round #{cfg.current_round_number}, duplex_mode={cfg.current_duplex_mode})",
        flush=True,
    )

    if cfg.current_duplex_mode:
        return await _streaming_generate_duplex(generate_request_time)
    else:
        return await _streaming_generate_simplex(generate_request_time)


# ====================== Prefill 辅助函数 ======================


async def _streaming_prefill_duplex(
    request,
    audio_np,
    pil_images,
    sr,
    audio_duration,
    omni_mode,
    timing_stats,
    prefill_start_time,
):
    assert cfg.http_client
    """双工模式 prefill：直接转发给 C++"""
    with cfg.session_lock:
        cnt = cfg.current_request_counter
        cfg.current_request_counter += 1

    t0 = time.time()
    temp_audio_path = ""
    if audio_np is not None and len(audio_np) > 0:
        MIN_AUDIO_SAMPLES = 1600
        if len(audio_np) < MIN_AUDIO_SAMPLES:
            padding_len = MIN_AUDIO_SAMPLES - len(audio_np)
            audio_np = np.pad(
                audio_np, (0, padding_len), mode="constant", constant_values=0
            )

        temp_audio_path = os.path.join(
            cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}.wav"
        )
        audio_to_save = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
        sf.write(temp_audio_path, audio_to_save, 16000, format="WAV", subtype="PCM_16")
    timing_stats["audio_save"] = (time.time() - t0) * 1000

    t0 = time.time()
    temp_image_paths = []
    if len(pil_images) > 0:
        if cfg.current_high_fps_mode and len(pil_images) > 1:
            main_image = pil_images[0]
            rest_images = pil_images[1:]
            main_path = os.path.join(
                cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}_main.png"
            )
            main_image.save(main_path, format="PNG")
            temp_image_paths.append(main_path)
            if len(rest_images) > 0:
                stacked_image = cppmgr.stack_images(rest_images)
                stack_path = os.path.join(
                    cfg.TEMP_DIR,
                    f"prefill_{cfg.current_active_session_id}_{cnt}_stack.png",
                )
                stacked_image.save(stack_path, format="PNG")
                temp_image_paths.append(stack_path)
                print(f"[高刷模式] 处理 {len(pil_images)} 帧", flush=True)
        else:
            img_path = os.path.join(
                cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}.png"
            )
            pil_images[0].save(img_path, format="PNG")
            temp_image_paths.append(img_path)
    timing_stats["image_save"] = (time.time() - t0) * 1000

    t0 = time.time()
    cpp_success = True
    if len(temp_image_paths) == 0:
        cpp_request = {
            "audio_path_prefix": temp_audio_path,
            "img_path_prefix": "",
            "cnt": cnt,
            "prompt_text": request.prompt_text or "",
        }
        resp = await cfg.http_client.post(
            f"{cfg.CPP_SERVER_URL}/v1/stream/prefill", json=cpp_request, timeout=30.0
        )
        cpp_success = resp.status_code == 200
    else:
        for i, img_path in enumerate(temp_image_paths):
            cpp_request = {
                "audio_path_prefix": temp_audio_path if i == 0 else "",
                "img_path_prefix": img_path,
                "cnt": cnt + i,
                "prompt_text": request.prompt_text if i == 0 else "",
            }
            resp = await cfg.http_client.post(
                f"{cfg.CPP_SERVER_URL}/v1/stream/prefill",
                json=cpp_request,
                timeout=30.0,
            )
            if resp.status_code != 200:
                cpp_success = False
                break
        with cfg.session_lock:
            cfg.current_request_counter += len(temp_image_paths) - 1
    timing_stats["cpp_http"] = (time.time() - t0) * 1000

    total_prefill_time = (time.time() - prefill_start_time) * 1000
    timing_stats["total"] = total_prefill_time

    num_images = len(temp_image_paths)
    has_image = f"✓({num_images}张)" if num_images > 0 else "✗"
    if cpp_success:
        print(
            f"[Prefill #{cnt}] ✓ {total_prefill_time:.0f}ms (音频:{audio_duration:.2f}s 图片:{has_image}) [双工]",
            flush=True,
        )
    else:
        print(f"[Prefill #{cnt}] ✗ C++ prefill 失败 [双工]", flush=True)

    # 清理临时文件
    try:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        for img_path in temp_image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
    except Exception:
        pass

    return {
        "success": cpp_success,
        "session_id": cfg.current_active_session_id,
        "cnt": cnt,
        "audio_duration_seconds": float(audio_duration),
        "timing": timing_stats,
        "backend": "cpp_duplex",
    }


async def _streaming_prefill_highfps_direct(
    request,
    audio_np,
    pil_images,
    sr,
    audio_duration,
    omni_mode,
    timing_stats,
    prefill_start_time,
    is_main_image=False,
):
    """高刷单工模式 prefill：直接 prefill，不延迟"""
    assert cfg.http_client
    with cfg.session_lock:
        cnt = cfg.current_request_counter
        cfg.current_request_counter += 1

    t0 = time.time()
    temp_audio_path = ""
    if audio_np is not None and len(audio_np) > 0:
        MIN_AUDIO_SAMPLES = 1600
        if len(audio_np) < MIN_AUDIO_SAMPLES:
            padding_len = MIN_AUDIO_SAMPLES - len(audio_np)
            audio_np = np.pad(
                audio_np, (0, padding_len), mode="constant", constant_values=0
            )
        temp_audio_path = os.path.join(
            cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}.wav"
        )
        audio_to_save = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
        sf.write(temp_audio_path, audio_to_save, 16000, format="WAV", subtype="PCM_16")
    timing_stats["audio_save"] = (time.time() - t0) * 1000

    t0 = time.time()
    temp_image_paths = []
    if len(pil_images) > 0:
        if len(pil_images) > 1:
            main_image = pil_images[0]
            rest_images = pil_images[1:]
            main_path = os.path.join(
                cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}_main.png"
            )
            main_image.save(main_path, format="PNG")
            temp_image_paths.append(main_path)
            if len(rest_images) > 0:
                stacked_image = cppmgr.stack_images(rest_images)
                stack_path = os.path.join(
                    cfg.TEMP_DIR,
                    f"prefill_{cfg.current_active_session_id}_{cnt}_stack.png",
                )
                stacked_image.save(stack_path, format="PNG")
                temp_image_paths.append(stack_path)
        else:
            img_path = os.path.join(
                cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}.png"
            )
            pil_images[0].save(img_path, format="PNG")
            temp_image_paths.append(img_path)
    timing_stats["image_save"] = (time.time() - t0) * 1000
    timing_stats["image_count"] = len(temp_image_paths)

    t0 = time.time()
    cpp_success = True
    for i, img_path in enumerate(temp_image_paths):
        current_audio_path = temp_audio_path if i == 0 else ""
        cpp_request = {
            "audio_path_prefix": current_audio_path,
            "img_path_prefix": img_path,
            "cnt": cnt + i,
            "prompt_text": request.prompt_text if i == 0 else "",
        }
        if cfg.current_high_quality_mode and is_main_image:
            cpp_request["max_slice_nums"] = 2
        resp = await cfg.http_client.post(
            f"{cfg.CPP_SERVER_URL}/v1/stream/prefill", json=cpp_request, timeout=30.0
        )
        if resp.status_code != 200:
            cpp_success = False
            break

    if len(temp_image_paths) > 1:
        with cfg.session_lock:
            cfg.current_request_counter += len(temp_image_paths) - 1

    timing_stats["cpp_http"] = (time.time() - t0) * 1000
    total_prefill_time = (time.time() - prefill_start_time) * 1000
    timing_stats["total"] = total_prefill_time

    num_images = len(temp_image_paths)
    has_image = f"✓({num_images}张)" if num_images > 0 else "✗"
    has_audio = f"✓({audio_duration:.2f}s)" if audio_np is not None else "✗"
    if cpp_success:
        print(
            f"[Prefill #{cnt}] ✓ {total_prefill_time:.0f}ms (音频:{has_audio} 图片:{has_image}) [高刷]",
            flush=True,
        )
    else:
        print(f"[Prefill #{cnt}] ✗ C++ prefill 失败 [高刷]", flush=True)

    try:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        for img_path in temp_image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
    except Exception:
        pass

    return {
        "success": cpp_success,
        "session_id": cfg.current_active_session_id,
        "cnt": cnt,
        "audio_duration_seconds": float(audio_duration),
        "timing": timing_stats,
        "backend": "cpp_highfps",
    }


async def _streaming_prefill_simplex(
    request,
    audio_np,
    pil_images,
    sr,
    audio_duration,
    omni_mode,
    timing_stats,
    prefill_start_time,
):
    """普通单工模式 prefill：使用延迟一拍机制"""
    assert cfg.http_client
    with cfg.session_lock:
        cnt = cfg.current_request_counter
        cfg.current_request_counter += 1

    t0 = time.time()
    temp_audio_path = ""
    if audio_np is not None and len(audio_np) > 0:
        MIN_AUDIO_SAMPLES = 1600
        if len(audio_np) < MIN_AUDIO_SAMPLES:
            padding_len = MIN_AUDIO_SAMPLES - len(audio_np)
            audio_np = np.pad(
                audio_np, (0, padding_len), mode="constant", constant_values=0
            )

        temp_audio_path = os.path.join(
            cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}.wav"
        )
        audio_to_save = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
        sf.write(temp_audio_path, audio_to_save, 16000, format="WAV", subtype="PCM_16")
    timing_stats["audio_save"] = (time.time() - t0) * 1000

    t0 = time.time()
    temp_image_paths = []
    if len(pil_images) > 0:
        img_path = os.path.join(
            cfg.TEMP_DIR, f"prefill_{cfg.current_active_session_id}_{cnt}.png"
        )
        pil_images[0].save(img_path, format="PNG")
        temp_image_paths.append(img_path)
    timing_stats["image_save"] = (time.time() - t0) * 1000

    if request.is_last_chunk:
        # 最后一片：发送到 C++
        t0 = time.time()
        cpp_request = {
            "audio_path_prefix": temp_audio_path,
            "img_path_prefix": temp_image_paths[0] if len(temp_image_paths) > 0 else "",
            "cnt": cnt,
            "prompt_text": request.prompt_text or "",
        }
        resp = await cfg.http_client.post(
            f"{cfg.CPP_SERVER_URL}/v1/stream/prefill", json=cpp_request, timeout=30.0
        )
        cpp_success = resp.status_code == 200
        timing_stats["cpp_http"] = (time.time() - t0) * 1000

        total_prefill_time = (time.time() - prefill_start_time) * 1000
        timing_stats["total"] = total_prefill_time
        print(
            f"[Prefill #{cnt}] ✓ {total_prefill_time:.0f}ms (is_last_chunk) [单工]",
            flush=True,
        )

        try:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            for img_path in temp_image_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
        except Exception:
            pass

        return {
            "success": cpp_success,
            "session_id": cfg.current_active_session_id,
            "cnt": cnt,
            "audio_duration_seconds": float(audio_duration),
            "timing": timing_stats,
            "backend": "cpp_simplex",
        }
    else:
        # 延迟一拍：缓存数据
        images_for_cache = []
        for img_path in temp_image_paths:
            try:
                img = Image.open(img_path).copy()
                images_for_cache.append(img)
                os.remove(img_path)
            except Exception:
                pass

        cfg.pending_prefill_data = {
            "audio_np": audio_np,
            "images": images_for_cache,
            "cnt": cnt,
        }
        print(f"[Prefill #{cnt}] 缓存数据 (延迟一拍) [单工]", flush=True)

        return {
            "success": True,
            "message": "数据已缓存 (延迟一拍)",
            "session_id": cfg.current_active_session_id,
            "cnt": cnt,
            "mode": "pending",
        }


# ====================== 流式生成器 ======================


async def _streaming_generate_simplex(generate_request_time):
    """单工模式 SSE 流式生成（轮询 WAV 目录）"""
    assert cfg.pending_prefill_data
    assert cfg.http_client
    has_pending = cfg.pending_prefill_data is not None
    pending_cnt = cfg.pending_prefill_data.get("cnt", -1) if has_pending else -1
    print(
        f"[streaming_generate] 开始, pending_data={has_pending}, pending_cnt={pending_cnt}, round={cfg.current_round_number} [单工]",
        flush=True,
    )

    # 处理延迟一拍的缓存数据
    if cfg.pending_prefill_data is not None:
        try:
            print(
                "[streaming_generate] 处理缓存的最后一片数据 (is_last_chunk=True)... [单工]",
                flush=True,
            )
            last_data = cfg.pending_prefill_data
            audio_np = last_data["audio_np"]
            if audio_np is not None and len(audio_np) > 0:
                MIN_AUDIO_SAMPLES = 1600
                if len(audio_np) < MIN_AUDIO_SAMPLES:
                    audio_np = np.pad(
                        audio_np,
                        (0, MIN_AUDIO_SAMPLES - len(audio_np)),
                        mode="constant",
                        constant_values=0,
                    )

                last_cnt = last_data["cnt"]
                temp_audio_path = os.path.join(
                    cfg.TEMP_DIR,
                    f"prefill_{cfg.current_active_session_id}_{last_cnt}.wav",
                )
                audio_to_save = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
                sf.write(
                    temp_audio_path,
                    audio_to_save,
                    16000,
                    format="WAV",
                    subtype="PCM_16",
                )

                temp_image_path = ""
                images = last_data.get("images", [])
                if len(images) > 0:
                    temp_image_path = os.path.join(
                        cfg.TEMP_DIR,
                        f"prefill_{cfg.current_active_session_id}_{last_cnt}.png",
                    )
                    images[0].save(temp_image_path, format="PNG")

                cpp_request = {
                    "audio_path_prefix": temp_audio_path,
                    "img_path_prefix": temp_image_path,
                    "cnt": last_cnt,
                }
                resp = await cfg.http_client.post(
                    f"{cfg.CPP_SERVER_URL}/v1/stream/prefill", json=cpp_request
                )
                if resp.status_code != 200:
                    print(f"C++ 最后一片 prefill 失败: {resp.text}", flush=True)
                else:
                    print(
                        f"[streaming_generate] 最后一片 prefill 成功 (cnt={last_cnt}) [单工]",
                        flush=True,
                    )

            cfg.pending_prefill_data = None
            print("[streaming_generate] 最后一片已处理 [单工]", flush=True)
        except Exception as e:
            print(f"[streaming_generate] 处理最后一片失败: {e}", flush=True)
            cfg.pending_prefill_data = None

    output_dir = os.path.join(
        cfg.TEMP_DIR,
        f"session_{cfg.current_active_session_id}",
        f"round_{cfg.current_round_number:04d}",
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    async def generate_stream():
        generate_start_time = time.time()
        first_chunk_time = None
        first_text_time = None
        chunk_durations = []
        sent_chunk_count = 0
        last_text_len = 0
        sr = 24000

        def sort_wav_files(files):
            def extract_num(f):
                match = re.search(r"wav_(\d+)\.wav", f)
                return int(match.group(1)) if match else 0

            return sorted(files, key=extract_num)

        try:
            cpp_request = {
                "debug_dir": output_dir,
                "stream": True,
                "round_idx": cfg.current_round_number,
            }
            print(
                f"[streaming_generate] 调用 C++ decode: {json.dumps(cpp_request)} [单工]",
                flush=True,
            )

            decode_task = asyncio.create_task(
                cfg.http_client.post(  # pyright: ignore[reportOptionalMemberAccess]
                    f"{cfg.CPP_SERVER_URL}/v1/stream/decode",
                    json=cpp_request,
                    timeout=600.0,
                )
            )

            cpp_output_base = cfg.CPP_OUTPUT_DIR
            round_dir = os.path.join(
                cpp_output_base, f"round_{cfg.current_round_number:03d}"
            )
            tts_wav_dir = os.path.join(round_dir, "tts_wav")
            llm_debug_dir = os.path.join(round_dir, "llm_debug")

            print(
                f"[streaming_generate] 当前轮次: {cfg.current_round_number} [单工]",
                flush=True,
            )
            print(f"  WAV 目录: {tts_wav_dir}", flush=True)

            max_wait = 1800
            check_interval = 0.01
            no_new_wav_count = 0
            max_no_new_wav = 1000
            decode_done = False
            chunk_texts = {}
            all_generated_text = []
            existing_wav_files = set()
            sent_wav_files = set()
            llm_chunk_idx = 0

            if os.path.exists(tts_wav_dir):
                existing_wav_files = set(
                    f
                    for f in os.listdir(tts_wav_dir)
                    if f.startswith("wav_") and f.endswith(".wav")
                )

            def read_chunk_text(llm_debug_dir, chunk_idx):
                chunk_dir = os.path.join(llm_debug_dir, f"chunk_{chunk_idx}")
                text_file = os.path.join(chunk_dir, "llm_text.txt")
                if os.path.exists(text_file):
                    try:
                        with open(
                            text_file, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            return f.read().strip()
                    except Exception:
                        pass
                return ""

            for _ in range(int(max_wait / check_interval)):
                await asyncio.sleep(check_interval)

                if cfg.is_breaking:
                    print(
                        "[streaming_generate] 检测到 break 标志，停止发送数据 [单工]",
                        flush=True,
                    )
                    yield f"data: {json.dumps({'break': True, 'done': True, 'message': '用户打断'}, ensure_ascii=False)}\n\n"
                    break

                if decode_task.done() and not decode_done:
                    decode_done = True
                    try:
                        resp = decode_task.result()
                        if resp.status_code != 200:
                            print(
                                f"[streaming_generate] C++ decode 返回错误: {resp.text}",
                                flush=True,
                            )
                        else:
                            print(
                                "[streaming_generate] C++ decode 完成 [单工]",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"[streaming_generate] C++ decode 异常: {e}", flush=True)

                if os.path.exists(tts_wav_dir):
                    wav_files = [
                        f
                        for f in os.listdir(tts_wav_dir)
                        if f.startswith("wav_") and f.endswith(".wav")
                    ]
                    wav_files = sort_wav_files(wav_files)
                    new_wav_files = [
                        f
                        for f in wav_files
                        if f not in existing_wav_files and f not in sent_wav_files
                    ]

                    for wav_file in new_wav_files:
                        wav_path = os.path.join(tts_wav_dir, wav_file)
                        if not os.path.exists(wav_path):
                            await asyncio.sleep(0.05)
                            if not os.path.exists(wav_path):
                                continue

                        match = re.search(r"wav_(\d+)\.wav", wav_file)
                        chunk_idx = int(match.group(1)) if match else sent_chunk_count

                        try:
                            await asyncio.sleep(0.01)
                            audio_data, audio_sr = sf.read(wav_path)
                            if len(audio_data) == 0:
                                sent_wav_files.add(wav_file)
                                continue

                            if first_chunk_time is None:
                                first_chunk_time = (
                                    time.time() - generate_start_time
                                ) * 1000
                                print(
                                    f"[Generate 音频首响] {first_chunk_time:.1f}ms [单工]",
                                    flush=True,
                                )

                            if audio_data.dtype != np.int16:
                                audio_data = (audio_data * 32767).astype(np.int16)
                            wav_base64 = base64.b64encode(audio_data.tobytes()).decode(
                                "utf-8"
                            )

                            chunk_duration = len(audio_data) / audio_sr
                            chunk_durations.append(chunk_duration)

                            if chunk_idx not in chunk_texts and os.path.exists(
                                llm_debug_dir
                            ):
                                chunk_text = read_chunk_text(
                                    llm_debug_dir, llm_chunk_idx
                                )
                                if chunk_text:
                                    chunk_texts[chunk_idx] = chunk_text
                                    all_generated_text.append(chunk_text)
                                    llm_chunk_idx += 1
                                    if first_text_time is None:
                                        first_text_time = (
                                            time.time() - generate_start_time
                                        ) * 1000
                                        print(
                                            f"[Generate 文本首响] {first_text_time:.1f}ms [单工]",
                                            flush=True,
                                        )

                            chunk_data = {
                                "chunk_idx": sent_chunk_count,
                                "chunk_data": {
                                    "wav": wav_base64,
                                    "sample_rate": int(audio_sr),
                                },
                            }
                            if chunk_idx in chunk_texts:
                                chunk_data["chunk_data"]["text"] = chunk_texts[
                                    chunk_idx
                                ]
                                last_text_len += len(chunk_texts[chunk_idx])

                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                            sent_wav_files.add(wav_file)
                            sent_chunk_count += 1

                        except FileNotFoundError:
                            pass
                        except Exception as e:
                            print(
                                f"[Chunk #{chunk_idx}] 读取失败: {e} [单工]", flush=True
                            )
                            sent_wav_files.add(wav_file)

                    done_flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
                    if os.path.exists(done_flag_path):
                        try:
                            with open(done_flag_path, "r") as f:
                                last_wav_idx = int(f.read().strip())
                            last_wav_file = f"wav_{last_wav_idx}.wav"
                            if (
                                last_wav_file in sent_wav_files
                                or last_wav_file in existing_wav_files
                            ):
                                print(
                                    "[streaming_generate] 所有 wav 已发送，立即结束 [单工]",
                                    flush=True,
                                )
                                break
                        except Exception:
                            pass

                    current_new_count = len(
                        [f for f in wav_files if f not in existing_wav_files]
                    )
                    if current_new_count == len(sent_wav_files):
                        no_new_wav_count += 1
                        if decode_done and no_new_wav_count >= 30000:
                            print("[streaming_generate] 超时退出 [单工]", flush=True)
                            break
                    else:
                        no_new_wav_count = 0

                if decode_done and sent_chunk_count == 0:
                    no_new_wav_count += 1
                    if no_new_wav_count >= max_no_new_wav:
                        print(
                            "[streaming_generate] decode完成但无wav输出，超时退出 [单工]",
                            flush=True,
                        )
                        break

            if not decode_task.done():
                print("[streaming_generate] 等待 C++ decode 完成... [单工]", flush=True)
                try:
                    await asyncio.wait_for(decode_task, timeout=30.0)
                except asyncio.TimeoutError:
                    print("[streaming_generate] C++ decode 超时 [单工]", flush=True)

            if all_generated_text:
                full_text = "".join(all_generated_text)
                print(f"\n[完整生成文本] {full_text}\n", flush=True)

            total_generate_time = (time.time() - generate_start_time) * 1000
            total_audio_duration = sum(chunk_durations) if chunk_durations else 0
            overall_rtf = (
                total_generate_time / 1000 / total_audio_duration
                if total_audio_duration > 0
                else 0
            )

            print(f"\n{'=' * 60}", flush=True)
            print("[Generate 性能总结] [单工]", flush=True)
            print(
                f"  音频首响: {first_chunk_time if first_chunk_time else 0:.1f}ms",
                flush=True,
            )
            print(
                f"  文本首响: {first_text_time if first_text_time else 0:.1f}ms",
                flush=True,
            )
            print(f"  总片数: {sent_chunk_count}", flush=True)
            print(f"  音频总长: {total_audio_duration:.2f}s", flush=True)
            print(f"  生成总时间: {total_generate_time:.0f}ms", flush=True)
            print(f"  文字总长: {last_text_len} 字符", flush=True)
            print(f"{'=' * 60}\n", flush=True)

        except Exception as e:
            error_msg = str(e) if str(e) else repr(e)
            print(
                f"[streaming_generate] 生成异常: {type(e).__name__}: {error_msg} [单工]",
                flush=True,
            )
            yield f"data: {json.dumps({'error': True, 'message': f'生成错误: {type(e).__name__}: {error_msg[:500]}'}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _streaming_generate_duplex(generate_request_time):
    """双工模式 SSE 流式生成（SSE 读 stream_decode 返回）"""
    print(
        f"[streaming_generate] 开始, round={cfg.current_round_number} [双工]",
        flush=True,
    )

    with cfg.session_lock:
        output_dir = os.path.join(
            cfg.TEMP_DIR,
            f"session_{cfg.current_active_session_id}",
            "round_0000",
            "output",
        )
    os.makedirs(output_dir, exist_ok=True)

    async def generate_stream():
        generate_start_time = time.time()
        first_chunk_time = None
        first_text_time = None
        last_text_len = 0
        sent_chunk_count = 0
        chunk_durations = []
        all_generated_text = []

        try:
            cpp_output_base = cfg.CPP_OUTPUT_DIR
            llm_debug_dir = os.path.join(cpp_output_base, "llm_debug")
            tts_wav_dir = os.path.join(cpp_output_base, "tts_wav")

            cpp_request = {
                "debug_dir": cpp_output_base,
                "stream": True,
                "round_idx": cfg.current_round_number,
            }
            print(
                f"[streaming_generate] 调用 C++ decode SSE: {json.dumps(cpp_request)} [双工]",
                flush=True,
            )

            async with cfg.http_client.stream(  # pyright: ignore[reportOptionalMemberAccess]
                "POST",
                f"{cfg.CPP_SERVER_URL}/v1/stream/decode",
                json=cpp_request,
                timeout=600.0,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(
                        f"[streaming_generate] C++ decode 错误: {error_text.decode()} [双工]",
                        flush=True,
                    )
                    yield f"data: {json.dumps({'error': True, 'message': f'C++ decode error: {response.status_code}'})}\n\n"
                    return

                sse_iterator = response.aiter_text().__aiter__()
                sent_wav_files = set()

                async for line in sse_iterator:
                    if cfg.is_breaking:
                        print(
                            "[streaming_generate] 检测到 break，停止 [双工]",
                            flush=True,
                        )
                        yield f"data: {json.dumps({'break': True, 'done': True, 'message': '用户打断'}, ensure_ascii=False)}\n\n"
                        break

                    if not line or not line.startswith("data:"):
                        continue

                    data_str = (
                        line[5:].strip()
                        if line.startswith("data: ")
                        else line[5:].strip()
                    )
                    if data_str == "[DONE]":
                        print("[streaming_generate] 收到 [DONE] [双工]", flush=True)
                        break

                    try:
                        event_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_data.get("done") or event_data.get("stop"):
                        break

                    content = event_data.get("content", "")
                    if content and content not in ("__IS_LISTEN__", "__END_OF_TURN__"):
                        all_generated_text.append(content)
                        if first_text_time is None:
                            first_text_time = (time.time() - generate_start_time) * 1000
                            print(
                                f"[streaming_generate] 文本首响: {first_text_time:.1f}ms [双工]",
                                flush=True,
                            )

                    # 检查 WAV 目录
                    if os.path.exists(tts_wav_dir):
                        wav_files = [
                            f
                            for f in os.listdir(tts_wav_dir)
                            if f.startswith("wav_") and f.endswith(".wav")
                        ]
                        for wav_file in sorted(
                            wav_files,
                            key=lambda f: (
                                int(re.search(r"wav_(\d+)\.wav", f).group(1))  # pyright: ignore[reportOptionalMemberAccess]
                                if re.search(r"wav_(\d+)\.wav", f)
                                else 0
                            ),
                        ):
                            if wav_file in sent_wav_files:
                                continue
                            wav_path = os.path.join(tts_wav_dir, wav_file)
                            if not os.path.exists(wav_path):
                                continue

                            try:
                                audio_data, audio_sr = sf.read(wav_path)
                                if len(audio_data) == 0:
                                    sent_wav_files.add(wav_file)
                                    continue

                                if first_chunk_time is None:
                                    first_chunk_time = (
                                        time.time() - generate_start_time
                                    ) * 1000
                                    print(
                                        f"[Generate 音频首响] {first_chunk_time:.1f}ms [双工]",
                                        flush=True,
                                    )

                                if audio_data.dtype != np.int16:
                                    audio_data = (audio_data * 32767).astype(np.int16)
                                wav_base64 = base64.b64encode(
                                    audio_data.tobytes()
                                ).decode("utf-8")

                                chunk_duration = len(audio_data) / audio_sr
                                chunk_durations.append(chunk_duration)

                                chunk_data = {
                                    "chunk_idx": sent_chunk_count,
                                    "chunk_data": {
                                        "wav": wav_base64,
                                        "sample_rate": int(audio_sr),
                                    },
                                }
                                if content:
                                    chunk_data["chunk_data"]["text"] = content
                                    last_text_len += len(content)

                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                                sent_wav_files.add(wav_file)
                                sent_chunk_count += 1
                            except Exception as e:
                                print(
                                    f"[WAV 读取失败] {wav_file}: {e} [双工]", flush=True
                                )
                                sent_wav_files.add(wav_file)

            if all_generated_text:
                full_text = "".join(all_generated_text)
                print(f"\n[完整生成文本] {full_text}\n", flush=True)

            total_generate_time = (time.time() - generate_start_time) * 1000
            total_audio_duration = sum(chunk_durations) if chunk_durations else 0
            print(f"\n{'=' * 60}", flush=True)
            print("[Generate 性能总结] [双工]", flush=True)
            print(
                f"  音频首响: {first_chunk_time if first_chunk_time else 0:.1f}ms",
                flush=True,
            )
            print(
                f"  文本首响: {first_text_time if first_text_time else 0:.1f}ms",
                flush=True,
            )
            print(f"  总片数: {sent_chunk_count}", flush=True)
            print(f"  音频总长: {total_audio_duration:.2f}s", flush=True)
            print(f"  生成总时间: {total_generate_time:.0f}ms", flush=True)
            print(f"{'=' * 60}\n", flush=True)

        except Exception as e:
            error_msg = str(e) if str(e) else repr(e)
            print(
                f"[streaming_generate] 生成异常: {type(e).__name__}: {error_msg} [双工]",
                flush=True,
            )
            yield f"data: {json.dumps({'error': True, 'message': f'生成错误: {type(e).__name__}: {error_msg[:500]}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
