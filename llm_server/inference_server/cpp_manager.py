"""
MiniCPMO C++ HTTP Server - C++ 进程管理模块
管理 llama-server 子进程生命周期、健康检查、服务注册
"""

import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional

import requests
from PIL import Image

from . import config

# ====================== GPU 内存监控 ======================
GPU_MEMORY_THRESHOLD_MB = 2000


def get_gpu_memory_info() -> Optional[dict]:
    """获取 GPU 显存信息"""
    try:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                total = int(parts[0].strip())
                used = int(parts[1].strip())
                free = int(parts[2].strip())
                return {
                    "total_mb": total,
                    "used_mb": used,
                    "free_mb": free,
                    "utilization": round(used / total * 100, 1) if total > 0 else 0,
                }
    except Exception as e:
        print(f"[显存监控] 获取显存信息失败: {e}", flush=True)
    return None


def check_gpu_memory_and_restart_if_needed(model_dir: str, gpu_devices: str) -> bool:
    """检查 GPU 显存，如果剩余不足则重启 C++ 服务器"""
    if not config.GPU_MEMORY_CHECK:
        return False

    mem_info = get_gpu_memory_info()
    if mem_info is None:
        return False

    free_mb = mem_info["free_mb"]
    print(
        f"[显存监控] 剩余显存: {free_mb} MB (阈值: {GPU_MEMORY_THRESHOLD_MB} MB)",
        flush=True,
    )

    if free_mb < GPU_MEMORY_THRESHOLD_MB:
        print(
            f"[显存监控] 显存不足 ({free_mb} MB < {GPU_MEMORY_THRESHOLD_MB} MB)，准备重启 C++ 服务器...",
            flush=True,
        )

        if _cpp_restart_lock is None:
            return False

        with _cpp_restart_lock:
            mem_info = get_gpu_memory_info()
            if mem_info and mem_info["free_mb"] >= GPU_MEMORY_THRESHOLD_MB:
                print("[显存监控] 显存已恢复，取消重启", flush=True)
                return False
            try:
                restart_cpp_server(model_dir, gpu_devices)
                return True
            except Exception as e:
                print(f"[显存监控] 重启失败: {e}", flush=True)
                return False
    return False


_cpp_restart_lock = threading.Lock()


def restart_cpp_server(model_dir: str, gpu_devices: str):
    """重启 C++ llama-server（保持相同配置）"""
    global _cpp_restart_lock

    print("=" * 60, flush=True)
    print("[重启] 开始重启 C++ llama-server...", flush=True)
    print("=" * 60, flush=True)

    config.cpp_restarting = True

    saved_duplex_mode = config.current_duplex_mode
    saved_msg_type = config.current_msg_type if config.current_msg_type else 2

    stop_cpp_server()
    time.sleep(2)
    reset_output_dir()

    config.model_state_initialized = False
    config.current_msg_type = None
    config.current_round_number = 0
    config.global_sent_wav_count = 0
    config.global_parsed_line_count = 0
    config.global_parsed_texts = []
    config.global_text_send_idx = 0
    config.global_sent_wav_files = set()

    start_cpp_server(
        model_dir=model_dir, gpu_devices=gpu_devices, port=config.CPP_SERVER_PORT
    )

    print("[重启] C++ llama-server 重启完成", flush=True)

    # 重新初始化 omni context
    try:
        print("[重启] 重新初始化 omni context...", flush=True)
        tts_bin_dir = os.path.join(model_dir, "tts")

        cpp_request = {
            "media_type": saved_msg_type,
            "use_tts": True,
            "duplex_mode": saved_duplex_mode,
            "model_dir": model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": config.TOKEN2WAV_DEVICE,
            "output_dir": config.CPP_OUTPUT_DIR,
        }
        cpp_request["vision_backend"] = config.VISION_BACKEND

        if os.path.exists(config.FIXED_TIMBRE_PATH):
            cpp_request["voice_audio"] = config.FIXED_TIMBRE_PATH

        resp = requests.post(
            f"{config.CPP_SERVER_URL}/v1/stream/omni_init",
            json=cpp_request,
            timeout=60.0,
        )

        if resp.status_code == 200:
            config.model_state_initialized = True
            config.current_msg_type = saved_msg_type
            config.current_duplex_mode = saved_duplex_mode
            print(f"[重启] omni context 初始化成功: {resp.json()}", flush=True)
        else:
            print(f"[重启] omni context 初始化失败: {resp.text}", flush=True)
    except Exception as e:
        print(f"[重启] omni context 初始化异常: {e}", flush=True)
    finally:
        config.cpp_restarting = False

    print("=" * 60, flush=True)


def stack_images(images: List[Image.Image]) -> Image.Image:
    """将多张图片 stack 成一张（横向拼接或 2x2 布局）"""
    if len(images) == 0:
        raise ValueError("images 列表不能为空")
    if len(images) == 1:
        return images[0]

    w, h = images[0].size

    if len(images) == 2:
        result = Image.new("RGB", (w * 2, h))
        result.paste(images[0], (0, 0))
        result.paste(images[1], (w, 0))
    elif len(images) == 3:
        result = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
        result.paste(images[0], (0, 0))
        result.paste(images[1], (w, 0))
        result.paste(images[2], (0, h))
    else:
        result = Image.new("RGB", (w * 2, h * 2))
        result.paste(images[0], (0, 0))
        result.paste(images[1], (w, 0))
        result.paste(images[2], (0, h))
        if len(images) >= 4:
            result.paste(images[3], (w, h))

    return result


# ====================== 独立健康检查服务器 ======================


class HealthCheckHandler(BaseHTTPRequestHandler):
    """独立的健康检查和打断HTTP处理器，运行在单独线程中"""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = json.dumps(
                {
                    "status": "healthy",
                    "message": "服务正常 (C++ backend)",
                    "backend": "cpp",
                }
            )
            self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/omni/break":
            print("======= [独立线程] 收到快速打断指令 =======", flush=True)
            config.is_breaking = True
            print("[独立线程] is_breaking 已设置为 True", flush=True)

            cpp_break_success = False
            if config.CPP_SERVER_URL and config.http_client:
                try:
                    # 使用同步 requests（在独立线程中）
                    break_resp = requests.post(
                        f"{config.CPP_SERVER_URL}/v1/stream/break",
                        json={"reason": "user_interrupt_from_health_thread"},
                        timeout=5.0,
                    )
                    if break_resp.status_code == 200:
                        print(
                            f"[独立线程] C++ 生成已中止: {break_resp.json()}",
                            flush=True,
                        )
                        cpp_break_success = True
                    else:
                        print(
                            f"[独立线程] C++ break 调用失败: {break_resp.status_code}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[独立线程] C++ break 调用异常: {e}", flush=True)

            self._send_json(
                200,
                {
                    "success": True,
                    "message": "当前轮对话已打断",
                    "state": "break",
                    "cpp_break": cpp_break_success,
                },
            )

        elif self.path == "/omni/stop":
            print("======= [独立线程] 收到快速停止指令 =======", flush=True)
            config.is_breaking = True

            cpp_break_success = False
            if config.CPP_SERVER_URL and config.http_client:
                try:
                    break_resp = requests.post(
                        f"{config.CPP_SERVER_URL}/v1/stream/break",
                        json={"reason": "session_stop_from_health_thread"},
                        timeout=5.0,
                    )
                    if break_resp.status_code == 200:
                        print(
                            f"[独立线程] C++ 生成已中止 (stop): {break_resp.json()}",
                            flush=True,
                        )
                        cpp_break_success = True
                    else:
                        print(
                            f"[独立线程] C++ break 调用失败 (stop): {break_resp.status_code}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[独立线程] C++ break 调用异常 (stop): {e}", flush=True)

            self._send_json(
                200,
                {
                    "success": True,
                    "message": "会话已停止",
                    "state": "session_stop",
                    "cpp_break": cpp_break_success,
                },
            )

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def _send_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def start_health_server(port: int):
    """在独立线程中启动健康检查和打断服务器"""
    health_port = port + 1
    server = HTTPServer(("0.0.0.0", health_port), HealthCheckHandler)
    print(f"独立健康检查/打断服务器已启动: http://0.0.0.0:{health_port}", flush=True)
    print("  - GET  /health     - 健康检查", flush=True)
    print("  - POST /omni/break - 快速打断", flush=True)
    print("  - POST /omni/stop  - 快速停止", flush=True)
    server.serve_forever()


# ====================== 服务注册 ======================


def get_local_ip() -> str:
    """获取本机 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def register_service_node(port: int, duplex_mode: bool):
    """注册服务节点到调度中心"""
    if not config.REGISTER_URL:
        print("跳过服务注册（未配置 REGISTER_URL）", flush=True)
        return

    try:
        url = f"{config.REGISTER_URL}/api/inference/register"
        local_ip = get_local_ip()
        model_type = "duplex" if duplex_mode else "simplex"
        data = {
            "ip": local_ip,
            "port": port,
            "model_port": port,
            "model_type": model_type,
            "session_type": "release",
            "service_name": "o45-cpp",
        }
        print(f"正在注册服务节点: url={url}, data={data}", flush=True)
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            print(f"服务节点注册成功: {response.text}", flush=True)
        else:
            print(
                f"服务节点注册失败: HTTP {response.status_code}, 响应: {response.text}",
                flush=True,
            )
    except Exception as e:
        import traceback

        print(f"服务节点注册异常: {e}", flush=True)
        traceback.print_exc()


# ====================== 输出目录管理 ======================


def reset_output_dir():
    """启动时重置 output 目录"""
    if os.path.exists(config.CPP_OUTPUT_DIR):
        try:
            shutil.rmtree(config.CPP_OUTPUT_DIR)
            print(f"[启动清理] 已删除 output 目录: {config.CPP_OUTPUT_DIR}", flush=True)
        except Exception as e:
            print(f"[启动清理] 删除 output 目录失败: {e}", flush=True)
    try:
        os.makedirs(config.CPP_OUTPUT_DIR, exist_ok=True)
        print(f"[启动清理] 已创建 output 目录: {config.CPP_OUTPUT_DIR}", flush=True)
    except Exception as e:
        print(f"[启动清理] 创建 output 目录失败: {e}", flush=True)


def clear_output_subfolders():
    """清空 output 目录下每个子文件夹的内容"""
    if not os.path.exists(config.CPP_OUTPUT_DIR):
        print(f"[清空输出] output 目录不存在: {config.CPP_OUTPUT_DIR}", flush=True)
        return

    cleared_count = 0
    for item in os.listdir(config.CPP_OUTPUT_DIR):
        item_path = os.path.join(config.CPP_OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                try:
                    if os.path.isdir(sub_item_path):
                        shutil.rmtree(sub_item_path)
                    else:
                        os.remove(sub_item_path)
                    cleared_count += 1
                except Exception as e:
                    print(f"[清空输出] 删除失败 {sub_item_path}: {e}", flush=True)

    print(
        f"[清空输出] 已清空 {config.CPP_OUTPUT_DIR} 下的子文件夹内容 (删除 {cleared_count} 项)",
        flush=True,
    )


# ====================== C++ 进程管理 ======================


def start_cpp_server(model_dir: str, gpu_devices: str, port: int):
    """启动 C++ llama-server"""
    llamacpp_root = os.path.abspath(config.LLAMACPP_ROOT)

    # 查找 llama-server 可执行文件
    server_bin = None
    candidates = [
        os.path.join(llamacpp_root, "build/bin/llama-server"),
        os.path.join(llamacpp_root, "build/bin/Release/llama-server.exe"),
        os.path.join(llamacpp_root, "build/bin/llama-server.exe"),
    ]
    if platform.system() == "Darwin":
        candidates.append(
            os.path.join(
                llamacpp_root, "build-arm64-apple-clang-release/bin/llama-server"
            )
        )
    elif platform.system() != "Windows":
        candidates.append(
            os.path.join(llamacpp_root, "build-x64-linux-cuda-release/bin/llama-server")
        )

    for c in candidates:
        if os.path.exists(c):
            server_bin = c
            break
    if server_bin is None:
        server_bin = candidates[0]

    if os.path.isabs(model_dir):
        model_path = os.path.join(model_dir, config.DEFAULT_LLM_MODEL)
    else:
        model_path = os.path.join(model_dir, config.DEFAULT_LLM_MODEL)

    if not os.path.exists(server_bin):
        raise RuntimeError(f"C++ server binary not found: {server_bin}")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    env = os.environ.copy()

    if platform.system() == "Darwin":
        dyld_paths = [os.path.dirname(server_bin), env.get("DYLD_LIBRARY_PATH", "")]
        env["DYLD_LIBRARY_PATH"] = ":".join(p for p in dyld_paths if p)
        print("Platform: macOS (Metal)", flush=True)
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
        cuda_env_path = os.environ.get("CUDA_LIB_PATH", "/usr/local/cuda/lib64")
        cuda_lib_paths = [
            cuda_env_path,
            llamacpp_root + "/build/bin",
            "/usr/lib/x86_64-linux-gnu",
            env.get("LD_LIBRARY_PATH", ""),
        ]
        env["LD_LIBRARY_PATH"] = ":".join(p for p in cuda_lib_paths if p)
        print("Platform: Linux (CUDA)", flush=True)
        print(f"CUDA_VISIBLE_DEVICES={gpu_devices}", flush=True)

    cmd = [
        server_bin,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model",
        model_path,
        "--ctx-size",
        str(config.DEFAULT_CTX_SIZE),
        "--n-gpu-layers",
        str(config.DEFAULT_N_GPU_LAYERS),
        "--repeat-penalty",
        "1.05",
        "--temp",
        "0.7",
    ]

    print(f"启动 C++ llama-server: {' '.join(cmd)}", flush=True)

    config.cpp_server_process = subprocess.Popen(
        cmd,
        env=env,
        cwd=llamacpp_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    # 启动日志读取线程
    def log_reader():
        try:
            assert config.cpp_server_process
            for line in config.cpp_server_process.stdout:
                print(f"[CPP] {line.rstrip()}", flush=True)
        except Exception as e:
            print(f"[CPP log_reader] 异常: {e}", flush=True)

    log_thread = threading.Thread(target=log_reader, daemon=True)
    log_thread.start()

    # 等待服务器启动
    max_wait = 180
    for i in range(max_wait):
        try:
            resp = requests.get(
                f"http://{config.CPP_SERVER_HOST}:{port}/health", timeout=2
            )
            if resp.status_code == 200:
                print(f"C++ llama-server 启动成功 (等待 {i + 1} 秒)", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError(f"C++ llama-server 启动超时 ({max_wait}秒)")


def stop_cpp_server():
    """停止 C++ llama-server"""
    if config.cpp_server_process:
        print("停止 C++ llama-server...", flush=True)
        config.cpp_server_process.terminate()
        try:
            config.cpp_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            config.cpp_server_process.kill()
        config.cpp_server_process = None
