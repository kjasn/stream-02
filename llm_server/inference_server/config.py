"""
MiniCPMO C++ HTTP Server - 配置模块
管理全局状态、环境变量、路径常量
"""

import os
import sys
import threading
import socket
from pathlib import Path
from typing import Optional, Dict, Any

from httpx import AsyncClient

# ====================== 路径常量 ======================
# 默认指向 apps/llm-server/ 下的同级目录
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent  # apps/llm-server/

# C++ 服务器配置
CPP_SERVER_HOST = "127.0.0.1"
CPP_SERVER_PORT = None  # 在 lifespan 中根据 Python 端口设置
CPP_SERVER_URL = None  # 在 lifespan 中根据 Python 端口设置

# 模型配置 - 可通过环境变量覆盖
LLAMACPP_ROOT = os.environ.get("LLAMACPP_ROOT", str(_DEFAULT_ROOT / "llama.cpp-omni"))
DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_DIR", str(_DEFAULT_ROOT / "models" / "openbmb" / "MiniCPM-o-4_5-gguf")
)
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "")
DEFAULT_GPU_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
DEFAULT_CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
DEFAULT_N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))

# 固定音色文件（用于 voice cloning）
_DEFAULT_REF_AUDIO = str(_DEFAULT_ROOT / "assets" / "default_ref_audio.wav")
FIXED_TIMBRE_PATH = os.environ.get("REF_AUDIO", _DEFAULT_REF_AUDIO)

# 视觉编码器后端: "metal"(默认，GPU) 或 "coreml"(ANE加速，macOS专用)
VISION_BACKEND = os.environ.get("VISION_BACKEND", "metal")

# Token2Wav device: "gpu:1"(默认，GPU加速) 或 "cpu"(节省GPU显存)
TOKEN2WAV_DEVICE = os.environ.get("TOKEN2WAV_DEVICE", "gpu:0")

# GPU 内存监控相关
GPU_MEMORY_CHECK = int(os.environ.get("GPU_MEMORY_CHECK", "0"))


def auto_detect_llm_model(model_dir: str) -> str:
    """自动从模型目录检测 LLM GGUF 文件

    优先级：Q4_K_M > Q8_0 > F16 > 其他 .gguf 文件
    """
    import glob

    if not model_dir or not os.path.isdir(model_dir):
        return ""

    priority_patterns = [
        "*Q4_K_M*.gguf",
        "*Q4_K_S*.gguf",
        "*Q8_0*.gguf",
        "*Q5_K_M*.gguf",
        "*F16*.gguf",
    ]

    for pattern in priority_patterns:
        matches = glob.glob(os.path.join(model_dir, pattern))
        root_matches = [m for m in matches if os.path.dirname(m) == model_dir]
        if root_matches:
            return os.path.basename(sorted(root_matches)[0])

    all_gguf = glob.glob(os.path.join(model_dir, "*.gguf"))
    if all_gguf:
        llm_candidates = [
            f
            for f in all_gguf
            if not any(
                x in os.path.basename(f).lower()
                for x in ["audio", "vision", "tts", "projector"]
            )
        ]
        if llm_candidates:
            return os.path.basename(sorted(llm_candidates)[0])

    return ""


# 临时文件目录
TEMP_DIR = os.environ.get(
    "LLM_SERVER_TEMP_DIR",
    os.path.join(os.path.dirname(__file__), "temp_streaming_prefill"),
)

# C++ llama-server 输出目录
DEFAULT_CPP_OUTPUT_DIR = os.path.join(LLAMACPP_ROOT, "tools/omni/output")
CPP_OUTPUT_DIR = DEFAULT_CPP_OUTPUT_DIR


def _get_default_register_url():
    """获取默认注册地址（本机 IP:8025）"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return f"http://{local_ip}:8025"
    except Exception:
        return "http://127.0.0.1:8025"


REGISTER_URL = os.environ.get("REGISTER_URL", _get_default_register_url())

# ====================== 全局状态 ======================
cpp_server_process: Optional[Any] = None  # subprocess.Popen
current_msg_type: Optional[int] = None  # 1=audio, 2=video/omni
current_duplex_mode: bool = False
current_high_quality_mode: bool = False
current_high_fps_mode: bool = False
current_active_session_id: Optional[str] = None
current_request_counter: int = 0
current_round_number: int = 0
session_lock = threading.Lock()
model_state_initialized: bool = False
pending_prefill_data: Optional[dict] = None
is_breaking: bool = False
health_server_thread: Optional[threading.Thread] = None
# C++ 重启标志
cpp_restarting: bool = False

# 高刷模式子图缓存
high_fps_subimage_cache: Dict[int, Dict[int, Any]] = {}  # PIL.Image
high_fps_cache_lock = threading.Lock()
high_fps_pending_audio: Dict[int, tuple] = {}
high_fps_audio_lock = threading.Lock()

# 双工模式全局状态
global_sent_wav_count: int = 0
global_parsed_line_count: int = 0
global_parsed_texts: list = []
global_text_send_idx: int = 0
global_sent_wav_files: set = set()

# WAV 发送时序日志
WAV_TIMING_LOG_PATH = os.path.join(
    os.environ.get("LLM_SERVER_OUTPUT_DIR", CPP_OUTPUT_DIR), "wav_timing.log"
)
wav_timing_log_file: Optional[Any] = None
last_wav_send_time: Optional[float] = None

# HTTP 客户端 (由 lifespan 初始化)
http_client: Optional[AsyncClient] = None  # httpx.AsyncClient


def init_from_args(args):
    """根据命令行参数初始化配置"""
    global CPP_SERVER_PORT, CPP_SERVER_URL, CPP_OUTPUT_DIR

    # 设置输出目录
    if hasattr(args, "output_dir") and args.output_dir:
        CPP_OUTPUT_DIR = args.output_dir

    # 设置模型目录
    if hasattr(args, "model_dir") and args.model_dir:
        global DEFAULT_MODEL_DIR
        DEFAULT_MODEL_DIR = args.model_dir

    # 设置 duplication mode
    if hasattr(args, "duplex") and args.duplex:
        global current_duplex_mode
        current_duplex_mode = True

    # 端口配置
    if hasattr(args, "port"):
        CPP_SERVER_PORT = args.port + 10000
        CPP_SERVER_URL = f"http://{CPP_SERVER_HOST}:{CPP_SERVER_PORT}"


def load_dotenv_if_needed():
    """加载 .env 文件（如果存在）"""
    try:
        from dotenv import load_dotenv

        _env_path = _DEFAULT_ROOT.parent / ".env"
        if _env_path.exists():
            load_dotenv(_env_path)
    except ImportError:
        pass


# 自动加载 .env
load_dotenv_if_needed()
