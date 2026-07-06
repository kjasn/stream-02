"""Entry point: python -m llm_server"""

import os
import sys

import uvicorn

from .inference_server import config as cfg
from .inference_server.server import app


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="MiniCPMO C++ HTTP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8060, help="服务器端口")
    parser.add_argument(
        "--llamacpp-root",
        type=str,
        default=cfg.LLAMACPP_ROOT,
        help="llama.cpp-omni 根目录",
    )
    parser.add_argument(
        "--model-dir", type=str, default=cfg.DEFAULT_MODEL_DIR, help="GGUF 模型目录"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=cfg.DEFAULT_LLM_MODEL,
        help="LLM 模型文件名（可选，默认自动检测）",
    )
    parser.add_argument(
        "--gpu-devices", type=str, default=cfg.DEFAULT_GPU_DEVICES, help="GPU 设备"
    )
    parser.add_argument("--duplex", action="store_true", help="默认使用双工模式")
    parser.add_argument("--simplex", action="store_true", help="默认使用单工模式")
    parser.add_argument("--output-dir", type=str, default=None, help="C++ 输出目录")
    parser.add_argument(
        "--vision-backend",
        type=str,
        default=cfg.VISION_BACKEND,
        choices=["metal", "coreml"],
        help="视觉编码器后端",
    )

    args = parser.parse_args()

    # 验证 LLAMACPP_ROOT
    if not args.llamacpp_root or not os.path.isdir(args.llamacpp_root):
        print(f"错误: LLAMACPP_ROOT 目录不存在: {args.llamacpp_root}", flush=True)
        sys.exit(1)
    cfg.LLAMACPP_ROOT = args.llamacpp_root

    # 验证 MODEL_DIR
    if not args.model_dir or not os.path.isdir(args.model_dir):
        print(f"错误: MODEL_DIR 目录不存在: {args.model_dir}", flush=True)
        sys.exit(1)
    cfg.DEFAULT_MODEL_DIR = args.model_dir

    # 自动检测或验证 LLM 模型
    llm_model = args.llm_model
    if not llm_model:
        llm_model = cfg.auto_detect_llm_model(args.model_dir)
        if llm_model:
            print(f"自动检测到 LLM 模型: {llm_model}", flush=True)
        else:
            print(f"错误: 在 {args.model_dir} 中未找到 LLM GGUF 模型", flush=True)
            sys.exit(1)
    else:
        llm_path = os.path.join(args.model_dir, llm_model)
        if not os.path.exists(llm_path):
            print(f"错误: LLM 模型文件不存在: {llm_path}", flush=True)
            sys.exit(1)
    cfg.DEFAULT_LLM_MODEL = llm_model

    # 设置参考音频路径
    if not cfg.FIXED_TIMBRE_PATH or not os.path.exists(cfg.FIXED_TIMBRE_PATH):
        cfg.FIXED_TIMBRE_PATH = os.path.join(
            args.llamacpp_root, "tools/omni/assets/default_ref_audio.wav"
        )

    # 视觉编码器后端
    if args.vision_backend == "coreml":
        vision_coreml = os.path.join(
            args.model_dir, "vision", "coreml_minicpmo45_vit_all_f16.mlmodelc"
        )
        if os.path.exists(vision_coreml):
            cfg.VISION_BACKEND = "coreml"
            print("Vision backend: CoreML/ANE", flush=True)
        else:
            print("CoreML model not found, falling back to Metal", flush=True)
            cfg.VISION_BACKEND = "metal"
    else:
        cfg.VISION_BACKEND = "metal"
        print("Vision backend: Metal (GPU)", flush=True)

    # 确定模式
    default_duplex_mode = args.duplex and not args.simplex

    # 设置输出目录
    if args.output_dir:
        cfg.CPP_OUTPUT_DIR = args.output_dir
    else:
        cfg.CPP_OUTPUT_DIR = os.path.join(
            args.llamacpp_root, f"tools/omni/output_{args.port}"
        )
    os.makedirs(cfg.CPP_OUTPUT_DIR, exist_ok=True)

    # 保存到 app.state
    app.state.port = args.port
    app.state.model_dir = args.model_dir
    app.state.gpu_devices = args.gpu_devices
    app.state.default_duplex_mode = default_duplex_mode
    app.state.output_dir = cfg.CPP_OUTPUT_DIR

    cfg.init_from_args(args)

    mode_name = "双工" if default_duplex_mode else "单工"
    print(f"\n{'=' * 60}", flush=True)
    print("MiniCPM-o C++ HTTP Server", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  HTTP 地址: http://{args.host}:{args.port}", flush=True)
    print(f"  健康检查: http://{args.host}:{args.port + 1}/health", flush=True)
    print(f"  默认模式: {mode_name}", flush=True)
    print(f"  LLAMACPP_ROOT: {args.llamacpp_root}", flush=True)
    print(f"  MODEL_DIR:     {args.model_dir}", flush=True)
    print(f"  LLM_MODEL:     {llm_model}", flush=True)
    print(f"  OUTPUT_DIR:    {cfg.CPP_OUTPUT_DIR}", flush=True)
    print(f"  REF_AUDIO:     {cfg.FIXED_TIMBRE_PATH}", flush=True)
    print(f"  VISION_BACKEND: {cfg.VISION_BACKEND}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
