## Quick Start

### 前提

安装 cmake, uv

### 本地部署(基于 `MiniCPM-o-4_5`)
1. 下载任意 MiniCPM-o-4_5-gguf 模型，`llm_server/models`文件内容如下
    ```txt
    .
    └── openbmb
        └── MiniCPM-o-4_5-gguf
            ├── audio
            │   └── MiniCPM-o-4_5-audio-F16.gguf
            ├── MiniCPM-o-4_5-Q4_K_M.gguf
            ├── README.md
            ├── token2wav-gguf
            │   ├── encoder.gguf
            │   ├── flow_extra.gguf
            │   ├── flow_matching.gguf
            │   ├── hifigan2.gguf
            │   └── prompt_cache.gguf
            ├── tts
            │   ├── MiniCPM-o-4_5-projector-F16.gguf
            │   └── MiniCPM-o-4_5-tts-F16.gguf
            └── vision
                ├── coreml_minicpmo45_vit_all_f16.mlmodelc
                │   ├── analytics
                │   │   └── coremldata.bin
                │   ├── coremldata.bin
                │   ├── metadata.json
                │   ├── model.mil
                │   └── weights
                │       └── weight.bin
                └── MiniCPM-o-4_5-vision-F16.gguf
    ```


2. 启动 `llm_server` 服务，基于 FastAPI 包装的多模态模型接口层
    ```sh
    # clone llama.cpp-omni for stream
    cd llm_server
    git clone -b feat/stream_with_text https://github.com/21ess/llama.cpp-omni
    make build-cpp # 编译 llama.cpp-omni

    uv sync # 安装依赖

    make run-llm-server # 启动 llm 服务

    ## Optional 启动 llm_server 测试脚本
    uv run ./scripts/infer_audio.py # 
    ```
    > 由于本地部署，注册服务启动默认失败

3.  `backend`服务

    1. 配置环境变量，**`.env` 文件模板**

        ```text
        # BiliLive 配置
        BILI_ID_CODE=xxx    # 
        BILI_APP_ID=xxx     #
        BILI_KEY=xxx        # bilibili 开放平台申请 
        BILI_SECRET=xxx     # bilibili 开放平台申请
        ```

    2. 启动服务
        ```sh
        uv run -m backend
        ```


## Development

```sh
uv run ruff format .
```