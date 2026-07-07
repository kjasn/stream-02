# stream-02

## Quick Start

### 前提

安装 cmake, uv

### 本地部署(基于 `MiniCPM-o-4_5`)

1. 下载任意 MiniCPM-o-4_5-gguf 模型，`llm_server/models`文件内容如下

   ```sh
    huggingface-cli download openbmb/MiniCPM-o-4_5-gguf \
    --include "MiniCPM-o-4_5-Q4_K_M.gguf" "vision/*" "audio/*" "tts/*" "token2wav-gguf/*" "*.md" ".git*" \
    --local-dir ./models/openbmb/MiniCPM-o-4_5-gguf
    
    # 下载完毕后，文件结构如下
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
    # 第一次使用（初始化子模块）
    git submodule update --init --recursive
    # 后续更新（拉取主仓库后同步子模块）
    git submodule update --recursive
    # 可选：跟踪 .gitmodules 中设置的分支更新到最新提交
    # git submodule update --remote --recursive

    make build-cpp # 编译 llama.cpp-omni

    uv sync # 安装依赖

    make run-llm-server # 启动 llm 服务
    ```
 3. 启动 `backend` 服务
    1. 配置环境变量，**`base.yaml` 文件模板 `/backend/config/default.yaml`**，通过 B 站开放平台获取

        ```yaml
        # BiliLive 配置
        bilibili:
            enabled: true
            id_code: '' # required
            app_id: # required
            key: '' # required
            secret: '' # required
            reconnect_delay: 5.0
            max_reconnect_attempts: 0
        ```

    2. 启动服务
        ```sh
        uv run -m backend
        ```


## Development

代码风格
```sh
uv run ruff format .
```
