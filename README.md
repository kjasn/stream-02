# stream-02

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
   # 第一次使用 init，后续更新只需要 update 即可
   git submodule update --init --recursive
   # git submodule update --remote

   make build-cpp # 编译 llama.cpp-omni

   uv sync # 安装依赖

   make run-llm-server # 启动 llm 服务

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

```sh
uv run ruff format .
```
