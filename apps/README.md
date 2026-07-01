## Quick Start

```sh
# 本地部署，下载任意 MiniCPM-o-4_5-gguf 模型
# path: llm_server/models/openbmb/MiniCPM-o-4_5-gguf


# clone llama.cpp-omni for stream
cd llm_server
git clone -b feat/stream_with_text https://github.com/21ess/llama.cpp-omni
make build-cpp # 编译 llama.cpp-omni

uv sync # 安装依赖

make run-llm-server # 启动 llm 服务

## optional 
uv run ./scripts/infer_audio.py # 
```


**`.env` 文件模板**


```text
# BiliLive 配置
BILI_ID_CODE=xxx    # 
BILI_APP_ID=xxx     #
BILI_KEY=xxx        # bilibili 开放平台申请 
BILI_SECRET=xxx     # bilibili 开放平台申请
```