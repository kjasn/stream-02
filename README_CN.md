# stream-02

`stream-02` 是一个面向直播场景的开源互动助手。它的定位类似主播身边的
“二号机”：在直播过程中持续读取观众消息、理解上下文，并帮助生成可播报的
回复。

> [!WARNING]
> 项目仍处于早期设计阶段。
>
> 当前文档大多由 AI 辅助生成，后续还需要人工持续整理和重写。

## 核心场景

在直播中，`stream-02` 能够完成以下工作：

1. 从直播间读取观众评论和弹幕。
2. 捕获并转写主播的语音。
3. 将有用的上下文发送给 LLM。
4. 为观众或主播生成回复。
5. 将回复文本转换为语音。
6. 把语音回复接入直播工作流。

## 架构

项目分为 Go 核心程序和 Python 模型服务两部分。

Go 核心负责长期运行的应用逻辑：

- 平台弹幕适配器
- OCR 适配器编排
- 事件总线和消息路由
- 对话状态
- LLM provider 适配器
- TTS provider 适配器
- 配置与插件式扩展点

Python 服务负责本地模型推理：

- 基于 PyTorch 的 OCR
- 可选的本地 STT
- 模型加载与缓存
- CPU/GPU 设备选择
- 面向 Go 核心的结构化推理结果

第一版实现应保持这两部分松耦合。Go 进程通过本地 HTTP API 与 Python 模型
服务通信，Python 侧预计使用 FastAPI 暴露接口。

## 主要技术栈

- **Go**：主应用运行时、平台适配器、事件总线、provider 客户端、配置与编排。
- **Python**：本地 AI 模型运行时，用于 OCR 和可选 STT。
- **PyTorch**：OCR 模型推理，以及后续本地模型实验。
- **FastAPI**：为 Go 核心提供本地 Python 服务 API。
- **LLM provider adapters**：可插拔的托管或本地 LLM 客户端。
- **TTS provider adapters**：可插拔的语音回复生成客户端。
- **Platform API/WebSocket adapters**：在平台集成可用时，提供高质量弹幕接入。
- **OCR screen reader adapter**：面向难以直接集成的平台，作为通用兜底方案。

## 项目流程

观众消息流程：

```text
Live Stream Screen / Platform API
        |
        v
Danmaku Sources
  - Platform API/WebSocket adapter
  - OCR screen reader adapter
        |
        v
Go Core Event Bus
        |
        +--> DanmakuEvent
        +--> AnchorVoiceTranscriptEvent
        |
        v
Conversation Engine
        |
        v
LLM Provider
        |
        v
TTS Provider
        |
        v
Voice Reply Output
```

Python 模型服务流程：

```text
Go Core
  |
  | local HTTP
  v
Python Model Service
  - OCR with PyTorch
  - optional local STT
  - model loading and device selection
  |
  v
Structured inference result
```

语音互动流程：

```text
Anchor Microphone
        |
        v
Voice Activity Detection
        |
        v
Speech-to-Text
        |
        v
Text Context for LLM
        |
        v
LLM Reply
        |
        v
Text-to-Speech
        |
        v
Voice Reply Output
```

## 弹幕输入策略

`stream-02` 将每一种弹幕来源都视为适配器。适配器的职责是输出统一格式的
标准化事件。

当平台 API 或 WebSocket 能够使用时，应优先采用它们，因为它们可以提供更
结构化的数据，例如用户名、时间戳、徽章、礼物事件、舰长/会员状态以及消息
元数据。

OCR 是重要的兜底路径。对于直接集成成本高、不稳定、非公开或暂时不可用的
平台，OCR 允许 `stream-02` 通过读取直播画面中可见的评论内容来生成标准化
弹幕事件，并附带置信度信息。

因此 v1 可以同时支持两条路径：

```text
Platform API/WebSocket -> DanmakuEvent
Screen OCR             -> DanmakuEvent
```

## 语音策略

v1 默认采用以文本为中心的语音路径：

```text
audio -> STT -> text -> LLM -> text -> TTS -> audio
```

相比直接的 audio-to-audio 流程，这条路径更容易调试、审核、定制和替换
provider。实时语音 API 可以在后续作为高级模式加入，但不应成为第一版的默认
设计目标。

## 为什么选择 Go + Python？

Go 适合作为主应用运行时：部署简单、并发能力内建，并且适合长期运行的服务。
大多数平台连接器、provider 客户端和事件路由逻辑都应放在 Go 侧。

Python 则更适合本地模型推理。OCR 和 STT 生态在 Python 中最成熟，PyTorch
模型支持也更完整。将 Python 放在独立的本地服务中，可以避免把 Go 应用逻辑
与 Python 运行时和包管理细节强耦合。

这种拆分也方便不同方向的贡献者独立工作：

- Go 贡献者可以改进平台适配器、事件流、LLM/TTS provider 和应用行为。
- Python 贡献者可以改进 OCR、STT、模型加载和推理性能。

## 当前方向

- 构建一个 local-first 的开源工具。
- 使用 Go 作为核心应用运行时。
- 使用 Python + FastAPI + PyTorch 承担本地 OCR 和可选 STT。
- 将 OCR 视为一等弹幕来源，而不是临时方案。
- 将平台 API 和 WebSocket 集成保留为更高质量的适配器。
- 默认采用 STT -> LLM -> TTS 的语音互动链路。
- 在核心流程更清晰之前，暂不决定 UI 技术栈。

## 路线图

- 定义弹幕、主播语音转写、LLM 回复和 TTS 输出的标准化事件类型。
- 构建 Go 事件总线和适配器接口。
- 构建 Python OCR 服务 API。
- 添加第一个基于 OCR 的弹幕来源。
- 添加一个平台 API/WebSocket 弹幕适配器。
- 添加至少一个 LLM provider 和一个 TTS provider 适配器。
- 创建最小可用的本地配置格式。
- 在核心管线可用后，添加一个简单的操作员 UI。
