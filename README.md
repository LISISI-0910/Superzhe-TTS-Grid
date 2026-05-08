# Superzhe TTS Grid

CosyVoice3 语音合成 + Whisper 语音识别 API 服务。支持零样本语音克隆、流式生成、字幕转录。

## 特性

- **零样本克隆** — 3 秒参考音频即可克隆任意说话人
- **流式生成** — Opus-OGG 流式输出，首块延迟 < 500ms
- **指令控制** — 通过自然语言控制语气、情感、语速
- **语音识别** — faster-whisper large-v3-turbo，支持 SRT/VTT 字幕
- **自适应并发** — 启动时基准测试动态确定 GPU 并发上限
- **横向扩展** — 完全无状态，增减实例即可扩缩容

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.1+
- NVIDIA GPU（建议 8GB+ 显存）

### 1. 克隆项目

```bash
git clone https://github.com/LISISI-0910/Superzhe-TTS-Grid.git
cd Superzhe-TTS-Grid
```

### 2. 一键部署

```bash
python setup.py
```

这会自动完成：系统依赖安装 → pip 依赖安装 → 下载 TTS/ASR 模型 → 安装 ttsfrd。

### 3. 启动服务

```bash
python run.py
```

默认监听 `0.0.0.0:6006`，Swagger 文档在 `http://localhost:6006/docs`。

```bash
python run.py --port 9000       # 指定端口
python run.py --no-vllm         # 关闭 vLLM
python run.py --no-fp16         # 关闭半精度
```

## API 概要

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/tts/extract` | POST | 提取说话人向量（需提供音频+文本） |
| `/api/v1/tts/enroll` | POST | 一键注册说话人（仅需音频） |
| `/api/v1/tts/generate` | POST | 流式语音生成（Opus-OGG） |
| `/api/v1/asr/transcribe` | POST | 语音识别（转录+字幕） |
| `/api/v1/health` | GET | 健康检查 |

详细文档见 [API-DOC.md](API-DOC.md)。

### 快速测试

```bash
# 1. 注册说话人
curl -X POST http://localhost:6006/api/v1/tts/enroll \
  -F "audio=@my_voice.wav"

# → {"speaker_b64": "gASVnwAAAAA..."}

# 2. 生成语音
curl -X POST http://localhost:6006/api/v1/tts/generate \
  -F "text=今天天气真好，我们出去玩吧。" \
  -F "speaker_b64=gASVnwAAAAA..." \
  -o output.ogg
```

## 配置

所有配置集中在 `config.yaml`，支持环境变量覆盖：

```yaml
server:
  host: "0.0.0.0"
  port: 6006

acceleration:
  trt: true        # TensorRT 加速
  vllm: true       # vLLM 推理
  fp16: true       # FP16 半精度

limits:
  tts_text_max: 20000       # 合成文本最大字数
  audio_size_max_mb: 50     # 上传音频最大 MB
  audio_duration_max_s: 30  # 说话人音频最大秒数

generation:
  hard_timeout_s: 3600      # 生成硬超时
  idle_timeout_s: 120       # 无产出空闲超时
```

环境变量覆盖（大写加 `TTS_` 前缀）：`TTS_PORT=9000`、`TTS_TRT=0` 等。

## 并发测试

```bash
python test.py -n 6           # 6 并发测试
python test.py -n 4 --text "自定义文本"
```

输出每个请求的耗时、加速比，音频保存到 `test_outputs/`。

## 目录结构

```
├── run.py                  # 启动脚本
├── setup.py                # 一键部署
├── config.yaml             # 统一配置
├── API-DOC.md              # 接口文档
├── test.py                 # 并发测试工具
├── cosyvoice/              # TTS 引擎
│   ├── engine.py           #   CosyVoiceEngine
│   ├── model.py            #   CosyVoice3Model
│   └── frontend.py         #   前端处理
├── whisper_asr/            # ASR 引擎
│   └── asr.py              #   WhisperASR
├── server/                 # API 服务
│   ├── main.py             #   FastAPI 路由
│   ├── manager.py          #   生命周期 + 并发控制
│   ├── audio.py            #   音频处理
│   ├── config.py           #   配置加载
│   └── schemas.py          #   数据模型
├── benchmark/              # 基准测试音频
├── pretrained_models/      # 模型文件（.gitignore）
└── third_party/            # 第三方依赖
```

## License

Apache 2.0
