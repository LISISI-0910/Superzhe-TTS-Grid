# CosyVoice3 引擎 API 文档

> 给后端开发者的引擎调用指南  
> 项目: [CosyVoice-3-TTS-API-service](https://github.com/LISISI-0910/CosyVoice-3-TTS-API-service)

---

## 目录

1. [整体架构](#1-整体架构)
2. [启动参数](#2-启动参数)
3. [认证机制（RSA 非对称签名）](#3-认证机制rsa-非对称签名)
4. [API 端点](#4-api-端点)
   - [4.1 POST /extract — 提取说话人向量](#41-post-extract--提取说话人向量)
   - [4.2 POST /tts — 生成语音](#42-post-tts--生成语音)
   - [4.3 GET /health — 健康检查](#43-get-health--健康检查)
5. [完整调用流程（后端视角）](#5-完整调用流程后端视角)
6. [音频编解码细节](#6-音频编解码细节)
7. [服务端统计与监控](#7-服务端统计与监控)
8. [常见问题](#8-常见问题)

---

## 1. 整体架构

```
┌──────────────────────┐      RSA-SHA256 签名      ┌─────────────────────┐
│                      │  ─────────────────────→    │                     │
│  后端服务 (调用方)     │  POST /extract            │  CosyVoice3 引擎    │
│                      │  X-Signature: <base64>     │  FastAPI :6006      │
│  · 持有 private_key  │  POST /tts                 │                     │
│  · 发送 base64 音频   │  X-Signature: <base64>     │  · 持有 public_key  │
│  · 接收 Opus 音频流   │  ←─────────────────────    │  · 验签             │
│                      │  audio/ogg (Opus)          │                     │
└──────────────────────┘                            └─────────────────────┘
```

### 引擎内部组件

```
run.py (FastAPI 入口)
 ├─ auth_middleware()         ← RSA 验签中间件
 ├─ POST /extract             提取说话人向量
 ├─ POST /tts                 生成语音（流式/非流式）
 ├─ GET /health               健康检查 + 基准报告
 └─ init_engine()             引擎初始化 + 预热 + 基准测试

cosyvoice/engine.py (核心引擎)
 ├─ CosyVoiceEngine.process()      统一入口
 │   ├─ mode='extract' → _process_extract()
 │   └─ mode='generate' → _process_generate()
 ├─ 模型加载: LLM(Qwen2) + Flow(DiT) + HiFiGAN
 └─ 加速: vLLM (LLM) / TensorRT (Flow) / FP16
```

---

## 2. 启动参数

```bash
python run.py [选项]

# 示例：默认启动（开启所有加速）
python run.py

# 示例：指定端口，关闭 vLLM
python run.py --port 6006 --no_vllm

# 示例：指定模型目录（调试用）
python run.py --model_dir /data/models/Fun-CosyVoice3-0.5B
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_dir` | str | `pretrained_models/Fun-CosyVoice3-0.5B` | 模型目录路径 |
| `--port` | int | 6006 | HTTP 监听端口 |
| `--host` | str | `0.0.0.0` | 监听地址 |
| `--no_trt` | 标志 | false | 关闭 TensorRT (Flow 加速) |
| `--no_vllm` | 标志 | false | 关闭 vLLM (LLM 加速) |
| `--no_fp16` | 标志 | false | 关闭 FP16 半精度 |

> **启动加速建议**：GPU 显存 ≥ 6GB 时使用默认值即可（同时开启 TRT + vLLM + FP16）

---

## 3. 认证机制（RSA 非对称签名）

### 3.1 原理

- **调用方（后端）**：持有 `private_key.pem`，对请求体签名
- **引擎**：持有 `public_key.pem`，验签
- 签名算法：**RSA-SHA256 + PKCS1v15 填充**
- 签名放在 HTTP Header `X-Signature` 中，值为 Base64 编码

### 3.2 安全特性

| 场景 | 结果 |
|------|------|
| 引擎被攻破，攻击者拿到公钥 | ❌ 无法伪造签名（公钥只能验签） |
| 调用方私钥泄露 | ❌ 需立即更换密钥对 |
| 多引擎部署 | ✅ 所有引擎共享同一公钥 |

### 3.3 放行规则

中间件对以下情况**跳过验签**：

| 条件 | 说明 |
|------|------|
| 公钥文件不存在 | 开发/调试模式 |
| 请求方法为 GET/HEAD/OPTIONS | `/health` 等读取操作 |

### 3.4 签名步骤（调用方实现）

```python
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64, json

def sign_request(body: dict) -> str:
    """对请求体进行 RSA-SHA256 签名，返回 Base64 字符串"""
    # 1. 请求体序列化（ensure_ascii=False 保证中文正确）
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    # 2. 加载私钥
    with open("private_key.pem", "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    # 3. 签名
    signature = private_key.sign(body_bytes, padding.PKCS1v15(), hashes.SHA256())

    # 4. Base64 编码
    return base64.b64encode(signature).decode()
```

### 3.5 验签步骤（引擎侧实现，仅供理解）

```python
def verify_signature(body: bytes, signature_b64: str) -> bool:
    public_key = serialization.load_pem_public_key(PUBLIC_KEY_PEM)
    signature = base64.b64decode(signature_b64)
    try:
        public_key.verify(signature, body, padding.PKCS1v15(), hashes.SHA256())
        return True
    except InvalidSignature:
        return False
```

### 3.6 请求示例（含签名）

```
POST /extract HTTP/1.1
Host: 127.0.0.1:6006
Content-Type: application/json
X-Signature: G7c3Xy9Qp2Rz5Ab8...

{"prompt_text": "...", "prompt_audio": "base64...", "voice_type": "zero_shot"}
```

---

## 4. API 端点

### 4.1 POST /extract — 提取说话人向量

从参考音频中提取说话人声纹特征向量，供后续 `/tts` 使用。

#### Request

```json
{
    "prompt_text": "这里是需要克隆的参考文本，要和音频内容一致",
    "prompt_audio": "UklGRiS...base64编码的音频数据...",
    "voice_type": "zero_shot"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt_text` | string | ✅ | 参考文本，需与音频内容匹配 |
| `prompt_audio` | string | ✅ | 音频文件 **Base64 编码**（支持 Opus/WAV/MP3/FLAC） |
| `voice_type` | string | 否 | `"zero_shot"`（默认）或 `"instruct2"` |

#### Response (200)

```json
{
    "status": "ok",
    "spk_vec": {
        "prompt_text_token": [50304, 50268, 23563, ...],
        "prompt_text_token_len": 37,
        "llm_prompt_speech_token": [1, 2, 3, ...],
        "llm_prompt_speech_token_len": 354,
        "flow_prompt_speech_token": [1, 2, 3, ...],
        "flow_prompt_speech_token_len": 354,
        "prompt_speech_feat": [[0.123, -0.456, ...], ...],
        "prompt_speech_feat_len": 354,
        "llm_embedding": [0.123, -0.456, ...],
        "flow_embedding": [0.123, -0.456, ...]
    }
}
```

> `spk_vec` 包含 8 个字段，其中以 `_len` 结尾的是 int 类型，其余均为 `list[int]` 或 `list[float]`，可直接 JSON 序列化。

#### Error Response

```json
// 401 — 签名认证失败
{"detail": "RSA 签名认证失败"}

// 503 — 引擎未就绪
{"detail": "引擎未就绪"}

// 400 — 参数错误（如音频格式不支持）
{"detail": "音频解码失败（支持 Opus/MP3/FLAC/WAV）: ..."}
```

---

### 4.2 POST /tts — 生成语音

使用已提取的说话人向量合成语音，返回 Opus/OGG 音频流。

#### Request

```json
{
    "text": "需要合成的文本内容，支持长文本",
    "spk_vec": {
        "prompt_text_token": [50304, ...],
        "prompt_text_token_len": 37,
        "llm_prompt_speech_token": [1, ...],
        "llm_prompt_speech_token_len": 354,
        "flow_prompt_speech_token": [1, ...],
        "flow_prompt_speech_token_len": 354,
        "prompt_speech_feat": [[0.123, ...], ...],
        "prompt_speech_feat_len": 354,
        "llm_embedding": [0.123, ...],
        "flow_embedding": [0.123, ...]
    },
    "voice_type": "zero_shot",
    "stream": false,
    "speed": 1.0,
    "instruct_text": ""
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | — | 合成文本，支持长文本（引擎自动分句） |
| `spk_vec` | object | ✅ | — | 从 `/extract` 获得，原样传回 |
| `voice_type` | string | 否 | `"zero_shot"` | `"zero_shot"` / `"cross_lingual"` / `"instruct2"` |
| `stream` | boolean | 否 | `false` | `true`=流式输出（逐帧 Opus）`false`=整段输出 |
| `speed` | number | 否 | `1.0` | 语速，取值范围 0.5~1.5 |
| `instruct_text` | string | 否 | `""` | instruct2 模式专用，传指令文本 |

#### Response (200)

- **Content-Type**: `audio/ogg`
- **Content-Disposition**: `attachment; filename=tts.opus`
- **Body**: Opus 编码的音频数据

#### voice_type 说明

| 类型 | 说明 | 要求 |
|------|------|------|
| `zero_shot` | 零样本克隆 | 需要参考音频的说话人向量 |
| `cross_lingual` | 跨语言合成 | 用 A 语言参考音频，合成 B 语言语音 |
| `instruct2` | 指令式控制 | 配合 `instruct_text` 参数控制语气/方言/情感 |

---

### 4.3 GET /health — 健康检查

#### Response (200)

```json
{
    "status": "ok",
    "version": "CosyVoice3-0.5B",
    "uptime_hours": null,
    "device": "cuda:0",
    "gpu": "NVIDIA A10G",
    "mem_gb": {
        "used": 4.52,
        "reserved": 5.87
    },
    "load": {
        "active_gen": 0,
        "total_gen": 128,
        "total_ext": 45,
        "by_mode": {
            "zero_shot": 150,
            "cross_lingual": 3,
            "instruct2": 20
        }
    },
    "rpm": {
        "requests_1m": 12,
        "requests_5m": 56
    },
    "benchmark": {
        "rtf_short": 0.18,
        "rtf_medium": 0.22,
        "rtf_long": 0.25,
        "rtf_avg": 0.22,
        "max_concurrency": 4,
        "gpu_name": "NVIDIA A10G"
    }
}
```

| 字段 | 说明 |
|------|------|
| `version` | 模型版本 |
| `device` | 运行设备（cuda / cpu） |
| `gpu` | GPU 型号 |
| `mem_gb.used` | 当前 GPU 显存占用 (GB) |
| `mem_gb.reserved` | 当前 GPU 缓存大小 (GB) |
| `load.active_gen` | 当前正在生成的请求数 |
| `load.total_gen` | 累计生成请求数 |
| `load.total_ext` | 累计提取请求数 |
| `load.by_mode` | 各模式使用统计 |
| `rpm.requests_1m` | 过去 1 分钟内请求数 |
| `rpm.requests_5m` | 过去 5 分钟内请求数 |
| `benchmark` | 启动时基准测试结果（RTF 越低越好） |

---

## 5. 完整调用流程（后端视角）

### 5.1 标准流程

```
后端                                   引擎
  │                                      │
  │ 1. 准备参考音频（Opus/WAV/MP3）       │
  │ 2. Base64 编码音频                    │
  │ 3. 构建请求体 + RSA 签名              │
  │ ─── POST /extract ─────────────────→  │
  │                                      │ 解码音频 → 提取声纹
  │ ←─── spk_vec (8个字段) ───────────── │
  │                                      │
  │ 4. 缓存 spk_vec（说话人→向量映射）     │
  │                                      │
  │ 5. 构建合成请求 + RSA 签名            │
  │ ─── POST /tts ────────────────────→  │
  │                                      │ 流式/非流式 Opus 编码
  │ ←─── audio/ogg (Opus) ───────────── │
  │                                      │
  │ 6. 播放/保存/转发 Opus 音频           │
```

### 5.2 后端缓存策略

```python
# 说话人向量缓存建议
spk_cache: dict[str, dict] = {}

def get_or_extract_spk(speaker_id: str, audio_b64: str, prompt_text: str):
    if speaker_id not in spk_cache:
        response = sign_and_post("/extract", {
            "prompt_text": prompt_text,
            "prompt_audio": audio_b64,
        })
        spk_cache[speaker_id] = response["spk_vec"]
    return spk_cache[speaker_id]
```

> `spk_vec` 在说话人不更换时只需提取**一次**，后续 `/tts` 可复用。

### 5.3 instruct2 模式特殊流程

```
后端                                   引擎
  │                                      │
  │ POST /extract (普通音频+文本)         │
  │ ←── spk_vec ──────────────────────  │ spk_vec 含原始 prompt_text_token
  │                                      │
  │ POST /tts (传 instruct_text)          │
  │ { spk_vec, instruct_text: "用湖南话" }│
  │ ──────────────────────────────────→  │ 引擎自动替换 prompt_text_token
  │                                      │ 为指令文本编码后的 token
```

---

## 6. 音频编解码细节

### 6.1 输入音频（/extract）

| 项 | 说明 |
|----|------|
| 格式 | Opus/OGG、MP3、FLAC、WAV 均可 |
| 传输方式 | Base64 编码放在 JSON 的 `prompt_audio` 字段 |
| 引擎解码 | `ffmpeg` 子进程 → `pcm_s16le` / 16kHz / 单声道 / WAV 临时文件 |
| 解码后 | 引擎自动清理临时文件 |

### 6.2 输出音频（/tts）

| 项 | 说明 |
|----|------|
| 格式 | **Opus/OGG** (libopus 编码器，比特率 32kbps) |
| 采样率 | 22050 Hz（引擎 `sample_rate`） |
| 声道 | 单声道 (mono) |
| Content-Type | `audio/ogg` |
| Content-Disposition | `attachment; filename=tts.opus` |

### 6.3 流式 vs 非流式

| 模式 | 特点 | 适合场景 |
|------|------|----------|
| `stream: false` | 引擎完全生成后一次性返回 | API 调用、短文本 |
| `stream: true` | 逐帧 Opus 编码，Chunked 传输 | 实时语音、长文本 |

---

## 7. 服务端统计与监控

### 7.1 关键指标

- **RTF (Real-Time Factor)**：合成耗时 / 音频时长。RTF < 1 表示比实时快。
  - 短文本 RTF ≈ 0.18（每秒合成约 5.5 秒音频）
  - 长文本 RTF ≈ 0.25（每秒合成约 4 秒音频）
- **最大并发数**：`int(1.0 / avg_rtf)`，超过此值可能显存溢出
- **每分钟请求数 (RPM)**：通过 `/health` 的 `rpm` 字段获取

### 7.2 引擎启动日志

```
🚀 CosyVoice3 引擎启动中...
   📂 pretrained_models/Fun-CosyVoice3-0.5B
   [1/3] 文本前端...   ✓ <class 'cosyvoice.frontend.CosyVoice3FrontEnd'>
   [2/3] vLLM 引擎... ✓
   [3/3] TensorRT...   ✓
   模型: LLM=Qwen2 0.5B +vLLM | Flow=DiT +TRT | HiFiGAN=CausalHiFT
   耗时 12.3s
   💾 显存占用: 已分配 4.52 GB / 缓存 5.87 GB
   ✅ 引擎就绪

═╦═ 基准测试开始... ╦═
   📄 参考文本: "这里是参考文本..."
   🎵 参考音频: benchmark/ref.wav
   [1/4] 提取说话人向量... ✓ (312ms)
   [2/4] 短文本基准...
         第1次: TTFB= 152.3ms 总耗时= 180.5ms 音频=0.98s RTF=0.1836
         ...
```

---

## 8. 常见问题

### Q: 为什么 /extract 返回 401？

A：签名认证失败。检查：
1. 引擎是否部署了与调用方**匹配**的 `public_key.pem`
2. 调用的请求体 JSON 和签名时的 body_bytes 是否**完全一致**（注意编码）
3. 私钥没有损坏

### Q: 为什么音频解码失败？

A：引擎依赖 `ffmpeg` 进行音频解码（替代 PyAV）。确保：
```bash
# 检查 ffmpeg 是否可用
ffmpeg -version
```
ffmpeg 必须存在于 `PATH` 中。

### Q: 长文本合成有什么限制？

A：引擎内部按标点分句后逐句合成，理论上无长度限制。但建议：
- 单次请求不要超过 5000 字符
- 超长文本建议客户端分片，每片 1000~2000 字符

### Q: 如何估算硬件需求？

| 场景 | 最低显存 | 推荐显存 | 建议 GPU |
|------|---------|---------|---------|
| 单并发推理 | 4 GB | 6 GB | RTX 3060 / A10 |
| 3~5 并发 | 8 GB | 12 GB | RTX 4090 / A10G |
| 10+ 并发 | 16 GB | 24 GB | A100 / A6000 |

### Q: 跨语言合成要注意什么？

A：`voice_type: "cross_lingual"` 模式下：
- 参考音频使用 A 语言（如中文）
- `text` 使用 B 语言（如英文）
- 引擎自动为 B 语言文本加 Chat 前缀编码
- 合成质量取决于目标语言的 tokenizer 支持情况

---

> 文档版本: v1.0 | 最后更新: 2025-05-05
