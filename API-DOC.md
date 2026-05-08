# CosyVoice3 TTS + WhisperASR API 文档

**版本**: 1.0  
**协议**: HTTP/1.1  
**Content-Type**: `multipart/form-data`（所有 POST 端点）

---

## 目录

- [1. 健康检查](#1-健康检查)
- [2. 提取说话人向量（需手动提供文本）](#2-提取说话人向量须手动提供文本)
- [3. 注册说话人（仅需音频，自动转录）](#3-注册说话人仅需音频自动转录)
- [4. 流式语音生成](#4-流式语音生成)
- [5. 语音识别（转录）](#5-语音识别转录)
- [附：错误码速查](#附错误码速查)
- [附：典型调用流程](#附典型调用流程)

---

## 1. 健康检查

```
GET /api/v1/health
```

**响应** `200`

```json
{
  "status": "ok",
  "tts_available": true,
  "asr_available": true,
  "tts_concurrency": "1/4"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ok"` 可用，`"initializing"` 启动中 |
| `tts_available` | bool | TTS 引擎是否就绪 |
| `asr_available` | bool | ASR 引擎是否就绪 |
| `tts_concurrency` | string | 当前 TTS 并发占用情况，如 `"1/4"` = 4 个槽位用了 1 个 |

**用途**：负载均衡器探活 + 判断是否有空闲槽位。当 `tts_concurrency` 达到上限时（如 `"4/4"`），新请求将返回 503。

---

## 2. 提取说话人向量（须手动提供文本）

从参考音频 + 对应文本中提取说话人特征，返回 base64 编码的向量包。

```
POST /api/v1/tts/extract
```

### 请求（multipart/form-data）

| 字段 | 类型 | 必填 | 限制 | 说明 |
|------|------|------|------|------|
| `audio` | file | ✓ | ≤50MB, ≤30s | 参考音频，支持 MP3/WAV/OGG/FLAC/AAC 等 |
| `prompt_text` | string | ✓ | 1-1000 字 | 音频对应的文本内容 |

### 响应 `200`

```json
{
  "speaker_b64": "gASVnwAAAAAAAACMFG51bXB5LmNvcmUubXVsdGlhcnJheZSM..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `speaker_b64` | string | 说话人向量包（base64 编码的 pickle），约 80KB |

### 错误

| 状态码 | 条件 |
|--------|------|
| 400 | 音频 >50MB / >30s / 文本为空或超长 |
| 503 | 服务未就绪 |

### 示例

```bash
curl -X POST http://localhost:6006/api/v1/tts/extract \
  -F "audio=@speaker.wav" \
  -F "prompt_text=希望你以后能够做的比我还好呦。"
```

---

## 3. 注册说话人（仅需音频，自动转录）

只传一个音频文件，内部先用 ASR 转录文本，再提取说话人向量。适合"扔一段音频就注册"的场景。

```
POST /api/v1/tts/enroll
```

### 请求（multipart/form-data）

| 字段 | 类型 | 必填 | 限制 | 说明 |
|------|------|------|------|------|
| `audio` | file | ✓ | ≤50MB, ≤30s | 参考音频，需包含清晰语音 |

### 响应 `200`

```json
{
  "speaker_b64": "gASVnwAAAAAAAACMFG51bXB5LmNvcmUubXVsdGlhcnJheZSM..."
}
```

### 错误

| 状态码 | 条件 |
|--------|------|
| 400 | 音频 >50MB / >30s / 音频中未识别到有效语音 |
| 503 | 服务未就绪 |

### 示例

```bash
curl -X POST http://localhost:6006/api/v1/tts/enroll \
  -F "audio=@speaker.wav"
```

---

## 4. 流式语音生成

传入目标文本 + 说话人向量，流式返回 Opus-OGG 音频。

```
POST /api/v1/tts/generate
```

### 请求（multipart/form-data）

| 字段 | 类型 | 必填 | 限制 | 说明 |
|------|------|------|------|------|
| `text` | string | ✓ | 1-20000 字 | 要合成的目标文本 |
| `speaker_b64` | string | ✓ | — | 从 extract/enroll 获取的说话人向量 |
| `mode` | string | — | `zero_shot` / `cross_lingual` / `instruct2` | 生成模式，默认 `zero_shot` |
| `instruct_text` | string | — | 1-500 字 | 指令控制文本，仅 `mode=instruct2` 时生效 |

### mode 说明

| 值 | 行为 |
|------|------|
| `zero_shot`（默认） | 零样本语音克隆，使用参考音频的语速和情感 |
| `cross_lingual` | 跨语言合成，文本语言可与参考音频不同 |
| `instruct2` | 指令控制，通过 `instruct_text` 控制语气/情感/语速等 |

### 响应 `200`

- **Content-Type**: `audio/ogg`
- **Header**: `X-TTS-SampleRate: 24000`
- **Body**: 流式 Opus-OGG 编码的音频数据（int16, mono, 24000Hz）

每个 chunk 是合法的 OGG 页，客户端可边收边播。

### 错误

| 状态码 | 条件 |
|--------|------|
| 400 | 文本为空/超长/含非法字符 / speaker_b64 无效 |
| 503 | TTS 并发已满 |
| 504 | 生成超时（硬超时 1h 或 2min 无产出） |

### 客户端断开处理

如果客户端在生成过程中断开连接（Cancel），服务端会立即停止生成并释放 GPU 资源。

### 示例

```bash
# 零样本克隆
curl -X POST http://localhost:6006/api/v1/tts/generate \
  -F "text=今天天气真好，我们出去玩吧。" \
  -F "speaker_b64=gASVnwAAAAA..." \
  -F "mode=zero_shot" \
  -o output.ogg

# 指令控制模式
curl -X POST http://localhost:6006/api/v1/tts/generate \
  -F "text=欢迎来到语音合成系统。" \
  -F "speaker_b64=gASVnwAAAAA..." \
  -F "mode=instruct2" \
  -F "instruct_text=用热情洋溢的语气" \
  -o output.ogg
```

### 后端集成要点

1. **speaker_b64 需原样传递**，不要做 URL decode/encode 或任何字符串转换
2. **并发控制**：先调 `/api/v1/health` 查看 `tts_concurrency`，满时不要硬等，直接返回"排队中"给前端
3. **断连传递**：如果前端断开 WebSocket/SSE，立即取消对 generate 的请求，GPU 会被释放
4. **音频解码**：返回的 OGG 数据可直接写入文件播放，或用 `av`/`ffmpeg` 转码为其他格式

---

## 5. 语音识别（转录）

将音频转录为文本 + SRT/VTT 字幕。

```
POST /api/v1/asr/transcribe
```

### 请求（multipart/form-data）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | file | ✓ | 音频文件，支持 MP3/WAV/OGG/FLAC/AAC 等，时长不限 |

### 响应 `200`

```json
{
  "text": "希望你以后能够做的比我还好呦。",
  "language": "zh",
  "duration": 5.12,
  "segments": [
    {"start": 0.02, "end": 5.10, "text": "希望你以后能够做的比我还好呦。"}
  ],
  "srt": "1\n00:00:00,020 --> 00:00:05,100\n希望你以后能够做的比我还好呦\n",
  "vtt": "WEBVTT\n\n00:00:00.020 --> 00:00:05.100\n希望你以后能够做的比我还好呦\n"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 完整转录文本（带标点） |
| `language` | string | 检测到的语言代码，如 `"zh"` `"en"` `"ja"` |
| `duration` | float | 音频时长（秒） |
| `segments` | array | 句子级分段，每个含 `start` `end` `text` |
| `srt` | string | SubRip 字幕格式（去标点） |
| `vtt` | string | WebVTT 字幕格式（去标点） |

### 示例

```bash
curl -X POST http://localhost:6006/api/v1/asr/transcribe \
  -F "audio=@speech.mp3"
```

---

## 附：错误码速查

| 状态码 | 含义 | 重试策略 |
|--------|------|----------|
| 200 | 成功 | — |
| 400 | 参数错误（文本超长、音频太大、speaker_b64 无效等） | 不重试，检查输入 |
| 503 | 服务繁忙（TTS 并发满） | 1-3s 后退重试，最多 3 次 |
| 504 | 生成超时 | 检查文本长度是否合理 |

---

## 附：典型调用流程

### 流程 A：零样本克隆（手动提供文本）

```
1. POST /api/v1/tts/extract  { audio + prompt_text }
   → {"speaker_b64": "..."}

2. POST /api/v1/tts/generate  { text + speaker_b64, mode="zero_shot" }
   → 流式 audio/ogg
```

### 流程 B：一键注册 + 生成

```
1. POST /api/v1/tts/enroll   { audio }
   → {"speaker_b64": "..."}

2. POST /api/v1/tts/generate  { text + speaker_b64 }
   → 流式 audio/ogg
```

### 流程 C：语音识别

```
1. POST /api/v1/asr/transcribe  { audio }
   → {"text": "...", "srt": "...", "vtt": "..."}
```

### 流程 D：并发控制最佳实践

```
while True:
    health = GET /api/v1/health
    used, max = health.tts_concurrency.split("/")
    if int(used) < int(max):
        break
    sleep(1)

POST /api/v1/tts/generate  { ... }
→ 流式 audio/ogg
```
