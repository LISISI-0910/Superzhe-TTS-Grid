"""
CosyVoice3 + WhisperASR API 服务
===============================
启动: uvicorn server.main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import logging
import re
import threading
import time
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from server.manager import EngineManager

# ── 日志静音 ──
for _name in ("vllm", "multipart", "httptools", "asyncio",
               "faster_whisper", "faster_whisper.audio", "ctranslate2"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.transformers_utils.tokenizer").setLevel(logging.ERROR)

from server.schemas import TTSExtractResponse, ASRResponse, HealthResponse
from server.audio import AudioProcessor
from server.config import load_config

# ── 全局单例 ───────────────────────────────────────────
manager = EngineManager()

# ── 配置 ───────────────────────────────────────────────
_config = load_config()
_CFG = _config["limits"]
_CFG_GEN = _config["generation"]

_BAD_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

GENERATION_HARD_TIMEOUT = _CFG_GEN["hard_timeout_s"]
GENERATION_IDLE_TIMEOUT = _CFG_GEN["idle_timeout_s"]
AUDIO_MAX_BYTES = _CFG["audio_size_max_mb"] * 1024 * 1024
AUDIO_MAX_DURATION = _CFG["audio_duration_max_s"]


def _check_text(value: str, key: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(400, f"{key} 不能为空")
    limit_map = {"prompt": _CFG["prompt_text_max"],
                 "tts": _CFG["tts_text_max"],
                 "instruct": _CFG["instruct_text_max"]}
    limit = limit_map.get(key, 2000)
    if len(value) > limit:
        raise HTTPException(400, f"{key} 最长 {limit} 字，当前 {len(value)} 字")
    if _BAD_RE.search(value):
        raise HTTPException(400, f"{key} 包含非法字符")
    return value


# ── Lifespan ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    m = _config["models"]
    a = _config["acceleration"]
    b = _config["benchmark"]
    await manager.startup(
        tts_model_dir=m["tts_dir"],
        asr_model_dir=m.get("asr_dir", ""),
        load_trt=a["trt"], load_vllm=a["vllm"], fp16=a["fp16"],
        asr_concurrency=_config["asr"]["concurrency"],
        warmup_rounds=b["warmup_rounds"],
        coarse_rounds=b["coarse_rounds"],
        fine_rounds=b["fine_rounds"],
    )
    yield
    await manager.shutdown()


app = FastAPI(title="CosyVoice3 TTS + ASR API", version="1.0", lifespan=lifespan)


# ═══════════════════  TTS  ══════════════════════════════

@app.post("/api/v1/tts/extract", response_model=TTSExtractResponse)
async def tts_extract(
    audio: UploadFile = File(...),
    prompt_text: str = Form(...),
):
    """提取说话人向量 → speaker_b64（需手动提供文本）。"""
    prompt_text = _check_text(prompt_text, "prompt")
    audio_bytes = await audio.read()
    if len(audio_bytes) > AUDIO_MAX_BYTES:
        raise HTTPException(400, f"音频文件不能超过 {_CFG['audio_size_max_mb']}MB")
    dur = AudioProcessor.probe_duration(audio_bytes)
    if dur > AUDIO_MAX_DURATION:
        raise HTTPException(400,
            f"音频最长 {AUDIO_MAX_DURATION:.0f} 秒，当前 {dur:.0f}s")
    path = manager.audio.normalize(audio_bytes, audio.filename or "audio")
    try:
        spk_b64 = await asyncio.to_thread(
            manager.tts.extract_speaker, path, prompt_text)
    finally:
        manager.audio.clean(path)
    return {"speaker_b64": spk_b64}


@app.post("/api/v1/tts/enroll", response_model=TTSExtractResponse)
async def tts_enroll(audio: UploadFile = File(...)):
    """只传一个音频文件，ASR 自动转文字 → 提取说话人向量。"""
    audio_bytes = await audio.read()
    if len(audio_bytes) > AUDIO_MAX_BYTES:
        raise HTTPException(400, f"音频文件不能超过 {_CFG['audio_size_max_mb']}MB")
    dur = AudioProcessor.probe_duration(audio_bytes)
    if dur > AUDIO_MAX_DURATION:
        raise HTTPException(400,
            f"音频最长 {AUDIO_MAX_DURATION:.0f} 秒，当前 {dur:.0f}s")
    path = manager.audio.normalize(audio_bytes, audio.filename or "audio")
    try:
        # ASR 转出文本
        async with manager.asr_sem:
            asr_result = await asyncio.to_thread(manager.asr.transcribe, path)
        prompt_text = asr_result["text"].strip()
        if not prompt_text:
            raise HTTPException(400, "音频未识别到有效语音内容")
        # 提取说话人向量
        spk_b64 = await asyncio.to_thread(
            manager.tts.extract_speaker, path, prompt_text)
        return {"speaker_b64": spk_b64}
    finally:
        manager.audio.clean(path)


@app.post("/api/v1/tts/generate")
async def tts_generate(
    text: str = Form(...),
    speaker_b64: str = Form(...),
    mode: str = Form(default="zero_shot"),
    instruct_text: str = Form(default=""),
):
    """流式生成语音 (Opus-OGG)。"""
    text = _check_text(text, "tts")
    if instruct_text:
        instruct_text = _check_text(instruct_text, "instruct")

    # 预校验 speaker_b64，避免流式开始后才发现损坏
    import base64, pickle
    try:
        pickle.loads(base64.b64decode(speaker_b64))
    except Exception:
        raise HTTPException(400, "speaker_b64 无效，请重新提取")

    # 并发控制：快速失败
    try:
        await asyncio.wait_for(manager.tts_sem.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="TTS server at capacity, try later")

    async def stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in _run_tts_stream(manager.tts, mode, text, speaker_b64, instruct_text):
                yield chunk
        except asyncio.CancelledError:
            raise
        finally:
            manager.tts_sem.release()

    return StreamingResponse(stream(), media_type="audio/ogg",
                             headers={"X-TTS-SampleRate": str(manager.tts.sample_rate)})


async def _run_tts_stream(engine, mode: str, text: str,
                          speaker_b64: str, instruct_text: str = "") -> AsyncGenerator[bytes, None]:
    """桥接同步 TTS → 异步 Opus-OGG 流，超时自动取消。"""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    cancelled = threading.Event()

    def _worker():
        try:
            encode, flush = AudioProcessor.create_opus_encoder()
            for pcm in engine.generate(mode=mode, text=text,
                                       speaker_b64=speaker_b64,
                                       instruct_text=instruct_text):
                if cancelled.is_set():
                    break
                data = encode(pcm)
                if data:
                    asyncio.run_coroutine_threadsafe(q.put(data), loop)
            if not cancelled.is_set():
                remaining = flush()
                if remaining:
                    asyncio.run_coroutine_threadsafe(q.put(remaining), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(e), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    thread = threading.Thread(target=_worker)
    thread.start()
    hard_deadline = time.monotonic() + GENERATION_HARD_TIMEOUT
    idle_deadline = time.monotonic() + GENERATION_IDLE_TIMEOUT

    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if now > hard_deadline or now > idle_deadline:
                    cancelled.set()
                    raise HTTPException(status_code=504,
                                        detail="生成超时")
                continue
            # 收到数据，重置空闲超时
            idle_deadline = time.monotonic() + GENERATION_IDLE_TIMEOUT
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    except asyncio.CancelledError:
        cancelled.set()
        raise


# ═══════════════════  ASR  ══════════════════════════════

@app.post("/api/v1/asr/transcribe", response_model=ASRResponse)
async def asr_transcribe(audio: UploadFile = File(...)):
    """转录 → {text, srt, vtt, language, segments}。"""
    audio_bytes = await audio.read()
    path = manager.audio.normalize(audio_bytes, audio.filename or "audio")
    try:
        async with manager.asr_sem:
            result = await asyncio.to_thread(manager.asr.transcribe, path)
    finally:
        manager.audio.clean(path)
    return result


# ═══════════════════  Health  ═══════════════════════════

@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok" if manager.ready else "initializing",
        "tts_available": manager.tts is not None,
        "asr_available": manager.asr is not None,
        "tts_concurrency": manager.tts_status(),
    }
