"""
CosyVoice3 Engine Server — FastAPI 端点。
"""
import av
import argparse
import base64
import contextlib
import io
import os
import sys
import tempfile
import time
import uuid
from typing import Optional, AsyncGenerator

# === 噪音压制 ===
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'ERROR')
os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('VLLM_CONFIGURE_LOGGING', '0')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TQDM_DISABLE'] = '1'

import logging as _logging
for _noisy in ['vllm', 'httpx', 'httpcore', 'urllib3', 'modelscope',
               'lightning', 'lightning.fabric', 'torch.distributed']:
    _logging.getLogger(_noisy).setLevel(_logging.ERROR)

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
_THIRD_PARTY = os.path.join(_PROJ_ROOT, 'third_party', 'Matcha-TTS')
if os.path.isdir(_THIRD_PARTY) and _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cosyvoice.engine import CosyVoiceEngine


engine: Optional[CosyVoiceEngine] = None
GEN_MODE_CN = {'zero_shot': '克隆', 'cross_lingual': '跨语言', 'instruct2': '指令'}

# ─── RSA 非对称签名认证 ────────────────────────────────
# 用法：
#   1. 调用方（后端）用 private_key.pem 对请求体签名 → base64
#      → 放在 HTTP Header X-Signature 中
#   2. 引擎收到请求后，用 public_key.pem 验签
#   3. private_key.pem 仅在调用方持有，不泄露；public_key.pem 分发到所有引擎
#
# 文件结构：
#   /home/li/new/public_key.pem   ← 引擎（验签）
#   调用方后端保留 private_key.pem   ← 后端（签名）
#
# 安全性：即使引擎被攻破，攻击者拿到公钥也无法伪造签名（只能验签）
# 多引擎共享公钥：完全安全

def _load_rsa_public_key() -> bytes | None:
    """读取 RSA 公钥文件。文件不存在时返回 None（跳过认证）。"""
    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public_key.pem')
    if os.path.exists(key_path):
        with open(key_path, 'rb') as f:
            return f.read()
    return None

_RSA_PUBLIC_KEY_PEM = _load_rsa_public_key()

def _verify_signature(body: bytes, signature_b64: str) -> bool:
    """用 RSA 公钥验证签名。公钥不存在时跳过认证（开发环境）。"""
    if _RSA_PUBLIC_KEY_PEM is None:
        return True  # 没配公钥就不做认证
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        from cryptography.exceptions import InvalidSignature

        public_key = serialization.load_pem_public_key(_RSA_PUBLIC_KEY_PEM)
        signature = base64.b64decode(signature_b64)
        public_key.verify(
            signature,
            body,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return True
    except (InvalidSignature, Exception):
        return False


# ── 提取说话人向量 ──
class ExtractRequest(BaseModel):
    prompt_text: str          # 参考文本
    prompt_audio: str         # 音频文件（base64 编码，支持 Opus/MP3/FLAC/WAV）
    voice_type: str = 'zero_shot'  # zero_shot | instruct2

# ── 生成语音 ──
class GenerateRequest(BaseModel):
    text: str                 # 合成文本
    spk_vec: dict             # 说话人向量（从 extract 获得）
    voice_type: str = 'zero_shot'  # zero_shot | cross_lingual | instruct2
    stream: bool = False
    speed: float = 1.0
    instruct_text: str = ''


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global engine
    if engine is None:
        raise RuntimeError("引擎未初始化")
    yield
    # 服务器关闭时清理 GPU 资源
    if engine is not None:
        del engine
        engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _decode_audio_to_wav(audio_bytes: bytes) -> str:
    """
    通用音频解码：用 ffmpeg 子进程将 Opus/MP3/FLAC/WAV/… 解码为
    16kHz 16-bit 单声道 WAV，保存到临时文件并返回路径。

    使用 subprocess 而非 PyAV，避免 PyAV 版本兼容性问题。
    """
    import subprocess as sp

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        proc = sp.Popen(
            ['ffmpeg', '-y', '-i', 'pipe:0',
             '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
             tmp.name],
            stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.DEVNULL
        )
        proc.communicate(input=audio_bytes)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 返回码 {proc.returncode}")
    except Exception as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"音频解码失败（支持 Opus/MP3/FLAC/WAV）: {e}")

    return tmp.name


app = FastAPI(title="CosyVoice3 Engine", version="1.0.0", lifespan=lifespan)


# ─── RSA 验签中间件 ──────────────────────────────────
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    RSA 非对称验签，以下情况跳过认证：
      - 公钥不存在（开发环境）
      - 请求方法为 GET/HEAD/OPTIONS（读取操作，无需验签）
    """
    # ── 放行条件 ──
    if _RSA_PUBLIC_KEY_PEM is None:
        return await call_next(request)
    if request.method in ('GET', 'HEAD', 'OPTIONS'):
        return await call_next(request)

    # ── 读取请求体并验签 ──
    body = await request.body()
    sig = request.headers.get('X-Signature', '')
    if not sig or not _verify_signature(body, sig):
        # ← 静默返回 401，不触发 FastAPI 的告警日志
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "RSA 签名认证失败"})

    return await call_next(request)


def log_req(req_id: str, mode_cn: str, text: str, elapsed_ms: float, ttft_ms: float = 0):
    """
    格式化请求日志
    ttft_ms=0 时表示非流式或非生成任务
    """
    chars = len(text)
    preview = text[:10] + ('...' if len(text) > 10 else '')
    
    # 构建耗时描述字符串
    # 如果 ttft_ms > 0，我们就显示首包时间；否则只显示总耗时
    time_info = f"⏱总计:{elapsed_ms:.2f}ms"
    if ttft_ms > 0:
        time_info = f"⚡首包:{ttft_ms:.2f}ms | " + time_info

    print(f'  [{req_id}] {mode_cn} "{preview}"({chars}字) {time_info}')

# ─── 提取说话人向量 ─────────────────────────────────────
@app.post("/extract")
async def extract(req: ExtractRequest):
    global engine
    if engine is None:
        raise HTTPException(503, "引擎未就绪")

    req_id = uuid.uuid4().hex[:6]
    t0 = time.perf_counter()

    # 解码前端传来的音频（Opus/MP3/FLAC/WAV 均可）
    audio_bytes = base64.b64decode(req.prompt_audio)
    wav_path = _decode_audio_to_wav(audio_bytes)
    try:
        spk_vec = engine.process(mode='extract', prompt_text=req.prompt_text,
                                 prompt_wav=wav_path, gen_type=req.voice_type)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_req(req_id, '提取', req.prompt_text, elapsed_ms)
        return {'status': 'ok', 'spk_vec': spk_vec}
    except (ValueError, AssertionError) as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        print(f"提取失败: {e}")
        raise HTTPException(500, detail="引擎提取向量失败")
    finally:
        os.unlink(wav_path)


@app.post("/tts")
async def generate(req: GenerateRequest):
    global engine
    if engine is None:
        raise HTTPException(503, "引擎未就绪")
    if not req.text:
        raise HTTPException(400, "需要合成文本")
    if not req.spk_vec:
        raise HTTPException(400, "需要说话人向量")

    req_id = uuid.uuid4().hex[:6]
    t0 = time.perf_counter()
    mode_cn = GEN_MODE_CN.get(req.voice_type, req.voice_type)

    raw_gen = engine.process(mode='generate', text=req.text, spk_vec=req.spk_vec,
                             gen_type=req.voice_type, stream=req.stream,
                             speed=req.speed, instruct_text=req.instruct_text)

    # 追踪首包和总耗时
    def timed_gen():
        nonlocal t0
        t0 = time.perf_counter()
        first = True
        ttft = 0.0
        try:
            for o in raw_gen:
                if first:
                    ttft = (time.perf_counter() - t0) * 1000
                    first = False
                yield o
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            log_req(req_id, mode_cn, req.text, elapsed, ttft)

    timed = timed_gen()

    if req.stream:
        # ── 流式 Opus ──
        def stream_opus():
            sample_rate = engine.sample_rate
            out_buf = io.BytesIO()
            container = av.open(out_buf, mode='w', format='ogg')
            s = container.add_stream('libopus', rate=sample_rate)
            s.bit_rate = 32000
            s.layout = 'mono'
            try:
                for output in timed:
                    chunk = output['tts_speech'].numpy()
                    chunk_int16 = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
                    frame = av.AudioFrame.from_ndarray(chunk_int16, format='s16', layout='mono')
                    frame.sample_rate = sample_rate
                    for packet in s.encode(frame):
                        container.mux(packet)
                    data = out_buf.getvalue()
                    if data:
                        yield data
                        out_buf.seek(0)
                        out_buf.truncate()
                for packet in s.encode(None):
                    container.mux(packet)
                container.close()
                final_data = out_buf.getvalue()
                if final_data:
                    yield final_data
            except Exception as e:
                print(f"Opus 编码异常: {e}")
            finally:
                out_buf.close()
        return StreamingResponse(stream_opus(), media_type='audio/ogg',
                                 headers={'Content-Disposition': 'attachment; filename=tts.opus'})
    else:
        # ── 非流式 Opus ──
        pcm = []
        for output in timed:
            pcm.append(output['tts_speech'].numpy())
        audio = np.concatenate(pcm, axis=1)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        sample_rate = engine.sample_rate
        out_buf = io.BytesIO()
        with av.open(out_buf, mode='w', format='ogg') as container:
            s = container.add_stream('libopus', rate=sample_rate)
            s.bit_rate = 32000
            s.layout = 'mono'
            frame = av.AudioFrame.from_ndarray(audio_int16, format='s16', layout='mono')
            frame.sample_rate = sample_rate
            for packet in s.encode(frame):
                container.mux(packet)
            for packet in s.encode(None):
                container.mux(packet)
        out_buf.seek(0)
        return StreamingResponse(out_buf, media_type='audio/ogg',
                                 headers={'Content-Disposition': 'attachment; filename=tts.opus'})


@app.get("/health")
async def health():
    if engine is None:
        return {'status': '未就绪'}

    stats = engine.stats
    rate = engine.rate_stats

    # GPU 信息（防御性获取，避免 torch 内部断言抛异常）
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        mem_used = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else None
        mem_reserved = round(torch.cuda.memory_reserved() / (1024**3), 2) if torch.cuda.is_available() else None
    except Exception:
        gpu_name = 'unknown'
        mem_used = None
        mem_reserved = None

    return {
        'status': 'ok',
        'version': 'CosyVoice3-0.5B',
        'uptime_hours': None,  # 暂不追踪
        'device': str(engine.model.device),
        'gpu': gpu_name,
        'mem_gb': {'used': mem_used, 'reserved': mem_reserved} if mem_used else None,
        'load': {
            'active_gen': stats['active'],
            'total_gen': stats['total_gen'],
            'total_ext': stats['total_ext'],
            'by_mode': stats['by_mode'],
        },
        'rpm': {
            'requests_1m': rate['requests_1m'],
            'requests_5m': rate['requests_5m'],
        },
        'benchmark': getattr(engine, 'benchmark', None),
    }


def _run_benchmark(engine_obj, spk_vec, label: str, text: str, n: int):
    """运行 n 次推理，返回 (avg_rtf, avg_ttfb_ms, avg_total_ms, avg_audio_sec, raw_results)"""
    sample_rate = engine_obj.sample_rate
    times = []

    for i in range(n):
        t_start = time.perf_counter()
        first_chunk_time = None
        total_samples = 0
        last_chunk_time = None

        gen = engine_obj.process(mode='generate', text=text, spk_vec=spk_vec,
                                 gen_type='zero_shot', stream=True)
        for output in gen:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            chunk = output['tts_speech']
            total_samples += chunk.shape[-1]
            last_chunk_time = time.perf_counter()

        t_end = time.perf_counter()

        ttfb_ms = (first_chunk_time - t_start) * 1000
        total_ms = (t_end - t_start) * 1000
        audio_sec = total_samples / sample_rate
        rtf = total_ms / 1000.0 / audio_sec if audio_sec > 0 else 0

        times.append({
            'ttfb_ms': round(ttfb_ms, 2),
            'total_ms': round(total_ms, 2),
            'audio_sec': round(audio_sec, 3),
            'rtf': round(rtf, 4),
        })

    avg_rtf = sum(t['rtf'] for t in times) / n
    avg_ttfb = sum(t['ttfb_ms'] for t in times) / n
    avg_total = sum(t['total_ms'] for t in times) / n
    avg_audio = sum(t['audio_sec'] for t in times) / n

    return avg_rtf, avg_ttfb, avg_total, avg_audio, times


def init_engine(model_dir: str, load_trt=False, load_vllm=False, fp16=False):
    global engine
    print('🚀 CosyVoice3 引擎启动中...')
    print(f'   📂 {model_dir}')
    feat = []
    if load_vllm: feat.append('vLLM')
    if load_trt: feat.append('TRT')
    if fp16: feat.append('FP16')
    if torch.cuda.is_available():
        feat.append(torch.cuda.get_device_name(0))
    print(f'   ⚙️  {" | ".join(feat) if feat else "PyTorch CPU"}')
    engine = CosyVoiceEngine(model_dir, load_trt=load_trt, load_vllm=load_vllm, fp16=fp16)
    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f'   💾 显存占用: 已分配 {mem_allocated:.2f} GB / 缓存 {mem_reserved:.2f} GB')
    print('   ✅ 引擎就绪')

    # ==================== 预热 & 基准测试 ====================
    print()
    print('═' * 50)
    print('🔬 基准测试开始...')
    print('═' * 50)

    bm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark')
    ref_wav = os.path.join(bm_dir, 'ref.wav')
    ref_txt = os.path.join(bm_dir, 'ref.txt')

    if not os.path.exists(ref_wav) or not os.path.exists(ref_txt):
        print('   ⚠️  benchmark/ref.wav 或 ref.txt 不存在，跳过基准测试')
        print()
        return

    with open(ref_txt, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()

    print(f'   📄 参考文本: "{prompt_text}"')
    print(f'   🎵 参考音频: {ref_wav}')

    # --- 提取 spk_vec ---
    print()
    print('   [1/4] 提取说话人向量... ', end='', flush=True)
    t0 = time.perf_counter()
    spk_vec = engine.process(mode='extract', prompt_text=prompt_text, prompt_wav=ref_wav)
    extract_ms = (time.perf_counter() - t0) * 1000
    print(f'✓ ({extract_ms:.0f}ms)')

    SHORT_TEXT = "你好，今天天气不错。"
    MEDIUM_TEXT = (
        "国家卫健委今天通报，截至本月二十日，我国已完成新冠疫苗加强免疫接种超过八亿剂次，"
    )
    LONG_TEXT = (
        "人工智能作为新一轮科技革命和产业变革的重要驱动力，正深刻改变着人类生产生活方式。"
        "在医疗领域，人工智能辅助诊断系统能够快速分析医学影像，帮助医生更早发现肺部结节、"
        "眼底病变等早期病灶，大幅提升筛查效率和准确率。"
    )
    N = 3
    # --- 短文本 ---
    print('   [2/4] 短文本基准...')
    for _ in engine.process(mode='generate', text=SHORT_TEXT, spk_vec=spk_vec,
                            gen_type='zero_shot', stream=True):
        pass
    rtf_short, ttfb_short, total_short, audio_short, raw_short = _run_benchmark(
        engine, spk_vec, '短文本', SHORT_TEXT, N)
    for i, r in enumerate(raw_short):
        print(f'         第{i+1}次: TTFB={r["ttfb_ms"]:6.1f}ms '
              f'总耗时={r["total_ms"]:6.1f}ms '
              f'音频={r["audio_sec"]:.2f}s '
              f'RTF={r["rtf"]:.4f}')
    print(f'         平均:   TTFB={ttfb_short:6.1f}ms '
          f'总耗时={total_short:6.1f}ms '
          f'音频={audio_short:.2f}s '
          f'RTF={rtf_short:.4f}')

    # --- 中文本 ---
    print('   [3/4] 中文本基准...')
    for _ in engine.process(mode='generate', text=MEDIUM_TEXT, spk_vec=spk_vec,
                            gen_type='zero_shot', stream=True):
        pass
    rtf_medium, ttfb_medium, total_medium, audio_medium, raw_medium = _run_benchmark(
        engine, spk_vec, '中文本', MEDIUM_TEXT, N)
    for i, r in enumerate(raw_medium):
        print(f'         第{i+1}次: TTFB={r["ttfb_ms"]:6.1f}ms '
              f'总耗时={r["total_ms"]:6.1f}ms '
              f'音频={r["audio_sec"]:.2f}s '
              f'RTF={r["rtf"]:.4f}')
    print(f'         平均:   TTFB={ttfb_medium:6.1f}ms '
          f'总耗时={total_medium:6.1f}ms '
          f'音频={audio_medium:.2f}s '
          f'RTF={rtf_medium:.4f}')

    # --- 长文本 ---
    print('   [4/4] 长文本基准...')
    for _ in engine.process(mode='generate', text=LONG_TEXT, spk_vec=spk_vec,
                            gen_type='zero_shot', stream=True):
        pass
    rtf_long, ttfb_long, total_long, audio_long, raw_long = _run_benchmark(
        engine, spk_vec, '长文本', LONG_TEXT, N)
    for i, r in enumerate(raw_long):
        print(f'         第{i+1}次: TTFB={r["ttfb_ms"]:6.1f}ms '
              f'总耗时={r["total_ms"]:6.1f}ms '
              f'音频={r["audio_sec"]:.2f}s '
              f'RTF={r["rtf"]:.4f}')
    print(f'         平均:   TTFB={ttfb_long:6.1f}ms '
          f'总耗时={total_long:6.1f}ms '
          f'音频={audio_long:.2f}s '
          f'RTF={rtf_long:.4f}')

    # --- 计算最大并发 ---
    rtf_avg = (rtf_short + rtf_medium + rtf_long) / 3
    max_concurrency = max(1, int(1.0 / rtf_avg))
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'

    engine.benchmark = {
        'rtf_short': round(rtf_short, 4),
        'rtf_medium': round(rtf_medium, 4),
        'rtf_long': round(rtf_long, 4),
        'rtf_avg': round(rtf_avg, 4),
        'max_concurrency': max_concurrency,
        'gpu_name': gpu_name,
    }

    # --- 汇总 ---
    print()
    print('═' * 50)
    print('📊 基准测试汇总')
    print('═' * 50)
    print(f'   GPU:                  {gpu_name}')
    print(f'   短文本 RTF (平均):     {rtf_short:.4f}')
    print(f'   中文本 RTF (平均):     {rtf_medium:.4f}')
    print(f'   长文本 RTF (平均):     {rtf_long:.4f}')
    print(f'   三场景平均 RTF:        {rtf_avg:.4f}')
    print(f'   建议最大并发数:        {max_concurrency}')
    print()
    print('✅ 基准测试已全部完成')
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B",
                        help="Path to CosyVoice3 model directory")
    p.add_argument('--port', type=int, default=6006)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--no_trt', action='store_true', default=False,
                   help='Disable TensorRT acceleration for Flow decoder')
    p.add_argument('--no_vllm', action='store_true', default=False,
                   help='Disable vLLM acceleration for LLM inference')
    p.add_argument('--no_fp16', action='store_true', default=False,
                   help='Disable FP16 half-precision inference')
    args = p.parse_args()
    init_engine(args.model_dir, load_trt=not args.no_trt, load_vllm=not args.no_vllm, fp16=not args.no_fp16)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning', access_log=False)


if __name__ == '__main__':
    main()
