"""
EngineManager — TTS + ASR 双实例生命周期、并发基准测试、信号量限流。
"""
import os
import time
import asyncio
import logging
import threading
import torch
from cosyvoice.engine import CosyVoiceEngine
from whisper_asr import WhisperASR
from server.audio import AudioProcessor


def _silence_noisy_loggers():
    """压制 vLLM / multipart 等第三方库的 DEBUG/WARNING 噪音。"""
    for name in ("vllm", "multipart", "multipart.multipart",
                 "httptools", "asyncio", "uvicorn",
                 "faster_whisper", "ctranslate2"):
        logging.getLogger(name).setLevel(logging.ERROR)


class EngineManager:
    """管理 TTS/ASR 单例生命周期 + 并发控制。"""

    def __init__(self):
        self.tts: CosyVoiceEngine | None = None
        self.asr: WhisperASR | None = None
        self.audio = AudioProcessor()
        self.tts_sem: asyncio.Semaphore | None = None
        self.asr_sem: asyncio.Semaphore | None = None
        self._tts_max_concurrency: int = 1
        self._bench_warmup = 2
        self._bench_coarse = 2
        self._bench_fine = 3

    # ======================= Lifecycle =======================

    async def startup(self,
                      tts_model_dir: str = "",
                      asr_model_dir: str = "",
                      load_trt: bool = False,
                      load_vllm: bool = False,
                      fp16: bool = False,
                      asr_concurrency: int = 2,
                      warmup_rounds: int = 2,
                      coarse_rounds: int = 2,
                      fine_rounds: int = 3):
        loop = asyncio.get_running_loop()

        # --- 1. TTS ---
        if not tts_model_dir:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tts_model_dir = os.path.join(root, "pretrained_models", "Fun-CosyVoice3-0.5B")

        print("[manager] 加载 TTS 引擎...")
        self.tts = await loop.run_in_executor(
            None, lambda: CosyVoiceEngine(tts_model_dir, load_trt=load_trt,
                                          load_vllm=load_vllm, fp16=fp16))
        _silence_noisy_loggers()

        # --- 2. ASR ---
        print("[manager] 加载 ASR 引擎...")
        asr_kwargs = {}
        if asr_model_dir:
            asr_kwargs["model_path"] = asr_model_dir
        self.asr = await loop.run_in_executor(None, lambda: WhisperASR(**asr_kwargs))

        # --- 3. TTS 基准测试 ---
        self._bench_warmup = warmup_rounds
        self._bench_coarse = coarse_rounds
        self._bench_fine = fine_rounds
        print("[manager] TTS 并发基准测试...")
        self._tts_max_concurrency = await loop.run_in_executor(None, self._benchmark_tts)
        print(f"[manager] TTS 最大并发: {self._tts_max_concurrency}")

        self.tts_sem = asyncio.Semaphore(self._tts_max_concurrency)
        self.asr_sem = asyncio.Semaphore(asr_concurrency)

        print(f"[manager] 就绪  |  TTS 并发={self._tts_max_concurrency}  "
              f" |  ASR 并发={asr_concurrency}")

    async def shutdown(self):
        print("[manager] 关闭引擎...")
        if self.tts:
            del self.tts
            self.tts = None
        if self.asr:
            await asyncio.get_running_loop().run_in_executor(None, self.asr.close)
            self.asr = None
        self.audio.clean()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[manager] 已关闭")

    @property
    def ready(self) -> bool:
        return self.tts is not None and self.asr is not None

    def tts_status(self) -> str:
        if self.tts_sem is None:
            return "0/0"
        used = self._tts_max_concurrency - self.tts_sem._value
        return f"{used}/{self._tts_max_concurrency}"

    # ======================= Benchmark =======================

    def _benchmark_tts(self) -> int:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ref_wav = os.path.join(root, "benchmark", "ref.wav")
        ref_txt = os.path.join(root, "benchmark", "ref.txt")

        with open(ref_txt, encoding="utf-8") as f:
            ref_text = f.read().strip()

        print("  [bench] 提取说话人向量...", flush=True)
        spk_b64 = self.tts.extract_speaker(ref_wav, ref_text)
        sr = self.tts.sample_rate

        # ── 预热 ──
        warmup_text = (
            "这是一段用于预热的较长文本，目的是触发所有CUDA内核编译和内存分配，"
            "包括LLM的自注意力缓存、Flow模型的DiT解码器缓存，"
            "以及HiFiGAN的流式缓存等。确保后续基准测试结果稳定可靠。"
        )
        wr = self._bench_warmup
        print(f"  [bench] 预热中（{len(warmup_text)} 字, {wr} 轮）...", flush=True)
        for i in range(wr):
            list(self.tts.generate(mode="zero_shot", text=warmup_text,
                                   speaker_b64=spk_b64))
            print(f"  [bench]   预热 {i + 1}/{wr} 完成", flush=True)

        test_text = (
            "这是一段用于并发基准测试的语音合成文本，长度约为六十个字左右，"
            "可以更好地模拟真实使用场景的负载压力。短文本会低估 GPU 争抢程度，"
            "导致并发上限偏乐观，所以这里用更长的文本。"
        )
        print(f"  [bench] 测试文本: \"{test_text}\" ({len(test_text)} 字)", flush=True)

        # ── 阶段 1：指数探测（1,2,4,8,16,32,64...）找上限 ──
        last_good = 1
        n = 1
        print("  [bench] 阶段1 指数探测...", flush=True)
        while n <= 64:
            result = self._measure_concurrency(n, test_text, spk_b64, sr,
                                                  rounds=self._bench_coarse)
            if result is None:
                return max(1, n - 1)
            avg, max_rtf = result
            mark = "✗" if max_rtf > 1.0 else "✓"
            print(f"  [bench]   并发={n:<3}  avg={avg:.2f}  max={max_rtf:.2f}  {mark}", flush=True)
            if max_rtf > 1.0:
                break
            last_good = n
            n = min(n * 2, 64)
        else:
            print(f"  [bench] 并发=64 仍全 RTF≤1，上限 64", flush=True)
            return 64

        # ── 阶段 2：二分搜索定精确上限 ──
        lo, hi = last_good, n
        print(f"  [bench] 阶段2 二分搜索 ({lo} → {hi})...", flush=True)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            result = self._measure_concurrency(mid, test_text, spk_b64, sr,
                                                  rounds=self._bench_fine)
            if result is None:
                return lo
            avg, max_rtf = result
            mark = "✗" if max_rtf > 1.0 else "✓"
            print(f"  [bench]   并发={mid:<3}  avg={avg:.2f}  max={max_rtf:.2f}  {mark}", flush=True)
            if max_rtf > 1.0:
                hi = mid
            else:
                lo = mid

        return lo

    # ── 测单个并发级别，返回 (avg_rtf, max_rtf) ──

    def _measure_concurrency(self, n: int, text: str, spk_b64: str,
                             sr: int, rounds: int = 3) -> tuple[float, float] | None:
        """返回 (avg_rtf, max_rtf)。任一测量超 1.0 即判定该级别失败。"""
        all_rtfs = []
        for _ in range(rounds):
            barrier = threading.Barrier(n)
            lock = threading.Lock()
            errors = []
            round_rtfs = []

            def worker():
                try:
                    barrier.wait()
                    t0 = time.perf_counter()
                    chunks = list(self.tts.generate(
                        mode="zero_shot", text=text,
                        speaker_b64=spk_b64))
                    latency = time.perf_counter() - t0
                    total_bytes = sum(len(c) for c in chunks)
                    duration = total_bytes / (sr * 2)
                    with lock:
                        round_rtfs.append(latency / duration if duration > 0 else 999)
                except Exception as e:
                    errors.append(e)

            t_wall_start = time.perf_counter()
            threads = [threading.Thread(target=worker) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall_sec = time.perf_counter() - t_wall_start

            if errors:
                print(f"  [bench]   并发={n} 出错: {errors[0]}", flush=True)
                return None

            all_rtfs.extend(round_rtfs)
            print(f"  [bench]     wall={wall_sec:.1f}s  "
                  f"rtf=[{min(round_rtfs):.2f}..{max(round_rtfs):.2f}]  "
                  f"({n} 线程并发)", flush=True)

        avg = sum(all_rtfs) / len(all_rtfs) if all_rtfs else 999
        max_ = max(all_rtfs) if all_rtfs else 999
        return avg, max_
