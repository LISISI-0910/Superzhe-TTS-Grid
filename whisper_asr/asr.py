import os
from typing import Optional, Dict, Any
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio


class WhisperASR:
    """Whisper ASR engine wrapping faster-whisper (CTranslate2 backend).

    Usage:
        # 中文转录（默认）
        with WhisperASR() as asr:
            r = asr.transcribe("audio.wav")
            print(r["text"])   # 完整文本，带标点
            print(r["srt"])    # SRT 字幕
            print(r["vtt"])    # WebVTT 字幕

        # 自动检测语言 + 翻译
        with WhisperASR(language=None) as asr:
            r = asr.translate("audio.wav")
            print(r["text"])   # 英文翻译
    """

    _SPLIT_PUNCTUATION = frozenset("。！？.!?，,、；;")

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        language: Optional[str] = "zh",
        compute_type: str = "int8_float16",
        cpu_threads: int = 0,
        num_workers: int = 1,
    ):
        if not model_path:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(root, "faster-whisper-large-v3-turbo-ct2")

        if cpu_threads == 0:
            cpu_threads = max(1, os.cpu_count() // 2)

        self._language = language           # None = auto-detect
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type

        self._model = WhisperModel(
            model_path,
            device="cuda" if device == "cuda" else "cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        self._warmup()

    # ======================= Public API =======================

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """转录，返回 {text, language, duration, segments, srt, vtt}。"""
        return self._run(audio_path)

    @property
    def info(self) -> Dict[str, Any]:
        """引擎元信息。"""
        return {
            "model_path": self._model_path,
            "device": self._device,
            "compute_type": self._compute_type,
            "language": self._language,
        }

    def close(self) -> None:
        """释放底层模型持有的 GPU 资源。"""
        if hasattr(self, "_model"):
            del self._model

    def __enter__(self) -> "WhisperASR":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ======================= Internal =======================

    def _run(self, audio_path: str) -> Dict[str, Any]:
        audio = decode_audio(audio_path)

        lang = self._language
        initial_prompt = None
        if self._language in ("zh", "zh-CN"):
            initial_prompt = (
                "以下是普通话的句子，可能包含中英文混合，"
                "如CosyVoice、iPhone、AI等英文词汇请保留原文。"
                "请包含正确的标点符号。"
            )

        segments_generator, info = self._model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            initial_prompt=initial_prompt,
            vad_filter=True,
            word_timestamps=True,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        segments, text_parts = self._collect_segments(segments_generator)

        join_char = "" if info.language in ("zh", "ja") else " "
        text = join_char.join(text_parts)

        return {
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "segments": segments,
            "srt": self._fmt_srt(segments),
            "vtt": self._fmt_vtt(segments),
        }

    # ---- Segment collection ----

    def _collect_segments(self, generator):
        segments, text_parts = [], []
        for seg in generator:
            if not seg.text.strip():
                continue
            if seg.words:
                buf_words, buf_start = [], None
                for w in seg.words:
                    if not w.word.strip():
                        continue
                    if buf_start is None:
                        buf_start = w.start
                    buf_words.append(w.word)
                    if w.word.strip()[-1] in self._SPLIT_PUNCTUATION:
                        sentence = "".join(buf_words).strip()
                        text_parts.append(sentence)
                        segments.append({"start": buf_start, "end": w.end, "text": sentence})
                        buf_words, buf_start = [], None
                if buf_words:
                    sentence = "".join(buf_words).strip()
                    text_parts.append(sentence)
                    segments.append({"start": buf_start, "end": seg.end, "text": sentence})
            else:
                text_parts.append(seg.text.strip())
                segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        return segments, text_parts

    # ---- Subtitle formatters ----

    @staticmethod
    def _fmt_srt(segments: list) -> str:
        """segments → SRT (SubRip) 字幕（去标点）。"""
        parts = []
        for i, seg in enumerate(segments, 1):
            start = WhisperASR._fmt_time_srt(seg["start"])
            end = WhisperASR._fmt_time_srt(seg["end"])
            text = seg["text"].strip("".join(WhisperASR._SPLIT_PUNCTUATION) + " ")
            parts.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(parts)

    @staticmethod
    def _fmt_vtt(segments: list) -> str:
        """segments → WebVTT 字幕（去标点）。"""
        parts = ["WEBVTT", ""]
        for seg in segments:
            start = WhisperASR._fmt_time_vtt(seg["start"])
            end = WhisperASR._fmt_time_vtt(seg["end"])
            text = seg["text"].strip("".join(WhisperASR._SPLIT_PUNCTUATION) + " ")
            parts.append(f"{start} --> {end}\n{text}\n")
        return "\n".join(parts)

    @staticmethod
    def _fmt_time_srt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _fmt_time_vtt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    # ---- Warmup ----

    def _warmup(self):
        """触发模型权重加载 + VAD 懒初始化，避免首次调用卡顿。"""
        try:
            # 用随机噪声代替静音，确保 VAD 不会直接跳过
            rng = np.random.RandomState(42)
            dummy = rng.randn(16000).astype(np.float32) * 0.01  # 1s @ 16kHz
            list(self._model.transcribe(
                dummy,
                language=self._language or "zh",
                vad_filter=True,
                word_timestamps=False,
                temperature=[0.0],
            ))
            print(f"  [asr] warmup 完成，模型已加载到 {self._device}")
        except Exception as e:
            print(f"  [asr] warmup 失败: {e}")
