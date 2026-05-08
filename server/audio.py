"""
AudioProcessor — 任意格式音频标准化 + Opus-OGG 流式编码。
"""
import os
import io
import tempfile
import numpy as np
from typing import Tuple, Callable, Generator


class AudioProcessor:
    """音频格式转换：入参标准化 → 引擎消费；PCM → Opus-OGG 流式输出。"""

    def __init__(self, output_sample_rate: int = 24000):
        self.sample_rate = output_sample_rate
        self._temp_files: list[str] = []

    # ======================= Validation =======================

    @staticmethod
    def probe_duration(audio_bytes: bytes) -> float:
        """用 ffmpeg 探针获取音频时长（秒），不完整解码。"""
        import av
        try:
            suffix = ".tmp"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(path, "wb") as f:
                f.write(audio_bytes)
            container = av.open(path, 'r')
            duration = float(container.duration) if container.duration else 0.0
            container.close()
            os.unlink(path)
            return duration
        except Exception:
            return 0.0

    # ======================= Input Normalize =======================

    def normalize(self, audio_bytes: bytes, original_name: str = "audio") -> str:
        """任意格式 → 24kHz mono s16le WAV 临时文件。"""
        suffix = os.path.splitext(original_name)[1] or ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="audio_")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(audio_bytes)

        import av
        try:
            # 如果 ffmpeg 能直接解码输入格式，就重采样到标准格式
            wav_path = path + ".wav"
            input_ = av.open(path, 'r')
            output = av.open(wav_path, 'w', 'wav')

            in_stream = input_.streams.audio[0]
            out_stream = output.add_stream('pcm_s16le', self.sample_rate)
            out_stream.layout = 'mono'

            for frame in input_.decode(in_stream):
                frame.pts = None
                for packet in out_stream.encode(frame):
                    output.mux(packet)

            for packet in out_stream.encode(None):
                output.mux(packet)

            output.close()
            input_.close()

            os.unlink(path)
            self._temp_files.append(wav_path)
            return wav_path
        except Exception:
            # 无法解码时保留原文件（让引擎自己尝试）
            if os.path.exists(path):
                self._temp_files.append(path)
            return path

    def clean(self, path: str = ""):
        """删除某个临时文件，或清空所有。"""
        if path:
            if path in self._temp_files:
                self._temp_files.remove(path)
            if os.path.exists(path):
                os.unlink(path)
        else:
            for p in self._temp_files:
                if os.path.exists(p):
                    os.unlink(p)
            self._temp_files.clear()

    # ======================= Output: PCM → Opus-OGG =======================

    @staticmethod
    def create_opus_encoder(sample_rate: int = 24000,
                            bitrate: int = 24000) -> Tuple[Callable[[bytes], bytes],
                                                          Callable[[], bytes]]:
        """创建流式 Opus-OGG 编码器。

        返回 (encode, flush):
          - encode(pcm_bytes) → ogg_bytes    每收到一块 PCM，返回新的 OGG 数据
          - flush() → ogg_bytes              刷新缓冲区，获取剩余 OGG 数据

        用法:
            encode, flush = AudioProcessor.create_opus_encoder()
            for pcm in pcm_chunks:
                data = encode(pcm)
                if data:
                    yield data
            yield flush()
        """
        import av

        class _OGGWriter(io.RawIOBase):
            def __init__(self):
                self._buf = io.BytesIO()

            def write(self, b):
                self._buf.write(b)
                return len(b)

            def writable(self):
                return True

            def flush(self):
                data = self._buf.getvalue()
                self._buf.seek(0)
                self._buf.truncate(0)
                return data

        writer = _OGGWriter()
        container = av.open(writer, 'w', 'ogg')
        ogg_stream = container.add_stream('libopus', sample_rate)
        ogg_stream.layout = 'mono'
        ogg_stream.bit_rate = bitrate

        def encode(pcm_bytes: bytes) -> bytes:
            arr = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(arr, format='s16', layout='mono')
            frame.sample_rate = sample_rate
            for packet in ogg_stream.encode(frame):
                container.mux(packet)
            return writer.flush()

        def flush() -> bytes:
            for packet in ogg_stream.encode(None):
                container.mux(packet)
            container.close()
            return writer.flush()

        return encode, flush
