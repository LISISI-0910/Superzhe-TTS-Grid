"""
CosyVoice3 引擎 SDK — 供外部代码以函数方式调用引擎 API。

用法:
    from cosyvoice.client import CosyVoiceClient

    client = CosyVoiceClient('http://localhost:6006')

    # 提取说话人向量
    spk_vec = client.extract('ref.wav', '参考文本')

    # 生成语音（返回 WAV bytes）
    wav_bytes = client.generate('合成文本', spk_vec, gen_type='zero_shot')

    # 生成并保存到文件
    client.generate('文本', spk_vec, save_to='output.wav')

    # 流式生成（生成器逐 chunk 返回）
    for chunk in client.generate_stream('文本', spk_vec):
        save_chunk(chunk)
"""

import base64
import io
import json
import os
import struct
import warnings
from typing import Dict, Generator, List, Optional, Union

import requests


class CosyVoiceClient:
    """CosyVoice3 引擎客户端"""

    def __init__(self, base_url: str = 'http://localhost:6006'):
        self.url = base_url.rstrip('/')

    # ---- 健康检查 ----

    def health(self) -> dict:
        resp = requests.get(f'{self.url}/health', timeout=10)
        return resp.json()

    # ---- 提取 ----

    def extract(self, prompt_wav: str, prompt_text: str,
                gen_type: str = 'zero_shot') -> dict:
        """
        提取说话人向量。

        参数:
            prompt_wav:   参考音频路径
            prompt_text:  参考文本（和音频内容一致）
            gen_type:     zero_shot | instruct2（影响前缀编码方式）

        返回:
            spk_vec dict — 包含 prompt_text_token、speech_token、embedding 等
        """
        with open(prompt_wav, 'rb') as f:
            wav_b64 = base64.b64encode(f.read()).decode()

        resp = requests.post(
            f'{self.url}/engine/process',
            json={'mode': 'extract', 'prompt_text': prompt_text,
                  'prompt_wav_b64': wav_b64, 'gen_type': gen_type},
            timeout=60,
        )
        data = resp.json()
        if data.get('status') != 'ok':
            raise RuntimeError(f"提取失败: {data}")
        return data['spk_vec']

    # ---- 生成（非流式） ----

    def generate(self, text: str, spk_vec: dict,
                 gen_type: str = 'zero_shot',
                 speed: float = 1.0,
                 save_to: Optional[str] = None) -> bytes:
        """
        非流式生成语音。

        参数:
            text:      合成文本
            spk_vec:   提取得到的说话人向量
            gen_type:  zero_shot | cross_lingual | instruct2
            speed:     语速（默认 1.0）
            save_to:   可选，保存 WAV 文件路径

        返回:
            WAV bytes（16-bit PCM）
        """
        resp = requests.post(
            f'{self.url}/engine/process',
            json={'mode': 'generate', 'gen_type': gen_type,
                  'text': text, 'spk_vec': spk_vec,
                  'stream': False, 'speed': speed},
            stream=False,
            timeout=300,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"生成失败 [{resp.status_code}]: {resp.text[:200]}")

        wav_bytes = resp.content
        if save_to:
            with open(save_to, 'wb') as f:
                f.write(wav_bytes)
        return wav_bytes

    # ---- 生成（流式） ----

    def generate_stream(self, text: str, spk_vec: dict,
                        gen_type: str = 'zero_shot',
                        speed: float = 1.0,
                        strip_header: bool = False) -> Generator[bytes, None, None]:
        """
        流式生成语音，逐 chunk 返回 WAV 字节流。

        参数:
            text:          合成文本
            spk_vec:       说话人向量
            gen_type:      zero_shot | cross_lingual | instruct2
            speed:         语速
            strip_header:  True 时过滤掉 WAV header，只保留 PCM 裸数据

        返回:
            Generator[bytes] — 每 chunk 是 WAV 数据片段
        """
        resp = requests.post(
            f'{self.url}/engine/process',
            json={'mode': 'generate', 'gen_type': gen_type,
                  'text': text, 'spk_vec': spk_vec,
                  'stream': True, 'speed': speed},
            stream=True,
            timeout=300,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"生成失败 [{resp.status_code}]: {resp.text[:200]}")

        header_skipped = False
        for chunk in resp.iter_content(chunk_size=8192):
            if strip_header and not header_skipped:
                # 跳过 WAV header（前 44 字节）
                if len(chunk) <= 44:
                    continue
                chunk = chunk[44:]
                header_skipped = True
            yield chunk

    # ---- 生成并播放 ----

    def generate_and_play(self, text: str, spk_vec: dict,
                          gen_type: str = 'zero_shot',
                          speed: float = 1.0):
        """
        生成语音并尝试用系统播放器播放。
        需要安装 ffplay / aplay / mpv。
        """
        import shutil
        import subprocess
        import tempfile

        wav_bytes = self.generate(text, spk_vec, gen_type, speed)
        player = None
        for cmd in ['ffplay', 'aplay', 'paplay', 'mpv']:
            if shutil.which(cmd):
                player = cmd
                break

        if not player:
            path = f'/tmp/tts_{os.urandom(4).hex()}.wav'
            with open(path, 'wb') as f:
                f.write(wav_bytes)
            print(f'没有播放器，已保存: {path}')
            return

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name

        try:
            if player == 'ffplay':
                subprocess.run([player, '-nodisp', '-autoexit', tmp],
                               stderr=subprocess.DEVNULL)
            elif player == 'aplay':
                subprocess.run([player, '-q', tmp])
            elif player == 'mpv':
                subprocess.run([player, '--no-video', tmp],
                               stderr=subprocess.DEVNULL)
        finally:
            os.unlink(tmp)

    # ---- 工具：spk_vec 序列化 ----

    @staticmethod
    def save_spk_vec(spk_vec: dict, path: str):
        """保存说话人向量到 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(spk_vec, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_spk_vec(path: str) -> dict:
        """从 JSON 文件加载说话人向量"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
