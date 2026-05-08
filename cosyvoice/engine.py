# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 CosyVoice Engine Refactor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""
CosyVoice3 无状态引擎入口。
"""

import os
import time
import pickle
import base64
from typing import Generator, Dict
import torch
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.frontend import CosyVoiceFrontEnd
from cosyvoice.model import CosyVoice3Model
from cosyvoice.utils.class_utils import get_model_type


class CosyVoiceEngine:

    def __init__(self, model_dir: str, load_trt: bool = False,
                 load_vllm: bool = False, fp16: bool = False,
                 trt_concurrent: int = 1):
        self.model_dir = model_dir
        self.fp16 = fp16
        t0 = time.perf_counter()

        # --- 0. 模型目录 ---
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)

        # --- 1. 读 YAML ---
        for yaml_name in ['cosyvoice3.yaml', 'cosyvoice2.yaml', 'cosyvoice.yaml']:
            hyper_yaml_path = os.path.join(model_dir, yaml_name)
            if os.path.exists(hyper_yaml_path):
                break
        else:
            raise ValueError('no cosyvoice*.yaml found in {}'.format(model_dir))

        overrides = {}
        if 'cosyvoice2' in yaml_name or 'cosyvoice3' in yaml_name:
            overrides = {'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')}

        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=overrides)
        assert get_model_type(configs) == CosyVoice3Model, \
            '仅支持 CosyVoice3 模型: {}'.format(model_dir)

        self.sample_rate = configs['sample_rate']

        # speech tokenizer 版本
        if 'cosyvoice3' in yaml_name:
            speech_tokenizer = '{}/speech_tokenizer_v3.onnx'.format(model_dir)
        elif 'cosyvoice2' in yaml_name:
            speech_tokenizer = '{}/speech_tokenizer_v2.onnx'.format(model_dir)
        else:
            speech_tokenizer = '{}/speech_tokenizer_v1.onnx'.format(model_dir)

        # --- 2. 加载前端 ---
        ttsfrd_resource = os.path.join(os.path.dirname(model_dir), 'CosyVoice-ttsfrd', 'resource')
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            '{}/campplus.onnx'.format(model_dir),
            speech_tokenizer,
            configs['allowed_special'],
            ttsfrd_resource=ttsfrd_resource)
        print(f'   [1/3] 文本前端...   \u2713 {self.frontend.text_frontend}')

        # --- 3. 加载模型 ---
        if torch.cuda.is_available() is False and (load_trt or load_vllm or fp16):
            load_trt = load_vllm = fp16 = False

        self.model = CosyVoice3Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        llm_tag = 'Qwen2 0.5B'
        flow_tag = 'DiT'
        hifi_tag = 'CausalHiFT'

        if load_vllm:
            print('   [2/3] vLLM 引擎... ', end='', flush=True)
            self.model.load_vllm('{}/vllm'.format(model_dir))
            llm_tag += ' +vLLM'
            print('\u2713')
        else:
            llm_tag += ' PyTorch'

        if load_trt:
            print('   [3/3] TensorRT...   ', end='', flush=True)
            self.model.load_trt(
                '{}/flow.decoder.estimator.{}.mygpu.plan'.format(
                    model_dir, 'fp16' if fp16 else 'fp32'),
                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                trt_concurrent, fp16)
            flow_tag += ' +TRT'
            print('\u2713')
        else:
            flow_tag += ' PyTorch'

        del configs

        elapsed = time.perf_counter() - t0
        print(f'   模型: LLM={llm_tag} | Flow={flow_tag} | HiFiGAN={hifi_tag}')
        print(f'   耗时 {elapsed:.1f}s\n')

    # ======================= Public API =======================

    # CV3 特定编码
    CHAT_PREFIX = 'You are a helpful assistant.'
    END_MARKER = '<|endofprompt|>'
    ZS_PREFIX = f'{CHAT_PREFIX}{END_MARKER}'  # zero_shot / cross_lingual 前缀

    def extract_speaker(self, audio_path: str, prompt_text: str) -> str:
        """提取说话人向量，返回 base64 编码字符串。"""
        prompt_text = self._format_prompt(prompt_text, 'zero_shot')
        try:
            spk_vec = self.frontend.extract_spk_vectors(prompt_text, audio_path, self.sample_rate)
        except Exception:
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self._pack_spk_vec(spk_vec)

    def generate(self, mode: str, text: str, speaker_b64: str,
                 instruct_text: str = '') -> Generator[bytes, None, None]:
        """流式生成语音，返回 int16 PCM bytes (mono)。"""
        try:
            spk_tensors = self._unpack_spk_vec(speaker_b64)
        except Exception as e:
            raise ValueError(f"speaker_b64 无效，请重新提取: {e}") from e

        if mode == 'instruct2' and instruct_text:
            instruct_text = self._format_prompt(instruct_text, 'instruct2')
            instruct_token, instruct_len = self.frontend._extract_text_token(instruct_text)
            spk_tensors['prompt_text_token'] = instruct_token.squeeze(0)
            spk_tensors['prompt_text_token_len'] = instruct_len.item()

        segments = self.frontend.text_normalize(text, split=True, text_frontend=True)

        for seg in segments:
            seg = self._format_tts_text(seg, mode)
            model_input = self.frontend.build_model_input(seg, spk_tensors, mode=mode)
            for output in self.model.tts(**model_input, stream=True, speed=1.0):
                yield self._tensor_to_pcm(output['tts_speech'])
            del model_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del spk_tensors

    # ---- Internal format helpers ----

    def _format_prompt(self, text: str, gen_type: str) -> str:
        """按 CV3 格式编码 prompt_text / instruct_text。"""
        if self.END_MARKER in text:
            return text

        if gen_type == 'instruct2':
            if self.CHAT_PREFIX in text:
                return f'{text}{self.END_MARKER}'
            return f'{self.CHAT_PREFIX} {text}{self.END_MARKER}'
        else:
            return f'{self.ZS_PREFIX}{text}'

    def _format_tts_text(self, text: str, gen_type: str) -> str:
        """按 CV3 格式编码目标文本（cross_lingual 需要前缀）。"""
        if gen_type == 'cross_lingual':
            if self.ZS_PREFIX not in text:
                return f'{self.ZS_PREFIX}{text}'
        return text

    # ======================= Serialization Helpers =======================

    def _pack_spk_vec(self, spk_vec: dict) -> str:
        tensors = self._deserialize_spk_vec(spk_vec)
        return base64.b64encode(pickle.dumps(tensors, protocol=pickle.HIGHEST_PROTOCOL)).decode('ascii')

    @staticmethod
    def _unpack_spk_vec(spk_b64: str) -> dict:
        """base64 → pickle → tensor dict。"""
        return pickle.loads(base64.b64decode(spk_b64))

    @staticmethod
    def _deserialize_spk_vec(spk_vec: dict) -> Dict[str, torch.Tensor]:
        """将 list/bytes dict 还原为 tensor dict。"""
        result = {}
        for key in ['prompt_text_token', 'llm_prompt_speech_token',
                     'flow_prompt_speech_token']:
            if key in spk_vec:
                result[key] = torch.tensor(spk_vec[key], dtype=torch.int32)

        if 'prompt_speech_feat' in spk_vec:
            shape = spk_vec['prompt_speech_feat_shape']
            arr = np.frombuffer(spk_vec['prompt_speech_feat'], dtype=np.float32).reshape(shape)
            result['prompt_speech_feat'] = torch.tensor(arr)

        for key in ['llm_embedding', 'flow_embedding']:
            if key in spk_vec:
                result[key] = torch.tensor(spk_vec[key], dtype=torch.float32)

        for key in ['prompt_text_token_len', 'llm_prompt_speech_token_len',
                     'flow_prompt_speech_token_len', 'prompt_speech_feat_len']:
            if key in spk_vec:
                result[key] = spk_vec[key]

        return result

    @staticmethod
    def _tensor_to_pcm(audio: torch.Tensor) -> bytes:
        """[1, samples] float32 → int16 PCM bytes。"""
        audio = audio.squeeze(0).cpu()
        audio = audio.clamp(-1.0, 1.0) * 32767.0
        audio = audio.round().clamp(-32768, 32767).to(torch.int16)
        return audio.numpy().tobytes()
