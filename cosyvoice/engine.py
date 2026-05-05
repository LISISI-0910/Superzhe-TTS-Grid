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
import threading
from typing import Generator, Dict, Any, List, Union
import torch
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
        print(f'   [1/3] 文本前端...   ✓ {self.frontend.text_frontend}')

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
            print('✓')
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
            print('✓')
        else:
            flow_tag += ' PyTorch'

        del configs

        elapsed = time.perf_counter() - t0
        print(f'   模型: LLM={llm_tag} | Flow={flow_tag} | HiFiGAN={hifi_tag}')
        print(f'   耗时 {elapsed:.1f}s\n')

        # 统计计数器
        self._stats_lock = threading.Lock()
        self._active = 0           # 当前正在生成
        self._total_gen = 0        # 总生成
        self._total_ext = 0        # 总提取
        self._gen_by_mode = {'zero_shot': 0, 'cross_lingual': 0, 'instruct2': 0}

        # 速率统计：记录每次 process() 调用的时间戳
        self._timestamps_lock = threading.Lock()
        self._request_timestamps: List[float] = []

    # ======================= Stats =======================

    @property
    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                'active': self._active,
                'total_gen': self._total_gen,
                'total_ext': self._total_ext,
                'by_mode': dict(self._gen_by_mode),
            }

    @property
    def rate_stats(self) -> Dict[str, int]:
        """请求速率统计：1分钟内、5分钟内的请求数"""
        now = time.time()
        with self._timestamps_lock:
            # 移除 5 分钟前的旧时间戳
            cutoff = now - 300
            self._request_timestamps = [t for t in self._request_timestamps if t >= cutoff]
            total = len(self._request_timestamps)
            one_min = sum(1 for t in self._request_timestamps if t >= now - 60)
            five_min = total
        return {
            'requests_1m': one_min,
            'requests_5m': five_min,
        }

    # ======================= Process Entry =======================

    # CV3 特定编码
    CHAT_PREFIX = 'You are a helpful assistant.'
    END_MARKER = '<|endofprompt|>'
    ZS_PREFIX = f'{CHAT_PREFIX}{END_MARKER}'  # zero_shot / cross_lingual 前缀

    def process(self, mode: str, **kwargs) -> Union[Dict[str, Any], Generator]:
        # 记录请求时间戳用于速率统计
        with self._timestamps_lock:
            self._request_timestamps.append(time.time())
        if mode == 'extract':
            with self._stats_lock:
                self._total_ext += 1
            return self._process_extract(**kwargs)
        elif mode == 'generate':
            return self._generate_counted(**kwargs)
        else:
            raise ValueError("mode must be 'extract' or 'generate', got '{}'".format(mode))

    def _generate_counted(self, **kwargs) -> Generator:
        """Generator 包装：统计在生成器耗尽时递减 active。"""
        gen_type = kwargs.get('gen_type', 'zero_shot')
        with self._stats_lock:
            self._active += 1
            self._total_gen += 1
            if gen_type in self._gen_by_mode:
                self._gen_by_mode[gen_type] += 1
        try:
            yield from self._process_generate(**kwargs)
        finally:
            with self._stats_lock:
                self._active -= 1

    # ---- Format helpers ----

    def _format_prompt(self, text: str, gen_type: str) -> str:
        """按 CV3 格式编码 prompt_text / instruct_text。"""
        if self.END_MARKER in text:
            return text  # 用户已带前缀，不重复加

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

    # ---- Extract / Generate ----

    def _process_extract(self, prompt_text: str, prompt_wav,
                         gen_type: str = 'zero_shot') -> Dict[str, Any]:
        """提取说话人向量，自动加 CV3 前缀编码。"""
        prompt_text = self._format_prompt(prompt_text, gen_type)
        try:
            result = self.frontend.extract_spk_vectors(prompt_text, prompt_wav, self.sample_rate)
        except Exception:
            # 任何异常都要确保释放 GPU 中间张量，防止显存泄漏
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result

    def _process_generate(self, text: str, spk_vec: dict,
                          gen_type: str = 'zero_shot',
                          stream: bool = False, speed: float = 1.0,
                          instruct_text: str = '') -> Generator:
        """生成语音。cross_lingual 自动加前缀；长文本拆分后每句也带前缀。
        
        instruct_text: instruct2 模式专用，传纯指令文本（如"请用广东话表达"），
                       引擎自动格式化并替换 spk_vec 中的 prompt_text_token，
                       无需为每条指令单独提取向量。
        """
        spk_tensors = self._deserialize_spk_vec(spk_vec)

        # instruct2 模式：用传入的指令文本替换 spk_vec 中的 prompt_text
        if gen_type == 'instruct2' and instruct_text:
            instruct_text = self._format_prompt(instruct_text, 'instruct2')
            instruct_token, instruct_len = self.frontend._extract_text_token(instruct_text)
            spk_tensors['prompt_text_token'] = instruct_token.squeeze(0)
            spk_tensors['prompt_text_token_len'] = instruct_len.item()

        # cross_lingual: 目标文本加 Chat 前缀
        # text = self._format_tts_text(text, gen_type)
        segments = self.frontend.text_normalize(text, split=True, text_frontend=True)

        for seg in segments:
            seg = self._format_tts_text(seg, gen_type)
            model_input = self.frontend.build_model_input(seg, spk_tensors, mode=gen_type)
            for output in self.model.tts(**model_input, stream=stream, speed=speed):
                yield output
            # 每句推理完立即释放 GPU 张量
            del model_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 循环结束，释放 CPU 侧的 spk_tensors
        del spk_tensors

    # ======================= Serialization Helpers =======================

    @staticmethod
    def _deserialize_spk_vec(spk_vec: dict) -> Dict[str, torch.Tensor]:
        """将后端传来的 list-based dict 还原为 tensor dict。"""
        result = {}
        for key in ['prompt_text_token', 'llm_prompt_speech_token',
                     'flow_prompt_speech_token']:
            if key in spk_vec:
                result[key] = torch.tensor(spk_vec[key], dtype=torch.int32)

        if 'prompt_speech_feat' in spk_vec:
            result['prompt_speech_feat'] = torch.tensor(spk_vec['prompt_speech_feat'],
                                                        dtype=torch.float32)

        for key in ['llm_embedding', 'flow_embedding']:
            if key in spk_vec:
                result[key] = torch.tensor(spk_vec[key], dtype=torch.float32)

        # scalar lengths
        for key in ['prompt_text_token_len', 'llm_prompt_speech_token_len',
                     'flow_prompt_speech_token_len', 'prompt_speech_feat_len']:
            if key in spk_vec:
                result[key] = spk_vec[key]

        return result

    def serialize_spk_vec(self, spk_vec: dict) -> Dict[str, Any]:
        """确保所有值可 JSON 序列化（用于 extract 返回）。"""
        out = {}
        for k, v in spk_vec.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu().tolist()
            elif isinstance(v, list):
                out[k] = v
            else:
                out[k] = v
        return out
