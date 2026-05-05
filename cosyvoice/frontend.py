# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
精简版 CosyVoiceFrontEnd — 用于分布式引擎。

引擎不存储 spk2info，只负责：
  1. 向量提取（extract_spk_vectors）→ 用完即丢
  2. 文本规范化（text_normalize）→ ttsfrd 保留
  3. model_input 组装（build_model_input）→ zero_shot / cross_lingual / instruct2
"""

from functools import partial
from typing import Generator, Callable
import json
import onnxruntime
import torch
import numpy as np
import whisper
import os
import re
import inflect
from cosyvoice.utils.file_utils import logging, load_wav
from cosyvoice.utils.frontend_utils import (
    contains_chinese, replace_blank, replace_corner_mark,
    remove_bracket, spell_out_number, split_paragraph, is_only_punctuation
)


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 allowed_special: str = 'all',
                 ttsfrd_resource: str = ''):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model, sess_options=option,
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"])
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()

        # --- ttsfrd 文本前端 ---
        try:
            import ttsfrd
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            # 优先用引擎传入的路径，其次用相对路径（往前一级到项目根）
            if ttsfrd_resource:
                resource_path = ttsfrd_resource
            else:
                resource_path = '{}/../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)
            assert self.frd.initialize(resource_path) is True, \
                'failed to initialize ttsfrd resource at {}'.format(resource_path)
            self.frd.set_lang_type('pinyinvg')
            self.text_frontend = 'ttsfrd'
            logging.info('use ttsfrd frontend')
        except Exception:
            try:
                from wetext import Normalizer as ZhNormalizer
                from wetext import Normalizer as EnNormalizer
                self.zh_tn_model = ZhNormalizer(remove_erhua=False)
                self.en_tn_model = EnNormalizer()
                self.text_frontend = 'wetext'
                logging.info('use wetext frontend')
            except Exception:
                self.text_frontend = ''
                logging.info('no frontend is available')

    # ======================= Tokenize =======================

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will return _extract_text_token_generator!')
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32).to(self.device)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    # ======================= Feature Extraction =======================

    def _extract_speech_token(self, prompt_wav):
        speech = load_wav(prompt_wav, 16000)
        duration = speech.shape[1] / 16000
        if duration > 30:
            raise ValueError('do not support extract speech token for audio longer than 30s, got {:.1f}s'.format(duration))
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(
            None, {
                self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
            })[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        import torchaudio.compliance.kaldi as kaldi
        speech = load_wav(prompt_wav, 16000)
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(
            None, {
                self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
            })[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, prompt_wav):
        speech = load_wav(prompt_wav, 24000)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    # ======================= Text Normalization =======================

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will skip text_normalize!')
            return [text]
        if '<|' in text and '|>' in text:
            text_frontend = False
        if text_frontend is False or text == '':
            return [text] if split is True else text
        text = text.strip()
        if self.text_frontend == 'ttsfrd':
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                if self.text_frontend == 'wetext':
                    text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(
                    text, partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                    "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
            else:
                if self.text_frontend == 'wetext':
                    text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(
                    text, partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                    "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    # ======================= Engine API =======================

    def extract_spk_vectors(self, prompt_text: str, prompt_wav, resample_rate: int = 24000):
        """
        提取说话人向量 — 用于 mode='extract'。
        提取完毕不存入任何缓存，直接返回序列化友好的 dict。

        返回:
            {
                'prompt_text_token':          List[int],        # (1, N)
                'prompt_text_token_len':      int,
                'llm_prompt_speech_token':    List[int],        # (1, N)
                'llm_prompt_speech_token_len': int,
                'flow_prompt_speech_token':   List[int],        # = 上面同一个
                'flow_prompt_speech_token_len': int,
                'prompt_speech_feat':         List[List[float]],  # (1, N, 80)
                'prompt_speech_feat_len':     int,
                'llm_embedding':              List[float],       # (1, 192)
                'flow_embedding':             List[float],       # = 上面同一个
            }
        """
        prompt_text = self.text_normalize(prompt_text, split=False, text_frontend=True)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
        speech_token, speech_token_len = self._extract_speech_token(prompt_wav)

        # CV2/3: force speech_feat % speech_token = 2
        if resample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat = speech_feat[:, :2 * token_len]
            speech_feat_len = torch.tensor([2 * token_len], dtype=torch.int32).to(self.device)
            speech_token = speech_token[:, :token_len]
            speech_token_len = torch.tensor([token_len], dtype=torch.int32).to(self.device)

        embedding = self._extract_spk_embedding(prompt_wav)

        return {
            'prompt_text_token': prompt_text_token.squeeze(0).cpu().tolist(),
            'prompt_text_token_len': prompt_text_token_len.item(),
            'llm_prompt_speech_token': speech_token.squeeze(0).cpu().tolist(),
            'llm_prompt_speech_token_len': speech_token_len.item(),
            'flow_prompt_speech_token': speech_token.squeeze(0).cpu().tolist(),
            'flow_prompt_speech_token_len': speech_token_len.item(),
            'prompt_speech_feat': speech_feat.squeeze(0).cpu().tolist(),
            'prompt_speech_feat_len': speech_feat_len.item(),
            'llm_embedding': embedding.squeeze(0).cpu().tolist(),
            'flow_embedding': embedding.squeeze(0).cpu().tolist(),
        }

    def build_model_input(self, tts_text: str, spk_vec: dict, mode: str = 'zero_shot'):
        """
        根据 mode 组装 model_input dict，给 model.tts() 用。

        spk_vec: extract_spk_vectors() 返回的 dict（由后端传来且已反序列化为 tensor）

        mode:
          - 'zero_shot'      → 完整 prompt
          - 'cross_lingual'  → 去掉 llm 侧的 prompt_text + speech_token
          - 'instruct2'      → 去掉 llm 侧的 speech_token
        """
        # 规范化文本——和原始 inference_zero_shot 一致：
        # text_normalize(tts_text, split=True) → 逐句 → _extract_text_token(每句)
        # 引擎收到的是已拆分的单句，所以 split=False，但必须规范化
        tts_text = self.text_normalize(tts_text, split=False, text_frontend=True)
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)

        spk_vec = {k: v.clone().to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in spk_vec.items()}

        model_input = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
            'prompt_text': spk_vec['prompt_text_token'].unsqueeze(0),
            'prompt_text_len': torch.tensor([spk_vec['prompt_text_token_len']], dtype=torch.int32).to(self.device),
            'llm_prompt_speech_token': spk_vec['llm_prompt_speech_token'].unsqueeze(0),
            'llm_prompt_speech_token_len': torch.tensor([spk_vec['llm_prompt_speech_token_len']], dtype=torch.int32).to(self.device),
            'flow_prompt_speech_token': spk_vec['flow_prompt_speech_token'].unsqueeze(0),
            'flow_prompt_speech_token_len': torch.tensor([spk_vec['flow_prompt_speech_token_len']], dtype=torch.int32).to(self.device),
            'prompt_speech_feat': spk_vec['prompt_speech_feat'].unsqueeze(0),
            'prompt_speech_feat_len': torch.tensor([spk_vec['prompt_speech_feat_len']], dtype=torch.int32).to(self.device),
            'llm_embedding': spk_vec['llm_embedding'].unsqueeze(0),
            'flow_embedding': spk_vec['flow_embedding'].unsqueeze(0),
        }

        if mode == 'cross_lingual':
            del model_input['prompt_text']
            del model_input['prompt_text_len']
            del model_input['llm_prompt_speech_token']
            del model_input['llm_prompt_speech_token_len']
        elif mode == 'instruct2':
            del model_input['llm_prompt_speech_token']
            del model_input['llm_prompt_speech_token_len']

        return model_input
