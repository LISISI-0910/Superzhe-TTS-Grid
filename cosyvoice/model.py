# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
精简版 CosyVoice3Model — 无继承链，单类直达。

合并了原始三层的必要方法：
  - CosyVoiceModel:   load / load_trt / get_trt_kwargs / llm_job
  - CosyVoice2Model:  load_vllm / tts
  - CosyVoice3Model:  __init__ / token2wav
"""

import os
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper


class CosyVoice3Model:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16

        # CV3 streaming params
        self.token_hop_len = 25
        self.token_max_hop_len = 4 * self.token_hop_len  # 100
        self.stream_scale_factor = 2
        assert self.stream_scale_factor >= 1, \
            'stream_scale_factor should be greater than 1, change it according to your actual rtf'

        # rtf and decoding
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) \
            if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()

        # per-session dicts
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        # CV3 FSQ silent / breath tokens (CV1 would be [])
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

    # ======================= Load =======================

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        hift_state_dict = {k.replace('generator.', ''): v
                           for k, v in torch.load(hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(),
                                flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent,
                                                        device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape,
                'max_shape': max_shape, 'input_names': input_names}

    # ======================= LLM Job =======================

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uid):
        cur_silent_token_num, max_silent_token_num = 0, 5
        with self.llm_context, torch.amp.autocast('cuda',
                enabled=self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            if isinstance(text, Generator):
                assert not hasattr(self.llm, 'vllm'), \
                    'streaming input text is only implemented for CosyVoice2/3 and do not support vllm!'
                token_generator = self.llm.inference_bistream(
                    text=text,
                    prompt_text=prompt_text.to(self.device),
                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_speech_token=llm_prompt_speech_token.to(self.device),
                    prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]],
                                                         dtype=torch.int32).to(self.device),
                    embedding=llm_embedding.to(self.device))
            else:
                token_generator = self.llm.inference(
                    text=text.to(self.device),
                    text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_text=prompt_text.to(self.device),
                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_speech_token=llm_prompt_speech_token.to(self.device),
                    prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]],
                                                         dtype=torch.int32).to(self.device),
                    embedding=llm_embedding.to(self.device),
                    uuid=uid)
            for token in token_generator:
                if token in self.silent_tokens:
                    cur_silent_token_num += 1
                    if cur_silent_token_num > max_silent_token_num:
                        continue
                else:
                    cur_silent_token_num = 0
                self.tts_speech_token_dict[uid].append(token)
        self.llm_end_dict[uid] = True

    # ======================= Token2Wav =======================

    def token2wav(self, token, prompt_token, prompt_feat, embedding,
                  token_offset, uid, stream=False, finalize=False, speed=1.0):
        with torch.amp.autocast('cuda', enabled=self.fp16):
            tts_mel, _ = self.flow.inference(
                token=token.to(self.device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_token.to(self.device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                embedding=embedding.to(self.device),
                streaming=stream,
                finalize=finalize)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            if self.hift_cache_dict[uid] is not None:
                hift_cache_mel = self.hift_cache_dict[uid]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[uid]['mel'] = tts_mel
            else:
                self.hift_cache_dict[uid] = {'mel': tts_mel, 'speech_offset': 0}
            if speed != 1.0:
                assert token_offset == 0 and finalize is True, \
                    'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[uid]['speech_offset']:]
            self.hift_cache_dict[uid]['speech_offset'] += tts_speech.shape[1]
        return tts_speech

    # ======================= TTS (CV2 version, used by CV3) =======================

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32),
            flow_embedding=torch.zeros(0, 192),
            llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            stream=False, speed=1.0, **kwargs):
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job,
                                 args=(text, prompt_text, llm_prompt_speech_token,
                                       llm_embedding, this_uuid))
        else:
            # VC job — keep for compatibility but not used in CV3 normal flow
            self.tts_speech_token_dict[this_uuid] = source_speech_token.flatten().tolist()
            self.llm_end_dict[this_uuid] = True
            p = threading.Thread(target=lambda: None)
        p.start()
        if stream is True:
            token_offset = 0
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len)
                                   * self.token_hop_len - flow_prompt_speech_token.shape[1])
            while True:
                time.sleep(0.1)
                this_token_hop_len = (self.token_hop_len + prompt_token_pad
                                      if token_offset == 0 else self.token_hop_len)
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset \
                        >= this_token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid]
                        [:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]
                    ).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        token_offset=token_offset,
                        uid=this_uuid,
                        stream=stream,
                        finalize=False)
                    token_offset += this_token_hop_len
                    self.token_hop_len = min(self.token_max_hop_len,
                                             self.token_hop_len * self.stream_scale_factor)
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True \
                        and len(self.tts_speech_token_dict[this_uuid]) - token_offset \
                        < this_token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # remaining tokens
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=token_offset,
                uid=this_uuid,
                finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            p.join()
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=0,
                uid=this_uuid,
                finalize=True,
                speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        # cleanup
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()
