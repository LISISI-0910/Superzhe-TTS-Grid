# CosyVoice 3 Distributed Engine
import os
import sys

# 确保 third_party 依赖在 cosyvoice 任何子模块 import 前可访问
# matcha 被 flow_matching / decoder / hifigan 引用
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD_PARTY = os.path.join(_PROJ_ROOT, 'third_party', 'Matcha-TTS')
if os.path.isdir(_THIRD_PARTY) and _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from cosyvoice.engine import CosyVoiceEngine
from cosyvoice.model import CosyVoice3Model
from cosyvoice.frontend import CosyVoiceFrontEnd
from cosyvoice.client import CosyVoiceClient

__all__ = ['CosyVoiceEngine', 'CosyVoice3Model', 'CosyVoiceFrontEnd', 'CosyVoiceClient']
