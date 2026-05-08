from pydantic import BaseModel, Field


# ======================= TTS =======================

class TTSExtractResponse(BaseModel):
    speaker_b64: str = Field(..., description="说话人向量 base64 编码")


class TTSGenerateRequest(BaseModel):
    text: str = Field(..., description="要合成的目标文本")
    speaker_b64: str = Field(..., description="说话人向量 base64")
    mode: str = Field(default="zero_shot", description="zero_shot | cross_lingual | instruct2")


# ======================= ASR =======================

class ASRSegment(BaseModel):
    start: float
    end: float
    text: str


class ASRResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[ASRSegment]
    srt: str
    vtt: str


# ======================= Health =======================

class HealthResponse(BaseModel):
    status: str
    tts_available: bool
    asr_available: bool
    tts_concurrency: str = ""  # e.g. "2/4"
