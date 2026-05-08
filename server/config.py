"""
统一配置加载 —— config.yaml + 环境变量覆盖。
"""
import os
import yaml
from typing import Any


def _merge_env(cfg: dict, prefix: str, keys: list[str]):
    """环境变量覆盖同名字段（全大写，例: TTS_HOST → cfg["server"]["host"]）。"""
    for key in keys:
        env_val = os.environ.get(f"{prefix}_{key}".upper())
        if env_val is not None:
            # 尝试转换类型
            current = cfg.get(key)
            if isinstance(current, bool):
                cfg[key] = env_val.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                cfg[key] = int(env_val)
            elif isinstance(current, float):
                cfg[key] = float(env_val)
            else:
                cfg[key] = env_val


def load_config(path: str = "") -> dict[str, Any]:
    """读取 config.yaml，合并环境变量覆盖。

    搜索顺序:
      1. 参数 path
      2. 环境变量 CONFIG_PATH
      3. 项目根目录 config.yaml
    """
    if not path:
        path = os.environ.get("CONFIG_PATH", "")
    if not path:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "config.yaml")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # 环境变量覆盖
    srv = config.get("server", {})
    _merge_env(srv, "TTS", ["host", "port"])

    models = config.get("models", {})
    _merge_env(models, "TTS", ["tts_dir", "asr_dir"])

    accel = config.get("acceleration", {})
    _merge_env(accel, "TTS", ["trt", "vllm", "fp16"])

    limits = config.get("limits", {})
    _merge_env(limits, "TTS", ["tts_text_max", "prompt_text_max",
                "instruct_text_max", "audio_size_max_mb", "audio_duration_max_s"])

    gen = config.get("generation", {})
    _merge_env(gen, "TTS", ["hard_timeout_s", "idle_timeout_s"])

    asr_cfg = config.get("asr", {})
    _merge_env(asr_cfg, "TTS", ["concurrency"])

    bench = config.get("benchmark", {})
    _merge_env(bench, "TTS", ["warmup_rounds", "coarse_rounds", "fine_rounds"])

    return config
