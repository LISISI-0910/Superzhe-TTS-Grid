#!/usr/bin/env python3
"""
CosyVoice3 TTS + ASR 一键部署脚本
=================================

用法:
  python setup.py                              # 完整部署（下载模型 + 安装依赖）
  python setup.py --skip-deps                  # 跳过系统依赖和 pip install
  python setup.py --model-dir ./my_models      # 自定义模型目录
  python setup.py --no-tts-model               # 跳过 TTS 模型下载
  python setup.py --no-asr-model               # 跳过 ASR 模型下载
  python setup.py --help
"""
import argparse
import os
import sys
import subprocess
import platform
import zipfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_MODEL_DIR_DEFAULT = os.path.join(SCRIPT_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
TTSFRD_DIR_DEFAULT = os.path.join(SCRIPT_DIR, "pretrained_models", "CosyVoice-ttsfrd")
ASR_MODEL_DIR_DEFAULT = os.path.join(SCRIPT_DIR, "faster-whisper-large-v3-turbo-ct2")


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def ok(msg):    print(f"  ✓ {msg}")
def info(msg):  print(f"  → {msg}")
def warn(msg):  print(f"  ⚠ {msg}")
def fail(msg):  print(f"  ✗ {msg}"); sys.exit(1)


def run(cmd, desc=""):
    """执行 shell 命令，打印并检查返回码。"""
    if desc:
        info(desc)
    print(f"     $ {cmd}")
    try:
        subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR)
    except subprocess.CalledProcessError:
        fail(f"命令失败: {cmd}")


def pip_install(packages, index_url=None):
    """pip install 封装，支持镜像源。"""
    cmd = f"{sys.executable} -m pip install {packages}"
    if index_url:
        cmd += f" -i {index_url} --trusted-host={index_url.split('://')[1].split('/')[0]}"
    run(cmd, f"安装 {packages}")


# ══════════════════════════════════════════════════════════
#  Step 1: 系统依赖（仅 Linux）
# ══════════════════════════════════════════════════════════

def install_system_deps():
    if platform.system() != "Linux":
        info("非 Linux，跳过系统依赖安装")
        return
    run("sudo apt-get update", "更新 apt 源")
    run("sudo apt-get install -y sox libsox-dev", "安装 sox")


# ══════════════════════════════════════════════════════════
#  Step 2-3: Python 依赖
# ══════════════════════════════════════════════════════════

def install_python_deps():
    # Step 2: requirements.txt
    req_path = os.path.join(SCRIPT_DIR, "requirements.txt")
    if os.path.exists(req_path):
        pip_install(f"-r {req_path}")
    else:
        warn("未找到 requirements.txt，跳过")

    # Step 3: 特定版本（阿里云镜像）
    pip_install(
        "vllm==0.11.0 transformers==4.57.1 numpy==1.26.4",
        index_url="https://mirrors.aliyun.com/pypi/simple/",
    )


# ══════════════════════════════════════════════════════════
#  Step 4: 下载模型
# ══════════════════════════════════════════════════════════

def download_tts_model(target_dir: str):
    """从 ModelScope 下载 CosyVoice3 模型。"""
    flag = os.path.join(target_dir, "llm.pt")
    if os.path.exists(flag):
        ok(f"TTS 模型已存在: {target_dir}")
        return

    from modelscope import snapshot_download
    info(f"下载 CosyVoice3 模型 → {target_dir}")
    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=target_dir)
    if os.path.exists(os.path.join(target_dir, "llm.pt")):
        ok("TTS 模型下载完成")
    else:
        fail("TTS 模型下载失败")


def download_asr_model(target_dir: str):
    """从 HuggingFace 下载 faster-whisper CTranslate2 模型。"""
    flag = os.path.join(target_dir, "model.bin")
    if os.path.exists(flag):
        ok(f"ASR 模型已存在: {target_dir}")
        return

    try:
        from huggingface_hub import snapshot_download
        info(f"下载 ASR 模型 → {target_dir}")
        snapshot_download(
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
    except ImportError:
        # huggingface_hub 不可用时，用 modelscope 或 git clone 兜底
        info("huggingface_hub 不可用，尝试 modelscope ...")
        try:
            from modelscope import snapshot_download as ms_snapshot
            ms_snapshot("deepdml/faster-whisper-large-v3-turbo-ct2", local_dir=target_dir)
        except Exception:
            repo = "https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2"
            info(f"使用 git clone: {repo}")
            if os.path.exists(target_dir):
                import shutil
                shutil.rmtree(target_dir)
            run(f"git clone {repo} {target_dir}", "git clone ASR 模型")

    if os.path.exists(os.path.join(target_dir, "model.bin")):
        ok("ASR 模型下载完成")
    else:
        fail("ASR 模型下载失败")


def install_ttsfrd(target_dir: str):
    """下载 ttsfrd 资源 + Linux wheel 安装。"""
    resource_dir = os.path.join(target_dir, "resource")
    if os.path.isdir(resource_dir) and os.listdir(resource_dir):
        ok(f"ttsfrd 资源已存在: {resource_dir}")
    else:
        from modelscope import snapshot_download
        info(f"下载 ttsfrd → {target_dir}")
        snapshot_download("iic/CosyVoice-ttsfrd", local_dir=target_dir)

        # 解压 resource.zip
        zip_path = os.path.join(target_dir, "resource.zip")
        if os.path.exists(zip_path):
            info("解压 resource.zip ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
            ok("解压完成")

    # Linux wheel 安装
    if platform.system() != "Linux":
        info("非 Linux，跳过 ttsfrd wheel")
        return

    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for fname in sorted(os.listdir(target_dir)):
        if fname.endswith(".whl") and py_ver in fname and "linux" in fname:
            info(f"安装 ttsfrd wheel: {fname}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", os.path.join(target_dir, fname)])
                ok("ttsfrd wheel 安装完成")
            except subprocess.CalledProcessError:
                warn("ttsfrd wheel 安装失败（不影响核心功能）")
            return
    warn("未找到匹配的 ttsfrd wheel（不影响核心功能）")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 TTS + ASR 一键部署")
    parser.add_argument("--skip-deps", action="store_true", help="跳过所有依赖安装")
    parser.add_argument("--model-dir", default="",
                        help="TTS 模型下载目录（默认 pretrained_models/Fun-CosyVoice3-0.5B）")
    parser.add_argument("--ttsfrd-dir", default="",
                        help="ttsfrd 资源目录（默认 pretrained_models/CosyVoice-ttsfrd）")
    parser.add_argument("--asr-model-dir", default="",
                        help="ASR 模型目录（默认 faster-whisper-large-v3-turbo-ct2）")
    parser.add_argument("--no-tts-model", action="store_true", help="跳过 TTS 模型下载")
    parser.add_argument("--no-asr-model", action="store_true", help="跳过 ASR 模型下载")
    args = parser.parse_args()

    tts_dir = args.model_dir or TTS_MODEL_DIR_DEFAULT
    ttsfrd_dir = args.ttsfrd_dir or TTSFRD_DIR_DEFAULT
    asr_dir = args.asr_model_dir or ASR_MODEL_DIR_DEFAULT

    print()
    print("=" * 55)
    print("  CosyVoice3 TTS + ASR 一键部署")
    print(f"  系统: {platform.system()}  Python: {sys.version_info.major}.{sys.version_info.minor}")
    print("=" * 55)
    print()

    # ── 依赖 ──
    if not args.skip_deps:
        print("── [1/4] 系统依赖 ──")
        install_system_deps()

        print("\n── [2/4] Python 依赖 ──")
        install_python_deps()
    else:
        info("跳过依赖安装 (--skip-deps)")

    # ── 模型 ──
    print("\n── [3/4] TTS 模型 + ttsfrd ──")
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(ttsfrd_dir, exist_ok=True)
    if not args.no_tts_model:
        download_tts_model(tts_dir)
    install_ttsfrd(ttsfrd_dir)

    print("\n── [4/4] ASR 模型 ──")
    os.makedirs(asr_dir, exist_ok=True)
    if not args.no_asr_model:
        download_asr_model(asr_dir)

    # ── 完成 ──
    print()
    print("=" * 55)
    print("  部署完成！")
    print()
    print("  启动服务:")
    print("    python serve.py")
    print()
    print("  查看文档:")
    print("    http://localhost:6006/docs")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
