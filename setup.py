#!/usr/bin/env python3
"""
CosyVoice3 模型 & ttsfrd 资源下载脚本（Windows / Linux / macOS 通用）
============================================================

功能：
  1. 自动安装缺失的 Python 依赖
  2. 从 ModelScope 下载 CosyVoice3 模型
  3. 从 ModelScope 下载 ttsfrd 资源并解压

用法：
  python setup.py                        # 完整安装
  python setup.py --no-deps              # 跳过 pip install
  python setup.py --model-dir PATH       # 自定义模型目录
"""

import argparse
import os
import sys
import zipfile
import subprocess
import platform

# ─── 路径 ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
TTSFRD_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "CosyVoice-ttsfrd")
MODEL_FLAG = os.path.join(MODEL_DIR, "llm.pt")
TTSFRD_FLAG = os.path.join(TTSFRD_DIR, "resource")  # 解压后的目录


def info(msg):
    print(f"  [INFO] {msg}")


def ok(msg):
    print(f"  [ OK] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


# ═══════════════════════════════════════════════════
#  1. 依赖安装
# ═══════════════════════════════════════════════════
def install_deps():
    """自动安装必要的 Python 包"""
    needed = []

    try:
        import modelscope  # noqa
    except ImportError:
        needed.append("modelscope")

    if needed:
        info(f"发现缺失依赖: {', '.join(needed)}，正在安装...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + needed
            )
            ok("依赖安装完成")
        except subprocess.CalledProcessError as e:
            fail(f"pip install 失败: {e}")
    else:
        ok("所有依赖已就绪")


# ═══════════════════════════════════════════════════
#  2. 下载模型
# ═══════════════════════════════════════════════════
def download_model(model_dir: str):
    from modelscope import snapshot_download

    if os.path.exists(os.path.join(model_dir, "llm.pt")):
        ok(f"模型已存在，跳过下载: {model_dir}")
        return

    info(f"正在下载 CosyVoice3 模型...")
    info(f"  本地路径: {model_dir}")
    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=model_dir)

    if os.path.exists(os.path.join(model_dir, "llm.pt")):
        ok("模型下载完成")
    else:
        fail("模型下载失败，请检查网络或手动下载")


# ═══════════════════════════════════════════════════
#  3. 下载 ttsfrd
# ═══════════════════════════════════════════════════
def download_ttsfrd(ttsfrd_dir: str):
    from modelscope import snapshot_download

    resource_dir = os.path.join(ttsfrd_dir, "resource")
    if os.path.exists(resource_dir) and os.listdir(resource_dir):
        ok(f"ttsfrd 资源已存在，跳过下载: {resource_dir}")
        return

    info("正在下载 ttsfrd 资源...")
    info(f"  本地路径: {ttsfrd_dir}")
    snapshot_download("iic/CosyVoice-ttsfrd", local_dir=ttsfrd_dir)

    # 解压 resource.zip
    resource_zip = os.path.join(ttsfrd_dir, "resource.zip")
    if os.path.exists(resource_zip) and not os.path.exists(resource_dir):
        info("正在解压 resource.zip...")
        with zipfile.ZipFile(resource_zip, "r") as zf:
            zf.extractall(ttsfrd_dir)
        ok("解压完成")
    elif os.path.exists(resource_dir):
        ok("resource 已就绪")
    else:
        warn("未找到 resource.zip，请检查下载是否完整")


# ═══════════════════════════════════════════════════
#  4. 安装 ttsfrd wheel（Linux 专用）
# ═══════════════════════════════════════════════════
def install_ttsfrd_wheel(ttsfrd_dir: str):
    """安装 ttsfrd 的 .whl 包（仅 Linux，Windows 没有预编译包）"""
    if platform.system() != "Linux":
        info("非 Linux 系统，跳过 ttsfrd wheel 安装")
        return

    # 找到匹配当前 Python 版本的 wheel
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for fname in os.listdir(ttsfrd_dir):
        if fname.endswith(".whl") and py_ver in fname and "linux" in fname:
            wheel_path = os.path.join(ttsfrd_dir, fname)
            info(f"安装 ttsfrd wheel: {fname}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", wheel_path]
                )
                ok("ttsfrd 安装完成")
            except subprocess.CalledProcessError as e:
                warn(f"ttsfrd 安装失败（可忽略，不影响核心功能）: {e}")
            return

    # 没找到匹配的 wheel，尝试任意 cp310 版本
    for fname in os.listdir(ttsfrd_dir):
        if fname.endswith(".whl") and "linux" in fname:
            wheel_path = os.path.join(ttsfrd_dir, fname)
            info(f"尝试安装 ttsfrd wheel（版本可能不匹配）: {fname}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", wheel_path]
                )
                return
            except subprocess.CalledProcessError:
                pass

    info("未找到匹配的 ttsfrd wheel，跳过（引擎运行不依赖此包）")


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CosyVoice3 模型 & ttsfrd 资源下载脚本（跨平台）"
    )
    parser.add_argument(
        "--no-deps", action="store_true", help="跳过依赖安装"
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help=f"模型下载目录 (默认: {MODEL_DIR})",
    )
    parser.add_argument(
        "--ttsfrd-dir",
        default=TTSFRD_DIR,
        help=f"ttsfrd 下载目录 (默认: {TTSFRD_DIR})",
    )
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  CosyVoice3 模型 & ttsfrd 资源下载")
    print(f"  系统: {platform.system()} | Python: {sys.version_info.major}.{sys.version_info.minor}")
    print("=" * 55)
    print()

    # 1. 安装依赖
    if not args.no_deps:
        install_deps()
    else:
        info("跳过依赖安装 (--no-deps)")

    print()

    # 2. 下载模型
    print("─" * 45)
    print("  [1/3] 下载 CosyVoice3 模型")
    print("─" * 45)
    os.makedirs(args.model_dir, exist_ok=True)
    download_model(args.model_dir)

    print()

    # 3. 下载 ttsfrd
    print("─" * 45)
    print("  [2/3] 下载 ttsfrd 资源")
    print("─" * 45)
    os.makedirs(args.ttsfrd_dir, exist_ok=True)
    download_ttsfrd(args.ttsfrd_dir)

    print()

    # 4. 安装 ttsfrd wheel（仅 Linux）
    print("─" * 45)
    print("  [3/3] 安装 ttsfrd（Linux 专用）")
    print("─" * 45)
    install_ttsfrd_wheel(args.ttsfrd_dir)

    print()
    print("=" * 55)
    print("  全部完成！")
    print()
    print("  启动引擎:")
    print("    python run.py")
    print()
    print("  或使用 Docker:")
    print("    docker compose up -d --build")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
