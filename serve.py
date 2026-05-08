#!/usr/bin/env python3
"""
CosyVoice3 TTS + ASR API 启动脚本。配置在 config.yaml。
用法:
  python serve.py                    # 默认
  python serve.py --port 9000        # 覆盖端口
  python serve.py --no-trt           # 关闭 TensorRT
  python serve.py --help
"""
import argparse
import os
import sys
import yaml
import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(ROOT, "config.yaml")


def main():
    # 读取默认值
    cfg = {}
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)

    srv = cfg.get("server", {})
    accel = cfg.get("acceleration", {})

    parser = argparse.ArgumentParser(description="CosyVoice3 TTS + ASR API 服务")
    parser.add_argument("--host", default=srv.get("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=srv.get("port", 6006))
    parser.add_argument("--no-trt", action="store_true",
                        help="关闭 TensorRT（默认开启）")
    parser.add_argument("--no-vllm", action="store_true",
                        help="关闭 vLLM（默认开启）")
    parser.add_argument("--no-fp16", action="store_true",
                        help="关闭 FP16（默认开启）")
    parser.add_argument("--reload", action="store_true", help="开发热重载")
    args = parser.parse_args()

    # 覆盖环境变量（给 main.py 用）
    os.environ["TTS_TRT"] = "0" if args.no_trt else "1"
    os.environ["TTS_VLLM"] = "0" if args.no_vllm else "1"
    os.environ["TTS_FP16"] = "0" if args.no_fp16 else "1"

    on = lambda b: "v" if b else "x"
    print("=" * 55)
    print("  CosyVoice3 TTS + ASR API 服务")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    print(f"  [{on(not args.no_trt)}] TensorRT   "
          f"[{on(not args.no_vllm)}] vLLM   "
          f"[{on(not args.no_fp16)}] FP16")
    print("=" * 55)
    sys.stdout.flush()

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
