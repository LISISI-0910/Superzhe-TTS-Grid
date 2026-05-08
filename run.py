#!/usr/bin/env python3
"""
CosyVoice3 TTS + WhisperASR API 服务启动脚本。

默认开启 TRT + vLLM + FP16 全加速。
用法:
  python serve.py               # 全加速，0.0.0.0:6006
  python serve.py --no-trt      # 关闭 TensorRT
  python serve.py --no-vllm     # 关闭 vLLM
  python serve.py --no-fp16     # 关闭半精度
  python serve.py --port 9000   # 指定端口
"""
import argparse
import os
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 TTS + ASR API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model-dir", default="pretrained_models/Fun-CosyVoice3-0.5B",
                        help="TTS 模型目录")
    parser.add_argument("--no-trt", action="store_true", help="关闭 TensorRT 加速")
    parser.add_argument("--no-vllm", action="store_true", help="关闭 vLLM 加速")
    parser.add_argument("--no-fp16", action="store_true", help="关闭 FP16 半精度")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    os.environ["TTS_MODEL_DIR"] = args.model_dir
    os.environ["TTS_LOAD_TRT"] = "0" if args.no_trt else "1"
    os.environ["TTS_LOAD_VLLM"] = "0" if args.no_vllm else "1"
    os.environ["TTS_FP16"] = "0" if args.no_fp16 else "1"

    on = lambda b: "v" if b else "x"
    print("=" * 55)
    print("  CosyVoice3 TTS + ASR API 服务")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    print(f"  模型: {args.model_dir}")
    print(f"  [{on(not args.no_trt)}] TensorRT   [{on(not args.no_vllm)}] vLLM   [{on(not args.no_fp16)}] FP16")
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
