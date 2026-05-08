#!/usr/bin/env python3
"""
TTS 并发测试工具 —— 向服务端发起 N 个并发请求，验证是否真正并行。

用法:
  python test_concurrency.py                     # 默认 4 并发，localhost:6006
  python test_concurrency.py -n 6                # 6 并发
  python test_concurrency.py --host 10.0.0.1     # 指定地址
  python test_concurrency.py -n 6 --text "你好世界" # 自定义文本
"""
import argparse
import os
import time
import threading
import requests
import sys


def main():
    parser = argparse.ArgumentParser(description="TTS 并发测试")
    parser.add_argument("-n", type=int, default=4, help="并发请求数")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=6006, help="服务端口")
    parser.add_argument("--text", default="路灯又亮了，昏黄的光晕照着这片略显破败却充满生机的土地。老李知道，几个小时后，他的闹钟又会响起。他会再次推开那扇沉重的卷帘门，再次揉开那团发酵好的面。这就是他的生活。它不精致，甚至有些寒酸；它不伟大，甚至有些机械。但就在这一揉一炸之间，在这一递一接之余，那些琐碎的瞬间构成了生命的全部意义。平凡，却有着像面团一样的韧劲；粗糙，却有着像豆浆一样的温热。老李闭上眼，在酒精的微醺中想：明天，要把油锅洗得再亮一点。",
                        help="测试文本")
    parser.add_argument("--mode", default="zero_shot", help="生成模式")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    # ── Step 1: 提取说话人向量 ──
    print(f"[1] 提取说话人向量...")
    try:
        with open("benchmark/ref.wav", "rb") as f:
            resp = requests.post(f"{base}/api/v1/tts/extract",
                                 files={"audio": ("ref.wav", f, "audio/wav")},
                                 data={"prompt_text": open("benchmark/ref.txt", encoding="utf-8").read().strip()})
        resp.raise_for_status()
        spk_b64 = resp.json()["speaker_b64"]
        print(f"    ✓ speaker_b64 长度={len(spk_b64)}")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        sys.exit(1)

    # ── Step 2: 单次调用，测基准 ──
    print(f"\n[2] 单次生成（基准）...")
    baseline = None
    try:
        t0 = time.perf_counter()
        resp = requests.post(f"{base}/api/v1/tts/generate",
                             data={"text": args.text, "speaker_b64": spk_b64, "mode": args.mode},
                             stream=True)
        resp.raise_for_status()
        total_bytes = 0
        for chunk in resp.iter_content(chunk_size=4096):
            total_bytes += len(chunk)
        baseline = time.perf_counter() - t0
        print(f"    ✓ 耗时={baseline:.2f}s  下载={total_bytes}B  状态={resp.status_code}")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        sys.exit(1)

    # ── Step 3: N 并发请求 ──
    out_dir = "test_outputs"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[3] {args.n} 并发请求（音频保存到 {out_dir}/）...")
    print(f"    {'id':<5} {'开始(s)':<8} {'完成(s)':<8} {'耗时(s)':<8} {'文件':<30}")
    print(f"    {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*30}")

    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(args.n)
    t_global_start = time.perf_counter()

    def worker(idx):
        barrier.wait()  # 同时出发
        t_start = time.perf_counter()
        filepath = ""
        try:
            resp = requests.post(f"{base}/api/v1/tts/generate",
                                 data={"text": args.text, "speaker_b64": spk_b64, "mode": args.mode},
                                 stream=True, timeout=120)
            status = resp.status_code
            if status != 200:
                body = resp.text[:200]
                elapsed = time.perf_counter() - t_start
                with lock:
                    results.append((idx, t_start - t_global_start, time.perf_counter() - t_global_start,
                                   elapsed, f"{status}: {body}"))
                return
            chunks = []
            for chunk in resp.iter_content(chunk_size=4096):
                chunks.append(chunk)
            elapsed = time.perf_counter() - t_start
            size = sum(len(c) for c in chunks)
            filepath = os.path.join(out_dir, f"concurrent_{idx:02d}.ogg")
            with open(filepath, "wb") as f:
                for c in chunks:
                    f.write(c)
            detail = filepath
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            detail = str(e)[:40]

        with lock:
            results.append((idx, t_start - t_global_start, time.perf_counter() - t_global_start,
                           elapsed, detail))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    t_global_end = time.perf_counter()

    # 排序输出
    results.sort(key=lambda r: r[0])
    for idx, start_rel, end_rel, elapsed, detail in results:
        print(f"    {idx:<5} {start_rel:<8.2f} {end_rel:<8.2f} {elapsed:<8.2f} {detail:<30}")

    # ── 总结 ──
    latencies = [r[3] for r in results]
    print(f"\n[4] 总结")
    print(f"    wall-clock: {t_global_end - t_global_start:.2f}s")
    print(f"    单次基准:   {baseline:.2f}s")
    print(f"    并发 {args.n} 个:")
    print(f"      平均耗时: {sum(latencies)/len(latencies):.2f}s")
    print(f"      最快:     {min(latencies):.2f}s")
    print(f"      最慢:     {max(latencies):.2f}s")
    print(f"    音频目录:   {os.path.abspath(out_dir)}/")

    serial_est = baseline * args.n
    speedup = serial_est / (t_global_end - t_global_start) if t_global_end > t_global_start else 0
    print(f"\n    串行预估: {serial_est:.1f}s")
    print(f"    加速比:   {speedup:.1f}x")
    if speedup >= 1.3:
        print(f"    ✓ 并发生效！{args.n} 个请求同时处理")
    else:
        print(f"    ✗ 加速比不足，GPU 可能已饱和")


if __name__ == "__main__":
    main()
