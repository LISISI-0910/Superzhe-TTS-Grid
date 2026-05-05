#!/usr/bin/env python3
"""
CosyVoice3 带 RSA 签名的 API 测试脚本

RSA 签名认证逻辑（对应 run.py 的 auth_middleware）：
  1. 调用方用 private_key.pem 对请求体进行 RSA-SHA256 签名
  2. 签名结果 Base64 编码后放入 HTTP Header X-Signature
  3. 引擎收到请求后用 public_key.pem 验签

私钥路径：..\private_key.pem（上级目录）
公钥路径：.\public_key.pem（当前目录，用于验签）

用法：
  python test_extract.py                        # 默认测试完整流程
  python test_extract.py --health               # 测试 /health（GET，不需要签名）
  python test_extract.py --no-sign              # 不加签名（预期返回 401）
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests")
    sys.exit(1)

# ─── 路径配置 ─────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()          # 当前目录 e:\Superzhe-TTS-Grid-main
PARENT_DIR = BASE_DIR.parent                        # 上级目录 e:\
PRIVATE_KEY_PATH = PARENT_DIR / "private_key.pem"   # 上级目录的私钥
BENCHMARK_DIR = BASE_DIR / "benchmark"

HOST = "http://127.0.0.1:6006"
SIGN_ENABLED = True  # 可通过 --no-sign 关闭


# ═══════════════════════════════════════════════════
#  RSA 签名（与 run.py 的 _verify_signature 对应）
# ═══════════════════════════════════════════════════
def sign_body(body_bytes: bytes) -> str:
    """
    用 private_key.pem 对请求体进行 RSA-SHA256 签名，返回 Base64 字符串。
    签名算法与 run.py 中 _verify_signature 的验签逻辑完全对应：
      - 哈希: SHA256
      - 填充: PKCS1v15
    """
    if not SIGN_ENABLED:
        return ""

    if not PRIVATE_KEY_PATH.exists():
        print(f"  [WARN] 私钥文件不存在: {PRIVATE_KEY_PATH}，跳过签名")
        return ""

    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        with open(PRIVATE_KEY_PATH, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)

        signature = private_key.sign(body_bytes, padding.PKCS1v15(), hashes.SHA256())
        return base64.b64encode(signature).decode()
    except Exception as e:
        print(f"  [ERROR] 签名失败: {e}")
        return ""


def send_request(method: str, path: str, body: dict = None,
                 timeout: int = 60, stream: bool = False) -> requests.Response:
    """
    发送带 RSA 签名的 HTTP 请求。
    关键：签名的 body_bytes 必须和实际发送的 data 完全一致。
    """
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else b""
    sig = sign_body(body_bytes)

    headers = {"Content-Type": "application/json"}
    if sig:
        headers["X-Signature"] = sig
        sig_preview = sig[:20] + "..."
    else:
        sig_preview = "(无签名)"

    print(f"  -> {method} {path}")
    print(f"    X-Signature: {sig_preview}")

    return requests.request(
        method, f"{HOST}{path}",
        data=body_bytes if body else None,
        headers=headers,
        timeout=timeout,
        stream=stream
    )


# ═══════════════════════════════════════════════════
#  API 测试函数
# ═══════════════════════════════════════════════════

def test_health() -> bool:
    """
    测试 GET /health（不需要签名，中间件对 GET/HEAD/OPTIONS 放行）
    """
    print("\n" + "=" * 60)
    print("1. 健康检查 GET /health")
    print("=" * 60)

    try:
        r = requests.get(f"{HOST}/health", timeout=5)
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"    状态:   {data.get('status')}")
            print(f"    版本:   {data.get('version')}")
            if data.get("gpu"):
                print(f"    GPU:    {data['gpu']}")
            if data.get("mem_gb") and data["mem_gb"]:
                m = data["mem_gb"]
                print(f"    显存:   已分配 {m['used']}GB / 缓存 {m['reserved']}GB")
            if data.get("benchmark"):
                b = data["benchmark"]
                print(f"    基准:   RTF_avg={b['rtf_avg']} 并发={b['max_concurrency']}")
            return True
        else:
            print(f"    [失败] 响应: {r.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"    [失败] 无法连接 {HOST}，请先启动引擎: python run.py")
        return False
    except Exception as e:
        print(f"    [失败] 异常: {e}")
        return False


def test_extract_without_sign() -> bool:
    """
    测试不携带签名时是否返回 401。
    引擎有公钥时应拒绝；无公钥时应放行。
    """
    print("\n" + "=" * 60)
    print("2. 测试无签名请求（预期 401 或 200）")
    print("=" * 60)

    audio_path = BENCHMARK_DIR / "ref.wav"
    text_path = BENCHMARK_DIR / "ref.txt"
    if not audio_path.exists() or not text_path.exists():
        print("    找不到 benchmark 文件，跳过")
        return False

    prompt_text = text_path.read_text(encoding="utf-8").strip()
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

    body = {"prompt_text": prompt_text, "prompt_audio": audio_b64, "voice_type": "zero_shot"}
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    # 发送不带签名的请求
    r = requests.post(
        f"{HOST}/extract",
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    print(f"    状态码: {r.status_code}")
    if r.status_code == 401:
        print("    [通过] 无签名请求被拒绝（401）- 认证正常工作")
        return True
    elif r.status_code == 200:
        print("    [提示] 请求通过（无公钥模式或认证已关闭）")
        return True
    else:
        print(f"    [失败] 意外响应: {r.text}")
        return False


def test_extract() -> dict | None:
    """
    测试 POST /extract（带 RSA 签名）
    返回 spk_vec 供后续 /tts 使用
    """
    print("\n" + "=" * 60)
    print("3. 提取说话人向量 POST /extract（带签名）")
    print("=" * 60)

    audio_path = BENCHMARK_DIR / "ref.wav"
    text_path = BENCHMARK_DIR / "ref.txt"
    if not audio_path.exists() or not text_path.exists():
        print("    [失败] 找不到 benchmark/ref.wav 或 benchmark/ref.txt")
        return None

    prompt_text = text_path.read_text(encoding="utf-8").strip()
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
    print(f"    参考文本: \"{prompt_text}\"")
    print(f"    参考音频: {audio_path} ({audio_path.stat().st_size / 1024:.1f} KB)")

    body = {
        "prompt_text": prompt_text,
        "prompt_audio": audio_b64,
        "voice_type": "zero_shot"
    }

    t0 = time.perf_counter()
    r = send_request("POST", "/extract", body, timeout=60)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"    状态码: {r.status_code}  [耗时 {elapsed:.0f}ms]")
    if r.status_code == 200:
        data = r.json()
        spk_vec = data.get("spk_vec", {})
        emb_keys = [k for k in spk_vec if 'embedding' in k]
        lens = {k: len(v) for k, v in spk_vec.items() if isinstance(v, list)}
        print(f"    [成功] 提取成功！共 {len(spk_vec)} 个字段")
        for k, v in sorted(spk_vec.items()):
            if isinstance(v, list):
                print(f"      {k}: list[{len(v)}]")
            else:
                print(f"      {k}: {v}")
        return spk_vec
    elif r.status_code == 401:
        print(f"    [失败] 签名认证失败: {r.text}")
        print("    提示: 请确认引擎已部署 public_key.pem，且私钥正确")
        return None
    else:
        print(f"    [失败] 错误: {r.text}")
        return None


def test_tts(spk_vec: dict, text: str, stream: bool = False,
             label: str = "") -> bool:
    """
    测试 POST /tts（带 RSA 签名）
    """
    mode = "流式" if stream else "非流式"
    output = f"output_{label}.opus" if label else "output.opus"

    print(f"\n    -- {label or mode} TTS --")
    print(f"    合成文本: \"{text[:30]}{'...' if len(text) > 30 else ''}\"")
    print(f"    输出文件: {output}")

    body = {
        "text": text,
        "spk_vec": spk_vec,
        "voice_type": "zero_shot",
        "stream": stream,
        "speed": 1.0,
        "instruct_text": ""
    }

    t0 = time.perf_counter()
    r = send_request("POST", "/tts", body, timeout=300, stream=stream)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"    状态码: {r.status_code}  [耗时 {elapsed:.0f}ms]")
    if r.status_code != 200:
        print(f"    [失败] 错误: {r.text}")
        return False

    with open(output, "wb") as f:
        f.write(r.content)
    size_kb = os.path.getsize(output) / 1024
    print(f"    [成功] 保存成功！{output} ({size_kb:.1f} KB)")
    return True


def test_tts_flow(spk_vec: dict):
    """
    测试多种文本长度下的 TTS 生成
    """
    print("\n" + "=" * 60)
    print("4. 生成语音 POST /tts（带签名）")
    print("=" * 60)

    test_cases = [
        ("短文本", "这是短文本：今天天气真不错，适合出去走走。"),
        ("中文本", "这是中文本：老李的闹钟定在凌晨四点。在这个南方小城的旧巷子里，当路灯还没熄灭，空气里还带着浓重的水汽时，老李已经熟练地拉开了卷帘门。“哗啦”一声，金属碰撞的声音在寂静的巷子里传得很远，像是某种古老的唤醒仪式。"),
        ("长文本", "这是长文本：老李的闹钟定在凌晨四点。在这个南方小城的旧巷子里，当路灯还没熄灭，空气里还带着浓重的水汽时，老李已经熟练地拉开了卷帘门。“哗啦”一声，金属碰撞的声音在寂静的巷子里传得很远，像是某种古老的唤醒仪式。老李是个做早餐的。他的铺子没有名字，门口挂着一块早就褪了色的红布，上面手写着“豆浆油条”四个字。揉面、醒面、烧水、磨豆浆。这些动作老李做了三十年，早就刻进了骨子里。面团在他的手里像是有了生命，揉捏、按压、拉伸，最后被切成匀称的长条。等油锅里的油开始冒起细小的青烟，他指尖一捻，面条滑入锅中，瞬间激起一阵密集的“滋啦”声。原本干瘪的面条在热油中迅速膨胀、翻滚，变成金黄诱人的色泽。第一个客人照例是环卫工老张。老张推着清扫车，身上还带着凌晨的寒气。老李头也不抬，顺手从蒸笼里抓起两个大肉包子，装袋，再舀上一碗热气腾腾的浓豆浆，递过去：“还是老样，包子皮软点儿。”老张嘿嘿一笑，粗糙的手接过早餐，蹲在路边的马路牙子上吃得热汗淋漓。这一刻，早起的辛劳似乎都被这口热乎劲儿给化解了。六点半到八点，是巷子里最热闹的时候。穿校服的中学生嚼着油条急匆匆地赶路，耳机里还放着英语单词；穿职业装的年轻姑娘踩着高跟鞋，在湿漉漉的石板路上走得小心翼翼，却不忘在老李的摊位前等上一份刚出锅的粢饭团。老李观察着这些人。那个总是皱着眉头的程序员，最近似乎放松了些；那个每天都要买两份早餐的单亲妈妈，孩子应该已经上幼儿园了吧。老李话不多，但他记得住几乎所有熟客的口味。谁家的豆浆不加糖，谁家的煎饼要多放葱，谁家的油条喜欢炸得老一点。这些琐碎的信息在他脑子里编织成了一张巨大的网，网住的是这方圆一公里内最真实的人情味。九点以后，喧嚣渐退。剩下的多是些提着菜篮子慢悠悠晃荡的老人。老王头拄着拐棍挪到了老李的铺子前。老王头以前是厂里的老技术员，现在病了，腿脚不灵便。老李见他走得吃力，赶紧抹了抹凳子扶他坐下。老王头叹了口气：“老李啊，这巷子听说要拆了。”老李手里的动作顿了顿，沉默了一会儿，说：“传了好几年了，谁知道呢。拆了就歇着，不拆就接着干。”“舍不得你这口油条啊。”老王头苦笑着，从兜里摸出几张皱巴巴的零钱。老李没接钱，摆摆手：“今儿这顿算我的，面剩了一点，不卖也浪费了。”生活其实就是这样，没有什么惊天动地的大事，多的是这种微不足道的体谅。中午时分，老李收了摊。洗刷蒸笼、清理油锅，这些活儿甚至比炸油条还要累人。他的腰早就落下了职业病，直起来的时候总会发出一声轻微的响动。回到楼上那间窄小的屋子里，老李收到了儿子的微信。儿子在省城读大学，发来一张图书馆的照片，说在准备考研。老李看着手机屏幕，粗糙的大拇指反复摩挲着儿子的头像，最后只回复了简短的两个字：“加油。”然后，他从床头的铁罐子里摸出这一天的收入。五块的、十块的、一块的，他一张张捋平，整整齐齐地叠好。这些带着油烟味和面粉白印的钞票，是他生活的底气，也是儿子未来的路标。午后的阳光穿过窗棂，洒在老李满是老茧的手上。他感到一阵深深的疲惫袭来，靠在藤椅上打起了盹。梦里，他似乎又回到了年轻的时候，也是这样的巷子，也是这样的清晨，他第一次炸出一根完美的油条，兴奋得像个孩子。夕阳西下，巷子里的烟火气又转为了另一种形态。炒菜的香味从各家的窗户里飘出来，混杂着孩童的嬉闹声和电视机的背景音。老李下楼倒垃圾，顺便去巷口的小店买了一瓶最便宜的二锅头。他坐在店门口，看着天色一点点暗下去，看着那些白天奔波的人们一个个归巢。路灯又亮了，昏黄的光晕照着这片略显破败却充满生机的土地。老李知道，几个小时后，他的闹钟又会响起。他会再次推开那扇沉重的卷帘门，再次揉开那团发酵好的面。这就是他的生活。它不精致，甚至有些寒酸；它不伟大，甚至有些机械。但就在这一揉一炸之间，在这一递一接之余，那些琐碎的瞬间构成了生命的全部意义。平凡，却有着像面团一样的韧劲；粗糙，却有着像豆浆一样的温热。老李闭上眼，在酒精的微醺中想：明天，要把油锅洗得再亮一点。")
    ]

    all_ok = True
    for label, text in test_cases:
        ok = test_tts(spk_vec, text, stream=False, label=label)
        if not ok:
            all_ok = False

    if all_ok:
        print("\n    [成功] 所有 TTS 测试通过！")
    else:
        print("\n    [失败] 部分 TTS 测试失败")
    return all_ok


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def run_all():
    """完整测试流程：health > 无签名测试 > extract > tts"""
    print("=" * 60)
    print("  CosyVoice3 带 RSA 签名认证的完整 API 测试")
    print("=" * 60)
    print(f"  私钥: {PRIVATE_KEY_PATH}")
    print(f"  公钥: {BASE_DIR / 'public_key.pem'}")
    print(f"  签名: {'已启用' if SIGN_ENABLED else '已禁用'}")
    print()

    # 1. 健康检查
    if not test_health():
        print("\n引擎未就绪，退出测试")
        return

    # 2. 测试无签名是否被拒绝
    test_extract_without_sign()

    # 3. 提取说话人向量
    spk_vec = test_extract()
    if spk_vec is None:
        return

    # 4. TTS 生成测试
    test_tts_flow(spk_vec)

    print("\n" + "=" * 60)
    print("  测试完成！")
    print("=" * 60)


def main():
    global HOST, SIGN_ENABLED

    parser = argparse.ArgumentParser(description="CosyVoice3 RSA 签名测试脚本")
    parser.add_argument("--host", default=HOST, help=f"服务地址 (默认: {HOST})")
    parser.add_argument("--no-sign", action="store_true", help="关闭 RSA 签名")
    parser.add_argument("--health", action="store_true", help="仅测试健康检查")

    args = parser.parse_args()

    HOST = args.host
    SIGN_ENABLED = not args.no_sign

    if args.health:
        test_health()
    else:
        run_all()


if __name__ == "__main__":
    main()
