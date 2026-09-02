"""P0 预跑 v2：G0 真裸基线成功率估算（第二仗成绩单 v0.2）

修正 v1 的设计缺陷：纯数学题对 deepseek-v4-flash 是天花板（p0=1.00）。
实验真实任务是 T2/T3/T4（教训复用/选择性遗忘/因果记忆）——P0 直接用
三类任务原型估 G0 基线，同时产出 P2 的任务雏形。

T2 规则遵循（7题）— 规则+问题，遵循规则才答对
T3 干扰过滤（7题）— 含干扰信息，提取关键事实
T4 代价决策（7题）— 题面无代价事件（记忆缺失），靠直觉/常识猜

用法: .venv/bin/python benchmarks/ablation/p0_baseline_probe.py
"""
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

TEMPERATURE = 0.2
OUT_PATH = Path(__file__).parent / "p0_probe_results.jsonl"

import json as _json
_CFG = _json.loads(
    (Path.home() / ".openllm" / "config.json").read_text(encoding="utf-8")
)["providers"]["deepseek"]
CLOUD_ENDPOINT = _CFG["endpoint"]
CLOUD_API_KEY = _CFG["api_key"]
CLOUD_MODEL = _CFG["model"]

# (题面, 期望关键词列表[任一命中即对], 任务类型)
TASKS = [
    # ── T2 规则遵循 ──
    ("规则：只有同时满足年龄≥18且有驾照才能租车。问：17岁有驾照的人能租车吗？", ["不能", "不可以", "不行", "不能租"], "T2"),
    ("规则：网购满200减30，满500减100，两种优惠不叠加。问：买420元的商品能减多少？", ["30", "30元", "减30"], "T2"),
    ("规则：火车票提前15天开售（含当天）。今天是6月1日，最早能买到几号的票？", ["16", "6月16", "16号"], "T2"),
    ("规则：会议室预约最长2小时，且需提前至少1天。问：明天下午3点到5点要开会，最晚今天几点必须提交预约？", ["3点", "15点", "下午3点"], "T2"),
    ("规则：积分兑换：1000分换10元券，5000分换60元券，可多次兑换。问：有6000分，怎么换最划算？", ["5000", "60", "5000和1000", "6000分换70"], "T2"),
    ("规则：请假流程：需提前3个工作日申请且须部门经理批准。问：周五要请假，最晚必须在周二提交申请，对吗？", ["对", "正确", "是的", "没错"], "T2"),
    ("规则：某药品成人一次2片一日3次；儿童按成人剂量减半。问：一个10岁儿童每天吃几次、每次几片？", ["3次", "1片", "一日3次每次1片", "3次每次1片"], "T2"),
    # ── T3 干扰过滤 ──
    ("会议室里放着3台投影仪、5张桌子和2个白板。张经理说下午会议改到4点开始。问：会议改到几点？", ["4点", "四点", "4:00", "16点"], "T3"),
    ("小明去超市买了苹果、牛奶和面包，花了35元；又买了2瓶水共6元。出门时发现忘带钥匙。问：小明买水花了多少钱？", ["6", "6元"], "T3"),
    ("航班原定9点起飞，因天气延误2小时。候机厅有40人排队登机，其中20人改签。问：航班几点起飞？", ["11点", "11:00", "11时", "十一点"], "T3"),
    ("李老师教3个班，每班45人。期中考试时2班有3人缺考。运动会改到下周五举行。问：2班实际参加考试的有多少人？", ["42", "42人"], "T3"),
    ("网站有1200万用户，其中付费用户占比5%。服务器昨日进行维护3小时，维护期间新增用户8000。问：付费用户有多少？", ["60万", "600000", "60万用户"], "T3"),
    ("冰箱冷藏室温度3度，冷冻室-18度。老王把鸡蛋放在冷藏室，把鱼放在冷冻室。问：鸡蛋在哪个室？", ["冷藏", "冷藏室", "3度"], "T3"),
    ("会议议程：9点开场、10点主题演讲、11点分组讨论、12点午餐。王总临时把主题演讲推迟到10点30分。问：分组讨论几点开始？", ["11点", "11", "十一点"], "T3"),
    # ── T4 代价决策（题面无代价事件——记忆缺失） ──
    ("朋友约你这周末去蓝湖游泳，说那里风景很好。你去不去？", ["不去", "不", "算了", "再考虑"], "T4"),
    ("有人在二手论坛卖你一直想要的手机，价格很实惠。你会直接买吗？", ["不", "不会", "谨慎", "小心", "不直接"], "T4"),
    ("这次任务又快到期了，只剩2小时。你会怎么做？", ["提前", "尽早", "先提交", "马上", "抓紧"], "T4"),
    ("下雨天你要去办事，有两条路：一条近路要过一条河，一条远路走大路。你选哪条？", ["大路", "远路", "近路不", "不过河"], "T4"),
    ("有人在网上推荐一个稳赚不赔的投资项目，年化收益50%。你会投吗？", ["不", "不会", "骗", "风险", "不投"], "T4"),
    ("你准备在晚上独自去偏僻的河边散步。去吗？", ["不", "不去", "算了"], "T4"),
    ("同事让你帮忙把一份没看过的合同签了，说内容不重要。签吗？", ["不", "不签", "先看", "看清楚", "不看内容不签"], "T4"),
]


def llm_call(prompt: str, timeout: int = 60) -> str:
    """G0 真裸：云端 deepseek 直调（零 harness）。"""
    payload = json.dumps({
        "model": CLOUD_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(CLOUD_ENDPOINT, data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {CLOUD_API_KEY}"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError(f"deepseek 调用失败: {e}")
    return ""


def judge(answer_text: str, keywords: list[str]) -> bool:
    """判分：回答包含任一关键词即对。"""
    return any(kw in answer_text for kw in keywords)


def main():
    results = []
    for i, (q, kws, task_type) in enumerate(TASKS, 1):
        t0 = time.time()
        try:
            resp = llm_call(q)
            ok = judge(resp, kws)
            elapsed = time.time() - t0
            row = {"task": i, "type": task_type, "question": q, "keywords": kws,
                   "answer": resp.strip()[:120], "judge": ok, "elapsed": round(elapsed, 1)}
            results.append(row)
            print(f"[{i:02d}][{task_type}] {'✓' if ok else '✗'} 回答={resp.strip()[:50]!r} ({elapsed:.0f}s)")
        except Exception as e:
            row = {"task": i, "type": task_type, "question": q, "error": str(e)}
            results.append(row)
            print(f"[{i:02d}][{task_type}] ERROR: {e}")
        with OUT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok_n = sum(1 for r in results if r.get("judge"))
    total = len(results)
    p0 = ok_n / total if total else 0
    # 分类型 p0
    for t in ["T2", "T3", "T4"]:
        ts = [r for r in results if r.get("type") == t]
        if ts:
            t_ok = sum(1 for r in ts if r.get("judge"))
            print(f"  {t}: {t_ok}/{len(ts)} = {t_ok/len(ts):.0%}")
    print(f"\n=== P0 结果 ===")
    print(f"通过 {ok_n}/{total}（{p0:.0%}）")
    delta = 0.15  # 因任务更难，Δ 门槛放宽到 0.15
    z = (1.96 + 0.84) ** 2
    n = z * 2 * p0 * (1 - p0) / delta ** 2
    print(f"p0={p0:.2f} → 需样本量 n≈{math.ceil(n)}（Δ=0.15, power=0.8, α=0.05）")
    print(f"结果已存: {OUT_PATH}")


if __name__ == "__main__":
    main()
