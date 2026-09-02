"""品尝师真实任务：指挥openLLM统计自身六体代码行数分布。
对比openLLM的答案与bash基准真值，评估工具调用+数字准确性。
"""
import subprocess
import re
import os

TASK = ("统计/home/zcs/projects/openllm/src/openllm/目录下各子目录（iai、core、isa、ios、isn、iko、"
        "memory、identity、tools、bridge、security、governance、retrieval）的Python代码总行数，"
        "用terminal工具执行wc -l命令统计，然后报告每个子目录的行数。")

env = dict(os.environ)
env.update({
    "HOME": "/home/zcs",
    "PYTHONPATH": "/home/zcs/projects/openllm/src",
    "HF_HUB_OFFLINE": "1",
})

r = subprocess.run(
    ["/usr/bin/python3", "-m", "openllm.core.main_loop", "--once", TASK],
    capture_output=True, text=True, timeout=280,
    env=env, cwd="/home/zcs/projects/openllm",
)
print("EXIT:", r.returncode)
out = r.stdout.strip()
print("=== openLLM回答（尾部1200字） ===")
print(out[-1200:])

# 提取openLLM报的数字
truth = {"iai": 329, "core": 21234, "isa": 220, "ios": 151, "isn": 7827, "iko": 2611,
         "memory": 8850, "identity": 570, "tools": 2072, "bridge": 242,
         "security": 792, "governance": 5186, "retrieval": 121}
hits, misses = [], []
for d, tv in truth.items():
    m = re.search(rf"{d}[^0-9\n]{{0,20}}(\d[\d,]{{3,}})", out)
    if m:
        got = int(m.group(1).replace(",", ""))
        (hits if abs(got - tv) / tv < 0.02 else misses).append((d, got, tv))
    else:
        misses.append((d, None, tv))
print("\n=== 数字核对（±2%容差） ===")
print("对:", len(hits), "错:", len(misses))
for d, got, tv in misses:
    print(f"  ✗ {d}: openLLM={got} 真值={tv}")

# 运行时证据：本次会话的工具trace与落盘
print("\n=== 运行时指标 ===")
import glob, os, time
now = time.time()
fresh = [f for f in glob.glob("/home/zcs/.openllm/output/iko/ticks/ticks.jsonl")]
if fresh:
    tail = open(fresh[0]).readlines()[-60:]
    tool_calls = [l for l in tail if '"terminal"' in l or 'tool' in l.lower()]
    print("IKO trace尾部60条中tool相关:", len(tool_calls))
caps_new = sorted(glob.glob("/home/zcs/projects/openllm/caps/history_*.json"), key=os.path.getmtime)
if caps_new and now - os.path.getmtime(caps_new[-1]) < 400:
    h = open(caps_new[-1]).read()
    print("落盘: history有新文件", os.path.basename(caps_new[-1]))
