"""DR-20260829-01 军师冒烟验证——P0三修落地检查。
修复后应满足：
  1. `你是谁`类冒烟的回答自称openLLM（identity注入生效）→ 军师人工跑
  2. --once会话结束后caps/出现新v06/v07/history文件 → 本脚本
  3. 新v07胶囊vector非全零、norm>0、384维 → 本脚本
"""
import glob
import json
import os
import subprocess
import time

CAPS_DIR = "/home/zcs/projects/openllm/caps"


def latest_files():
    """列出caps目录当前文件+mtime快照。"""
    snap = {}
    for pat in ("v06_*.json", "v07_*.json", "history_*.json"):
        for f in glob.glob(os.path.join(CAPS_DIR, pat)):
            snap[f] = os.path.getmtime(f)
    return snap


def main():
    before = latest_files()
    print(f"[before] caps文件数: {len(before)}")

    # 冒烟：--once身份问答（军师另跑人工核验回答内容；本脚本只触发一次会话收尾）
    env = dict(os.environ)
    env["HOME"] = "/home/zcs"
    env["PYTHONPATH"] = "/home/zcs/projects/openllm/src"
    env["HF_HUB_OFFLINE"] = "1"
    r = subprocess.run(
        ["/usr/bin/python3", "-m", "openllm.core.main_loop", "--once", "你好"],
        capture_output=True, text=True, timeout=280, env=env,
        cwd="/home/zcs/projects/openllm",
    )
    print("[smoke] exit:", r.returncode)
    print("[smoke] stdout-tail:", (r.stdout or "").strip()[-200:])
    if r.returncode != 0:
        print("[smoke] stderr-tail:", (r.stderr or "").strip()[-500:])
        return

    after = latest_files()
    new = {f: t for f, t in after.items() if f not in before}
    print(f"[after] 新增文件: {[os.path.basename(f) for f in new]}")

    if not new:
        print("FAIL: 会话收尾未落盘任何新文件（P0-B未生效）")
        return

    v07 = [f for f in new if os.path.basename(f).startswith("v07_")]
    if not v07:
        print("FAIL: 新增文件中没有v07胶囊")
        return

    for f in sorted(v07):
        d = json.load(open(f))
        v = d.get("vector", [])
        zero = all(x == 0.0 for x in v)
        print(f"[v07] {os.path.basename(f)}: dim={len(v)} norm={d.get('norm')} "
              f"all-zero={zero} encoding={d.get('metadata', {}).get('encoding', d.get('metadata', {}).get('model'))}")
        if zero:
            print("FAIL: v07向量仍全零（P0-A未生效）")
        elif len(v) == 384 and d.get("norm", 0) > 0:
            print("PASS: Δ胶囊心脏复跳（P0-A✓ P0-B✓）")

    hist = [f for f in new if os.path.basename(f).startswith("history_")]
    if hist:
        content = json.load(open(sorted(hist)[-1]))
        print(f"[history] {os.path.basename(sorted(hist)[-1])}: {len(content)}条消息")


if __name__ == "__main__":
    main()
