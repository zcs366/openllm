#!/usr/bin/env python3
"""M3 trace 自分析器 — 读取 IO-S 全部 trace 源，产出分析报告。

数据源：
  ~/.io-s/traces/*.jsonl        — 工具/模型trace（trace.py输出）
  ~/.io-s/hindsight/verify_*.json — IO-S verify+hindsight记录（engine.py输出）
  ~/.io-s/signals/*.json         — ISN→ISA技能信号
  ~/.hermes/isa/verify_records.jsonl — ISA belief_update记录

用法：
    python3 trace_analyzer.py              # 标准报告
    python3 trace_analyzer.py --json       # JSON输出
    python3 trace_analyzer.py --watch      # 持续监控
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

IO_S = Path.home() / ".io-s"
TRACES_DIR = IO_S / "traces"
HINDSIGHT_DIR = IO_S / "hindsight"
SIGNALS_DIR = IO_S / "signals"
ISA_RECORDS = Path.home() / ".hermes" / "isa" / "verify_records.jsonl"


def read_jsonl(path: Path, limit: int = 10000) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-limit:]


def read_json_dir(directory: Path, glob: str = "*.json") -> list:
    """读取目录下所有JSON文件。"""
    if not directory.exists():
        return []
    records = []
    for f in sorted(directory.glob(glob)):
        try:
            with open(f, encoding="utf-8") as fh:
                records.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return records


# ════════════════════════════════════════════════════
# 分析函数
# ════════════════════════════════════════════════════

def analyze_tools() -> dict:
    """工具调用分析（来自 traces/tools.jsonl）。"""
    records = read_jsonl(TRACES_DIR / "tools.jsonl")
    if not records:
        return {"status": "no_data", "total": 0, "source": "traces/tools.jsonl"}

    total = len(records)
    success = sum(1 for r in records if r.get("success"))
    before_fails = sum(1 for r in records if not r.get("verify_before", {}).get("pass", True))
    after_fails = sum(1 for r in records if not r.get("verify_after", {}).get("pass", True))

    by_tool = {}
    for r in records:
        name = r.get("tool_name", "?")
        if name not in by_tool:
            by_tool[name] = {"calls": 0, "success": 0, "latencies": []}
        by_tool[name]["calls"] += 1
        if r.get("success"):
            by_tool[name]["success"] += 1
        by_tool[name]["latencies"].append(r.get("latency_ms", 0))

    return {
        "status": "ok",
        "total_calls": total,
        "success_rate": round(success / total * 100, 1) if total else 0,
        "verify_before_fails": before_fails,
        "verify_after_fails": after_fails,
        "by_tool": {
            name: {
                "calls": v["calls"],
                "success_rate": round(v["success"] / v["calls"] * 100, 1),
                "avg_latency_ms": round(sum(v["latencies"]) / len(v["latencies"]), 1),
            }
            for name, v in sorted(by_tool.items(), key=lambda x: -x[1]["calls"])
        },
    }


def analyze_costs() -> dict:
    """成本分析（来自 traces/models.jsonl）。"""
    records = read_jsonl(TRACES_DIR / "models.jsonl")
    if not records:
        return {"status": "no_data", "total": 0, "source": "traces/models.jsonl"}

    total = len(records)
    total_input = sum(r.get("input_tokens", 0) for r in records)
    total_output = sum(r.get("output_tokens", 0) for r in records)
    total_cost = sum(r.get("cost_usd", 0.0) for r in records)
    total_latency = sum(r.get("latency_ms", 0.0) for r in records)

    by_model = {}
    for r in records:
        m = r.get("model", "?")
        if m not in by_model:
            by_model[m] = {"calls": 0, "tokens": 0, "cost": 0.0}
        by_model[m]["calls"] += 1
        by_model[m]["tokens"] += r.get("input_tokens", 0) + r.get("output_tokens", 0)
        by_model[m]["cost"] += r.get("cost_usd", 0.0)

    return {
        "status": "ok",
        "total_calls": total,
        "total_tokens": total_input + total_output,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": round(total_latency / total, 1) if total else 0,
        "by_model": {
            name: {"calls": v["calls"], "tokens": v["tokens"], "cost_usd": round(v["cost"], 6)}
            for name, v in sorted(by_model.items(), key=lambda x: -x[1]["calls"])
        },
    }


def analyze_hindsight() -> dict:
    """IO-S verify+hindsight 分析（来自 hindsight/verify_*.json）。"""
    records = read_json_dir(HINDSIGHT_DIR, "verify_*.json")
    if not records:
        return {"status": "no_data", "total": 0, "source": "hindsight/verify_*.json"}

    total = len(records)
    passed = sum(1 for r in records if r.get("verify_pass"))
    failed = total - passed

    by_tool = {}
    for r in records:
        name = r.get("tool", "?")
        if name not in by_tool:
            by_tool[name] = {"total": 0, "pass": 0}
        by_tool[name]["total"] += 1
        if r.get("verify_pass"):
            by_tool[name]["pass"] += 1

    return {
        "status": "ok",
        "total": total,
        "pass": passed,
        "fail": failed,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "by_tool": {
            name: {
                "total": v["total"],
                "pass_rate": round(v["pass"] / v["total"] * 100, 1),
            }
            for name, v in sorted(by_tool.items(), key=lambda x: -x[1]["total"])
        },
    }


def analyze_signals() -> dict:
    """ISN→ISA 技能信号分析（来自 signals/*.json）。"""
    records = read_json_dir(SIGNALS_DIR, "*.json")
    if not records:
        return {"status": "no_data", "total": 0, "source": "signals/*.json"}

    total = len(records)
    by_type = Counter(r.get("type", "?") for r in records)
    return {
        "status": "ok",
        "total": total,
        "by_type": dict(by_type),
        "latest": records[-1] if records else None,
    }


def analyze_isa_verdicts() -> dict:
    """ISA belief_update 记录分析。"""
    records = read_jsonl(ISA_RECORDS)
    if not records:
        return {"status": "no_data", "total": 0, "source": "isa/verify_records.jsonl"}

    total = len(records)
    verdicts = Counter(r.get("verdict", "?") for r in records)
    by_tool = Counter(r.get("tool_name", "?") for r in records)

    return {
        "status": "ok",
        "total": total,
        "verdicts": dict(verdicts),
        "by_tool": dict(by_tool.most_common(10)),
    }


def health_check() -> dict:
    """健康检查。"""
    issues = []
    sources = {}

    for name, path in [
        ("traces", TRACES_DIR),
        ("hindsight", HINDSIGHT_DIR),
        ("signals", SIGNALS_DIR),
    ]:
        if path.exists():
            count = len(list(path.iterdir()))
            sources[name] = count
            if count == 0:
                issues.append(f"{name}/ 为空")
        else:
            sources[name] = 0
            issues.append(f"{name}/ 不存在")

    isa_ok = ISA_RECORDS.exists()
    sources["isa_verdicts"] = isa_ok

    return {
        "status": "healthy" if not issues else "degraded",
        "issues": issues,
        "sources": sources,
    }


def full_report() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "health": health_check(),
        "tools": analyze_tools(),
        "costs": analyze_costs(),
        "hindsight": analyze_hindsight(),
        "signals": analyze_signals(),
        "isa_verdicts": analyze_isa_verdicts(),
    }


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps(full_report(), indent=2, ensure_ascii=False))
    elif "--watch" in sys.argv:
        print("📊 M3 trace 监控模式 (每 10 秒刷新, Ctrl+C 退出)")
        try:
            while True:
                r = full_report()
                ts = datetime.now().strftime("%H:%M:%S")
                h = r["health"]
                hind = r["hindsight"]
                sig = r["signals"]
                print(f"[{ts}] 健康:{h['status']} | hindsight:{hind['total']}条 pass率:{hind.get('pass_rate','N/A')}% | signals:{sig['total']}条 | isa:{r['isa_verdicts']['total']}条")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n监控已退出")
    else:
        r = full_report()
        h = r["health"]
        t = r["tools"]
        c = r["costs"]
        hind = r["hindsight"]
        sig = r["signals"]
        isa = r["isa_verdicts"]

        print("=" * 55)
        print("📊 IO-S Trace 自分析报告 · 六源全扫")
        print("=" * 55)

        print(f"\n🔴 健康: {h['status']}")
        for issue in h.get("issues", []):
            print(f"   ⚠️  {issue}")

        print(f"\n🔧 工具调用 (traces/tools.jsonl)")
        if t.get("total_calls", 0) > 0:
            print(f"   总调用: {t['total_calls']} | 成功率: {t['success_rate']}%")
            print(f"   verify拦截(前): {t['verify_before_fails']} | (后): {t['verify_after_fails']}")
        else:
            print("   (无数据)")

        print(f"\n💰 成本 (traces/models.jsonl)")
        if c.get("total_calls", 0) > 0:
            print(f"   总调用: {c['total_calls']} | tokens: {c['total_tokens']:,} | ${c['total_cost_usd']:.4f}")
        else:
            print("   (无数据)")

        print(f"\n🧠 Verify+Hindsight (hindsight/verify_*.json)")
        if hind.get("total", 0) > 0:
            print(f"   总记录: {hind['total']} | pass率: {hind['pass_rate']}%")
            for name, stats in list(hind.get("by_tool", {}).items())[:5]:
                print(f"     {name}: {stats['total']}次, {stats['pass_rate']}%pass")
        else:
            print("   (无数据)")

        print(f"\n📡 ISN信号 (signals/*.json)")
        if sig.get("total", 0) > 0:
            print(f"   总信号: {sig['total']}")
            for t, cnt in sig.get("by_type", {}).items():
                print(f"     {t}: {cnt}")
        else:
            print("   (无数据)")

        print(f"\n📝 ISA verdicts (verify_records.jsonl)")
        if isa.get("total", 0) > 0:
            print(f"   总记录: {isa['total']}")
            for v, cnt in isa.get("verdicts", {}).items():
                print(f"     {v}: {cnt}")
        else:
            print("   (无数据)")
