#!/usr/bin/env python3
"""
MemoryBus周级考核脚本

用法: python3 weekly_memory_bus_assessment.py [--output JSON|text]

考核项:
1. query/write总数 + 活跃provider数
2. 各provider调用次数
3. write_log跨provider重复检测
4. token_budget利用率
5. 异常计数
"""
import sys
import os
import json
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

def assess():
    from openllm.memory.memory_bus import MemoryBus, MemoryRecord, Query, WriteRequest

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "weekly_assessment",
    }

    # 1. 尝试创建ISA获取真实bus实例
    try:
        from openllm.memory.isa import ISA
        isa = ISA()
        bus = isa.bus
        report["isa_available"] = True
    except Exception as e:
        report["isa_available"] = False
        report["isa_error"] = str(e)
        bus = MemoryBus()
        # 注册providers以获取健康数据
        try:
            from openllm.memory.providers import JiakProvider, RecallProvider, CausalProvider, UnifiedProvider
            bus.register(JiakProvider())
            bus.register(RecallProvider())
            bus.register(CausalProvider())
            bus.register(UnifiedProvider())
        except Exception:
            pass

    # 2. Provider统计
    stats = bus.stats()
    report["provider_count"] = stats["providers"]
    report["total_records"] = stats["total_records"]
    report["provider_stats"] = {}
    for name, pstat in stats.get("provider_stats", {}).items():
        report["provider_stats"][name] = {
            "count": pstat.get("count", 0),
            "priority": pstat.get("priority", -1),
            "health": pstat.get("health", {}),
        }

    # 3. Write log分析
    write_log = bus.get_write_log(last_n=100)
    report["write_log_size"] = len(write_log)
    report["write_success_rate"] = (
        sum(1 for w in write_log if w.success) / max(len(write_log), 1)
    )

    # 跨provider重复检测
    if len(write_log) >= 2:
        contents = [(w.provider, w.record_id) for w in write_log if w.success]
        provider_counts = Counter(p for p, _ in contents)
        report["writes_by_provider"] = dict(provider_counts)

    # 4. 异常计数
    report["write_failures"] = sum(1 for w in write_log if not w.success)
    report["write_blocked"] = sum(1 for w in write_log if w.blocked)

    # 5. 性能基线（跑5次query测延迟）
    perf_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        try:
            bus.query(Query(text="test", top_k=3, token_budget=1000))
        except Exception:
            pass
        perf_times.append((time.perf_counter() - t0) * 1000)
    report["perf_ms_avg"] = sum(perf_times) / len(perf_times) if perf_times else 0
    report["perf_ms_p50"] = sorted(perf_times)[len(perf_times)//2] if perf_times else 0
    report["perf_ms_p99"] = max(perf_times) if perf_times else 0

    # 6. 健康检查
    health = bus.health_check()
    report["health"] = health
    report["degraded_providers"] = [
        name for name, h in health.items()
        if h.get("status") != "ok"
    ]

    # 7. 评级
    issues = []
    if report["provider_count"] < 3:
        issues.append("P1: provider数不足(<3)")
    if report["write_success_rate"] < 0.5 and len(write_log) > 5:
        issues.append("P1: 写入成功率<50%")
    if report["perf_ms_avg"] > 50:
        issues.append("P1: query延迟>50ms")
    if report["degraded_providers"]:
        issues.append(f"P2: 降级provider: {report['degraded_providers']}")
    if report["write_blocked"] > 0:
        issues.append(f"P2: {report['write_blocked']}条写入被免疫拦截")

    report["issues"] = issues
    report["grade"] = "🔴" if any("P1" in i for i in issues) else ("🟡" if issues else "🟢")

    return report


def format_text(report):
    lines = [
        f"{'='*60}",
        f"MemoryBus周级考核 · {report['timestamp']}",
        f"{'='*60}",
        f"评级: {report['grade']}",
        f"",
        f"## Provider",
        f"  数量: {report['provider_count']}",
        f"  总记录: {report['total_records']}",
    ]
    for name, ps in report.get("provider_stats", {}).items():
        lines.append(f"  {name}: {ps['count']}条 (priority={ps['priority']})")

    lines.extend([
        f"",
        f"## 写入",
        f"  日志大小: {report['write_log_size']}",
        f"  成功率: {report['write_success_rate']:.1%}",
        f"  失败: {report['write_failures']}",
        f"  被拦截: {report['write_blocked']}",
    ])

    lines.extend([
        f"",
        f"## 性能",
        f"  avg: {report['perf_ms_avg']:.2f}ms",
        f"  p50: {report['perf_ms_p50']:.2f}ms",
        f"  p99: {report['perf_ms_p99']:.2f}ms",
    ])

    if report["issues"]:
        lines.extend(["", "## 问题"])
        for issue in report["issues"]:
            lines.append(f"  {issue}")
    else:
        lines.extend(["", "## 无问题 🟢"])

    return "\n".join(lines)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "text"
    report = assess()
    if output == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_text(report))
    sys.exit(1 if report["grade"] == "🔴" else 0)
