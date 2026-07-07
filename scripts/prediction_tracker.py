#!/usr/bin/env python3
"""
MemoryBus可证伪预测追踪器

用法:
  python3 prediction_tracker.py                  # 查看所有预测状态
  python3 prediction_tracker.py --check P1       # 检查单个预测
  python3 prediction_tracker.py --record P1 pass  # 记录预测结果
  python3 prediction_tracker.py --json           # JSON输出
"""
import sys
import os
import json
import time
from pathlib import Path

TRACKER_PATH = Path.home() / ".hermes" / "jiak" / "cards" / "memory_bus_predictions.json"

PREDICTIONS = {
    "P1": {
        "title": "检索质量不低于旧ICE",
        "description": "MemoryBus检索结果≥旧ICE硬编码4-selector",
        "method": "A/B测试：同一query，MemoryBus vs ICE旧结果，人工评分",
        "deadline_days": 14,
        "falsified_if": "MemoryBus得分 < 旧ICE",
        "status": "pending",
    },
    "P2": {
        "title": "jiak recall@5 ≥ 0.6",
        "description": "5条结果中至少3条相关",
        "method": "20个真实query，人工标注相关性",
        "deadline_days": 7,
        "falsified_if": "recall@5 < 0.6",
        "status": "pending",
    },
    "P3": {
        "title": "写入零重复率",
        "description": "同一事实不被两个provider各写一次",
        "method": "审计write_log，Jaccard>0.8跨provider→重复",
        "deadline_days": 7,
        "falsified_if": "Jaccard>0.8的跨provider写入存在",
        "status": "pending",
    },
    "P4": {
        "title": "100 provider规模性能",
        "description": "100个provider下query延迟仍<50ms",
        "method": "注册100个mock provider，测query延迟",
        "deadline_days": 1,
        "falsified_if": "延迟 > 50ms",
        "status": "pending",
    },
    "P5": {
        "title": "30天内≥3个Hermes插件使用",
        "description": "至少3个Hermes插件实际调用bus_write/bus_query",
        "method": "grep代码库中bus_write/bus_query调用者",
        "deadline_days": 30,
        "falsified_if": "调用者 < 3",
        "status": "pending",
    },
}


def load_tracker():
    if TRACKER_PATH.exists():
        try:
            return json.loads(TRACKER_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"predictions": {}, "history": []}


def save_tracker(data):
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def check_prediction(pred_id):
    """检查单个预测的当前状态"""
    if pred_id == "P4":
        # P4可以自动检查
        try:
            sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))
            from openllm.memory.memory_bus import MemoryBus, Query
            import time as t

            bus = MemoryBus()
            class MockP:
                @property
                def name(self): return f"mock_{id(self)}"
                @property
                def priority(self): return 0
                def search(self, q): return []
                def store(self, r): from openllm.memory.memory_bus import WriteResult; return WriteResult(success=False)
                def count(self): return 0
                def health(self): return {"status": "ok"}

            for i in range(100):
                bus.register(MockP())

            times = []
            for _ in range(10):
                t0 = t.perf_counter()
                bus.query(Query(text="test", top_k=5))
                times.append((t.perf_counter() - t0) * 1000)

            avg = sum(times) / len(times)
            return {"auto_check": True, "avg_ms": avg, "passed": avg < 50}
        except Exception as e:
            return {"auto_check": True, "error": str(e), "passed": False}

    if pred_id == "P5":
        # P5可以自动检查
        try:
            result = os.popen(
                'grep -rn "bus_write\\|bus_query" ~/projects/openllm/src/ '
                '| grep -v "def bus_" '
                '| grep -v "memory_bus.py" '
                '| grep -v "isa.py" '
                '| grep -v "__pycache__" '
                '| cut -d: -f1 | sort -u | wc -l'
            ).read().strip()
            count = int(result) if result.isdigit() else 0
            return {"auto_check": True, "callers": count, "passed": count >= 3}
        except Exception as e:
            return {"auto_check": True, "error": str(e), "passed": False}

    return {"auto_check": False, "message": "需要人工检查"}


def record_result(pred_id, passed, detail=""):
    tracker = load_tracker()
    if pred_id not in tracker["predictions"]:
        tracker["predictions"][pred_id] = {
            "created": time.strftime("%Y-%m-%d"),
            "status": "pending",
            "checks": [],
        }

    entry = tracker["predictions"][pred_id]
    entry["checks"].append({
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "passed": passed,
        "detail": detail,
    })

    if passed:
        entry["status"] = "verified"
    else:
        entry["status"] = "falsified"
        # 自动触发整改预案
        entry["remediation_triggered"] = True

    tracker["history"].append({
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "prediction": pred_id,
        "passed": passed,
        "detail": detail,
    })

    save_tracker(tracker)
    return entry


def format_status(tracker=None):
    if tracker is None:
        tracker = load_tracker()

    lines = [
        f"{'='*60}",
        f"MemoryBus可证伪预测追踪",
        f"{'='*60}",
    ]

    for pid, pred in PREDICTIONS.items():
        record = tracker.get("predictions", {}).get(pid, {})
        status = record.get("status", "pending")
        checks = record.get("checks", [])

        status_icon = {"verified": "✅", "falsified": "❌", "pending": "⏳"}.get(status, "❓")
        lines.append(f"\n{status_icon} {pid}: {pred['title']}")
        lines.append(f"   证伪条件: {pred['falsified_if']}")
        lines.append(f"   截止: {pred['deadline_days']}天 | 方法: {pred['method']}")

        if checks:
            last = checks[-1]
            lines.append(f"   最近检查: {last['date']} → {'PASS' if last['passed'] else 'FAIL'} {last.get('detail','')}")

        if status == "falsified" and record.get("remediation_triggered"):
            lines.append(f"   🔔 整改预案已触发: 见remediation_plans.md")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


if __name__ == "__main__":
    if "--check" in sys.argv:
        idx = sys.argv.index("--check")
        if idx + 1 < len(sys.argv):
            pid = sys.argv[idx + 1].upper()
            result = check_prediction(pid)
            print(json.dumps(result, indent=2, default=str))
        else:
            print("Usage: --check P1-P5")
    elif "--record" in sys.argv:
        idx = sys.argv.index("--record")
        if idx + 2 < len(sys.argv):
            pid = sys.argv[idx + 1].upper()
            passed = sys.argv[idx + 2].lower() in ("pass", "true", "1", "yes")
            detail = sys.argv[idx + 3] if idx + 3 < len(sys.argv) else ""
            entry = record_result(pid, passed, detail)
            print(f"Recorded {pid}: {'PASS' if passed else 'FAIL'}")
        else:
            print("Usage: --record P1 pass/fail [detail]")
    elif "--json" in sys.argv:
        tracker = load_tracker()
        print(json.dumps(tracker, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_status())
