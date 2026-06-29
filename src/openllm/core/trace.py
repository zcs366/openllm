"""Trace — IO-S结构化追踪系统

每次工具验证、模型调用、Agent turn都吐出结构化追踪数据。
追踪数据不替代审计日志——审计日志是"谁做了什么"，追踪是"怎么做、结果如何"。
"""

from __future__ import annotations
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("io-s.trace")

TRACE_DIR = Path.home() / ".io-s" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# ── 数据结构 ──────────────────────────────────


@dataclass
class ToolTrace:
    """单次工具调用追踪。"""
    turn_id: str                # 关联的Agent turn ID
    tool_name: str              # 工具名称
    params: dict                # 调用参数
    verify_before: dict         # 调用前验证结果 {pass, reason}
    success: bool               # 执行是否成功
    output_size: int            # 输出字节数
    verify_after: dict          # 调用后验证结果 {pass, reason}
    latency_ms: float           # 执行耗时
    timestamp: float = 0.0      # 时间戳

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ModelTrace:
    """单次模型调用追踪。"""
    turn_id: str                # 关联的Agent turn ID
    model: str                  # 模型名称
    provider: str               # 提供商
    input_tokens: int           # 输入token数
    output_tokens: int          # 输出token数
    latency_ms: float           # 延迟
    cost_usd: float = 0.0       # 费用
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class TurnTrace:
    """单轮Agent turn的完整追踪。"""
    turn_id: str                # 唯一标识
    user_input_preview: str     # 用户输入摘要
    phase_sequence: list[str]   # PLan→ACT→OBSERVE→REFLECT
    tool_traces: list[dict] = field(default_factory=list)  # 本轮的tool调用
    model_trace: Optional[dict] = None  # 本轮的模型调用
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    verify_count: int = 0
    verify_fail_count: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def record_tool(self, tt: ToolTrace):
        self.tool_traces.append(asdict(tt))
        if tt.verify_after and not tt.verify_after.get("pass", True):
            self.verify_fail_count += 1
        self.verify_count += 1

    def record_model(self, mt: ModelTrace):
        self.model_trace = asdict(mt)
        self.total_tokens += mt.input_tokens + mt.output_tokens

    def summary(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "tools_called": len(self.tool_traces),
            "tools_ok": len([t for t in self.tool_traces if t.get("success")]),
            "verify_passed": self.verify_count - self.verify_fail_count,
            "verify_failed": self.verify_fail_count,
            "tokens": self.total_tokens,
            "latency_ms": self.total_latency_ms,
        }


# ── I/O ───────────────────────────────────────


def write_tool_trace(tt: ToolTrace):
    """写入单次工具调用trace。"""
    path = TRACE_DIR / "tools.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(tt), ensure_ascii=False) + "\n")


def write_model_trace(mt: ModelTrace):
    """写入单次模型调用trace（=成本记录）。"""
    path = TRACE_DIR / "models.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(mt), ensure_ascii=False) + "\n")


def write_turn_trace(tt: TurnTrace):
    """写入单轮Agent turn完整trace。"""
    path = TRACE_DIR / "turns.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(tt), ensure_ascii=False) + "\n")


def read_tool_traces(limit: int = 50) -> list[dict]:
    """读取最近N条工具trace。"""
    return _read_jsonl(TRACE_DIR / "tools.jsonl", limit)


def read_model_traces(limit: int = 50) -> list[dict]:
    """读取最近N条模型trace（=成本数据）。"""
    return _read_jsonl(TRACE_DIR / "models.jsonl", limit)


def _read_jsonl(path: Path, limit: int) -> list[dict]:
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


# ── 聚合查询 ──────────────────────────────────


def cost_summary() -> dict:
    """成本汇总。"""
    records = read_model_traces(limit=10000)
    if not records:
        return {"total_calls": 0, "total_tokens": 0, "total_cost_usd": 0.0}
    return {
        "total_calls": len(records),
        "total_tokens": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0)
                            for r in records),
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in records), 6),
        "total_latency_ms": round(sum(r.get("latency_ms", 0.0) for r in records), 1),
        "avg_latency_ms": round(
            sum(r.get("latency_ms", 0.0) for r in records) / max(len(records), 1), 1
        ),
    }


def verify_stats() -> dict:
    """verify拦截统计。"""
    records = read_model_traces(limit=10000)  # 复用model trace统计
    tool_records = read_tool_traces(limit=10000)
    before_fails = sum(1 for r in tool_records
                       if not r.get("verify_before", {}).get("pass", True))
    after_fails = sum(1 for r in tool_records
                      if not r.get("verify_after", {}).get("pass", True))
    return {
        "total_tool_calls": len(tool_records),
        "verify_before_fails": before_fails,
        "verify_after_fails": after_fails,
        "success_rate": round(
            (len(tool_records) - before_fails - after_fails) / max(len(tool_records), 1) * 100, 1
        ) if tool_records else 0.0,
    }
