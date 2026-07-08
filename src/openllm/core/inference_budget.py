"""IAX Layer 7 推理预算管理 — Kamera

纯统计，零LLM调用。记录推理token消耗和耗时，
提供使用统计、预算检查和限流建议。
数据JSON持久化到 ~/.hermes/inference_budget_state.json。

架构归属：IAX (心跳层) → Layer 7 推理预算
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

@dataclass
class InferenceRecord:
    """单次推理记录"""
    model: str
    tokens_in: int
    tokens_out: int
    duration_ms: float
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 默认每token单价（USD/1M tokens）: (input_price, output_price)
DEFAULT_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o":      (2.50,  10.00),
    "gpt-4o-mini": (0.15,  0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3-haiku":    (0.25, 1.25),
    "deepseek-chat":     (0.14, 0.28),
    "deepseek-coder":    (0.14, 0.28),
    "qwen-turbo":        (0.02, 0.06),
    "qwen-plus":         (0.40, 1.20),
    "qwen-max":          (2.40, 9.60),
}

class InferenceBudgetManager:
    """推理预算管理器 — 记录/统计/预算检查/限流建议"""

    STATE_FILE = Path.home() / ".hermes" / "inference_budget_state.json"
    def __init__(self, price_table: Optional[dict] = None):
        self._prices = price_table or DEFAULT_PRICE_TABLE
        self._records: list[InferenceRecord] = []
        self._load()

    # ── 持久化 ──────────────────────────────────────

    def _load(self) -> None:
        if self.STATE_FILE.exists():
            try:
                data = json.loads(self.STATE_FILE.read_text(encoding="utf-8"))
                self._records = [InferenceRecord(**r) for r in data.get("records", [])]
            except (json.JSONDecodeError, TypeError, KeyError):
                self._records = []

    def _save(self) -> None:
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"records": [r.to_dict() for r in self._records[-10000:]]}
        self.STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _match_price(self, model: str) -> tuple[float, float]:
        model_lower = model.lower()
        for key, price in self._prices.items():
            if key in model_lower:
                return price
        return (0.0, 0.0)

    def _calc_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        in_price, out_price = self._match_price(model)
        return (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    # ── 公开API ─────────────────────────────────────

    def record_inference(self, model: str, tokens_in: int, tokens_out: int, duration_ms: float) -> InferenceRecord:
        """记录一次推理调用，自动估算成本"""
        cost = self._calc_cost(model, tokens_in, tokens_out)
        rec = InferenceRecord(model=model, tokens_in=tokens_in, tokens_out=tokens_out, duration_ms=duration_ms, cost_usd=cost)
        self._records.append(rec)
        self._save()
        return rec

    def get_usage_stats(self, window_hours: float = 24) -> dict[str, Any]:
        """统计指定时间窗口内的使用量（总量/平均延迟/成本/模型列表）"""
        cutoff = time.time() - window_hours * 3600
        recent = [r for r in self._records if r.timestamp >= cutoff]
        if not recent:
            return {"total_inferences": 0, "total_tokens_in": 0, "total_tokens_out": 0, "avg_latency_ms": 0.0, "total_cost_usd": 0.0, "models_used": [], "period_hours": window_hours}
        return {
            "total_inferences": len(recent),
            "total_tokens_in": sum(r.tokens_in for r in recent),
            "total_tokens_out": sum(r.tokens_out for r in recent),
            "avg_latency_ms": round(sum(r.duration_ms for r in recent) / len(recent), 1),
            "total_cost_usd": round(sum(r.cost_usd for r in recent), 6),
            "models_used": sorted(set(r.model for r in recent)),
            "period_hours": window_hours,
        }

    def check_budget(self, daily_limit: int = 1_000_000) -> dict[str, Any]:
        """检查今日预算：是否超限、使用率、剩余额度、成本"""
        stats = self.get_usage_stats(window_hours=24)
        total = stats["total_tokens_in"] + stats["total_tokens_out"]
        return {
            "total_tokens": total,
            "limit": daily_limit,
            "usage_ratio": round(total / daily_limit, 4) if daily_limit > 0 else 1.0,
            "over_budget": total > daily_limit,
            "remaining": max(0, daily_limit - total),
            "cost_usd": stats["total_cost_usd"],
        }

    def suggest_throttle(self, trend_hours: float = 6) -> bool:
        """基于近期趋势建议限流：比较前后两段窗口的token消耗速率，增长率>50%则建议限流"""
        now = time.time()
        recent_cutoff = now - trend_hours * 3600
        earlier_cutoff = recent_cutoff - trend_hours * 3600
        recent = [r for r in self._records if r.timestamp >= recent_cutoff]
        earlier = [r for r in self._records if earlier_cutoff <= r.timestamp < recent_cutoff]
        if not earlier or not recent:
            return False
        recent_rate = sum(r.tokens_in + r.tokens_out for r in recent) / trend_hours
        earlier_rate = sum(r.tokens_in + r.tokens_out for r in earlier) / trend_hours
        if earlier_rate == 0:
            return recent_rate > 100_000
        return (recent_rate - earlier_rate) / earlier_rate > 0.5

    def get_total_records(self) -> int:
        """获取总记录数"""
        return len(self._records)

    def clear_old_records(self, keep_days: int = 30) -> int:
        """清理超过N天的旧记录，返回清理数"""
        cutoff = time.time() - keep_days * 86400
        before = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        removed = before - len(self._records)
        if removed > 0:
            self._save()
        return removed
