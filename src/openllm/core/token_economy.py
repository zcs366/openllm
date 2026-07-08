"""IOS Layer 15 Token经济 — TokenEconomy

Token成本追踪与预算管理。记录LLM调用的token消耗，
按模型计算成本，支持日/周粒度查询和预算告警。
JSON持久化到 ~/.hermes/token_economy_state.json。
架构归属：IOS (决策层) → Layer 15 Token经济
"""
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

@dataclass
class UsageRecord:
    """单次Token使用记录"""
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    task: str
    timestamp: float = field(default_factory=time.time)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

# 默认价格表：(input_per_1M, output_per_1M) USD
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15,  0.60),
    "deepseek-chat": (0.14,  0.28),
    "deepseek-coder":(0.14,  0.28),
}

class TokenEconomy:
    """IOS Layer 15 Token经济管理器 — 记录/统计/预算/报告"""
    STATE_FILE = Path.home() / ".hermes" / "token_economy_state.json"
    def __init__(self, prices: Optional[dict[str, tuple[float, float]]] = None):
        self._prices = prices or DEFAULT_PRICES
        self._records: list[UsageRecord] = []
        self._daily_limit: float = 10.0
        self._weekly_limit: float = 50.0
        self._load()

    def _load(self) -> None:
        if self.STATE_FILE.exists():
            try:
                data = json.loads(self.STATE_FILE.read_text(encoding="utf-8"))
                self._records = [UsageRecord(**r) for r in data.get("records", [])]
                self._daily_limit = data.get("daily_limit", 10.0)
                self._weekly_limit = data.get("weekly_limit", 50.0)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    def _save(self) -> None:
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"daily_limit": self._daily_limit, "weekly_limit": self._weekly_limit,
                   "records": [r.to_dict() for r in self._records[-10000:]]}
        self.STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _match_prices(self, model: str) -> tuple[float, float]:
        """模糊匹配模型价格（最长键优先），未找到返回(0, 0)"""
        ml = model.lower()
        best: tuple[float, float] = (0.0, 0.0)
        best_len = 0
        for key, price in self._prices.items():
            if key in ml and len(key) > best_len:
                best, best_len = price, len(key)
        return best

    def compute_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """计算指定模型+token量的美元成本"""
        p_in, p_out = self._match_prices(model)
        return round((tokens_in * p_in + tokens_out * p_out) / 1_000_000, 6)

    def record_usage(self, model: str, tokens_in: int, tokens_out: int, task: str = "") -> UsageRecord:
        """记录一次Token使用，自动计算成本并持久化"""
        cost = self.compute_cost(model, tokens_in, tokens_out)
        rec = UsageRecord(model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost, task=task)
        self._records.append(rec)
        self._save()
        return rec

    def get_daily_cost(self, date: Optional[date] = None) -> float:
        """查询指定日期的总成本（默认今日）"""
        target = date or datetime.now().date()
        start = datetime.combine(target, datetime.min.time()).timestamp()
        return round(sum(r.cost_usd for r in self._records if start <= r.timestamp < start + 86400), 6)

    def get_weekly_cost(self) -> float:
        """查询最近7天的总成本"""
        cutoff = time.time() - 7 * 86400
        return round(sum(r.cost_usd for r in self._records if r.timestamp >= cutoff), 6)

    def set_budget(self, daily_limit: float, weekly_limit: float) -> None:
        """设置日/周预算上限（USD）"""
        self._daily_limit = daily_limit
        self._weekly_limit = weekly_limit
        self._save()

    def check_budget(self) -> tuple[bool, dict[str, Any]]:
        """检查预算状态，返回(是否超限, 详情字典)"""
        daily, weekly = self.get_daily_cost(), self.get_weekly_cost()
        d_pct = round(daily / self._daily_limit * 100, 1) if self._daily_limit > 0 else 100.0
        w_pct = round(weekly / self._weekly_limit * 100, 1) if self._weekly_limit > 0 else 100.0
        return daily >= self._daily_limit or weekly >= self._weekly_limit, {
            "daily_cost": daily, "daily_limit": self._daily_limit, "daily_usage_pct": d_pct,
            "weekly_cost": weekly, "weekly_limit": self._weekly_limit, "weekly_usage_pct": w_pct,
        }

    def get_cost_report(self) -> dict[str, Any]:
        """生成成本报告：总额、按模型/任务分布、预算状态"""
        by_model: dict[str, float] = {}
        by_task: dict[str, float] = {}
        for r in self._records:
            by_model[r.model] = by_model.get(r.model, 0) + r.cost_usd
            by_task[r.task or "未分类"] = by_task.get(r.task or "未分类", 0) + r.cost_usd
        over, budget = self.check_budget()
        return {
            "total_cost_usd": round(sum(r.cost_usd for r in self._records), 6),
            "total_tokens_in": sum(r.tokens_in for r in self._records),
            "total_tokens_out": sum(r.tokens_out for r in self._records),
            "record_count": len(self._records),
            "by_model": {k: round(v, 6) for k, v in sorted(by_model.items(), key=lambda x: -x[1])},
            "by_task": {k: round(v, 6) for k, v in sorted(by_task.items(), key=lambda x: -x[1])},
            "budget_over": over, "budget": budget,
        }

    def get_total_records(self) -> int:
        """获取总记录数"""
        return len(self._records)

    def clear_old_records(self, keep_days: int = 30) -> int:
        """清理超过N天的旧记录，返回清理数量"""
        cutoff = time.time() - keep_days * 86400
        before = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        removed = before - len(self._records)
        if removed > 0:
            self._save()
        return removed
