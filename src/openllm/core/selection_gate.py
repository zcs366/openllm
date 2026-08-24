"""
SelectionGate — 技能/提案的选择闸门（V驱动生死判决）。

核心语义：提案必须通过所有validator才能被接受。
threshold=0.0时是monotone safety——V不下降即可通过，不要求V上升。

接口契约（来自 signal_bus.make_v_validator 文档字符串）：
    validator签名 = (proposal) -> (passed: bool, score: float, details: dict)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Proposal 数据模型 ──

@dataclass
class Proposal:
    """被闸门判决的提案对象。"""

    proposal_id: str
    description: str
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending / accepted / rejected
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── SelectionGate 核心 ──

class SelectionGate:
    """提案选择闸门——依次跑所有validator，全票通过才accepted，一票否决则rejected。"""

    def __init__(self, validators: Optional[List[Callable]] = None) -> None:
        """
        初始化闸门。

        Args:
            validators: validator函数列表。每个签名须为
                        (proposal) -> (passed: bool, score: float, details: dict)
        """
        self.validators: List[Callable] = validators if validators is not None else []
        self.decisions: List[Dict[str, Any]] = []  # append-only日志

    def evaluate(self, proposal: Proposal) -> Dict[str, Any]:
        """
        对提案跑所有validator，收集结果。

        规则：全部passed才算accepted，任一failed则rejected。

        Args:
            proposal: 待判决的提案

        Returns:
            {proposal, accepted: bool, results: [{validator_name, passed, score, details}]}
        """
        results: List[Dict[str, Any]] = []
        all_passed = True

        for idx, validator in enumerate(self.validators):
            name = getattr(validator, "__name__", f"validator_{idx}")
            try:
                passed, score, details = validator(proposal)
            except Exception as exc:
                passed, score, details = False, 0.0, {"error": str(exc)}
            if not passed:
                all_passed = False
            results.append({
                "validator_name": name,
                "passed": bool(passed),
                "score": float(score),
                "details": details,
            })

        accepted = all_passed
        proposal.status = "accepted" if accepted else "rejected"

        # append-only日志
        self.decisions.append({
            "proposal_id": proposal.proposal_id,
            "accepted": accepted,
            "scores": [r["score"] for r in results],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "proposal": proposal,
            "accepted": accepted,
            "results": results,
        }


# ── 与signal_bus对接的工厂 ──

def make_v_gate(threshold: float = 0.0) -> SelectionGate:
    """
    便捷工厂：用signal_bus.make_v_validator构造validator，包成SelectionGate。

    threshold=0.0 时是monotone safety——V不下降即通过。

    Raises:
        ImportError: 当 ~/io-s/signal_bus.py 不可导入时抛出清晰错误
    """
    io_s_path = str(Path.home() / "io-s")
    if io_s_path not in sys.path:
        sys.path.insert(0, io_s_path)

    try:
        from signal_bus import make_v_validator
    except ImportError as exc:
        raise ImportError(
            f"无法导入signal_bus.make_v_validator。"
            f"请确认 ~/io-s/signal_bus.py 存在且可导入。"
            f"原始错误: {exc}"
        ) from exc

    validator = make_v_validator(threshold=threshold)
    return SelectionGate(validators=[validator])
