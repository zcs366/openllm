"""Layer 10.b — IOS 自进化边界控制 (Self-Modification Guard)。

监控 Agent 对自身代码/配置的修改行为，防止失控自改。
纯规则 + 统计，零 LLM 调用。状态持久化到 ~/.hermes/governance/ 下的 JSON 文件。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

_STATE_FILE = Path.home() / ".hermes" / "governance" / "self_modification_state.json"
DEFAULT_WINDOW_HOURS: int = 24
DEFAULT_MAX_MODIFICATIONS: int = 3
DEGRADATION_THRESHOLD: float = 0.10   # reliability 下降 > 10% 触发
ROLLBACK_CONFIDENCE: float = 0.6     # 退化 + 高频 → 回滚


@dataclass
class ModificationEvent:
    """不可变的自修改事件记录。"""
    target_file: str
    change_type: str
    agent_id: str
    timestamp: float
    reliability_before: Optional[float] = None
    reliability_after: Optional[float] = None


class SelfModificationGuard:
    """IOS 自进化边界守卫。

    五个核心方法：
        record_modification — 记录每次自修改（不可变追加）
        check_rate_limit — 滑动窗口超频检测
        check_degradation — reliability 退化检测
        should_rollback — 综合决策是否应回滚
        get_modification_history — 获取某文件的完整修改历史
    """

    def __init__(
        self,
        state_path: Path | str = _STATE_FILE,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        max_modifications: int = DEFAULT_MAX_MODIFICATIONS,
    ):
        self._state_path = Path(state_path)
        self._window_seconds = window_hours * 3600
        self._max_mods = max_modifications
        self._state: dict[str, list[dict]] = self._load_state()

    def record_modification(
        self, target_file: str, change_type: str, agent_id: str,
        reliability_before: Optional[float] = None,
        reliability_after: Optional[float] = None,
    ) -> None:
        """记录一次自修改事件，不可变追加。"""
        event = ModificationEvent(
            target_file=target_file, change_type=change_type,
            agent_id=agent_id, timestamp=time.time(),
            reliability_before=reliability_before,
            reliability_after=reliability_after,
        )
        self._state.setdefault(target_file, []).append(asdict(event))
        self._save_state()

    def check_rate_limit(
        self, target_file: str,
        window_hours: int | None = None,
        max_modifications: int | None = None,
    ) -> bool:
        """检测目标文件在滑动窗口内是否超频，True = 已超频。"""
        window = (window_hours or self._window_seconds / 3600) * 3600
        limit = max_modifications or self._max_mods
        cutoff = time.time() - window
        return self._count_recent(target_file, cutoff) >= limit

    def check_degradation(
        self, target_file: str,
        reliability_before: float, reliability_after: float,
    ) -> tuple[bool, float]:
        """检测 reliability 退化是否超过阈值。

        Returns:
            (is_degraded, degradation_ratio) — 正值表示退化。
        """
        if reliability_before <= 0:
            return False, 0.0
        degradation = (reliability_before - reliability_after) / reliability_before
        return degradation >= DEGRADATION_THRESHOLD, max(degradation, 0.0)

    def should_rollback(self, target_file: str) -> bool:
        """综合判断是否应回滚。

        逻辑：退化 > 10% → 回滚；退化 + 超频 → 高置信度回滚；
        单纯超频无退化 → 不回滚（正常迭代）。
        """
        history = self._state.get(target_file, [])
        if not history:
            return False
        last = history[-1]
        has_degradation = (
            last.get("reliability_before") is not None
            and last.get("reliability_after") is not None
            and last["reliability_before"] > last["reliability_after"]
        )
        if has_degradation:
            is_degraded, _ = self.check_degradation(
                target_file, last["reliability_before"], last["reliability_after"],
            )
            if is_degraded:
                confidence = 0.8 if self.check_rate_limit(target_file) else ROLLBACK_CONFIDENCE
                return confidence >= ROLLBACK_CONFIDENCE
        return False

    def get_modification_history(
        self, target_file: str, limit: int | None = None,
    ) -> list[dict]:
        """获取目标文件的修改历史（时间正序）。"""
        history = self._state.get(target_file, [])
        return history[-limit:] if limit is not None else list(history)

    def _count_recent(self, target_file: str, cutoff: float) -> int:
        """统计 cutoff 之后的修改次数。"""
        return sum(1 for e in self._state.get(target_file, []) if e["timestamp"] >= cutoff)

    def _load_state(self) -> dict[str, list[dict]]:
        """从 JSON 加载状态，不存在则返回空字典。"""
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_state(self) -> None:
        """持久化状态到 JSON 文件。"""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8",
        )
