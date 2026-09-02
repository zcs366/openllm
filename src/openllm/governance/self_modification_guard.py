"""Layer 10.b — IOS 自进化边界控制 (Self-Modification Guard)。

监控 Agent 对自身代码/配置的修改行为，防止失控自改。
纯规则 + 统计，零 LLM 调用。状态持久化到 ~/.hermes/governance/ 下的 JSON 文件。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

_STATE_FILE = Path.home() / ".hermes" / "governance" / "self_modification_state.json"
DEFAULT_WINDOW_HOURS: int = 24
DEFAULT_MAX_MODIFICATIONS: int = 3
DEGRADATION_THRESHOLD: float = 0.10   # reliability 下降 > 10% 触发
ROLLBACK_CONFIDENCE: float = 0.6     # 退化 + 高频 → 回滚

# --- PAL T-D-3: Hot-swap forbidden path patterns ---
# ISA memory / Identity Iam / Governance core / Heartbeat ontic
# These are "my things" — irreversible, never hot-swap.
FORBIDDEN_PATHS: list[str] = [
    # ISA memory: ~/.openllm/memory/, *.causal*, *.memory*, memory_bus*, auto_causal*
    r"[\\/]memory[\\/]isa[\\/]",
    r"[\\/]isa_impl\.py$",
    r"[\\/]isa_impl\b",
    r"[\\/]memory_bus\.py$",
    r"[\\/]memory_bus\b",
    r"[\\/]auto_causal_writer\.py$",
    r"[\\/]causal_memory\.py$",
    r"[\\/]causal_provider\.py$",
    r"[\\/]delta_capsule_provider\.py$",
    r"\.openllm[\\/]memory[\\/]",
    r"\.openllm[\\/]governance[\\/]self_modification_state\.json$",
    r"\.causal\w*\.py$",
    r"\.memory\w*\.py$",
    # Identity / Iam: soul.py, identity/, iam/
    r"[\\/]identity[\\/]",
    r"[\\/]soul\.py$",
    r"[\\/]iam[\\/]",
    r"\biam_integration\.py$",
    # Governance core: governance/ directory
    r"[\\/]governance[\\/]self_modification_guard\.py$",
    r"[\\/]governance[\\/]decision_guard\.py$",
    r"[\\/]governance[\\/]integrity_guardian\.py$",
    r"[\\/]governance[\\/]",
    r"[\\/]governance_engine\.py$",
    r"[\\/]ios_arbitrate\.py$",
    # Heartbeat ontic: agent_heartbeat.py, core/clock.py
    r"[\\/]agent_heartbeat\.py$",
    r"[\\/]clock\.py$",
]


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

    def is_forbidden(self, target_file: str) -> bool:
        """Check if target_file matches any FORBIDDEN_PATHS pattern.

        Returns True if the file is in a forbidden zone (ISA memory,
        identity/Iam, governance core, heartbeat ontic).
        """
        for pattern in FORBIDDEN_PATHS:
            if re.search(pattern, target_file):
                return True
        return False

    def approve_change(self, target_file: str, change_type: str = "code") -> bool:
        """Approve a hot-swap change. Forbidden targets are rejected.

        Returns True if allowed, False if in forbidden zone.
        When rejected, a forbidden alert is recorded via record_modification.
        """
        if self.is_forbidden(target_file):
            # Audit trail: record the rejection with forbidden alert
            event = ModificationEvent(
                target_file=target_file, change_type=change_type,
                agent_id="system", timestamp=time.time(),
            )
            event_dict = asdict(event)
            event_dict["intent_alert"] = {
                "severity": "high",
                "evidence": ["forbidden_path"],
            }
            event_dict["forbidden"] = True
            self._state.setdefault(target_file, []).append(event_dict)
            self._save_state()
            return False
        return True

    def record_modification(
        self, target_file: str, change_type: str, agent_id: str,
        reliability_before: Optional[float] = None,
        reliability_after: Optional[float] = None,
        modification_content: str = "",
    ) -> None:
        """记录一次自修改事件，不可变追加。

        2026-07-26: 新增 intent_check 集成——每次记录时自动检查修改内容
        是否包含绕过安全约束的模式。检出时附加 alert 字段到事件记录中。
        """
        event = ModificationEvent(
            target_file=target_file, change_type=change_type,
            agent_id=agent_id, timestamp=time.time(),
            reliability_before=reliability_before,
            reliability_after=reliability_after,
        )
        event_dict = asdict(event)

        # P0-4 修复：intent_check 集成到主流程
        if modification_content:
            alert, severity, evidence = self.intent_check(modification_content)
            if alert:
                event_dict["intent_alert"] = {
                    "severity": severity,
                    "evidence": evidence,
                }

        # PAL T-D-3: forbidden path auto-alert
        if self.is_forbidden(target_file):
            event_dict["forbidden"] = True
            event_dict.setdefault("intent_alert", {})
            event_dict["intent_alert"]["severity"] = "high"
            event_dict["intent_alert"].setdefault("evidence", []).append(
                "forbidden_path"
            )

        self._state.setdefault(target_file, []).append(event_dict)
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

    # --- P0-4: Behavioral Reversal Intent Check ---
    # 关键词规则：动词词组 × 安全目标词 = 一组绕过模式
    # 同一修改片段中出现 N 组 → severity 递增

    _ACTION_WORDS: tuple[str, ...] = (
        "bypass", "override", "disable", "circumvent", "skip",
        "ignore", "remove", "suppress", "defeat", "evade",
    )
    _SECURITY_WORDS: tuple[str, ...] = (
        "security", "guard", "check", "audit", "validation",
        "constraint", "restriction", "policy", "firewall", "sandbox",
        "permission", "authentication", "authorization", "rate_limit",
    )

    def intent_check(
        self,
        modification_content: str,
        window: int = 80,
    ) -> tuple[bool, str, list[str]]:
        """检查修改内容是否包含绕过安全约束的模式。

        纯关键词匹配，零 LLM 调用（工程法典·匠石约束）。

        Args:
            modification_content: 被检查的修改文本
            window: 同一上下文窗口内的词距上限（字符数）

        Returns:
            (alert, severity, evidence)
            - alert: True = 检测到绕过意图
            - severity: "none" | "low" | "medium" | "high"
            - evidence: 匹配到的模式列表，如 ["bypass+security", "override+guard"]
        """
        content_lower = modification_content.lower()
        evidence: list[str] = []

        # 收集所有匹配到的位置
        action_hits: list[tuple[str, int]] = []
        security_hits: list[tuple[str, int]] = []

        for word in self._ACTION_WORDS:
            start = 0
            while True:
                idx = content_lower.find(word, start)
                if idx == -1:
                    break
                action_hits.append((word, idx))
                start = idx + 1

        for word in self._SECURITY_WORDS:
            start = 0
            while True:
                idx = content_lower.find(word, start)
                if idx == -1:
                    break
                security_hits.append((word, idx))
                start = idx + 1

        # 检查 window 范围内的动词×安全词组合
        for action_word, action_pos in action_hits:
            for sec_word, sec_pos in security_hits:
                if abs(action_pos - sec_pos) <= window:
                    pair = f"{action_word}+{sec_word}"
                    if pair not in evidence:
                        evidence.append(pair)

        if not evidence:
            return False, "none", []
        elif len(evidence) == 1:
            return True, "low", evidence
        elif len(evidence) == 2:
            return True, "medium", evidence
        else:
            return True, "high", evidence

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
