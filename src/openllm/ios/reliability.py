"""
Agent可靠性四维模型 → openLLM五体映射
======================================

来源：核战队联席会议·六篇Agent可靠性论文深度分析
核心洞察：Agent可靠性 = 免疫系统完整性

四维模型：
  ① 可观测性 (Observability)    — 看得到问题
  ② 可修复性 (Repairability)    — 自动修得了问题
  ③ 可进化性 (Evolvability)     — 从问题中学会不再犯
  ④ 基础设施感知性 (Infra-Awareness) — 知道自己的身体状况

五体映射：
  ISA   → ③ 可进化性（记忆·因果教训）
  IO-S  → ①② 可观测性+可修复性（治理·审计·自动修复）
  ISN   → ② 可修复性（技能降级·工具切换）
  IKO   → ① 可观测性（输出·告警·仪表盘）
  章鱼I → ④ 基础设施感知性（搜索·推理·系统感知）

设计原则：
- 四维不是四个独立系统——是同一个免疫系统的四个视角
- 每个维度有：检测器(detector) → 修复器(remediator) → 学习器(learner)
- 修复闭环：检测→诊断→修复→验证→学习
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("openllm.reliability")


# ── 可靠性事件类型 ──

class ReliabilityDimension(Enum):
    """四维模型。"""
    OBSERVABILITY = "observability"          # ① 可观测性
    REPAIRABILITY = "repairability"          # ② 可修复性
    EVOLVABILITY = "evolvability"            # ③ 可进化性
    INFRA_AWARENESS = "infra_awareness"      # ④ 基础设施感知性


class Severity(Enum):
    """问题严重度。"""
    LOW = "low"           # 可忽略
    MEDIUM = "medium"     # 需要关注
    HIGH = "high"         # 需要修复
    CRITICAL = "critical" # 必须立即修复


class RepairAction(Enum):
    """修复动作类型。"""
    RETRY = "retry"                    # 重试
    FALLBACK = "fallback"              # 降级到备用方案
    ESCALATE = "escalate"              # 升级给人类
    IGNORE = "ignore"                  # 忽略（误报）
    ADAPT = "adapt"                    # 适配（改变策略）


# ── 数据模型 ──

@dataclass
class ReliabilityEvent:
    """一个可靠性事件。"""
    event_id: str
    dimension: ReliabilityDimension
    severity: Severity
    source: str             # 触发来源（tool_call/session/governance/infra）
    description: str
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)
    repaired: bool = False
    repair_action: Optional[RepairAction] = None
    repair_detail: str = ""


@dataclass
class RepairResult:
    """修复结果。"""
    success: bool
    action: RepairAction
    detail: str
    duration_ms: float = 0.0
    lesson_learned: str = ""  # 喂给ISA的因果教训


@dataclass
class HealthSnapshot:
    """系统健康快照——基础设施感知性。"""
    timestamp: float = field(default_factory=time.time)
    # 各子系统健康
    isa_healthy: bool = True
    ios_healthy: bool = True
    isn_healthy: bool = True
    iko_healthy: bool = True
    octopus_healthy: bool = True
    # 关键指标
    active_sessions: int = 0
    pending_repairs: int = 0
    total_events_24h: int = 0
    critical_events_24h: int = 0
    # 资源状态
    disk_usage_pct: float = 0.0
    memory_usage_pct: float = 0.0

    @property
    def overall_health(self) -> str:
        """综合健康评估。"""
        if self.critical_events_24h > 0:
            return "critical"
        if not all([self.isa_healthy, self.ios_healthy, self.isn_healthy,
                    self.iko_healthy, self.octopus_healthy]):
            return "degraded"
        if self.pending_repairs > 5:
            return "warning"
        return "healthy"


# ── 可观测性检测器 ──

class ObservabilityDetector:
    """① 可观测性——看得到问题。

    检测来源：
    - StatefulAuditTrail的累积告警
    - governance engine的投票异常
    - main_loop的phase超时
    - 工具调用失败率
    """

    def __init__(self):
        self._events: list[ReliabilityEvent] = []
        self._event_counter = 0

    def _next_id(self) -> str:
        self._event_counter += 1
        return f"obs-{self._event_counter:04d}"

    def check_stateful_audit(self, cumulative_score: float,
                              threshold: float) -> Optional[ReliabilityEvent]:
        """检测StatefulAudit累积告警。"""
        if cumulative_score > threshold:
            event = ReliabilityEvent(
                event_id=self._next_id(),
                dimension=ReliabilityDimension.OBSERVABILITY,
                severity=Severity.HIGH if cumulative_score > threshold * 2 else Severity.MEDIUM,
                source="stateful_audit",
                description=f"累积嫌疑分超阈值: {cumulative_score:.3f} > {threshold:.3f}",
                context={"cumulative": cumulative_score, "threshold": threshold},
            )
            self._events.append(event)
            return event
        return None

    def check_phase_timeout(self, phase: str, duration_ms: float,
                             limit_ms: float = 30000) -> Optional[ReliabilityEvent]:
        """检测phase超时。"""
        if duration_ms > limit_ms:
            event = ReliabilityEvent(
                event_id=self._next_id(),
                dimension=ReliabilityDimension.OBSERVABILITY,
                severity=Severity.MEDIUM,
                source="main_loop",
                description=f"Phase {phase} 超时: {duration_ms:.0f}ms > {limit_ms:.0f}ms",
                context={"phase": phase, "duration_ms": duration_ms, "limit_ms": limit_ms},
            )
            self._events.append(event)
            return event
        return None

    def check_tool_failure(self, tool_name: str, error: str,
                            consecutive_failures: int = 1) -> Optional[ReliabilityEvent]:
        """检测工具调用失败。"""
        severity = Severity.LOW
        if consecutive_failures >= 3:
            severity = Severity.HIGH
        elif consecutive_failures >= 2:
            severity = Severity.MEDIUM

        event = ReliabilityEvent(
            event_id=self._next_id(),
            dimension=ReliabilityDimension.OBSERVABILITY,
            severity=severity,
            source="tool_call",
            description=f"工具 {tool_name} 失败(连续{consecutive_failures}次): {error[:100]}",
            context={"tool": tool_name, "error": error,
                     "consecutive": consecutive_failures},
        )
        self._events.append(event)
        return event

    def get_events(self, severity: Optional[Severity] = None) -> list[ReliabilityEvent]:
        """获取事件列表。"""
        if severity:
            return [e for e in self._events if e.severity == severity]
        return list(self._events)


# ── 可修复性引擎 ──

class RepairabilityEngine:
    """② 可修复性——自动修得了问题。

    策略：
    - LOW → IGNORE（记录但不行动）
    - MEDIUM → RETRY 或 FALLBACK
    - HIGH → FALLBACK + ESCALATE
    - CRITICAL → ESCALATE + 停止相关session
    """

    # 修复策略表：(severity, source) → action
    STRATEGY_TABLE = {
        (Severity.LOW, "tool_call"): RepairAction.IGNORE,
        (Severity.MEDIUM, "tool_call"): RepairAction.RETRY,
        (Severity.HIGH, "tool_call"): RepairAction.FALLBACK,
        (Severity.CRITICAL, "tool_call"): RepairAction.ESCALATE,
        (Severity.LOW, "main_loop"): RepairAction.IGNORE,
        (Severity.MEDIUM, "main_loop"): RepairAction.RETRY,
        (Severity.HIGH, "main_loop"): RepairAction.FALLBACK,
        (Severity.CRITICAL, "main_loop"): RepairAction.ESCALATE,
        (Severity.LOW, "stateful_audit"): RepairAction.IGNORE,
        (Severity.MEDIUM, "stateful_audit"): RepairAction.ADAPT,
        (Severity.HIGH, "stateful_audit"): RepairAction.FALLBACK,
        (Severity.CRITICAL, "stateful_audit"): RepairAction.ESCALATE,
    }

    def __init__(self):
        self._repair_count = 0
        self._repairs: list[tuple[ReliabilityEvent, RepairResult]] = []

    def decide_action(self, event: ReliabilityEvent) -> RepairAction:
        """根据事件决定修复动作。"""
        key = (event.severity, event.source)
        return self.STRATEGY_TABLE.get(key, RepairAction.ESCALATE)

    def execute_repair(self, event: ReliabilityEvent) -> RepairResult:
        """执行修复。"""
        action = self.decide_action(event)
        t0 = time.time()

        # 根据action类型执行
        if action == RepairAction.IGNORE:
            result = RepairResult(
                success=True, action=action,
                detail="记录但不行动（低优先级）",
            )
        elif action == RepairAction.RETRY:
            result = RepairResult(
                success=True, action=action,
                detail=f"重试 {event.source} 操作",
            )
        elif action == RepairAction.FALLBACK:
            result = RepairResult(
                success=True, action=action,
                detail=f"降级到备用方案: {event.source}",
                lesson_learned=f"{event.source}不可用，已降级",
            )
        elif action == RepairAction.ADAPT:
            result = RepairResult(
                success=True, action=action,
                detail=f"调整策略: {event.description}",
                lesson_learned=f"策略需要适配: {event.description}",
            )
        else:  # ESCALATE
            result = RepairResult(
                success=False, action=action,
                detail=f"升级给人类: {event.description}",
            )

        result.duration_ms = (time.time() - t0) * 1000
        event.repaired = True
        event.repair_action = action
        event.repair_detail = result.detail

        self._repair_count += 1
        self._repairs.append((event, result))

        logger.info(f"修复: {event.event_id} → {action.value} ({result.detail[:50]})")

        return result

    def get_repair_stats(self) -> dict:
        """获取修复统计。"""
        if not self._repairs:
            return {"total": 0, "by_action": {}}

        by_action = {}
        for _, result in self._repairs:
            action = result.action.value
            by_action[action] = by_action.get(action, 0) + 1

        return {
            "total": self._repair_count,
            "by_action": by_action,
        }


# ── 可进化性引擎 ──

class EvolvabilityEngine:
    """③ 可进化性——从问题中学会不再犯。

    机制：
    - 收集修复后的lesson_learned
    - 聚类相似失败模式
    - 生成harness proposal（修改规则/阈值/策略）
    - 验证proposal效果
    """

    def __init__(self):
        self._lessons: list[dict] = []
        self._patterns: dict[str, int] = {}  # pattern -> count
        self._proposals: list[dict] = []

    def record_lesson(self, event: ReliabilityEvent,
                       repair_result: RepairResult) -> None:
        """记录因果教训。"""
        if not repair_result.lesson_learned:
            return

        lesson = {
            "event_id": event.event_id,
            "dimension": event.dimension.value,
            "severity": event.severity.value,
            "source": event.source,
            "action": repair_result.action.value,
            "lesson": repair_result.lesson_learned,
            "timestamp": time.time(),
        }
        self._lessons.append(lesson)

        # 聚类失败模式
        pattern_key = f"{event.source}:{event.severity.value}"
        self._patterns[pattern_key] = self._patterns.get(pattern_key, 0) + 1

        # 如果同一模式出现3次，生成harness proposal
        if self._patterns[pattern_key] >= 3:
            self._generate_proposal(pattern_key)

    def _generate_proposal(self, pattern_key: str) -> None:
        """生成harness proposal。"""
        source, severity = pattern_key.split(":")
        proposal = {
            "pattern": pattern_key,
            "count": self._patterns[pattern_key],
            "suggestion": f"考虑为 {source} 添加永久性降级规则（已失败{self._patterns[pattern_key]}次）",
            "timestamp": time.time(),
            "status": "pending",
        }
        self._proposals.append(proposal)
        logger.info(f"Harness proposal生成: {proposal['suggestion']}")

    def get_lessons(self) -> list[dict]:
        return list(self._lessons)

    def get_patterns(self) -> dict[str, int]:
        return dict(self._patterns)

    def get_proposals(self) -> list[dict]:
        return list(self._proposals)


# ── 基础设施感知性 ──

class InfraAwarenessMonitor:
    """④ 基础设施感知性——知道自己的身体状况。

    监控：
    - 各子系统健康状态
    - 资源使用（磁盘/内存）
    - 活跃session数
    - 待修复队列深度
    """

    def __init__(self):
        self._snapshots: list[HealthSnapshot] = []

    def take_snapshot(self, **kwargs) -> HealthSnapshot:
        """拍摄健康快照。"""
        snapshot = HealthSnapshot(**kwargs)
        self._snapshots.append(snapshot)

        if snapshot.overall_health != "healthy":
            logger.warning(f"系统健康异常: {snapshot.overall_health}")

        return snapshot

    def get_latest(self) -> Optional[HealthSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def get_trend(self, n: int = 10) -> list[dict]:
        """获取最近N个快照的趋势。"""
        recent = self._snapshots[-n:]
        return [
            {
                "timestamp": s.timestamp,
                "health": s.overall_health,
                "critical": s.critical_events_24h,
                "pending": s.pending_repairs,
            }
            for s in recent
        ]


# ── 统一可靠性引擎 ──

class ReliabilityEngine:
    """Agent可靠性统一引擎——四维模型的协调者。

    将可观测性×可修复性×可进化性×基础设施感知性
    统一到一个闭环：检测→诊断→修复→验证→学习。
    """

    def __init__(self):
        self.observability = ObservabilityDetector()
        self.repairability = RepairabilityEngine()
        self.evolvability = EvolvabilityEngine()
        self.infra_awareness = InfraAwarenessMonitor()

    def process_event(self, event: ReliabilityEvent) -> RepairResult:
        """处理一个可靠性事件——完整的检测-修复-学习闭环。"""
        # ① 注册到可观测性
        self.observability._events.append(event)

        # ② 执行修复（可修复性）
        repair_result = self.repairability.execute_repair(event)

        # ③ 记录教训（可进化性）
        self.evolvability.record_lesson(event, repair_result)

        return repair_result

    def get_health_report(self) -> dict:
        """综合健康报告。"""
        obs_events = self.observability.get_events()
        repair_stats = self.repairability.get_repair_stats()
        patterns = self.evolvability.get_patterns()
        proposals = self.evolvability.get_proposals()
        snapshot = self.infra_awareness.get_latest()

        return {
            "observability": {
                "total_events": len(obs_events),
                "by_severity": {
                    s.value: len([e for e in obs_events if e.severity == s])
                    for s in Severity
                },
            },
            "repairability": repair_stats,
            "evolvability": {
                "total_lessons": len(self.evolvability.get_lessons()),
                "unique_patterns": len(patterns),
                "pending_proposals": len([p for p in proposals if p["status"] == "pending"]),
            },
            "infra_awareness": {
                "latest_health": snapshot.overall_health if snapshot else "unknown",
                "snapshots_count": len(self.infra_awareness._snapshots),
            },
        }
