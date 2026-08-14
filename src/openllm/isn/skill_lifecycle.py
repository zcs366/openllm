"""
ISN Skill Lifecycle Manager — 统一状态机
==========================================

管理技能生命周期的状态转换：active → dormant → deprecated → retired

架构：
  - SkillState: 枚举状态 (active, dormant, deprecated, retired)
  - SkillEvent: 事件基类 (SkillCreated, QualityThresholdHit, DormancyDetected, RetirementApproved)
  - TransitionGuard: 转换守卫（退休需双人审批）
  - EventBus: 事件总线，通知各框架
  - AuditTrail: 审计日志，记录所有转换

事件驱动：
  EventBus → 各框架事件处理器 (ISA, IO-S, ISN, IKO)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("openllm.isn.skill_lifecycle")

# ══════════════════════════════════════════════════════════════════════════════
# 状态定义
# ══════════════════════════════════════════════════════════════════════════════

class SkillState(Enum):
    """技能生命周期状态。"""
    ACTIVE = "active"
    DORMANT = "dormant"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


# ══════════════════════════════════════════════════════════════════════════════
# 事件定义
# ══════════════════════════════════════════════════════════════════════════════

class SkillEvent:
    """技能事件基类。"""
    def __init__(self, skill_name: str, **kwargs):
        self.skill_name = skill_name
        self.timestamp = time.time()
        self.metadata = kwargs

    @property
    def event_type(self) -> str:
        return self.__class__.__name__

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "skill_name": self.skill_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class SkillCreated(SkillEvent):
    """技能创建事件。"""
    def __init__(self, skill_name: str, author: str = "", **kwargs):
        super().__init__(skill_name, author=author, **kwargs)


class QualityThresholdHit(SkillEvent):
    """质量阈值命中事件（触发从dormant恢复到active）。"""
    def __init__(self, skill_name: str, quality_score: float = 0.0, **kwargs):
        super().__init__(skill_name, quality_score=quality_score, **kwargs)


class DormancyDetected(SkillEvent):
    """休眠检测事件（触发active → dormant）。"""
    def __init__(self, skill_name: str, dormant_days: int = 0, **kwargs):
        super().__init__(skill_name, dormant_days=dormant_days, **kwargs)


class RetirementApproved(SkillEvent):
    """退休批准事件（需双人审批）。"""
    def __init__(self, skill_name: str, approver: str = "", **kwargs):
        super().__init__(skill_name, approver=approver, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 状态转换规则
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionRule:
    """状态转换规则。"""
    source: SkillState
    target: SkillState
    event_type: str          # 事件类型名称
    guard: Optional[Callable] = None  # 守卫函数（返回bool）

    def __repr__(self):
        guard_str = f" + guard={self.guard.__name__}" if self.guard else ""
        return f"Transition({self.source.value} → {self.target.value} on {self.event_type}{guard_str})"


# ══════════════════════════════════════════════════════════════════════════════
# 守卫函数
# ══════════════════════════════════════════════════════════════════════════════

def dual_approval_guard(required_count: int = 2) -> Callable:
    """
    退休守卫：需双人审批。

    返回一个守卫函数，检查审批人数量是否达到required_count。
    使用 entry.retirement_approvers 作为审批人集合。
    """
    def guard(event: RetirementApproved, context: dict) -> bool:
        entry = context.get("skill")
        if entry is None:
            return False
        approver = event.metadata.get("approver", "")
        if approver:
            entry.retirement_approvers.add(approver)
        return len(entry.retirement_approvers) >= required_count
    return guard


def quality_threshold_guard(threshold: float = 0.6) -> Callable:
    """
    质量阈值守卫：质量分需超过阈值。

    返回守卫函数，检查quality_score是否超过threshold。
    """
    def guard(event: QualityThresholdHit, context: dict) -> bool:
        score = event.metadata.get("quality_score", 0.0)
        return score >= threshold
    return guard


# ══════════════════════════════════════════════════════════════════════════════
# 事件总线
# ══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """
    事件总线：发布/订阅模式。

    各框架（ISA, IO-S, ISN, IKO）注册处理器，状态转换时触发。
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._all_handlers: List[Callable] = []

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """订阅特定事件类型。"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"订阅事件: {event_type} → {handler.__name__}")

    def subscribe_all(self, handler: Callable) -> None:
        """订阅所有事件。"""
        self._all_handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """取消订阅。"""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def publish(self, event: SkillEvent, transition: Optional[Tuple[SkillState, SkillState]] = None) -> List[Any]:
        """
        发布事件到所有订阅者。

        Args:
            event: 技能事件
            transition: (source_state, target_state) 转换信息

        Returns:
            各处理器返回值列表
        """
        results = []
        event_type = event.event_type

        # 触发特定事件类型的处理器
        for handler in self._handlers.get(event_type, []):
            try:
                result = handler(event, transition=transition)
                results.append(result)
            except Exception as e:
                logger.error(f"事件处理器异常: {handler.__name__} — {e}")

        # 触发全局处理器
        for handler in self._all_handlers:
            try:
                result = handler(event, transition=transition)
                results.append(result)
            except Exception as e:
                logger.error(f"全局事件处理器异常: {handler.__name__} — {e}")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# 审计日志
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditEntry:
    """审计日志条目。"""
    timestamp: float
    skill_name: str
    event_type: str
    source_state: str
    target_state: str
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "skill_name": self.skill_name,
            "event_type": self.event_type,
            "source_state": self.source_state,
            "target_state": self.target_state,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
        }


class AuditTrail:
    """
    审计日志：记录所有状态转换。

    支持JSON持久化到文件。
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._entries: List[AuditEntry] = []
        self._persist_path = persist_path

    def record(
        self,
        skill_name: str,
        event_type: str,
        source_state: SkillState,
        target_state: SkillState,
        metadata: Optional[dict] = None,
        success: bool = True,
        error: str = "",
    ) -> AuditEntry:
        """记录一条审计日志。"""
        entry = AuditEntry(
            timestamp=time.time(),
            skill_name=skill_name,
            event_type=event_type,
            source_state=source_state.value,
            target_state=target_state.value,
            metadata=metadata or {},
            success=success,
            error=error,
        )
        self._entries.append(entry)

        # 日志记录
        status = "✅" if success else "❌"
        logger.info(
            f"{status} 审计: {skill_name} | {source_state.value} → {target_state.value} "
            f"| 事件: {event_type}"
        )

        # 持久化
        if self._persist_path:
            self._persist()

        return entry

    def get_entries(
        self,
        skill_name: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """查询审计日志。"""
        entries = self._entries

        if skill_name:
            entries = [e for e in entries if e.skill_name == skill_name]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        return entries[-limit:]

    def _persist(self) -> None:
        """持久化到JSON文件。"""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        self._persist_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════════════
# 框架事件处理器
# ══════════════════════════════════════════════════════════════════════════════

def create_isa_handler():
    """ISA框架事件处理器：记忆层同步。"""
    def handler(event: SkillEvent, transition: Optional[Tuple[SkillState, SkillState]] = None):
        skill_name = event.skill_name
        if transition:
            source, target = transition
            logger.info(f"[ISA] 记忆层同步: {skill_name} {source.value} → {target.value}")
            # ISA逻辑：活跃技能增加记忆权重，退休技能降低权重
            if target == SkillState.ACTIVE:
                return {"action": "increase_memory_weight", "skill": skill_name}
            elif target == SkillState.RETIRED:
                return {"action": "decrease_memory_weight", "skill": skill_name}
        return {"action": "noop", "skill": skill_name}
    return handler


def create_ios_handler():
    """IO-S框架事件处理器：决策层通知。"""
    def handler(event: SkillEvent, transition: Optional[Tuple[SkillState, SkillState]] = None):
        skill_name = event.skill_name
        if transition:
            source, target = transition
            logger.info(f"[IO-S] 决策层通知: {skill_name} {source.value} → {target.value}")
            if target == SkillState.DEPRECATED:
                return {"action": "mark_for_review", "skill": skill_name}
            elif target == SkillState.RETIRED:
                return {"action": "remove_from_routing", "skill": skill_name}
        return {"action": "noop", "skill": skill_name}
    return handler


def create_isn_handler():
    """ISN框架事件处理器：技能注册更新。"""
    def handler(event: SkillEvent, transition: Optional[Tuple[SkillState, SkillState]] = None):
        skill_name = event.skill_name
        if transition:
            source, target = transition
            logger.info(f"[ISN] 技能注册更新: {skill_name} {source.value} → {target.value}")
            if target == SkillState.RETIRED:
                return {"action": "unregister_skill", "skill": skill_name}
            elif target == SkillState.ACTIVE:
                return {"action": "register_skill", "skill": skill_name}
        return {"action": "noop", "skill": skill_name}
    return handler


def create_iko_handler():
    """IKO框架事件处理器：输出层适配。"""
    def handler(event: SkillEvent, transition: Optional[Tuple[SkillState, SkillState]] = None):
        skill_name = event.skill_name
        if transition:
            source, target = transition
            logger.info(f"[IKO] 输出层适配: {skill_name} {source.value} → {target.value}")
            if target == SkillState.DORMANT:
                return {"action": "reduce_output_frequency", "skill": skill_name}
            elif target == SkillState.ACTIVE:
                return {"action": "restore_output_frequency", "skill": skill_name}
        return {"action": "noop", "skill": skill_name}
    return handler


# ══════════════════════════════════════════════════════════════════════════════
# 技能条目
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SkillEntry:
    """技能条目：跟踪单个技能的生命周期。"""
    name: str
    state: SkillState = SkillState.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_transition_at: float = field(default_factory=time.time)
    retirement_approvers: Set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "created_at": self.created_at,
            "last_transition_at": self.last_transition_at,
            "retirement_approvers": list(self.retirement_approvers),
            "metadata": self.metadata,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SkillLifecycleManager — 核心状态机
# ══════════════════════════════════════════════════════════════════════════════

class SkillLifecycleManager:
    """
    ISN技能生命周期管理器。

    管理技能状态转换，支持事件驱动架构、守卫函数、审计日志。

    用法：
        manager = SkillLifecycleManager()
        manager.register_frameworks()  # 注册框架处理器

        # 创建技能
        event = SkillCreated("my_skill", author="zcs")
        manager.handle_event(event)

        # 检测休眠
        event = DormancyDetected("my_skill", dormant_days=30)
        manager.handle_event(event)

        # 双人审批退休
        event = RetirementApproved("my_skill", approver="agent_a")
        manager.handle_event(event)
        event = RetirementApproved("my_skill", approver="agent_b")
        manager.handle_event(event)
    """

    # 状态转换规则
    TRANSITIONS: List[TransitionRule] = [
        # SkillCreated → ACTIVE
        TransitionRule(SkillState.ACTIVE, SkillState.ACTIVE, "SkillCreated"),
        # ACTIVE → DORMANT (DormancyDetected)
        TransitionRule(SkillState.ACTIVE, SkillState.DORMANT, "DormancyDetected"),
        # ACTIVE → DEPRECATED (DormancyDetected when dormant_days > 90)
        TransitionRule(SkillState.ACTIVE, SkillState.DEPRECATED, "DormancyDetected"),
        # DORMANT → ACTIVE (QualityThresholdHit)
        TransitionRule(SkillState.DORMANT, SkillState.ACTIVE, "QualityThresholdHit", quality_threshold_guard()),
        # DORMANT → DEPRECATED (DormancyDetected)
        TransitionRule(SkillState.DORMANT, SkillState.DEPRECATED, "DormancyDetected"),
        # DEPRECATED → RETIRED (RetirementApproved + dual approval)
        TransitionRule(SkillState.DEPRECATED, SkillState.RETIRED, "RetirementApproved", dual_approval_guard()),
        # DEPRECATED → DORMANT (reactivation)
        TransitionRule(SkillState.DEPRECATED, SkillState.DORMANT, "QualityThresholdHit", quality_threshold_guard()),
    ]

    def __init__(self, audit_path: Optional[Path] = None):
        self._skills: Dict[str, SkillEntry] = {}
        self._event_bus = EventBus()
        self._audit = AuditTrail(persist_path=audit_path)
        self._transition_table: Dict[Tuple[SkillState, str], TransitionRule] = {}

        # 构建转换表
        for rule in self.TRANSITIONS:
            key = (rule.source, rule.event_type)
            if key not in self._transition_table:
                self._transition_table[key] = rule

    @property
    def event_bus(self) -> EventBus:
        """事件总线（用于外部订阅）。"""
        return self._event_bus

    @property
    def audit(self) -> AuditTrail:
        """审计日志（用于查询）。"""
        return self._audit

    def register_frameworks(self) -> None:
        """注册各框架事件处理器。"""
        self._event_bus.subscribe("SkillCreated", create_isa_handler())
        self._event_bus.subscribe("SkillCreated", create_isn_handler())
        self._event_bus.subscribe("QualityThresholdHit", create_isa_handler())
        self._event_bus.subscribe("QualityThresholdHit", create_isn_handler())
        self._event_bus.subscribe("QualityThresholdHit", create_iko_handler())
        self._event_bus.subscribe("DormancyDetected", create_isa_handler())
        self._event_bus.subscribe("DormancyDetected", create_isn_handler())
        self._event_bus.subscribe("DormancyDetected", create_iko_handler())
        self._event_bus.subscribe("RetirementApproved", create_isa_handler())
        self._event_bus.subscribe("RetirementApproved", create_ios_handler())
        self._event_bus.subscribe("RetirementApproved", create_isn_handler())
        self._event_bus.subscribe("RetirementApproved", create_iko_handler())
        logger.info("已注册所有框架事件处理器 (ISA, IO-S, ISN, IKO)")

    def handle_event(self, event: SkillEvent) -> dict:
        """
        处理技能事件，执行状态转换。

        Args:
            event: 技能事件

        Returns:
            转换结果字典

        Raises:
            ValueError: 如果事件无效或状态转换不允许
        """
        skill_name = event.skill_name
        event_type = event.event_type

        # 获取或创建技能条目
        if skill_name not in self._skills:
            if event_type == "SkillCreated":
                self._skills[skill_name] = SkillEntry(
                    name=skill_name,
                    state=SkillState.ACTIVE,
                    metadata=event.metadata,
                )
                # 记录审计
                self._audit.record(
                    skill_name=skill_name,
                    event_type=event_type,
                    source_state=SkillState.ACTIVE,
                    target_state=SkillState.ACTIVE,
                    metadata=event.metadata,
                )
                # 发布事件
                self._event_bus.publish(event, transition=(SkillState.ACTIVE, SkillState.ACTIVE))
                logger.info(f"✅ 技能创建: {skill_name}")
                return {"success": True, "skill": skill_name, "state": "active", "action": "created"}
            else:
                raise ValueError(f"技能 '{skill_name}' 不存在，无法处理 {event_type}")

        entry = self._skills[skill_name]
        source_state = entry.state

        # 查找转换规则
        key = (source_state, event_type)
        rule = self._transition_table.get(key)

        if rule is None:
            raise ValueError(
                f"不允许转换: {source_state.value} + {event_type} "
                f"(技能: {skill_name})"
            )

        # 特殊逻辑：DormancyDetected在active状态可能转到deprecated
        if source_state == SkillState.ACTIVE and event_type == "DormancyDetected":
            dormant_days = event.metadata.get("dormant_days", 0)
            if dormant_days > 90:
                target_state = SkillState.DEPRECATED
            else:
                target_state = SkillState.DORMANT
        else:
            target_state = rule.target

        # 检查守卫
        if rule.guard is not None:
            if not rule.guard(event, {"skill": entry}):
                # 退休审批未达到双人要求
                if event_type == "RetirementApproved":
                    approver = event.metadata.get("approver", "")
                    count = len(entry.retirement_approvers)
                    logger.info(
                        f"⏳ 退休审批中: {skill_name} ({count}/2 审批人)"
                    )
                    self._audit.record(
                        skill_name=skill_name,
                        event_type=event_type,
                        source_state=source_state,
                        target_state=source_state,  # 未转换
                        metadata={
                            **event.metadata,
                            "approval_count": count,
                            "required_count": 2,
                        },
                        success=False,
                        error=f"审批不足: {count}/2",
                    )
                    return {
                        "success": False,
                        "skill": skill_name,
                        "state": source_state.value,
                        "action": "approval_pending",
                        "approval_count": count,
                        "required_count": 2,
                    }
                else:
                    raise ValueError(
                        f"守卫未通过: {rule.guard.__name__} "
                        f"(技能: {skill_name})"
                    )

        # 执行状态转换
        entry.state = target_state
        entry.last_transition_at = time.time()

        # 记录审计
        self._audit.record(
            skill_name=skill_name,
            event_type=event_type,
            source_state=source_state,
            target_state=target_state,
            metadata=event.metadata,
        )

        # 发布事件到各框架
        self._event_bus.publish(event, transition=(source_state, target_state))

        logger.info(
            f"🔄 状态转换: {skill_name} {source_state.value} → {target_state.value} "
            f"(事件: {event_type})"
        )

        return {
            "success": True,
            "skill": skill_name,
            "state": target_state.value,
            "action": "transitioned",
            "from": source_state.value,
            "to": target_state.value,
        }

    def get_skill_state(self, skill_name: str) -> Optional[SkillState]:
        """获取技能当前状态。"""
        entry = self._skills.get(skill_name)
        return entry.state if entry else None

    def get_skill_entry(self, skill_name: str) -> Optional[SkillEntry]:
        """获取技能条目。"""
        return self._skills.get(skill_name)

    def list_skills(self, state: Optional[SkillState] = None) -> List[SkillEntry]:
        """列出所有技能。"""
        skills = list(self._skills.values())
        if state:
            skills = [s for s in skills if s.state == state]
        return skills

    def get_statistics(self) -> dict:
        """获取状态统计。"""
        stats = {s.value: 0 for s in SkillState}
        for entry in self._skills.values():
            stats[entry.state.value] += 1
        return {
            "total": len(self._skills),
            "by_state": stats,
        }
