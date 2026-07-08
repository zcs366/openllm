"""
治理规则模型 — P0-b-1

治理基质的基本单元。
十族治理机制的统一表示。

来源: arXiv:2607.01087 Table II + Agent Harness工程方法论 v1.0
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class RuleType(Enum):
    """治理规则类型（论文: architecture vs control）"""
    ARCHITECTURE = "architecture"   # 架构响应: 修改系统边界，消除失败类
    CONTROL = "control"             # 控制响应: 添加检测机制，捕获失败实例
    CONSTRAINT = "constraint"       # 约束: 限制Agent行动空间


class RuleFamily(Enum):
    """十族治理机制（论文 Table II）"""
    GOVERNANCE_DOC = "governance_doc"           # 1. 文档治理
    CONTEXT_DISPATCH = "context_dispatch"       # 2. 上下文+调度
    AGENT_OBSERVABILITY = "agent_observability"  # 3. Agent可观察性
    RESOURCE_MEDIATOR = "resource_mediator"     # 4. 资源中介
    INCORPORATION_GATE = "incorporation_gate"   # 5. 合入门控
    CANONICAL_SEAM = "canonical_seam"           # 6. 规范接缝
    VALIDATION = "validation"                   # 7. 验证+合规
    STATIC_DYNAMIC_ANALYSIS = "static_dynamic"  # 8. 静态+动态分析
    PROVENANCE = "provenance"                   # 9. 溯源
    REPAIR_VOCABULARY = "repair_vocabulary"     # 10. 修复词汇


class RuleStatus(Enum):
    """规则状态"""
    PENDING = "pending"         # 待验证
    ACTIVE = "active"           # 已激活
    SUPERSEDED = "superseded"   # 被更优规则取代
    ROLLED_BACK = "rolled_back" # 验证失败，已回滚
    CONFLICTED = "conflicted"   # 与已有规则冲突
    DEGRADED = "degraded"       # 冲突降级（阿瑞斯·保留拦截类）


@dataclass
class VerificationRecord:
    """单次验证记录"""
    timestamp: float
    passed: bool
    regression_result: str = ""
    side_effects: str = ""
    details: str = ""


@dataclass
class GovernanceRule:
    """治理规则: 治理基质的基本单元。
    
    论文依据:
    - "The Subject's governed environment encodes probabilistic controls 
       (e.g., agent harness) and deterministic controls (e.g., type system)."
    - "Governance mechanisms were layered. When a failure recurred across 
       one mechanism, the Subject's response was to add a complement."
    """
    rule_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    rule_type: RuleType = RuleType.CONTROL
    family: RuleFamily = RuleFamily.VALIDATION
    
    # 作用目标
    target: str = ""            # 文件/函数/组件
    condition: str = ""         # 触发条件（可机器验证的表达式）
    action: str = ""            # 执行动作
    
    # 来源
    source_failure: str = ""    # 来源失败signature key
    source_mechanism: str = ""  # 来源失败机制
    source_count: int = 0       # 触发治理转换的失败次数
    
    # 元数据
    description: str = ""       # 人类可读描述
    confidence: float = 0.5     # 置信度
    status: RuleStatus = RuleStatus.PENDING
    created_at: float = field(default_factory=time.time)
    
    # 冲突与验证（鲁班要求）
    conflicts: list[str] = field(default_factory=list)
    verification_history: list[VerificationRecord] = field(default_factory=list)
    
    # 衰减预留（克洛诺斯·P0预留，P1实现逻辑）
    last_triggered_at: float = 0.0    # 最后触发时间
    trigger_count: int = 0             # 累计触发次数
    
    def is_machine_verifiable(self) -> bool:
        """是否可机器验证（Agent可读定律）"""
        return bool(self.condition)
    
    def is_active(self) -> bool:
        """是否激活状态"""
        return self.status == RuleStatus.ACTIVE
    
    def add_verification(self, passed: bool, details: str = ""):
        """添加验证记录"""
        self.verification_history.append(VerificationRecord(
            timestamp=time.time(),
            passed=passed,
            details=details,
        ))
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        d = asdict(self)
        d["rule_type"] = self.rule_type.value
        d["family"] = self.family.value
        d["status"] = self.status.value
        d["verification_history"] = [
            asdict(v) for v in self.verification_history
        ]
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "GovernanceRule":
        """从字典反序列化"""
        d = dict(d)  # copy
        d["rule_type"] = RuleType(d["rule_type"])
        d["family"] = RuleFamily(d["family"])
        d["status"] = RuleStatus(d["status"])
        d["verification_history"] = [
            VerificationRecord(**v) for v in d.get("verification_history", [])
        ]
        return cls(**d)


# ── 规则存储 ──────────────────────────────────────────

class GovernanceRuleStore:
    """治理规则持久化存储。
    
    存储路径: ~/.openllm/output/ios/governance_rules.jsonl
    格式: 每行一个JSON（JSONL原子追加）
    """
    
    def __init__(self):
        self._path = Path.home() / ".openllm" / "output" / "ios" / "governance_rules.jsonl"
        self._rules: dict[str, GovernanceRule] = {}
        self._load()
    
    def _load(self):
        """从磁盘加载规则"""
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        rule = GovernanceRule.from_dict(d)
                        self._rules[rule.rule_id] = rule
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except IOError:
            pass
    
    def save(self, rule: GovernanceRule):
        """保存/更新规则（JSONL追加）"""
        self._rules[rule.rule_id] = rule
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(rule.to_dict(), ensure_ascii=False) + "\n")
    
    def get(self, rule_id: str) -> Optional[GovernanceRule]:
        """按ID查询"""
        return self._rules.get(rule_id)
    
    def query(
        self,
        family: Optional[RuleFamily] = None,
        rule_type: Optional[RuleType] = None,
        status: Optional[RuleStatus] = None,
        target: Optional[str] = None,
    ) -> list[GovernanceRule]:
        """多维查询"""
        results = list(self._rules.values())
        
        if family:
            results = [r for r in results if r.family == family]
        if rule_type:
            results = [r for r in results if r.rule_type == rule_type]
        if status:
            results = [r for r in results if r.status == status]
        if target:
            results = [r for r in results if target in r.target]
        
        return results
    
    def get_active(self) -> list[GovernanceRule]:
        """获取所有激活规则"""
        return self.query(status=RuleStatus.ACTIVE)
    
    def detect_conflicts(self, new_rule: GovernanceRule) -> list[GovernanceRule]:
        """冲突检测（鲁班要求）。
        
        冲突条件:
        1. 同target + 同condition + 不同action
        2. 同target + 同rule_type + 不同family（可能冲突）
        """
        conflicts = []
        
        for existing in self._rules.values():
            if existing.rule_id == new_rule.rule_id:
                continue
            if existing.status in (RuleStatus.ROLLED_BACK, RuleStatus.SUPERSEDED):
                continue
            
            # 条件1: 同target + 同condition + 不同action
            if (existing.target == new_rule.target and
                existing.condition == new_rule.condition and
                existing.action != new_rule.action):
                conflicts.append(existing)
            
            # 条件2: 同target + 同type + 同family（可能重复）
            elif (existing.target == new_rule.target and
                  existing.rule_type == new_rule.rule_type and
                  existing.family == new_rule.family and
                  existing.rule_id != new_rule.rule_id):
                conflicts.append(existing)
        
        return conflicts
    
    def get_stats(self) -> dict:
        """统计"""
        status_counts = {}
        family_counts = {}
        for r in self._rules.values():
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1
            family_counts[r.family.value] = family_counts.get(r.family.value, 0) + 1
        
        return {
            "total_rules": len(self._rules),
            "by_status": status_counts,
            "by_family": family_counts,
        }
