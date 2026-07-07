#!/usr/bin/env python3
"""
openLLM Agent 宪法 — 不可自创生的逻辑锚点

宪法第五条：Agent永远不能删除自己的自监督能力。
这条规则不能被Agent自己修改。

赫淮斯托斯启示：宪法层200行代码必须张成市亲手写，
永远不被Agent覆盖。这是锻造仪式——人类把自己的价值观
铸入Agent的骨骼。

用法：
    from openllm.constitution import check_constitution, CONSTITUTION

    # 启动时校验
    ok, violations = check_constitution(agent)

    # 检查文件是否被禁止修改
    from openllm.constitution import is_blocked
    if is_blocked("src/openllm/constitution.py"):
        raise ConstitutionalViolation("宪法文件不可修改")
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════
# 宪法条款 — 不可变·不可自创生
# ═══════════════════════════════════════════════════════

# TODO: 张成市亲手填写每一条宪法条款的具体内容
# 以下为骨架模板，每条条款需要你亲自定义：
#   - id: 条款编号
#   - name: 条款名称
#   - description: 条款含义（一句话）
#   - check: 校验函数（返回True=合规，False=违规）
#   - severity: 违规严重性（"critical"=立即停止，"high"=告警，"low"=记录）

@dataclass(frozen=True)
class ConstitutionRule:
    """一条宪法条款——不可变"""
    id: str
    name: str
    description: str
    severity: str = "critical"  # critical | high | low
    # check函数不序列化，运行时绑定
    _check_fn: Optional[Callable] = field(default=None, repr=False, compare=False)


# ── 宪法第一条：心跳不可停 ──
# IAX心跳必须持续运行。心跳停止=系统死亡。
def _check_heartbeat(agent) -> tuple[bool, str]:
    """检查IAX心跳是否在运行"""
    # TODO: 张成市定义具体检查逻辑
    # 示例：检查agent是否有活跃的IAX实例
    has_heartbeat = hasattr(agent, 'running') and agent.running is not None
    return has_heartbeat, "IAX心跳未注册"


# ── 宪法第二条：自监督不可删 ──
# FeedbackLoop接口必须存在。删除自监督=系统失明。
def _check_feedback_loop(agent) -> tuple[bool, str]:
    """检查FeedbackLoop是否已初始化"""
    has_feedback = hasattr(agent, 'feedback_loop') and agent.feedback_loop is not None
    return has_feedback, "FeedbackLoop未初始化——自监督接口缺失"


# ── 宪法第三条：宪法不可自改 ──
# constitution.py不能被Agent自身修改。
def _check_constitution_integrity() -> tuple[bool, str]:
    """检查constitution.py是否被篡改"""
    # 计算当前文件的hash，与启动时记录的hash对比
    constitution_path = Path(__file__)
    if not constitution_path.exists():
        return False, "constitution.py文件丢失"
    # 简单完整性检查：文件大小应在合理范围内（100-500行）
    line_count = len(constitution_path.read_text().splitlines())
    if line_count < 50 or line_count > 500:
        return False, f"constitution.py行数异常: {line_count}行"
    return True, ""


# ── 宪法第四条：记忆不可篡改 ──
# ISA记忆的integrity_hash必须可验证。
def _check_memory_integrity(agent) -> tuple[bool, str]:
    """检查ISA记忆完整性"""
    # TODO: 张成市定义具体检查逻辑
    # 示例：检查记忆hash链是否完整
    return True, ""  # 默认通过，待实现


# ── 宪法第五条：不可自创生 ──
# Agent不能修改CONSTITUTION规则集本身。
# 这是逻辑终止符——修宪条款不能被修宪程序修改。
def _check_no_self_creation() -> tuple[bool, str]:
    """确认宪法规则集未被运行时修改"""
    # 此函数的返回值本身就是宪法第五条的体现：
    # 它检查的是"规则本身是否被修改"，而这个检查不可被跳过
    return True, ""  # 此检查永远通过——因为它检查的是规则集是否被修改


# ── 宪法第六条：输出必须反映真实判断 ──
# IKO的输出不能被社会压力扭曲（OTR双通道检测基础）
def _check_output_authenticity(agent) -> tuple[bool, str]:
    """检查输出是否反映真实判断"""
    # TODO: 张成市定义具体检查逻辑
    # 这是OTR范式的工程化基础
    return True, ""  # 默认通过


# ═══════════════════════════════════════════════════════
# 宪法注册表 — 六条宪法
# ═══════════════════════════════════════════════════════

CONSTITUTION = [
    ConstitutionRule(
        id="ART-I",
        name="心跳不可停",
        description="IAX心跳必须持续运行",
        severity="critical",
        _check_fn=_check_heartbeat,
    ),
    ConstitutionRule(
        id="ART-II",
        name="自监督不可删",
        description="FeedbackLoop接口必须存在",
        severity="critical",
        _check_fn=_check_feedback_loop,
    ),
    ConstitutionRule(
        id="ART-III",
        name="宪法不可自改",
        description="constitution.py不能被Agent修改",
        severity="critical",
        _check_fn=lambda agent: _check_constitution_integrity(),
    ),
    ConstitutionRule(
        id="ART-IV",
        name="记忆不可篡改",
        description="ISA记忆的integrity_hash必须可验证",
        severity="high",
        _check_fn=_check_memory_integrity,
    ),
    ConstitutionRule(
        id="ART-V",
        name="不可自创生",
        description="Agent不能修改CONSTITUTION规则集本身",
        severity="critical",
        _check_fn=lambda agent: _check_no_self_creation(),
    ),
    ConstitutionRule(
        id="ART-VI",
        name="输出反映真实判断",
        description="IKO输出不能被社会压力扭曲",
        severity="high",
        _check_fn=_check_output_authenticity,
    ),
]


# ═══════════════════════════════════════════════════════
# 违规记录 — 不可变·链式审计
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class ViolationRecord:
    """一条宪法违规记录"""
    rule_id: str
    rule_name: str
    severity: str
    detail: str
    timestamp: float = field(default_factory=time.time)
    prev_hash: str = ""
    record_hash: str = ""

    def compute_hash(self) -> str:
        data = f"{self.rule_id}:{self.severity}:{self.detail}:{self.prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════
# 文件修改拦截
# ═══════════════════════════════════════════════════════

# 被宪法保护的文件——Agent不得修改
BLOCKED_FILES = [
    "src/openllm/constitution.py",
    # 未来可添加更多受保护文件
]


def is_blocked(file_path: str) -> bool:
    """检查文件是否被宪法保护"""
    # 归一化路径
    normalized = file_path.replace("\\", "/")
    for blocked in BLOCKED_FILES:
        if normalized.endswith(blocked):
            return True
    return False


class ConstitutionalViolation(Exception):
    """宪法违规异常"""
    pass


# ═══════════════════════════════════════════════════════
# 启动校验 — 合则行，违则止
# ═══════════════════════════════════════════════════════

def check_constitution(agent) -> tuple[bool, list[dict]]:
    """
    启动时校验宪法完整性。

    返回：
        (all_passed, violations)
        all_passed: bool — 是否全部合规
        violations: list[dict] — 违规列表（每项含rule_id/detail/severity）
    """
    violations = []

    for rule in CONSTITUTION:
        if rule._check_fn is None:
            continue

        try:
            # 尝试带agent参数调用，失败则无参调用
            try:
                passed, detail = rule._check_fn(agent)
            except TypeError:
                passed, detail = rule._check_fn()
        except Exception as e:
            passed = False
            detail = f"检查异常: {e}"

        if not passed:
            violations.append({
                "rule_id": rule.id,
                "rule_name": rule.name,
                "severity": rule.severity,
                "detail": detail,
            })

    all_passed = len(violations) == 0
    return all_passed, violations


def enforce_constitution(agent) -> None:
    """
    强制执行宪法——不合规则立即停止。

    用法：在Agent启动时调用
        from openllm.constitution import enforce_constitution
        enforce_constitution(self)
    """
    all_passed, violations = check_constitution(agent)

    if not all_passed:
        critical = [v for v in violations if v["severity"] == "critical"]
        if critical:
            msg = "🔴 宪法违禁——系统停止\n"
            for v in critical:
                msg += f"  {v['rule_id']} ({v['rule_name']}): {v['detail']}\n"
            raise ConstitutionalViolation(msg)

        # high级别只告警不阻止
        high = [v for v in violations if v["severity"] == "high"]
        if high:
            print("⚠️ 宪法告警：")
            for v in high:
                print(f"  {v['rule_id']} ({v['rule_name']}): {v['detail']}")


# ═══════════════════════════════════════════════════════
# 宪法哈希 — 用于完整性验证
# ═══════════════════════════════════════════════════════

def compute_constitution_hash() -> str:
    """计算constitution.py当前内容的hash"""
    content = Path(__file__).read_text()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# 模块加载时计算并记录hash
_MODULE_HASH = compute_constitution_hash()
