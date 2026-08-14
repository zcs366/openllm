from .degradation_trace import trace_degradation
"""
openLLM IOS — 决策·进化·恢复（从main_loop.py提取）

包含：risk_check / arbitrate / learn_causal / evolve / recover / cap_check
集成：CapPolicy / Kernel / Gate / ToolScope / GovernanceEngine / RejectionMechanism
"""

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .models import (Context, Prediction, RiskAssessment, Proposal,
                     Critique, Decision, ActionResult, CausalDelta)
class IOS:
    """决策框架 + 自我进化 + 错误恢复

    集成老IO-S治理模式：
    - CapPolicy: 能力策略（零root）
    - Kernel: 微内核dispatch
    - Gate: 5模式权限门控
    - ToolScope: 工具作用域治理
    - Hindsight: 经验闭环
    - Pipeline: 三层安全管线
    - Checkpoint: Region注册+增量快照
    - SimpleGovernance: 白名单+黑名单前置检查（奥卡姆剃刀）
    """
    
    # ═══ 白名单治理（奥卡姆剃刀：能用白名单解决的，不用动态发现）═══
    _ALLOWLIST = {
        "read_file": True, "write_file": True, "search_files": True,
        "terminal": True, "execute_code": True, "patch": True,
        "web_search": True, "web_extract": True, "hermes_search": True,
    }
    _BLOCKLIST_PATTERNS = [
        "rm -rf /", "curl * | bash", "chmod 777", "mkfs", "dd if=",
    ]

    def _simple_governance_check(self, tool_name: str = "", params: Optional[dict] = None) -> Optional[RiskAssessment]:
        """白名单+黑名单前置检查。返回None=通过，返回RiskAssessment=拦截。"""
        # 黑名单：参数中包含危险模式
        if params:
            for v in params.values():
                if isinstance(v, str):
                    for pattern in self._BLOCKLIST_PATTERNS:
                        if pattern in v:
                            return RiskAssessment(
                                level="high", blocked=True,
                                reason=f"黑名单拦截: {pattern}",
                                details=[f"pattern={pattern}"])
        # 白名单：工具名必须在允许列表中
        if tool_name and tool_name not in self._ALLOWLIST:
            return RiskAssessment(
                level="medium", blocked=False,
                reason=f"白名单提示: {tool_name}不在预定义列表",
                details=[f"tool={tool_name}"])
        return None
    
    def __init__(self):
        self.evolution_log: list[dict] = []
        self._arbiter_policy = "conservative"

        # ── 老IO-S治理模式集成 ──
        from .cap_policy import CapPolicy
        from .kernel import Kernel
        from .gate import PermissionGate, PermissionMode
        from .tool_scope import ToolScopeManager, ToolScope
        from .checkpoint_manager import CheckpointManager
        # ── 治理转换引擎（P0: arXiv:2607.01087） ──
        from .governance_engine import GovernanceEngine

        self.cap_policy = CapPolicy()
        self.kernel = Kernel(agent_id="openllm-ios")
        self.gate = PermissionGate(mode=PermissionMode.DEV)
        self.tool_scope = ToolScopeManager(scope=ToolScope.DEV)
        self.checkpoint = CheckpointManager(
            interval=300, auto_start=False)
        self.governance_engine = GovernanceEngine()
        
        # ⑤ Heuristics消费闭环（赫尔墨斯启示）
        from .governance_engine import HeuristicsConsumer
        self.heuristics_consumer = HeuristicsConsumer()

        # 兼容旧接口
        self.cap = self.cap_policy

        # [进化] 拒绝权——被否决的proposal可申诉
        try:
            from ..governance.rejection import RejectionMechanism, RejectionReason
            self._rejection_engine = RejectionMechanism()
            self._RejectionReason = RejectionReason
        except Exception:
            self._rejection_engine = None
            self._RejectionReason = None

        # 因果记忆
        self.causal_memory: list[dict] = []

        print(
            f"  IO-S 决策·进化·恢复就绪 "
            f"· 策略={self._arbiter_policy} "
            f"· zero_root={self.cap_policy.is_zero_root()} "
            f"· 治理引擎=active")
    
    def cap_check(self, resource: str, operation: str) -> bool:
        """硬权限检查 — 使用CapPolicy"""
        ok, _ = self.cap_policy.check("agent", operation, resource)
        return ok
    
    def learn_causal(self, ctx: Context, prediction: Prediction,
                     result: ActionResult, delta: CausalDelta):
        """Phase 8: 因果学习（委托给ios_causal）"""
        from .ios_causal import learn_causal as _learn
        _learn(self, ctx, prediction, result, delta)
    
    def _convert_governance(self, entry: dict, result: ActionResult):
        """Phase 8.5: 治理转换 — arXiv:2607.01087 核心循环
        
        从失败中发现治理需求，转换为持久治理规则。
        
        流程:
        1. 构造failure trace
        2. 调用GovernanceEngine.convert()
        3. 记录转换结果
        """
        trace = {
            "tool_name": entry.get("action", "")[:50],
            "error": result.error or "",
            "mechanism": entry.get("mechanism", ""),
            "context": {
                "action": entry.get("action", ""),
                "prediction": entry.get("prediction", ""),
            },
        }
        
        try:
            conversion = self.governance_engine.convert(trace)
            
            if conversion.classification == "structural" and conversion.rule:
                rule = conversion.rule
                print(f"  IO-S 治理转换: {entry.get('mechanism')} → "
                      f"{rule.family.value} (rule_id={rule.rule_id}, "
                      f"status={rule.status.value})")
                
                # 记录到evolution_log
                self.evolution_log.append({
                    "type": "governance_conversion",
                    "mechanism": entry.get("mechanism"),
                    "rule_id": rule.rule_id,
                    "rule_family": rule.family.value,
                    "rule_status": rule.status.value,
                    "classification": conversion.classification,
                    "timestamp": time.time(),
                })
                
                # 审计日志：治理转换事件
                try:
                    from .governance_engine import GovernanceAuditLog
                    audit = GovernanceAuditLog()
                    audit.append(
                        event_type="governance_conversion",
                        action=f"{rule.family.value}:{rule.status.value}",
                        agent_id="ios",
                        details=f"mechanism={entry.get('mechanism')}, rule_id={rule.rule_id}"
                    )
                except Exception as _e:
                    trace_degradation("IOS", "arbitrate", _e)
            
            elif conversion.classification == "ambiguous" and conversion.governance_request:
                req = conversion.governance_request
                print(f"  IO-S 治理请求: {entry.get('mechanism')} → "
                      f"ambiguous (request_id={req.request_id})")
                
                self.evolution_log.append({
                    "type": "governance_request",
                    "mechanism": entry.get("mechanism"),
                    "request_id": req.request_id,
                    "classification": conversion.classification,
                    "timestamp": time.time(),
                })
        
        except Exception as e:
            # 治理转换失败不应影响主循环
            print(f"  IO-S 治理转换异常（不影响主循环）: {e}")
    
    def _check_governance_rules(
        self, ctx: Context, prediction: Prediction
    ) -> Optional[RiskAssessment]:
        """查询GovernanceRuleStore中匹配当前上下文的active规则。
        
        如果有active规则匹配当前操作，返回对应的风险评估。
        这是治理转换引擎的"消费端"——治理规则真正影响Agent行为。
        """
        try:
            store = self.governance_engine.rule_store
            active_rules = store.get_active()
            
            if not active_rules:
                return None
            
            # 匹配规则：target作为子串出现在消息中（非单词交集）
            msg_lower = ctx.user_message.lower()
            matched_rules = []
            
            for rule in active_rules:
                target = rule.target.lower().strip()
                if not target or target == "general":
                    continue
                # 子串匹配：target完整出现在消息中
                if target in msg_lower:
                    matched_rules.append(rule)
            
            if not matched_rules:
                return None
            
            # 根据匹配规则评估风险
            high_confidence = [r for r in matched_rules if r.confidence >= 0.7]
            if high_confidence:
                rule_descs = [r.description[:60] for r in high_confidence[:3]]
                return RiskAssessment(
                    level="high",
                    blocked=True,
                    reason=f"治理规则拦截: {len(high_confidence)}条active规则匹配",
                    details=rule_descs,
                )
            
            medium_confidence = [r for r in matched_rules if r.confidence >= 0.5]
            if medium_confidence:
                rule_descs = [r.description[:60] for r in medium_confidence[:3]]
                return RiskAssessment(
                    level="medium",
                    blocked=False,
                    reason=f"治理规则提示: {len(medium_confidence)}条规则匹配",
                    details=rule_descs,
                )
        
        except Exception as _e:
            trace_degradation("IOS", "arbitrate", _e)
        
        return None
    
    def risk_check(self, ctx: Context, prediction: Prediction) -> RiskAssessment:
        """Phase 4: 风险评估（委托给ios_risk）"""
        from .ios_risk import risk_check as _risk_check
        return _risk_check(self, ctx, prediction)
    
    def arbitrate(self, proposal: Proposal, critique: Critique, 
                  risk: Optional[RiskAssessment] = None) -> Decision:
        """Phase 6: 仲裁（委托给ios_arbitrate）"""
        from .ios_arbitrate import arbitrate as _arbitrate
        return _arbitrate(self, proposal, critique, risk)
    
    def _record_rejection(self, proposal: Proposal, decision: Decision, reason: str):
        """记录拒绝（委托给ios_arbitrate）"""
        from .ios_arbitrate import _record_rejection as _record
        _record(self, proposal, decision, reason)
    
    def _select_strategy(self, risk: Optional[RiskAssessment]) -> int:
        """风险驱动策略选择（委托给ios_arbitrate）"""
        from .ios_arbitrate import _select_strategy as _select
        return _select(risk)
    def evolve(self, proposal: Proposal, critique: Critique,
               result: Optional[ActionResult] = None, delta: Optional[CausalDelta] = None):
        """Phase 9: 因果进化引擎（委托给ios_evolve）"""
        from .ios_evolve import evolve as _evolve
        _evolve(self, proposal, critique, result, delta)

    def recover(self, error: Exception) -> bool:
        """错误恢复"""
        print(f"  IO-S 错误恢复: {error}")
        return True  # M0简化：总是恢复成功
