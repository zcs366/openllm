"""
openLLM IOS仲裁 — 从ios_impl.py提取

包含：arbitrate + _record_rejection + _select_strategy
"""
from typing import Optional

from .models import (Context, Prediction, RiskAssessment, Proposal,
                     Critique, Decision, ActionResult, CausalDelta)
from .degradation_trace import trace_degradation


def arbitrate(ios, proposal: Proposal, critique: Critique, 
              risk: Optional[RiskAssessment] = None) -> Decision:
    """
    Phase 6: 仲裁 — 5级策略栈 + 拒绝权记录
    
    Level 1: single — 只用左脑提案，跳过右脑
    Level 2: self_consistency — 左脑多次采样→投票
    Level 3: multi_persona — 左脑不同人格→聚合
    Level 4: debate — 左脑↔️右脑对弈（当前实现）
    Level 5: debate_then_verify — 对弈→第三方验证
    """
    # 风险拦截
    if risk and risk.is_blocked():
        decision = Decision(
            action="deny",
            approved=False,
            reason=f"安全拦截: {risk.reason}",
            risk_ref=risk,
        )
        _record_rejection(ios, proposal, decision, risk.reason)
        return decision
    
    # 风险驱动策略选择
    level = _select_strategy(risk)
    
    # Level 1: single — 快速·直接用左脑
    if level == 1:
        return Decision(
            action="execute",
            approved=True,
            reason=f"Level 1: 单左脑提案通过（风险={risk.level if risk else 'low'}）",
            risk_ref=risk,
            tool_calls=getattr(proposal, 'tool_calls', []) or [],
        )
    
    # Level 2: self_consistency — TODO: M1实现真正的self_consistency
    if level == 2:
        return Decision(
            action="execute",
            approved=True,
            reason=f"Level 2: self_consistency TODO·M1实现（降级为Level 1）",
            risk_ref=risk,
            tool_calls=getattr(proposal, 'tool_calls', []) or [],
        )
    
    # Level 3: multi_persona — TODO: M1实现真正的multi_persona
    if level == 3:
        return Decision(
            action="execute",
            approved=True,
            reason=f"Level 3: multi_persona TODO·M1实现（降级为Level 1）",
            risk_ref=risk,
            tool_calls=getattr(proposal, 'tool_calls', []) or [],
        )
    
    # Level 4: debate — 左脑↔️右脑对弈
    if level == 4:
        if critique.verdict == "approve":
            return Decision(action="execute", approved=True, reason="Level 4: 右脑通过", risk_ref=risk,
                            tool_calls=getattr(proposal, 'tool_calls', []) or [])
        if critique.verdict == "reject":
            decision = Decision(action="deny", approved=False, 
                          reason=f"Level 4: 右脑否决: {critique.concerns[0] if critique.concerns else '无理由'}",
                          risk_ref=risk)
            _record_rejection(ios, proposal, decision, decision.reason)
            return decision
        if critique.verdict == "revise":
            if ios._arbiter_policy == "conservative":
                return Decision(action="revise", approved=False, reason="Level 4: 右脑建议修改，暂缓", risk_ref=risk)
            return Decision(action="execute", approved=True, reason="Level 4: 策略允许存疑执行", risk_ref=risk,
                            tool_calls=getattr(proposal, 'tool_calls', []) or [])
    
    # Level 5: debate_then_verify — 对弈+验证
    if level == 5:
        if critique.verdict == "approve":
            return Decision(action="execute", approved=True, reason="Level 5: 对弈+验证通过", risk_ref=risk,
                            tool_calls=getattr(proposal, 'tool_calls', []) or [])
        else:
            decision = Decision(action="deny", approved=False,
                          reason=f"Level 5: critical风险需要明确批准，当前={critique.verdict}",
                          risk_ref=risk)
            _record_rejection(ios, proposal, decision, decision.reason)
            return decision
    
    # 默认放行
    return Decision(action="execute", approved=True, reason="默认放行", risk_ref=risk,
                    tool_calls=getattr(proposal, 'tool_calls', []) or [])


def _record_rejection(ios, proposal: Proposal, decision: Decision, reason: str):
    """记录拒绝——通过RejectionEngine创建不可变记录"""
    if not ios._rejection_engine:
        return
    try:
        # 映射到RejectionReason
        if "安全" in reason or "拦截" in reason:
            rej_reason = ios._RejectionReason.CONSTITUTIONAL
        elif "否决" in reason:
            rej_reason = ios._RejectionReason.OUT_OF_SCOPE
        else:
            rej_reason = ios._RejectionReason.UNCERTAIN_SAFETY
        
        ios._rejection_engine.reject(
            instruction=proposal.content[:200],
            reason=rej_reason,
            reasoning=reason,
            context={"proposal_confidence": proposal.confidence, "action": decision.action},
        )
    except Exception as _e:
        trace_degradation("IOS", "arbitrate", _e)


def _select_strategy(risk: Optional[RiskAssessment]) -> int:
    """风险驱动策略选择"""
    if risk is None:
        return 1
    if risk.level == "low":
        return 1  # 快速
    if risk.level == "medium":
        return 3  # 多样
    if risk.level == "high":
        return 4  # 对弈
    return 5  # critical→对弈+验证
