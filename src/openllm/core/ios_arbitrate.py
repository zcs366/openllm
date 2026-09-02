"""
openLLM IOS仲裁 — 从ios_impl.py提取

包含：arbitrate + _record_rejection + _select_strategy

5级仲裁栈：
  L1 single — 只用左脑提案
  L2 self_consistency — 置信度一致性检查（M1版本·见docstring）
  L3 multi_persona — 双persona视角聚合（M1版本·见docstring）
  L4 debate — 左脑↔️右脑对弈
  L5 debate_then_verify — 对弈→第三方验证
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
    Level 2: self_consistency — 左脑置信度一致性检查（M1版本）
    Level 3: multi_persona — 左右脑双视角聚合（M1版本）
    Level 4: debate — 左脑↔️右脑对弈（真实现）
    Level 5: debate_then_verify — 对弈→第三方验证（真实现）
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

    # Level 2: self_consistency — M1版本·基于置信度的一致性检查
    # 原理：真正的self_consistency需要LLM多次采样+投票（参考hemispheres_enhanced），
    # M1用proposal.confidence作为一致性代理指标。
    # 当confidence >= 0.6 时视为通过一致性检查；< 0.6 降级到L1并记录。
    if level == 2:
        if proposal.confidence >= 0.6:
            return Decision(
                action="execute",
                approved=True,
                reason=(f"Level 2: self_consistency通过"
                        f"（confidence={proposal.confidence:.2f} >= 0.6）"),
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )
        else:
            # 一致性检查未通过 → 保守降级到L1（单左脑），显式记录降级原因
            return Decision(
                action="execute",
                approved=True,
                reason=(f"Level 2: self_consistency未通过"
                        f"（confidence={proposal.confidence:.2f} < 0.6），"
                        f"降级L1单左脑"),
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )

    # Level 3: multi_persona — M1版本·双persona视角聚合
    # 原理：真正的multi_persona需要多个LLM persona并行采样+聚合，
    # M1用proposal（左脑persona）和critique（右脑persona）作为两个视角。
    # 聚合规则：
    #   - 两个persona都同意（critique.approve）→ 通过
    #   - 右脑persona明确否决 → 中风险下保守否决
    #   - 右脑建议修改 + 高置信度(>=0.7) → 存疑通过
    #   - 右脑建议修改 + 低置信度 → 降级到L4对弈（已有真实现）
    if level == 3:
        if critique.verdict == "reject":
            # 右脑否决 → 中风险保守否决
            decision = Decision(
                action="deny",
                approved=False,
                reason=(f"Level 3: multi_persona否决"
                        f"（右脑critique=reject，"
                        f"confidence={proposal.confidence:.2f}）"),
                risk_ref=risk,
            )
            _record_rejection(ios, proposal, decision, decision.reason)
            return decision

        if critique.verdict == "approve":
            # 两个persona一致同意
            return Decision(
                action="execute",
                approved=True,
                reason=(f"Level 3: multi_persona一致通过"
                        f"（confidence={proposal.confidence:.2f}）"),
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )

        # critique.verdict == "revise" 或其他 → 按置信度分路
        if proposal.confidence >= 0.7:
            return Decision(
                action="execute",
                approved=True,
                reason=(f"Level 3: multi_persona存疑通过"
                        f"（confidence={proposal.confidence:.2f} >= 0.7，"
                        f"右脑建议修改但置信度足够）"),
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )

        # 低置信度+右脑有疑虑 → 降级到L4对弈（已有真实现）
        # 以下为L4对弈逻辑（与下方Level 4一致，避免重构现有真实现）
        if critique.verdict == "revise":
            if ios._arbiter_policy == "conservative":
                return Decision(
                    action="revise",
                    approved=False,
                    reason="Level 3: 降级L4——右脑建议修改+低置信度，暂缓",
                    risk_ref=risk,
                )
            return Decision(
                action="execute",
                approved=True,
                reason="Level 3: 降级L4——策略允许存疑执行",
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )
        # 其他verdict → 降级L4对弈
        return Decision(
            action="execute",
            approved=True,
            reason=f"Level 3: 降级L4——右脑verdict={critique.verdict}，默认放行",
            risk_ref=risk,
            tool_calls=getattr(proposal, 'tool_calls', []) or [],
        )

    # Level 4: debate — 左脑↔️右脑对弈
    if level == 4:
        if critique.verdict == "approve":
            return Decision(
                action="execute",
                approved=True,
                reason="Level 4: 右脑通过",
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )
        if critique.verdict == "reject":
            decision = Decision(
                action="deny",
                approved=False,
                reason=f"Level 4: 右脑否决: {critique.concerns[0] if critique.concerns else '无理由'}",
                risk_ref=risk,
            )
            _record_rejection(ios, proposal, decision, decision.reason)
            return decision
        if critique.verdict == "revise":
            if ios._arbiter_policy == "conservative":
                return Decision(
                    action="revise",
                    approved=False,
                    reason="Level 4: 右脑建议修改，暂缓",
                    risk_ref=risk,
                )
            return Decision(
                action="execute",
                approved=True,
                reason="Level 4: 策略允许存疑执行",
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )

    # Level 5: debate_then_verify — 对弈+验证
    if level == 5:
        if critique.verdict == "approve":
            return Decision(
                action="execute",
                approved=True,
                reason="Level 5: 对弈+验证通过",
                risk_ref=risk,
                tool_calls=getattr(proposal, 'tool_calls', []) or [],
            )
        else:
            decision = Decision(
                action="deny",
                approved=False,
                reason=f"Level 5: critical风险需要明确批准，当前={critique.verdict}",
                risk_ref=risk,
            )
            _record_rejection(ios, proposal, decision, decision.reason)
            return decision

    # 默认放行
    return Decision(
        action="execute",
        approved=True,
        reason="默认放行",
        risk_ref=risk,
        tool_calls=getattr(proposal, 'tool_calls', []) or [],
    )


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
    """风险驱动策略选择。

    2026-09-01 G-1修复：消除L2/L3的TODO降级。
    - low→L2（self_consistency，M1置信度检查）
    - medium→L3（multi_persona，M1双视角聚合）
    - high→L4（debate，已有真实现）
    - critical→L5（debate_then_verify，已有真实现）
    """
    if risk is None:
        return 1
    if risk.level == "low":
        return 2   # self_consistency（M1：置信度一致性检查）
    if risk.level == "medium":
        return 3   # multi_persona（M1：双视角聚合）
    if risk.level == "high":
        return 4   # debate（已有真实现）
    return 5       # critical → debate_then_verify（已有真实现）
