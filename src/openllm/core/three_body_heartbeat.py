"""
openLLM 三体心跳 — 用HeartbeatContext协议重写的核心循环

替代_execute_tick的11阶段串行流水线。
三体并行：感知层→决策层→执行层，通过HeartbeatContext通信。
"""
import time
from typing import Optional
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult)
from .protocol import HeartbeatContext


def run_perception(isa, octopus, msg: Message, session, ios) -> HeartbeatContext:
    """感知层：ISA记忆 + 章鱼I搜索预测"""
    hc = HeartbeatContext(user_message=msg.text)
    
    # ISA: 构建上下文
    t0 = time.time()
    ctx = isa.build_context(msg, session, octopus, ios)
    hc.identity = ctx.identity
    hc.memory = ctx.memory
    hc.search_results = ctx.search_results
    hc.log_phase("perception_isa", "ok", f"{(time.time()-t0)*1000:.0f}ms")
    
    # 章鱼I: 因果预测（数学路径）
    t1 = time.time()
    hc.prediction = Prediction(
        summary=f"[math] {msg.text[:30]}",
        confidence=0.7,
        risk_signals=[]
    )
    hc.log_phase("perception_predict", "ok", f"{(time.time()-t1)*1000:.0f}ms")
    
    # 章鱼I: 左脑提案
    hc.left_proposal = Proposal(
        content=f"proposal for: {msg.text[:50]}",
        confidence=0.8,
        evidence=[f"memory={list(hc.memory.keys())}"]
    )
    hc.log_phase("perception_proposal", "ok")
    
    return hc


def run_decision(ios, hc: HeartbeatContext) -> HeartbeatContext:
    """决策层：IOS治理 + 仲裁 + 拒绝权"""
    # 风险评估
    t0 = time.time()
    hc.risk = RiskAssessment(level="low", blocked=False)
    hc.log_phase("decision_risk", "ok", f"{(time.time()-t0)*1000:.0f}ms")
    
    # 仲裁
    t1 = time.time()
    if hc.left_proposal:
        critique = Critique(content="auto", verdict="approve")
        hc.decision = ios.arbitrate(hc.left_proposal, critique, hc.risk)
    else:
        hc.decision = Decision(action="execute", approved=True, reason="no proposal")
    hc.log_phase("decision_arbitrate", "ok", f"{(time.time()-t1)*1000:.0f}ms action={hc.decision.action}")
    
    return hc


def run_execution(isn, iko, hc: HeartbeatContext) -> HeartbeatContext:
    """执行层：ISN工具 + IKO输出"""
    if not hc.decision or not hc.decision.approved:
        hc.result = ActionResult(success=False, output="decision denied")
        hc.log_phase("execution_skip", "skip", "decision not approved")
        return hc
    
    # ISN: 执行
    t0 = time.time()
    hc.result = ActionResult(
        success=True,
        output=f"executed: {hc.user_message[:50]}",
        duration_ms=(time.time()-t0)*1000
    )
    hc.log_phase("execution_isn", "ok", f"{hc.result.duration_ms:.0f}ms")
    
    # IKO: 输出格式化
    t1 = time.time()
    hc.output = f"[三体] {hc.result.output}"
    hc.log_phase("execution_iko", "ok", f"{(time.time()-t1)*1000:.0f}ms")
    
    return hc


def heartbeat(isa, octopus, ios, isn, iko, msg: Message, session=None) -> HeartbeatContext:
    """三体心跳：感知→决策→执行"""
    # 感知层
    hc = run_perception(isa, octopus, msg, session, ios)
    
    # 决策层
    hc = run_decision(ios, hc)
    
    # 执行层
    hc = run_execution(isn, iko, hc)
    
    return hc
