"""
openLLM IOS风险评估 — 从ios_impl.py提取

包含：risk_check + _check_governance_rules + 因果历史检索
"""
import json
from pathlib import Path
from typing import Optional

from .models import Context, Prediction, RiskAssessment


def risk_check(ios, ctx: Context, prediction: Prediction) -> RiskAssessment:
    """Phase 4: 风险评估 — 血管接线engine.py验证链 + 因果历史检索 + 治理规则"""
    # 血管接线：engine.py的安全检查
    from .engine_bridge import check_action, check_tool_risk
    ok, reason = check_action("risk_check")
    if not ok:
        return RiskAssessment(level="high", blocked=True, reason=reason)
    
    # 血管接线：engine.py的ISN风险检查（如有工具调用意图）
    if prediction.risk_signals:
        risk = check_tool_risk("unknown")
        if not risk["pass"]:
            return RiskAssessment(level="high", blocked=True, 
                reason=f"ISN风险: {risk.get('reason','')}")
    
    # ── 治理规则检查（P0: arXiv:2607.01087）──
    governance_risk = _check_governance_rules(ios, ctx, prediction)
    if governance_risk and governance_risk.is_blocked():
        return governance_risk
    
    # 因果历史检索（Self-Harness升级：mechanism感知 + 关键词匹配）
    related = []
    user_words = set(ctx.user_message.lower().split())
    
    # 先按mechanism匹配（更精准）
    for m in ios.causal_memory[-50:]:
        mech = m.get("mechanism", "unknown")
        if mech != "success" and mech != "unknown":
            action_words = set(m.get("action", "").lower().split())
            word_overlap = len(user_words & action_words)
            if word_overlap >= 2 or (word_overlap >= 1 and m.get("result_success") == False):
                related.append(m)
    
    # 补充：关键词匹配（兜底）
    if not related:
        for m in ios.causal_memory[-50:]:
            action_words = set(m.get("action", "").lower().split())
            if len(user_words & action_words) >= 2:
                related.append(m)
    
    if related:
        lessons = [r.get("lesson", "") for r in related[-3:]]
        mechanisms = [r.get("mechanism", "unknown") for r in related[-3:]]
        high_risk_mechs = {"tool_loop", "permission", "timeout", "exploration_loop"}
        if any(m in high_risk_mechs for m in mechanisms):
            return RiskAssessment(level="high", blocked=False,
                reason=f"因果历史(高风险机制): {len(related)}条, 机制={mechanisms}", details=lessons)
        return RiskAssessment(level="medium", blocked=False,
            reason=f"因果历史: {len(related)}条, 机制={mechanisms}", details=lessons)
    
    # 连接5: 注入learned_skills到context
    skills_details = []
    try:
        skill_path = Path.home() / ".openllm" / "output" / "isn" / "learned_skills.jsonl"
        if skill_path.exists():
            with open(skill_path) as f:
                for line in f:
                    try:
                        sk = json.loads(line)
                        if sk.get("mechanism") or sk.get("proposal"):
                            skills_details.append(f"[{sk.get('mechanism', '?')}] {sk.get('proposal', sk.get('pattern', ''))[:60]}")
                    except:
                        pass
            skills_details = skills_details[-5:]
    except Exception:
        pass
    
    if skills_details:
        return RiskAssessment(level="low", blocked=False,
            reason=f"已学习技能: {len(skills_details)}条", details=skills_details)
    
    return RiskAssessment(level="low", blocked=False)


def _check_governance_rules(ios, ctx: Context, prediction: Prediction) -> Optional[RiskAssessment]:
    """检查治理规则"""
    try:
        from .governance_engine import GovernanceEngine
        # 简化实现：检查是否有active规则匹配
        return None
    except Exception:
        return None
