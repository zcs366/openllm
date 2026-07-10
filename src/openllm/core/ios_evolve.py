"""
openLLM IOS进化引擎 — 从ios_impl.py提取

包含：evolve + _propose_harness_change + _extract_common
基于 Self-Harness (arXiv 2606.09498)
"""
import json
import time
from pathlib import Path
from typing import Optional

from .models import Context, Prediction, RiskAssessment, Proposal, \
                     Critique, Decision, ActionResult, CausalDelta
from .degradation_trace import trace_degradation


def evolve(ios, proposal: Proposal, critique: Critique,
           result: Optional[ActionResult] = None, delta: Optional[CausalDelta] = None):
    """
    Phase 9: 因果进化引擎 — Self-Harness Harness Proposal + Validation
    
    1. 记录（保留原有逻辑）
    2. 机制聚类：按mechanism字段聚合失败
    3. Harness修改提案：当高频机制≥阈值时，生成五体配置修改建议
    4. 策略自适应：根据失败率调整策略
    """
    # ① 记录
    ios.evolution_log.append({
        "timestamp": time.time(),
        "proposal": proposal.content[:100],
        "verdict": critique.verdict,
        "result_success": result.success if result else None,
        "delta_match": delta.prediction_match if delta else None,
    })
    
    # ② 机制聚类
    if result and not result.success and len(ios.causal_memory) >= 3:
        recent = ios.causal_memory[-20:]
        mechanism_counts: dict[str, int] = {}
        for m in recent:
            mech = m.get("mechanism", "unknown")
            if mech != "success":
                mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1
        
        # 高频机制（≥3次或占比≥30%）
        total_failures = sum(mechanism_counts.values())
        high_freq = {
            mech: count for mech, count in mechanism_counts.items()
            if count >= 3 or (total_failures > 0 and count / total_failures >= 0.3)
        }
        
        if high_freq:
            # ③ Harness修改提案
            prop_path = Path.home() / ".openllm" / "output" / "ios" / "harness_proposals.jsonl"
            existing_mechs = set()
            if prop_path.exists():
                try:
                    with open(prop_path) as f:
                        for line in f:
                            try:
                                p = json.loads(line)
                                if p.get("status") == "pending":
                                    existing_mechs.add(p.get("mechanism", ""))
                            except:
                                pass
                except:
                    pass
            
            top_mech = max(high_freq, key=lambda k: high_freq[k])
            if top_mech not in existing_mechs:
                proposal_text = _propose_harness_change(top_mech, high_freq[top_mech])
                if proposal_text:
                    ios.evolution_log.append({
                        "type": "harness_proposal",
                        "mechanism": top_mech,
                        "support": high_freq[top_mech],
                        "proposal": proposal_text,
                        "timestamp": time.time(),
                    })
                    print(f"  IO-S Self-Harness提案: [{top_mech}] {proposal_text[:80]}")
                    
                    # 持久化提案
                    prop_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(prop_path, "a") as f:
                        f.write(json.dumps({
                            "mechanism": top_mech,
                            "support": high_freq[top_mech],
                            "proposal": proposal_text,
                            "timestamp": time.time(),
                            "status": "pending",
                        }, ensure_ascii=False) + "\n")
                    
                    # 回流到causal_memory
                    ios.causal_memory.append({
                        "action": f"[Self-Harness提案] {top_mech}",
                        "prediction": "高频机制触发",
                        "actual": proposal_text[:100],
                        "lesson": f"机制={top_mech}, 支持={high_freq[top_mech]}次",
                        "timestamp": time.time(),
                    })


def _propose_harness_change(mechanism: str, support: int) -> str:
    """生成Harness修改提案"""
    proposals = {
        "tool_loop": "增加工具调用次数限制，单session最多10次同类工具",
        "missing_artifact": "执行后检查输出文件是否存在",
        "wrong_format": "添加输出格式验证器",
        "dependency_missing": "执行前检查依赖是否已安装",
        "timeout": "增加超时时间到60秒",
        "logic_error": "添加断言检查",
        "permission": "降级到只读模式",
        "state_corruption": "执行前备份状态",
        "exploration_loop": "限制搜索深度为3层",
        "premature_success": "要求二次验证",
    }
    return proposals.get(mechanism, f"针对{mechanism}机制的改进提案")


def _extract_common(texts: list[str]) -> list[str]:
    """提取共同关键词"""
    if not texts:
        return []
    
    words = [set(t.lower().split()) for t in texts]
    common = words[0]
    for w in words[1:]:
        common &= w
    
    return list(common)[:5]
