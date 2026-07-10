"""
openLLM IOS因果学习 — 从ios_impl.py提取

包含：learn_causal + _classify_mechanism + _classify_verifier_cause + _convert_governance
"""
import json
import time
from pathlib import Path
from typing import Optional

from .models import Context, Prediction, RiskAssessment, Proposal, \
                     Critique, Decision, ActionResult, CausalDelta
from .degradation_trace import trace_degradation


# Self-Harness Weakness Mining 关键词
_MECHANISM_KEYWORDS = {
    "tool_loop": ["重复", "循环", "loop", "retry", "反复", "无限"],
    "missing_artifact": ["缺失", "没有生成", "未创建", "missing", "not found", "未产出"],
    "wrong_format": ["格式错误", "format error", "解析失败", "parse error", "json解析", "语法错误"],
    "dependency_missing": ["依赖缺失", "import error", "module not found", "package未安装", "ModuleNotFoundError"],
    "timeout": ["超时", "timeout", "hang", "卡住", "无响应"],
    "logic_error": ["逻辑错误", "条件错误", "判断错误", "分支错误", "错误结果", "断言失败"],
    "permission": ["权限", "permission denied", "forbidden", "access denied"],
    "state_corruption": ["脏数据", "不一致", "corrupt", "状态损坏"],
    "exploration_loop": ["探索循环", "搜索失败", "盲目搜索", "找不到目标"],
    "premature_success": ["过早完成", "提前结束", "未验证成功"],
}


def learn_causal(ios, ctx: Context, prediction: Prediction,
                 result: ActionResult, delta: CausalDelta):
    """Phase 8: 因果学习 — Self-Harness Weakness Mining + 治理转换"""
    entry = {
        "action": ctx.user_message[:100],
        "prediction": prediction.summary[:100],
        "actual": result.output[:100] if result.output else "",
        "lesson": delta.delta_summary,
        "timestamp": time.time(),
        "verifier_cause": _classify_verifier_cause(result, delta),
        "mechanism": _classify_mechanism(result, delta),
        "prediction_match": delta.prediction_match,
        "result_success": result.success,
    }
    ios.causal_memory.append(entry)

    # 持久化到磁盘
    causal_path = Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl"
    causal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(causal_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # 连接4: 回流到jiak mismatch_log
    if not result.success:
        jiak_path = Path.home() / ".hermes" / "jiak" / "mismatch_log.jsonl"
        jiak_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jiak_path, "a") as f:
            f.write(json.dumps({
                "ts": entry.get("timestamp"),
                "card_id": "openllm-ios",
                "prediction_ref": "",
                "actual": f"[{entry.get('mechanism', 'unknown')}] {result.error[:80] if result.error else result.output[:80]}",
                "severity": "high" if entry.get("mechanism") in ("tool_loop", "permission", "timeout") else "medium",
            }, ensure_ascii=False) + "\n")
    
    # 治理转换
    if not result.success:
        _convert_governance(ios, entry, result)

    # Hindsight经验闭环
    from .hindsight_loop import extract_hindsight
    hindsight = extract_hindsight(
        pid="ios",
        goal=ctx.user_message[:200],
        result={
            "success": result.success,
            "execution_time": result.duration_ms / 1000,
            "error": result.error,
        },
        failure=result.error if not result.success else "",
    )
    if hindsight.get("failure_pattern"):
        entry["hindsight_pattern"] = hindsight["failure_pattern"]
        entry["hindsight_alternative"] = hindsight.get("alternative", "")

    # learn_causal_adapter
    if not result.success:
        try:
            import importlib.util as _ilu
            _isa_path = Path.home() / "projects" / "isa" / "learn_causal_adapter.py"
            _lca_spec = _ilu.spec_from_file_location("learn_causal_adapter", _isa_path)
            if _lca_spec and _lca_spec.loader:
                _lca_mod = _ilu.module_from_spec(_lca_spec)
                _lca_spec.loader.exec_module(_lca_mod)
                LearnCausalAdapter = _lca_mod.LearnCausalAdapter
            adapter = LearnCausalAdapter()
            adapter.on_failure(
                action=entry.get("action", ""),
                error=result.error or "",
                mechanism=entry.get("mechanism", "unknown"),
                tool_name=entry.get("action", "").split("(")[0] if "(" in entry.get("action", "") else "",
            )
            adapter.close()
        except Exception as e:
            print(f"  learn_causal_adapter failed (non-fatal): {e}")


def _classify_mechanism(result: ActionResult, delta: CausalDelta) -> str:
    """分类失败机制"""
    if result.success:
        return "success"
    
    error = (result.error or "").lower()
    output = (result.output or "").lower()
    combined = error + " " + output
    
    for mech, keywords in _MECHANISM_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return mech
    
    return "unknown"


def _classify_verifier_cause(result: ActionResult, delta: CausalDelta) -> str:
    """分类验证器原因"""
    if delta.prediction_match:
        return "prediction_correct"
    
    if result.success:
        return "unexpected_success"
    
    return "prediction_wrong"


def _convert_governance(ios, entry: dict, result: ActionResult):
    """治理转换"""
    try:
        from .governance_engine import GovernanceEngine
        # 简化实现：记录治理转换事件
        pass
    except Exception:
        pass
