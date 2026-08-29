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


def _build_causal_lesson(action_desc: str, prediction: Prediction,
                         result: ActionResult, delta: CausalDelta) -> str:
    """Fix P1-20260829-02: 因果结构lesson生成。
    
    成功: "{action} 因采取了{prediction中的关键词}策略而成功"
    失败: "{action} 因{mechanism/verifier_cause}而失败，教训：{actual前50字}"
    零信息lesson（同义反复）: 不进store（由调用方过滤）。
    """
    pred_summary = getattr(prediction, 'summary', '') or ''
    actual_text = getattr(result, 'output', '') or getattr(result, 'error', '') or ''
    mechanism = _classify_mechanism(result, delta)
    verifier = _classify_verifier_cause(result, delta)

    if result.success:
        # 成功：提取prediction中的关键词作为策略
        # DR-20260829-02R: strategy_kw必须是有实质内容的短语——
        # 过滤"[数学] 预测类型=text 置信=0.00"这类零信息预测头和纯噪声
        strategy_kw = ""
        for seg in pred_summary.split("|"):
            seg = seg.strip()
            if not seg or seg.startswith("[数学]") or "预测类型=" in seg:
                continue
            # 取LLM预测段（[LLM]后的实质内容）
            if seg.startswith("[LLM]"):
                seg = seg[4:].strip().lstrip("]：: ")
            if len(seg) >= 6:
                # 过滤markdown代码块/换行残留，保持lesson单行可读
                seg = seg.replace("```", "").replace("\n", " ").strip()
                if len(seg) >= 6:
                    strategy_kw = seg
                    break
        if not strategy_kw:
            strategy_kw = action_desc
        lesson = f"{action_desc} 因采取了{strategy_kw[:40]}策略而成功"
    else:
        # 失败：明确失败原因和教训
        cause = mechanism if mechanism != "unknown" else verifier
        lesson_tail = actual_text[:50] if actual_text else "需进一步分析"
        lesson = f"{action_desc} 因{cause}而失败，教训：{lesson_tail}"
    return lesson


def learn_causal(ios, ctx: Context, prediction: Prediction,
                 result: ActionResult, delta: CausalDelta):
    """Phase 8: 因果学习 — Self-Harness Weakness Mining + 治理转换"""
    # Fix P1-20260829-02: action记录agent行为，不是用户输入
    tool_name = getattr(result, 'tool_name', '') or ''
    if result.output and tool_name:
        action_desc = f"执行[{tool_name}] -> {'成功' if result.success else '失败'}"
    elif result.output:
        action_desc = f"执行[chat] -> {'成功' if result.success else '失败'}"
    else:
        action_desc = f"响应用户: {ctx.user_message[:100]}"

    # Fix P1-20260829-02: lesson升级为因果结构
    lesson = _build_causal_lesson(action_desc, prediction, result, delta)

    entry = {
        "action": action_desc,
        "prediction": prediction.summary[:100],
        "actual": result.output[:100] if result.output else "",
        "lesson": lesson,
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

    # 通电Phase 1c: AutoCausalWriter桥接 — 让因果数据流入CausalMemoryStore
    # MemoryBus通过CausalProvider检索因果记忆，但CausalMemoryStore只有6条数据。
    # 原因：learn_causal写到causal_memory.jsonl，没写到CausalMemoryStore。
    # 这一行桥接两个数据流。
    try:
        from ..memory.auto_causal_writer import AutoCausalWriter
        writer = AutoCausalWriter()
        writer.record(
            action=entry.get("action", ""),
            prediction=entry.get("prediction", ""),
            actual=entry.get("actual", ""),
            success=result.success,
            context=entry.get("mechanism", ""),
            lesson=entry.get("lesson", ""),  # DR-20260829-02R: 传递因果结构lesson，避免writer二次生成
        )
    except Exception:
        pass  # AutoCausalWriter不可用时静默降级

    # E2 缺口④：直通CausalMemoryStore写入（2026-08-23）
    # 补断：确保因果数据同时写入结构化store，不再只有jsonl
    # Fix P1-20260829-02: delta_magnitude=0不进store，防止流水账污染
    # 教训因果关键词检查仅对delta>0生效（确保失败记录有实质内容）
    _delta_mag = 0.0 if delta.prediction_match else min(1.0, len(delta.delta_summary) / 100.0)
    if _delta_mag > 0:
        try:
            from ..memory.causal_memory import CausalMemory, get_causal_store, TrustLevel, DEFAULT_STORE_DIR
            _store = get_causal_store(DEFAULT_STORE_DIR)
            _store.store(
                action_signature=entry.get("action", "")[:100],
                context_features=[entry.get("mechanism", "")],
                prediction=entry.get("prediction", ""),
                prediction_confidence=0.5,
                actual_result=entry.get("actual", ""),
                actual_success=result.success,
                delta=delta.delta_summary,
                delta_magnitude=_delta_mag,
                lesson=entry.get("lesson", ""),
                source="ios_learn_causal",
                trust_level=TrustLevel.INTERNAL,
                importance=0.5,
            )
        except Exception:
            pass  # 不阻塞主循环


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
