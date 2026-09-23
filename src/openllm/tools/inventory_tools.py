"""
Inventory Tools — 库存武器上膛（2026-09-22 军令：你决定）
==========================================================

背景：paper_knowledge/research_loop/experiment_engine/skill_evolution 四模块
（1028行）一直可导入却从未进注册表——造了武器没发到士兵手里。
本模块把它们包成8件工具，交给 presets 的剪枝池：默认不出鞘，
tool_load 按需装载，装载即用（函数原对象+闭包单例，行为与直调一致）。

四模块映射（每模块2件，状态用模块级惰性单例保持，会话内连续）：
  paper_knowledge  → pk_search(论文/概念检索)  pk_summary(知识库摘要)
  research_loop    → rl_step(研究循环推进)     rl_status(研究状态)
  experiment_engine→ ee_hypothesis(假设管理)   ee_summary(实验摘要)
  skill_evolution  → se_predict(预测登记)      se_confidence(置信查询)

注意：单例落在调用进程里；沙箱HOME下storage指向沙箱，不污染真库。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .executor import ToolRegistry

# ── 惰性单例（进程级，会话内状态连续）────────────────────────

_pk = None
_rl = None
_ee = None
_se = None


def _paper_knowledge():
    global _pk
    if _pk is None:
        from .paper_knowledge import PaperKnowledge
        _pk = PaperKnowledge(storage_dir=Path.home() / ".openllm" / "inventory" / "papers")
    return _pk


def _research_loop():
    global _rl
    if _rl is None:
        from .research_loop import ResearchLoop
        _rl = ResearchLoop()
    return _rl


def _experiment_engine():
    global _ee
    if _ee is None:
        from .experiment_engine import ExperimentEngine
        _ee = ExperimentEngine(storage_dir=Path.home() / ".openllm" / "inventory" / "experiments")
    return _ee


def _skill_evolution():
    global _se
    if _se is None:
        from .skill_evolution import SkillEvolution
        _se = SkillEvolution(auto_load=True)
    return _se


def _dump(obj: Any) -> str:
    """dict/list→json串；dataclass等对象→str。绝不静默吞。"""
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return json.dumps(obj, ensure_ascii=False, default=str)
    return str(obj)


# ── 工具函数 ─────────────────────────────────────────────

def tool_pk_search(query: str) -> str:
    """检索论文知识库（标题/摘要/概念）。返回JSON列表。"""
    papers = _paper_knowledge().search(query)
    return _dump([p.dict() if hasattr(p, "dict") else str(p) for p in papers])


def tool_pk_summary() -> str:
    """论文知识库摘要（论文数/概念数/主题分布）。"""
    return _dump(_paper_knowledge().summary())


def tool_rl_step(action: str, claim: str = "", prediction: str = "",
                 observation: str = "", finding: str = "",
                 name: str = "", method: str = "",
                 success: bool = True, confidence: str = "medium") -> str:
    """研究循环推进：action ∈ hypothesize/observe/design/run/conclude/iterate/status。"""
    rl = _research_loop()
    if action == "hypothesize":
        if not claim or not prediction:
            return "[rl_step] hypothesize 需要 claim 和 prediction"
        h = rl.hypothesize(claim, prediction)
        return _dump(h.dict() if hasattr(h, "dict") else {"claim": claim})
    if action == "observe":
        if not observation:
            return "[rl_step] observe 需要 observation"
        return _dump(rl.observe(observation))
    if action == "design":
        if not name or not method:
            return "[rl_step] design 需要 name 和 method"
        exp_id = rl.design_experiment(name, method)
        return _dump({"experiment_id": exp_id})
    if action == "run":
        return _dump(rl.run_experiment(success, conclusion=finding))
    if action == "conclude":
        if not finding:
            return "[rl_step] conclude 需要 finding"
        return _dump(rl.conclude(finding, confidence))
    if action == "iterate":
        return _dump(rl.iterate())
    if action == "status":
        return _dump(rl.status())
    return (f"[rl_step] 未知action '{action}'。"
            "可用: hypothesize/observe/design/run/conclude/iterate/status")


def tool_rl_status() -> str:
    """研究循环当前状态（阶段/迭代数/当前假设与实验）。"""
    return _dump(_research_loop().status())


def tool_ee_hypothesis(action: str, id: str = "", claim: str = "",
                       prediction: str = "", success: bool = True,
                       conclusion: str = "") -> str:
    """实验假设管理：action ∈ add/list/record/summary。"""
    ee = _experiment_engine()
    if action == "add":
        if not (id and claim and prediction):
            return "[ee_hypothesis] add 需要 id, claim, prediction"
        h = ee.add_hypothesis(id, claim, prediction)
        return _dump(h.dict() if hasattr(h, "dict") else {"id": id, "status": "created"})
    if action == "list":
        return _dump(ee.list_hypotheses())
    if action == "record":
        if not id:
            return "[ee_hypothesis] record 需要 id（实验id）"
        r = ee.record_result(id, success, conclusion=conclusion)
        return _dump(r.dict() if hasattr(r, "dict") else {"experiment_id": id})
    if action == "summary":
        return _dump(ee.summary())
    return f"[ee_hypothesis] 未知action '{action}'。可用: add/list/record/summary"


def tool_ee_summary() -> str:
    """实验引擎摘要（假设数/实验数/证实与反驳分布）。"""
    return _dump(_experiment_engine().summary())


def tool_se_predict(card_id: str, prediction: str, confidence: float = 0.7) -> str:
    """登记一次预测（技能进化引擎：预测→观察→校准）。"""
    return _dump(_skill_evolution().record_prediction(card_id, prediction, confidence))


def tool_se_confidence(card_id: str = "") -> str:
    """查询技能进化置信度：card_id=单卡查询；缺省=全局进化状态。"""
    se = _skill_evolution()
    if card_id:
        return _dump({"card_id": card_id, "confidence": se.get_confidence(card_id)})
    return _dump(se.get_evolution_status())


# ── 装配进注册表（由 presets 在剪枝时收编）────────────────

INVENTORY_TOOL_SPECS = [
    ("pk_search", tool_pk_search, "检索论文知识库(query)——标题/摘要/概念匹配"),
    ("pk_summary", tool_pk_summary, "论文知识库摘要()——论文数/概念数/主题分布"),
    ("rl_step", tool_rl_step, "研究循环推进(action,claim,prediction,...)——假设/观察/设计/运行/结论/迭代"),
    ("rl_status", tool_rl_status, "研究循环状态()——阶段/迭代数/当前假设"),
    ("ee_hypothesis", tool_ee_hypothesis, "实验假设管理(action,id,claim,prediction,...)——add/list/record/summary"),
    ("ee_summary", tool_ee_summary, "实验引擎摘要()——假设数/证实反驳分布"),
    ("se_predict", tool_se_predict, "登记预测(card_id,prediction,confidence)——技能进化校准链入口"),
    ("se_confidence", tool_se_confidence, "查询进化置信度(card_id)——单卡或全局状态"),
]

# 主参数名（engine._TOOL_ARG_MAP 同步用）
INVENTORY_TOOL_ARG = {
    "pk_search": "query",
    "pk_summary": "action",
    "rl_step": "action",
    "rl_status": "action",
    "ee_hypothesis": "action",
    "ee_summary": "action",
    "se_predict": "prediction",
    "se_confidence": "card_id",
}


def register_inventory_tools(registry: ToolRegistry) -> None:
    """把8件库存工具注册进registry（presets构建全量池时调用一次）。"""
    for name, func, desc in INVENTORY_TOOL_SPECS:
        registry.register(name, func, desc)
