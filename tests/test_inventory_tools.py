"""test_inventory_tools — 库存武器上膛钉子（pi工具池第三环）"""
import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openllm.tools.inventory_tools import (
    INVENTORY_TOOL_ARG,
    INVENTORY_TOOL_SPECS,
    register_inventory_tools,
    tool_pk_summary,
    tool_rl_step,
    tool_se_confidence,
    tool_se_predict,
)
from openllm.tools.presets import create_core_tools


def _names(registry):
    return {t["name"] for t in registry.list_tools()}


def test_eight_specs_present():
    assert len(INVENTORY_TOOL_SPECS) == 8
    names = {n for n, _, _ in INVENTORY_TOOL_SPECS}
    assert names == {"pk_search", "pk_summary", "rl_step", "rl_status",
                     "ee_hypothesis", "ee_summary", "se_predict", "se_confidence"}
    # 每件都有主参数名映射（防参数扭曲——engine._TOOL_ARG_MAP同步依据）
    assert set(INVENTORY_TOOL_ARG) == names


def test_registered_into_pruned_pool():
    """上膛=进剪枝池：默认被剪（不出鞘），tool_load可装载。"""
    r = create_core_tools()
    inv = {n for n, _, _ in INVENTORY_TOOL_SPECS}
    assert inv & _names(r) == set(), "库存武器不应常驻注册表"
    assert inv <= set(getattr(r, "_pruned_names", [])), "库存武器必须在待命清单里"
    # 全量池（含inventory）里它们在册
    r2 = create_core_tools(extra=sorted(inv))
    assert inv <= _names(r2)


def test_load_then_research_loop_roundtrip():
    """装载rl_step→假设→观察→结论→卸载，状态在会话内连续。"""
    r = create_core_tools()
    loaded = r.execute("tool_load", name="rl_step")
    assert "已装载 rl_step" in loaded.output
    h = json.loads(r.execute("rl_step", action="hypothesize",
                             claim="上下文压缩可逆", prediction="损失<2%").output)
    assert h["claim"] == "上下文压缩可逆" and h["status"] == "unverified"
    r.execute("rl_step", action="observe", observation="损失1.2%")
    st = json.loads(r.execute("rl_step", action="status").output)
    assert st["state"]["observations"] >= 1
    c = json.loads(r.execute("rl_step", action="conclude",
                             finding="初步证实", confidence="medium").output)
    assert c["finding"] == "初步证实"
    # 卸载后不可调
    r.execute("tool_unload", name="rl_step")
    gone = r.execute("rl_step", action="status")
    assert not gone.success and "未知工具" in gone.error


def test_pk_summary_json():
    out = json.loads(tool_pk_summary())
    assert "total_papers" in out and "total_concepts" in out


def test_se_predict_confidence_roundtrip():
    tool_se_predict(card_id="t-card", prediction="预测甲", confidence=0.65)
    c = json.loads(tool_se_confidence(card_id="t-card"))
    assert c["card_id"] == "t-card" and "confidence" in c


def test_rl_step_validates_action():
    r = tool_rl_step(action="no_such", )
    assert "未知action" in r
    r2 = tool_rl_step(action="hypothesize")  # 缺claim/prediction
    assert "需要 claim" in r2


def test_gate_mapping_covers_inventory():
    from openllm.security.gate import SecurityFoundation
    action_map = getattr(SecurityFoundation, "ACTION_MAP", None)
    if action_map is None:
        from openllm.security.gate import PermissionGate
        action_map = PermissionGate.ACTION_MAP
    missing = [n for n, _, _ in INVENTORY_TOOL_SPECS if n not in action_map]
    assert not missing, f"库存工具缺安全闸映射: {missing}"


def test_tool_arg_map_covers_inventory():
    """engine._TOOL_ARG_MAP 必须认识这8件（否则位置参数被扭曲成query/path）。"""
    from openllm.core.engine import OpenLLMEngine
    for n in INVENTORY_TOOL_ARG:
        assert OpenLLMEngine._TOOL_ARG_MAP.get(n) == INVENTORY_TOOL_ARG[n], \
            f"{n} 主参数名与engine映射不一致"
