"""test_tool_presets — CORE九件钉子（pi思路按需装配）"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openllm.tools.presets import CORE_TOOL_NAMES, create_core_tools
from openllm.tools.executor import create_default_tools


def _names(registry):
    return {t["name"] for t in registry.list_tools()}


def test_core_six_present():
    r = create_core_tools()
    assert set(CORE_TOOL_NAMES) <= _names(r), f"CORE缺件: {set(CORE_TOOL_NAMES) - _names(r)}"


def test_pruned_actually_gone():
    r = create_core_tools()
    names = _names(r)
    # 全量基线=32件（原生24+库存8）：用with_loader=False关掉元工具挂载，纯看"注册了什么"
    full_registry = create_core_tools(with_loader=False)
    full = set(full_registry._pruned_names) | _names(full_registry)
    assert len(full) == 32, f"全量池应为32件(24原生+8库存)，实为{len(full)}"
    assert len(names) == 8, f"工厂净出应8件(CORE6+元工具2)，实为{len(names)}"
    pruned = set(getattr(r, "_pruned_names", []))
    # 18件原生工具 + 8件库存武器（2026-09-22上膛，进剪枝池待命）
    assert len(pruned) == 26 and "fcrawl" in pruned and "pk_search" in pruned
    # 元工具不算被剪的（它们是钥匙，CORE常驻级）
    assert getattr(r, "_pruned_names", None) == sorted(pruned)
    # 剪掉的调它必须报未知工具，不能静默
    res = r.execute("fcrawl", url="http://x")
    assert not res.success and "未知工具" in (res.error or "")


def test_schema_count_matches_registry():
    r = create_core_tools()
    assert len(r.list_tools()) == 8  # CORE6 + tool_load/tool_unload
    schemas = [n for n in r._schemas if r._schemas.get(n)]
    # schema覆盖不作硬钉（默认schema生成在engine侧），只钉注册表与schema一致
    assert set(schemas) <= _names(r)


def test_extra_expands_core():
    r = create_core_tools(extra=["fcrawl", "ocr"])
    names = _names(r)
    assert "fcrawl" in names and "ocr" in names and len(names) == 10


def test_extra_unknown_warns_not_crashes():
    r = create_core_tools(extra=["no_such_tool"])
    assert len(r.list_tools()) == 8  # 忽略未知，CORE+元工具不变


def test_engine_default_is_core_nine(tmp_path, monkeypatch):
    """引擎默认装配=CORE九件；env扩编生效。"""
    home = tmp_path / "fakehome"
    (home / ".openllm").mkdir(parents=True)
    (home / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OPENLLM_TOOL_PRESET", raising=False)
    from openllm.core.engine import OpenLLMEngine, AgentConfig
    cfg = AgentConfig(provider="ollama", model="ilm-v6:latest",
                      capsule_dir=str(tmp_path / "caps"), security_level=3)
    eng = OpenLLMEngine(cfg)
    names_eng = _names(eng.tools)
    expected = (set(CORE_TOOL_NAMES)
                | {"tool_load", "tool_unload"}
                | {"memory_write", "memory_read", "memory_search"})
    assert names_eng == expected, f"引擎默认应=6文件+2元+3记忆=11件，实为: {sorted(names_eng)}"

    monkeypatch.setenv("OPENLLM_TOOL_PRESET", "fcrawl")
    eng2 = OpenLLMEngine(AgentConfig(provider="ollama", model="ilm-v6:latest",
                                     capsule_dir=str(tmp_path / "caps2"),
                                     security_level=3))
    names2 = _names(eng2.tools)
    assert "fcrawl" in names2 and len(names2) == 12


def test_memory_tools_still_registered(tmp_path, monkeypatch):
    """DR-20260917钉子契约：记忆三件必须在注册表里（test_dogfood_fixes同款断言）。"""
    home = tmp_path / "fakehome"
    (home / ".openllm").mkdir(parents=True)
    (home / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    from openllm.core.engine import OpenLLMEngine, AgentConfig
    eng = OpenLLMEngine(AgentConfig(provider="ollama", model="ilm-v6:latest",
                                    capsule_dir=str(tmp_path / "caps3"),
                                    security_level=3))
    names = {t["name"] for t in eng.tools.list_tools()}
    assert {"memory_write", "memory_read", "memory_search"} <= names


def test_gate_mapping_covers_core():
    """CORE九件都在安全闸ACTION_MAP里（对齐test_dogfood_fixes::TestGateCoverage）。"""
    from openllm.tools import create_default_tools
    from openllm.security.gate import SecurityFoundation
    r = create_core_tools()
    action_map = getattr(SecurityFoundation, "ACTION_MAP", None)
    if action_map is None:
        from openllm.security.gate import PermissionGate
        action_map = PermissionGate.ACTION_MAP
    missing = [n for n in _names(r) if n not in action_map]
    assert not missing, f"CORE工具缺安全闸映射: {missing}"
