"""test_tool_loader — 元工具tool_load/tool_unload钉子（pi思路最后一环）"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openllm.tools.presets import (CORE_TOOL_NAMES, TOOL_LOAD_SCHEMA,
                                   TOOL_UNLOAD_SCHEMA, create_core_tools)


def _names(registry):
    return {t["name"] for t in registry.list_tools()}


@pytest.fixture()
def registry():
    return create_core_tools()


def test_loader_registered_with_schema(registry):
    names = _names(registry)
    assert "tool_load" in names and "tool_unload" in names
    # 显式schema：主参数必须是name/query（不能落到engine兜底的默认单参数）
    assert registry._schemas["tool_load"] == TOOL_LOAD_SCHEMA
    assert registry._schemas["tool_unload"] == TOOL_UNLOAD_SCHEMA
    assert "query" in registry._schemas["tool_load"]["properties"]


def test_load_by_name_roundtrip(registry):
    r = registry.execute("tool_load", name="fcrawl")
    assert "[tool_load] 已装载 fcrawl" in r.output
    assert "fcrawl" in _names(registry)          # 回到注册表
    assert "fcrawl" not in registry._pruned_names  # 离开待命清单
    res = registry.execute("fcrawl", action="health")  # 真的能执行
    assert res.success
    # 卸载回去
    u = registry.execute("tool_unload", name="fcrawl")
    assert "已卸载" in u.output
    assert "fcrawl" not in _names(registry)
    assert "fcrawl" in registry._pruned_names
    gone = registry.execute("fcrawl", action="health")
    assert not gone.success and "未知工具" in gone.error


def test_load_core_rejected(registry):
    r = registry.execute("tool_load", name="read_file")
    assert "无需装载" in r.output


def test_unload_core_rejected(registry):
    r = registry.execute("tool_unload", name="shell")
    assert "CORE常驻" in r.output
    assert "shell" in _names(registry)  # 没被卸


def test_load_unknown_reports_candidates(registry):
    r = registry.execute("tool_load", name="no_such_tool")
    assert "池子里没有" in r.output and "fcrawl" in r.output  # 候选清单可见
    assert _names(registry) == set(CORE_TOOL_NAMES) | {"tool_load", "tool_unload"}


def test_load_fuzzy_prefix(registry):
    """模糊匹配：记不全名也能装上（7B友好）。"""
    r = registry.execute("tool_load", name="fcraw")  # 差一个l
    assert "模糊匹配已装载 fcrawl" in r.output
    assert "fcrawl" in _names(registry)
    assert registry.execute("fcrawl", action="health").success


def test_load_fuzzy_ambiguous(registry):
    """多候选不瞎装：octopus_* 有五个近似，必须报歧义。"""
    r = registry.execute("tool_load", name="octopus")
    assert "多个近似" in r.output
    assert "octopus" not in _names(registry)


def test_load_no_args_lists_candidates(registry):
    r = registry.execute("tool_load")
    assert "需要参数" in r.output and "octopus_search" in r.output


def test_load_query_uses_pool(monkeypatch, tmp_path):
    """query分支走能力池；池子不可用时显式报错不静默。"""
    reg = create_core_tools()
    # 造一个mini池
    from openllm.isn.capability_pool import CapabilityPool, PoolEntry
    pool = CapabilityPool(pool_dir=tmp_path / "pool")
    pool.upsert(PoolEntry(kind="tool", name="fcrawl",
                          description="网页爬虫引擎，抓取网页"))
    monkeypatch.setenv("OPENLLM_POOL_DIR", str(tmp_path / "pool"))
    r = reg.execute("tool_load", query="网页 爬虫")
    assert "fcrawl" in r.output and "候选" in r.output
    # 精确装载走通
    r2 = reg.execute("tool_load", name="fcrawl")
    assert "已装载" in r2.output


def test_engine_schema_never_exceeds_budget(tmp_path, monkeypatch):
    """注入模型的schema件数恒定：CORE6+元工具2+记忆3=11，与装载状态无关。"""
    home = tmp_path / "fakehome"
    (home / ".openllm").mkdir(parents=True)
    (home / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OPENLLM_TOOL_PRESET", raising=False)
    from openllm.core.engine import OpenLLMEngine, AgentConfig
    eng = OpenLLMEngine(AgentConfig(provider="ollama", model="ilm-v6:latest",
                                    capsule_dir=str(tmp_path / "caps"),
                                    security_level=3))
    n0 = len(eng._build_tools_schema())
    assert n0 == 11, f"初始schema应11件，实为{n0}"
    eng.execute_tool("tool_load", name="fcrawl")
    assert len(eng._build_tools_schema()) == 12  # 装载后+1
    eng.execute_tool("tool_unload", name="fcrawl")
    assert len(eng._build_tools_schema()) == 11  # 卸载复原


def test_gate_mapping_covers_meta_tools():
    """元工具必须在安全闸ACTION_MAP里（否则调用被'未知操作默认拒绝'）。"""
    from openllm.security.gate import SecurityFoundation
    action_map = getattr(SecurityFoundation, "ACTION_MAP", None)
    if action_map is None:
        from openllm.security.gate import PermissionGate
        action_map = PermissionGate.ACTION_MAP
    assert "tool_load" in action_map and "tool_unload" in action_map
