#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修补#3 provenance接线验收测试（2026-09-16）
⑥-q纪律：真装配真打标——build_context真跑（不mock装配逻辑），验证：
  1. Context.provenance字段存在且非空
  2. 七要素每块都有trust标签
  3. trust值在五级枚举内
  4. search_results=web（最低信任，固化门禁触发前提）
  5. web/mixed块不被标为creator（打标不能全绿——全creator=打标失效）
"""
import sys
sys.path.insert(0, "/mnt/i/openllm/src")

from openllm.core.models import Message, Context
from openllm.core.isa_impl import ISA

TRUST_LEVELS = {"creator", "user", "tool_output", "web", "mixed"}

def test_context_has_provenance_field():
    ctx = Context(user_message="test")
    assert hasattr(ctx, "provenance"), "Context缺provenance字段"
    assert ctx.provenance == {} or isinstance(ctx.provenance, dict)
    print("✓ Context.provenance字段存在")

def test_build_context_stamps_all_blocks():
    isa = ISA(mode="silent")
    msg = Message(text="验收探针：provenance接线测试")
    ctx = isa.build_context(msg, session=None, octopus=None, ios=None)
    assert isinstance(ctx.provenance, dict) and len(ctx.provenance) >= 10, \
        f"provenance标签数不足: {len(ctx.provenance)}"
    # 七要素主块必须在场
    required = {"identity", "tools", "causal_hints", "d0_report", "risk_context",
                "search_results", "identity_block"}
    missing = required - set(ctx.provenance.keys())
    assert not missing, f"缺标签块: {missing}"
    print(f"✓ build_context打标{len(ctx.provenance)}块，七要素主块全覆盖")

def test_trust_values_valid():
    isa = ISA(mode="silent")
    ctx = isa.build_context(Message(text="trust枚举验证"), None, None, None)
    bad = {k: v for k, v in ctx.provenance.items() if v not in TRUST_LEVELS}
    assert not bad, f"非法trust值: {bad}"
    print(f"✓ 全部trust值∈五级枚举{sorted(TRUST_LEVELS)}")

def test_search_results_is_web():
    isa = ISA(mode="silent")
    ctx = isa.build_context(Message(text="web级验证"), None, None, None)
    assert ctx.provenance.get("search_results") == "web", \
        "search_results必须=web（固化门trust<τ置零的前提）"
    assert ctx.provenance.get("identity") == "creator"
    print("✓ search_results=web / identity=creator 分级正确")

def test_not_all_creator():
    isa = ISA(mode="silent")
    ctx = isa.build_context(Message(text="打标有效性验证"), None, None, None)
    levels_used = set(ctx.provenance.values())
    assert len(levels_used) >= 3, f"打标全绿嫌疑：只用了{levels_used}"
    assert "web" in levels_used and "creator" in levels_used
    print(f"✓ 打标有区分度（用了{len(levels_used)}级）——非全creator橡皮章")

if __name__ == "__main__":
    test_context_has_provenance_field()
    test_build_context_stamps_all_blocks()
    test_trust_values_valid()
    test_search_results_is_web()
    test_not_all_creator()
    print("\n全部通过 ✓")
