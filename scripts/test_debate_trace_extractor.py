#!/usr/bin/env python3
"""
test_debate_trace_extractor.py — DebateTrace提取器测试
验证：表格解析/标题解析/终裁标记/空文件处理/幂等
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# 确保可以导入
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.debate_trace_extractor import (
    规范化神名, 提取日期, 生成trace_id, 计算共识比例, 估算偏好,
    解析河床表格行, 解析河床终裁, 提取河床文档, 提取jiak卡片,
    加载已处理索引, 扫描河床文档, 扫描jiak卡片,
)

# ─── 测试1: 表格行解析 ───────────────────────────────────────────────

def test_表格行解析():
    """验证markdown表格行正确解析"""
    line = "| 阿波罗 | 真理 | **O双向皆盲**——cosine测灵魂如同用体温测人格 |"
    result = 解析河床表格行(line)
    assert result is not None, "应该解析成功"
    assert result["name"] == "阿波罗", f"名字错误: {result['name']}"
    assert "O双向皆盲" in result["verdict"], f"裁决缺失: {result['verdict'][:50]}"
    print("✅ test_表格行解析")

def test_表格行分隔符跳过():
    """验证分隔行被跳过"""
    line = "|---|---|---|"
    result = 解析河床表格行(line)
    assert result is None, "分隔行应返回None"

def test_表格行多列():
    """验证3列+表格"""
    line = "| 雅典娜 | 战略 | **值得做但不该定\"第一公理\"** |"
    result = 解析河床表格行(line)
    assert result is not None
    assert result["name"] == "雅典娜"
    print("✅ test_表格行多列")

# ─── 测试2: 标题/段落解析 ─────────────────────────────────────────────

def test_终裁标记():
    """验证军师终裁被标记为is_final"""
    content = """# 七神终裁 · 测试议题

> 军师 · 2026-09-06

## 一、七神总览

| 神 | 裁决 |
|---|---|
| 阿波罗 | 测试裁决A |
| 雅典娜 | 测试裁决B |

## 二、军师终裁

采纳全七神，修正三条。
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        f.flush()
        try:
            result = 提取河床文档(Path(f.name))
            assert result is not None, "应该提取成功"
            agents = result["agents"]
            assert len(agents) >= 3, f"agents不足: {len(agents)}"
            # 检查终裁标记
            finals = [a for a in agents if a.get("is_final")]
            assert len(finals) >= 1, "应该有军师终裁标记"
            assert finals[0]["name"] == "军师"
            assert "采纳" in finals[0]["verdict"]
            print("✅ test_终裁标记")
        finally:
            os.unlink(f.name)

# ─── 测试3: 空文件处理 ───────────────────────────────────────────────

def test_空文件():
    """验证空文件不崩溃"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write("")
        f.flush()
        try:
            result = 提取河床文档(Path(f.name))
            assert result is None, "空文件应返回None"
            print("✅ test_空文件")
        finally:
            os.unlink(f.name)

def test_无合议内容文件():
    """验证非合议文件不提取"""
    content = "# 普通笔记\n\n这是一篇普通笔记，不含合议内容。\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        f.flush()
        try:
            result = 提取河床文档(Path(f.name))
            assert result is None, "非合议文件应返回None"
            print("✅ test_无合议内容文件")
        finally:
            os.unlink(f.name)

# ─── 测试4: 幂等 ────────────────────────────────────────────────────

def test_幂等():
    """验证相同trace_id不重复"""
    id1 = 生成trace_id("file_a", "topic1")
    id2 = 生成trace_id("file_a", "topic1")
    id3 = 生成trace_id("file_b", "topic1")
    assert id1 == id2, "相同输入应产生相同ID"
    assert id1 != id3, "不同输入应产生不同ID"
    print("✅ test_幂等")

# ─── 测试5: 规范化神名 ───────────────────────────────────────────────

def test_规范化神名():
    """验证各种格式的神名被统一"""
    assert 规范化神名("阿波罗(真理)") == "阿波罗"
    assert 规范化神名("阿波罗·真理") == "阿波罗"
    assert 规范化神名("阿波罗") == "阿波罗"
    assert 规范化神名("雅典娜(战略)") == "雅典娜"
    assert 规范化神名("赫淮斯托斯(锻造)") == "赫淮斯托斯"
    print("✅ test_规范化神名")

# ─── 测试6: 日期提取 ─────────────────────────────────────────────────

def test_提取日期():
    """验证日期提取"""
    assert 提取日期("2026-09-06 七神终裁") == "2026-09-06"
    assert 提取日期("七神启示-20260829.md") == "2026-08-29"
    assert 提取日期("无日期文本") == "unknown"
    print("✅ test_提取日期")

# ─── 测试7: 共识比例 ─────────────────────────────────────────────────

def test_计算共识比例():
    """验证共识比例计算"""
    agents = [
        {"name": "阿波罗", "verdict": "同意", "is_final": False},
        {"name": "雅典娜", "verdict": "同意", "is_final": False},
        {"name": "军师", "verdict": "裁决内容", "is_final": True},
    ]
    assert 计算共识比例(agents) == 1.0
    agents2 = [
        {"name": "阿波罗", "verdict": "同意", "is_final": False},
        {"name": "雅典娜", "verdict": "", "is_final": False},
    ]
    assert 计算共识比例(agents2) == 0.5
    assert 计算共识比例([]) == 0.0
    print("✅ test_计算共识比例")

# ─── 测试8: 偏好估算 ─────────────────────────────────────────────────

def test_估算偏好():
    """验证偏好估算"""
    agents = [
        {"name": "阿波罗", "verdict": "✅ 赞同", "is_final": False},
        {"name": "雅典娜", "verdict": "❌ 反对", "is_final": False},
        {"name": "军师", "verdict": "裁决", "is_final": True},
    ]
    hint = 估算偏好(agents)
    assert hint["chosen_count"] == 1
    assert "阿波罗" in hint["chosen_agents"]
    assert hint["rejected_count"] == 1
    assert "雅典娜" in hint["rejected_agents"]
    print("✅ test_估算偏好")

# ─── 测试9: 河床真实文档提取 ─────────────────────────────────────────

def test_真实河床文档():
    """验证能从真实河床文档提取≥5条Trace"""
    traces = 扫描河床文档()
    assert len(traces) >= 5, f"河床文档提取不足5条: {len(traces)}"
    # 至少一条有军师终裁
    has_final = any(
        any(a.get("is_final") for a in t["agents"])
        for t in traces
    )
    assert has_final, "应有至少一条含军师终裁"
    # 每条至少有2个agents
    for t in traces:
        assert len(t["agents"]) >= 2, f"{t['topic']} agents不足2个"
    print(f"✅ test_真实河床文档 ({len(traces)}条)")

# ─── 测试10: jiak卡片提取 ────────────────────────────────────────────

def test_jiak卡片():
    """验证jiak卡片提取能跑"""
    traces = 扫描jiak卡片()
    # jiak有740张decisions的卡片，但不是每张都是合议类
    assert len(traces) >= 1, f"jiak提取为0条"
    print(f"✅ test_jiak卡片 ({len(traces)}条合议类)")

# ─── 运行 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DebateTrace提取器测试")
    print("=" * 60)
    tests = [
        test_表格行解析, test_表格行分隔符跳过, test_表格行多列,
        test_终裁标记, test_空文件, test_无合议内容文件,
        test_幂等, test_规范化神名, test_提取日期,
        test_计算共识比例, test_估算偏好,
        test_真实河床文档, test_jiak卡片,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"结果: {passed}/{len(tests)} 通过, {failed} 失败")
    if failed:
        sys.exit(1)
