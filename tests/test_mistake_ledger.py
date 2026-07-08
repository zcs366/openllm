"""MistakeLedger单元测试——验证追加/查询/统计"""

import json
import os
import sys
import tempfile
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.memory.mistake_ledger import MistakeLedger


def test_append_creates_record():
    """测试：追加一条记录，验证字段完整"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        ledger = MistakeLedger(path)
        record = ledger.append(
            what="配错了端口，应该是8080",
            why="默认端口配置与实际环境不匹配",
            agent="tool_executor",
            severity="medium",
            session_id="test-session-001",
            tags=["config", "port"],
        )

        assert record["what"] == "配错了端口，应该是8080"
        assert record["why"] == "默认端口配置与实际环境不匹配"
        assert record["agent"] == "tool_executor"
        assert record["severity"] == "medium"
        assert record["session_id"] == "test-session-001"
        assert record["tags"] == ["config", "port"]
        assert "timestamp" in record
        assert "id" in record
        assert record["id"].startswith("mistake-")

        # 验证文件写入
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        saved = json.loads(lines[0])
        assert saved["what"] == record["what"]

        print("✅ test_append_creates_record passed")
    finally:
        os.unlink(path)


def test_query_by_keyword():
    """测试：按关键词查询"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        ledger = MistakeLedger(path)
        ledger.append(what="端口配错", why="配置不匹配", agent="executor")
        ledger.append(what="路径写错", why="文件不存在", agent="file_op")
        ledger.append(what="端口超时", why="网络不通", agent="executor")

        # 搜索"端口"
        results = ledger.query(keyword="端口")
        assert len(results) == 2
        assert all("端口" in r["what"] for r in results)

        # 搜索"executor"
        results = ledger.query(agent="executor")
        assert len(results) == 2

        # 搜索不存在的关键词（不会出现在what或why中）
        results = ledger.query(keyword="宇宙大爆炸")
        assert len(results) == 0

        print("✅ test_query_by_keyword passed")
    finally:
        os.unlink(path)


def test_get_stats():
    """测试：统计功能"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        ledger = MistakeLedger(path)

        # 空文件
        stats = ledger.get_stats()
        assert stats["total"] == 0

        # 添加多条记录
        ledger.append(what="错误1", why="原因1", severity="low")
        ledger.append(what="错误2", why="原因2", severity="high")
        ledger.append(what="错误3", why="原因3", severity="high")
        ledger.append(what="错误4", why="原因4", agent="agent_a")

        stats = ledger.get_stats()
        assert stats["total"] == 4
        assert stats["by_severity"]["low"] == 1
        assert stats["by_severity"]["high"] == 2
        assert stats["by_agent"]["agent_a"] == 1

        print("✅ test_get_stats passed")
    finally:
        os.unlink(path)


def test_empty_what_raises():
    """测试：空what应报错"""
    ledger = MistakeLedger("/tmp/test_mistake_ledger_empty.jsonl")
    try:
        ledger.append(what="", why="原因")
        assert False, "应该报ValueError"
    except ValueError as e:
        assert "what" in str(e)
        print("✅ test_empty_what_raises passed")
    finally:
        if os.path.exists("/tmp/test_mistake_ledger_empty.jsonl"):
            os.unlink("/tmp/test_mistake_ledger_empty.jsonl")


def test_invalid_severity_raises():
    """测试：无效severity应报错"""
    ledger = MistakeLedger("/tmp/test_mistake_ledger_invalid.jsonl")
    try:
        ledger.append(what="错误", why="原因", severity="invalid")
        assert False, "应该报ValueError"
    except ValueError as e:
        assert "severity" in str(e)
        print("✅ test_invalid_severity_raises passed")
    finally:
        if os.path.exists("/tmp/test_mistake_ledger_invalid.jsonl"):
            os.unlink("/tmp/test_mistake_ledger_invalid.jsonl")


if __name__ == "__main__":
    print("=== MistakeLedger 单元测试 ===\n")
    test_append_creates_record()
    test_query_by_keyword()
    test_get_stats()
    test_empty_what_raises()
    test_invalid_severity_raises()
    print("\n🎉 全部测试通过！")
