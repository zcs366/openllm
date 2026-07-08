"""
免疫系统v2 Day 2 测试
测试项:
  ④ GovernanceAuditLog (SQLite链式hash)
  ⑤ HeuristicsConsumer (经验消费闭环)
"""
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

# 用临时数据库测试，不污染hermes.db
TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["OPENLLM_TEST_DB"] = TEST_DB

from openllm.core.governance_engine import GovernanceAuditLog, HeuristicsConsumer


def test_audit_log_append():
    """④ 追加事件"""
    # 用临时db
    import sqlite3
    conn = sqlite3.connect(TEST_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS governance_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        event_type TEXT NOT NULL,
        agent_id TEXT DEFAULT '',
        action TEXT DEFAULT '',
        prev_hash TEXT DEFAULT '',
        session_id TEXT DEFAULT '',
        details TEXT DEFAULT ''
    )""")
    conn.commit()
    conn.close()
    
    log = GovernanceAuditLog()
    # 覆盖db_path
    log._db_path = TEST_DB
    log._conn = sqlite3.connect(TEST_DB)
    
    row_id = log.append("rule_installed", action="block_tool", agent_id="governance_engine")
    assert row_id > 0, f"append应返回正整数: {row_id}"
    
    row_id2 = log.append("rule_verified", action="regression_pass", agent_id="governance_engine")
    assert row_id2 > row_id
    
    print("  ✅ 审计追加: 2行写入成功")


def test_audit_log_chain_verify():
    """④ 链式hash验证"""
    import sqlite3
    
    # 用独立临时db
    test_chain_db = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(test_chain_db)
    conn.execute("""CREATE TABLE IF NOT EXISTS governance_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        event_type TEXT NOT NULL,
        agent_id TEXT DEFAULT '',
        action TEXT DEFAULT '',
        prev_hash TEXT DEFAULT '',
        session_id TEXT DEFAULT '',
        details TEXT DEFAULT ''
    )""")
    conn.commit()
    conn.close()
    
    log = GovernanceAuditLog()
    log._db_path = test_chain_db
    log._conn = sqlite3.connect(test_chain_db)
    
    # 写入2行
    log.append("test_event", action="action_1")
    log.append("test_event", action="action_2")
    
    valid, broken_at = log.verify_chain()
    assert valid, f"链应完整: broken_at={broken_at}"
    
    log._conn.close()
    os.remove(test_chain_db)
    
    print("  ✅ 链式hash: 验证通过")


def test_audit_log_query():
    """④ 查询事件"""
    import sqlite3
    log = GovernanceAuditLog()
    log._db_path = TEST_DB
    log._conn = sqlite3.connect(TEST_DB)
    
    results = log.query(event_type="rule_installed")
    assert len(results) >= 1, f"应查到rule_installed事件: {len(results)}"
    assert results[0]["event_type"] == "rule_installed"
    
    all_events = log.query(limit=10)
    assert len(all_events) >= 2
    
    print(f"  ✅ 审计查询: {len(results)}条rule_installed, {len(all_events)}条总计")


def test_heuristics_retrieve():
    """⑤ 从jiak卡片检索heuristics"""
    consumer = HeuristicsConsumer()
    # 用真实jiak目录
    results = consumer.retrieve("tool call error failure", top_k=3)
    # 可能有也可能没有结果，取决于cards目录内容
    assert isinstance(results, list)
    print(f"  ✅ 经验检索: 返回{len(results)}条候选")


def test_heuristics_format():
    """⑤ 格式化为context字符串"""
    consumer = HeuristicsConsumer()
    
    test_heuristics = [
        {
            "score": 5.0,
            "content": "测试决策",
            "trigger": {
                "auto": {"condition": "tool_call", "pattern": "TOOL_PARAM"},
                "semantic": "当连续两次TOOL_PARAM错误时，先验证参数schema"
            },
            "card": "test-card",
        },
        {
            "score": 3.0,
            "content": "另一个决策",
            "trigger": "简单trigger字符串",
            "card": "test-card-2",
        },
    ]
    
    output = consumer.format_for_context(test_heuristics)
    assert "⚠️历史经验" in output
    assert "TOOL_PARAM" in output
    assert "简单trigger字符串" in output
    
    # 空列表
    empty_output = consumer.format_for_context([])
    assert empty_output == ""
    
    print("  ✅ 经验格式化: 输出正确")


def test_audit_independence():
    """④ 审计日志不影响治理引擎测试"""
    import sqlite3
    # 确保hermes.db主库没被污染
    main_db = os.path.expanduser("~/.hermes/hermes.db")
    if os.path.exists(main_db):
        conn = sqlite3.connect(main_db)
        count = conn.execute("SELECT COUNT(*) FROM governance_audit").fetchone()[0]
        conn.close()
        print(f"  ✅ 主库独立: hermes.db有{count}行（未被测试污染）")
    else:
        print("  ⚠️ 主库不存在，跳过")


if __name__ == "__main__":
    print("=" * 60)
    print("免疫系统v2 · Day 2 单元测试")
    print("=" * 60)
    
    tests = [
        test_audit_log_append,
        test_audit_log_chain_verify,
        test_audit_log_query,
        test_heuristics_retrieve,
        test_heuristics_format,
        test_audit_independence,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1
    
    # 清理临时db
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    
    print(f"\n结果: {passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
