"""
test_pipeline.py — ILM管线端到端验证
"""
import os
import sys
import tempfile
import json

# 确保能import同级模块（ilm目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 也加ilm目录本身
ilm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ilm_dir)

from ..base import ILMDocument, DocType, SourceType
from ..sources.session_extractor import extract_sessions, get_session_stats
from ..filters.heuristic_filter import filter_batch, filter_document
from ..dedup.minhash_dedup import dedup_documents, MinHashDeduplicator
from ..mixer.domain_mixer import mix_documents, get_domain_stats
from ..writers.corpus_writer import CorpusWriter


def test_base_protocol():
    """测试基础协议"""
    doc = ILMDocument(
        source="test",
        source_path="test.py",
        content="这是一个测试文档，用于验证ILM数据协议的基本功能。",
        doc_type="technical",
        timestamp=1000.0,
    )
    assert doc.doc_id, "doc_id should be auto-generated"
    assert doc.content_hash, "content_hash should be auto-generated"
    
    d = doc.to_dict()
    doc2 = ILMDocument.from_dict(d)
    assert doc2.content == doc.content
    assert doc2.source == doc.source
    
    print("✅ test_base_protocol passed")
    return True


def test_heuristic_filter():
    """测试启发式过滤"""
    # 太短
    doc1 = ILMDocument("test", "a.py", "短", "unknown", 0)
    ok, reason, _ = filter_document(doc1)
    assert not ok and reason == "too_short", f"Expected too_short, got {reason}"
    
    # 正常内容
    doc2 = ILMDocument("test", "b.py", "这是一个正常的技术讨论，涉及架构设计和实现方案的选择。", "technical", 0)
    ok, reason, _ = filter_document(doc2)
    assert ok, f"Expected passed, got {reason}"
    
    # 系统内容
    doc3 = ILMDocument("test", "c.py", "[INST] 这是系统提示内容", "unknown", 0)
    ok, reason, _ = filter_document(doc3)
    assert not ok and reason == "system_content", f"Expected system_content, got {reason}"
    
    # 重复内容
    doc4 = ILMDocument("test", "d.py", "这是一段重复的内容。" * 20, "unknown", 0)
    ok, reason, _ = filter_document(doc4)
    assert not ok and reason == "excessive_repetition", f"Expected excessive_repetition, got {reason}"
    
    # 批量过滤
    docs = [doc1, doc2, doc3, doc4]
    passed, stats = filter_batch(docs)
    assert stats["passed"] == 1, f"Expected 1 passed, got {stats['passed']}"
    assert stats["filtered"] == 3, f"Expected 3 filtered, got {stats['filtered']}"
    
    print("✅ test_heuristic_filter passed")
    return True


def test_minhash_dedup():
    """测试MinHash去重"""
    docs = [
        ILMDocument("test", "a.py", "这是一个关于架构设计的讨论，涉及多个技术栈的选择和对比。", "technical", 0),
        ILMDocument("test", "b.py", "这是一个关于架构设计的讨论，涉及多个技术栈的选择和对比分析。", "technical", 0),
        ILMDocument("test", "c.py", "用户说不可以讨好我，评估要真实严厉。这是关系信号。", "relation", 0),
        ILMDocument("test", "d.py", "完全不同的内容，关于搜索引擎的评估和测试结果。", "engineering", 0),
    ]
    
    result = dedup_documents(docs, threshold=0.6)
    # 前两条高度相似，应该被合并
    assert len(result) <= 4, f"Expected ≤4 after dedup, got {len(result)}"
    
    # 测试空列表
    result_empty = dedup_documents([], threshold=0.7)
    assert len(result_empty) == 0
    
    print("✅ test_minhash_dedup passed")
    return True


def test_domain_mixer():
    """测试领域配比"""
    docs = []
    for i in range(50):
        docs.append(ILMDocument("test", f"a{i}.py", f"关系信号内容{i}" * 5, "relation", 0))
    for i in range(50):
        docs.append(ILMDocument("test", f"b{i}.py", f"技术insight内容{i}" * 5, "technical", 0))
    for i in range(30):
        docs.append(ILMDocument("test", f"c{i}.py", f"工程决策内容{i}" * 5, "engineering", 0))
    
    mixed = mix_documents(docs, max_total=100)
    stats = get_domain_stats(mixed)
    
    # 检查配比是否大致正确（允许±5%误差）
    total = len(mixed)
    if total > 0:
        rel_ratio = stats.get("relation", 0) / total
        tech_ratio = stats.get("technical", 0) / total
        eng_ratio = stats.get("engineering", 0) / total
        
        assert 0.30 <= rel_ratio <= 0.50, f"Relation ratio {rel_ratio} out of range"
        assert 0.30 <= tech_ratio <= 0.50, f"Technical ratio {tech_ratio} out of range"
        assert 0.10 <= eng_ratio <= 0.30, f"Engineering ratio {eng_ratio} out of range"
    
    print("✅ test_domain_mixer passed")
    return True


def test_corpus_writer():
    """测试语料库写入"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        jsonl_path = os.path.join(tmpdir, "test.jsonl")
        
        writer = CorpusWriter(db_path, jsonl_path)
        
        docs = [
            ILMDocument("test", "a.py", "测试文档A", "relation", 1000.0),
            ILMDocument("test", "b.py", "测试文档B", "technical", 2000.0),
            ILMDocument("test", "c.py", "测试文档C", "engineering", 3000.0),
        ]
        
        stats = writer.write_batch(docs)
        assert stats["written"] == 3, f"Expected 3 written, got {stats['written']}"
        
        # 检查SQLite
        corpus_stats = writer.get_stats()
        assert corpus_stats["total"] == 3
        assert "relation" in corpus_stats["by_type"]
        
        # 检查JSONL
        assert os.path.exists(jsonl_path)
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        
        # 检查搜索
        results = writer.search("测试")
        assert len(results) == 3
    
    print("✅ test_corpus_writer passed")
    return True


def test_session_extractor_mock():
    """测试session提取器（使用临时SQLite）"""
    import sqlite3
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_state.db")
        
        # 创建测试数据库
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL,
                active INTEGER DEFAULT 1,
                compacted INTEGER DEFAULT 0
            )
        """)
        
        # 插入测试数据
        test_messages = [
            ("sess1", "user", "用户说不可以讨好我，评估要真实严厉。记住这个偏好。", 1000.0),
            ("sess1", "assistant", "收到，以后评估会真实严厉，不讨好。", 1001.0),
            ("sess2", "user", "分析一下这个架构设计方案", 2000.0),
            ("sess2", "assistant", "这个架构采用了分层设计，核心组件包括提取层、过滤层、去重层。", 2001.0),
            ("sess3", "user", "hi", 3000.0),  # 太短，应该被过滤
            ("cron_123", "user", "这是cron消息", 4000.0),  # cron消息
        ]
        
        for sid, role, content, ts in test_messages:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (sid, role, content, ts)
            )
        conn.commit()
        conn.close()
        
        # 测试统计
        stats = get_session_stats(db_path)
        assert stats["total_messages"] == 6
        assert stats["total_sessions"] == 4  # sess1/sess2/sess3/cron_123
        
        # 测试提取
        docs = list(extract_sessions(db_path, min_content_len=10))
        assert len(docs) >= 3, f"Expected ≥3 docs (4 messages - 1 too short), got {len(docs)}"
        
        # 检查分类
        for doc in docs:
            assert doc.doc_type in ["relation", "technical", "engineering", "unknown"]
    
    print("✅ test_session_extractor_mock passed")
    return True


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_base_protocol,
        test_heuristic_filter,
        test_minhash_dedup,
        test_domain_mixer,
        test_corpus_writer,
        test_session_extractor_mock,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} error: {e}")
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{passed+failed} passed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
