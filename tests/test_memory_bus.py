#!/usr/bin/env python3
"""
MemoryBus测试 — 核心功能 + Provider集成

测试矩阵：
1. MemoryBus核心：注册/注销/写入/检索/路由/去重/token_budget
2. Mock Provider：验证协议正确性
3. 多源融合：排序/去重/贪心填充
4. 审计日志：写入记录追踪
"""

import sys
import os
import time
import tempfile
import json
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.memory.memory_bus import (
    MemoryBus, MemoryRecord, WriteRequest, WriteResult, Query,
    estimate_tokens
)
from openllm.memory.providers import JiakProvider, RecallProvider, CausalProvider, UnifiedProvider


# ═══════════════════════════════════════════════
# Mock Provider for unit tests
# ═══════════════════════════════════════════════

class MockProvider:
    """Mock provider for testing"""

    def __init__(self, name: str, priority: int, records=None, accept_writes=True):
        self._name = name
        self._priority = priority
        self._records = records or []
        self._accept_writes = accept_writes
        self._store_calls = []

    @property
    def name(self):
        return self._name

    @property
    def priority(self):
        return self._priority

    def search(self, query: Query):
        # 简单关键词匹配
        results = []
        for r in self._records:
            if query.text.lower() in r.content.lower():
                results.append(r)
        return results[:query.top_k]

    def store(self, request: WriteRequest):
        self._store_calls.append(request)
        if self._accept_writes:
            return WriteResult(
                success=True,
                record_id=f"{self._name}:{len(self._store_calls)}",
                provider=self._name,
            )
        return WriteResult(success=False, reason="mock rejects")

    def count(self):
        return len(self._records)

    def health(self):
        return {"status": "ok", "count": len(self._records)}


# ═══════════════════════════════════════════════
# 全局测试计数器
# ═══════════════════════════════════════════════
_results = {"passed": 0, "failed": 0}

def _check(name, cond, detail=""):
    if cond:
        print(f"  ✅ {name}")
        _results["passed"] += 1
    else:
        print(f"  ❌ {name}: {detail}")
        _results["failed"] += 1

# ═══════════════════════════════════════════════
# 测试函数
# ═══════════════════════════════════════════════

def test_memory_bus_core():
    """T1: MemoryBus核心功能"""
    print("\n[T1] MemoryBus核心...")

    bus = MemoryBus()

    # T1.1 注册/注销
    p1 = MockProvider("a", 10)
    p2 = MockProvider("b", 5)
    bus.register(p1)
    bus.register(p2)
    _check("register", bus.get_providers() == ["b", "a"])

    bus.unregister("a")
    _check("unregister", bus.get_providers() == ["b"])

    bus.register(p1)
    _check("reregister", len(bus.get_providers()) == 2)

    # T1.2 写入路由
    result = bus.write(WriteRequest(content="test", source="test"))
    _check("write success", result.success)
    _check("write routed to lowest priority",
          result.provider == "b")  # priority 5 < 10

    # T1.3 检索
    records = [
        MemoryRecord(
            record_id="m1", content="端口配错应该是8080",
            source="test", record_type="lesson",
            importance=0.8, temperature=0.9, trust_level="internal",
            tags=["config"], timestamp=time.time(),
            score=0.8, provider="a",
        ),
        MemoryRecord(
            record_id="m2", content="数据库连接超时",
            source="test", record_type="event",
            importance=0.6, temperature=0.7, trust_level="internal",
            tags=["db"], timestamp=time.time(),
            score=0.6, provider="a",
        ),
    ]
    p_search = MockProvider("search", 10, records=records)
    bus.register(p_search)

    query = Query(text="端口", top_k=5)
    results = bus.query(query)
    _check("query returns results", len(results) > 0)
    _check("query correct record", results[0].record_id == "m1" if results else False)

    # T1.4 审计日志
    log = bus.get_write_log()
    _check("write log", len(log) >= 1)



def test_dedup_and_budget():
    """T2: 去重 + token_budget"""
    print("\n[T2] 去重 + token_budget...")
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    bus = MemoryBus()

    # 两个provider返回相同record_id
    r1 = MemoryRecord(
        record_id="dup:1", content="短内容",
        source="a", record_type="lesson",
        importance=0.8, temperature=0.9, trust_level="internal",
        tags=[], timestamp=time.time(), score=0.9, provider="a",
    )
    r2 = MemoryRecord(
        record_id="dup:1", content="短内容重复",
        source="b", record_type="lesson",
        importance=0.8, temperature=0.9, trust_level="internal",
        tags=[], timestamp=time.time(), score=0.7, provider="b",
    )
    r3 = MemoryRecord(
        record_id="dup:2", content="另一条记录",
        source="a", record_type="event",
        importance=0.6, temperature=0.7, trust_level="internal",
        tags=[], timestamp=time.time(), score=0.6, provider="a",
    )

    p_a = MockProvider("a", 10, records=[r1, r3])
    p_b = MockProvider("b", 20, records=[r2])
    bus.register(p_a)
    bus.register(p_b)

    results = bus.query(Query(text="", top_k=5, token_budget=10000))
    _check("dedup: same record_id kept once",
          len(results) == 2 and results[0].record_id == "dup:1")

    # token_budget测试
    big_content = "这是一个很长的内容" * 100
    r_big = MemoryRecord(
        record_id="big:1", content=big_content,
        source="a", record_type="lesson",
        importance=0.8, temperature=0.9, trust_level="internal",
        tags=[], timestamp=time.time(), score=0.9, provider="a",
    )
    p_big = MockProvider("big", 10, records=[r_big])
    bus.register(p_big)

    # 设置很小的token_budget
    results = bus.query(Query(text="", top_k=10, token_budget=50))
    big_included = any(r.record_id == "big:1" for r in results)
    _check("token_budget: big content excluded", not big_included)



def test_stats_and_health():
    """T3: 统计和健康检查"""
    print("\n[T3] 统计和健康检查...")
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    bus = MemoryBus()
    bus.register(MockProvider("x", 10, records=[MemoryRecord(
        record_id="x1", content="test", source="x", record_type="lesson",
        importance=0.5, temperature=0.5, trust_level="internal",
        tags=[], timestamp=time.time(),
    )]))
    bus.register(MockProvider("y", 20))

    stats = bus.stats()
    _check("stats providers count", stats["providers"] == 2)
    _check("stats total records", stats["total_records"] == 1)

    health = bus.health_check()
    _check("health check all providers", len(health) == 2)
    _check("health status ok", all(h.get("status") == "ok" for h in health.values()))



def test_jiak_provider():
    """T4: JiakProvider集成（如果有jiak数据）"""
    print("\n[T4] JiakProvider...")
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    provider = JiakProvider()
    _check("name", provider.name == "jiak")
    _check("priority", provider.priority == 0)
    _check("health", provider.health().get("status") in ("ok", "degraded"))

    # 检索（可能有数据也可能没有）
    results = provider.search(Query(text="test", top_k=5))
    _check("search returns list", isinstance(results, list))

    # 写入应该被拒绝（只读）
    result = provider.store(WriteRequest(content="test", source="test"))
    _check("store rejected", not result.success)



def test_causal_provider():
    """T5: CausalProvider集成"""
    print("\n[T5] CausalProvider...")
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    provider = CausalProvider()
    _check("name", provider.name == "causal")
    _check("priority", provider.priority == 20)

    # 健康检查（可能没有数据）
    health = provider.health()
    _check("health returns dict", isinstance(health, dict))

    # 检索
    results = provider.search(Query(text="test", top_k=5))
    _check("search returns list", isinstance(results, list))



def test_token_estimation():
    """T6: Token估算"""
    print("\n[T6] Token估算...")
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    _check("empty", estimate_tokens("") == 0)
    _check("chinese", estimate_tokens("端口") == 3)  # 2字×1.5
    _check("english", estimate_tokens("hello world") == 2)  # 2词
    _check("mixed", estimate_tokens("端口配置 hello") == 7)  # 4×1.5+1=7



def test_write_reject_all():
    """T7: 所有provider拒绝写入"""
    print("\n[T7] 写入全拒绝...")
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    bus = MemoryBus()
    bus.register(MockProvider("a", 10, accept_writes=False))
    bus.register(MockProvider("b", 20, accept_writes=False))

    result = bus.write(WriteRequest(content="test", source="test"))
    _check("all reject → success=False", not result.success)
    _check("all reject → blocked", result.blocked)
    _check("all reject → reason set", len(result.reason) > 0)



# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MemoryBus 测试套件")
    print("=" * 60)


    for test_fn in [
        test_memory_bus_core,
        test_dedup_and_budget,
        test_stats_and_health,
        test_jiak_provider,
        test_causal_provider,
        test_token_estimation,
        test_write_reject_all,
    ]:
        test_fn()

    print("\n" + "=" * 60)
    print(f"结果: {_results['passed']}/{_results['passed']+_results['failed']} passed, {_results['failed']} failed")
    print("=" * 60)

    sys.exit(1 if _results["failed"] else 0)
