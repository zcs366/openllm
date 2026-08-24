"""CausalMemoryStore 单实例测试。

验证 get_causal_store 工厂函数 + __init__ 缓存：
  ① 同目录两次调用返回同一对象
  ② 写端写入后，长驻读端立即可见（修复缓存分裂）
  ③ 跨实例重载数据持久化正常
  ④ 并发调用工厂无重复实例
"""
import tempfile
import threading
from pathlib import Path

import pytest

from openllm.memory.causal_memory import (
    CausalMemoryStore,
    get_causal_store,
    _singleton_cache,
)


@pytest.fixture(autouse=True)
def _clean_singleton_cache():
    """每个测试前清空单例缓存，避免跨测试污染。"""
    _singleton_cache.clear()
    yield
    _singleton_cache.clear()


def _make_store_with_scar(store_dir: Path, sig: str = "test_scar") -> None:
    """向指定目录的 Store 写入一条测试疤痕。"""
    store = get_causal_store(store_dir)
    store.store(
        action_signature=sig,
        context_features=["test"],
        prediction="p",
        prediction_confidence=0.5,
        actual_result="r",
        actual_success=False,
        delta="d",
        delta_magnitude=0.8,
        lesson=f"lesson_{sig}",
    )


# ── 测试 ①：同目录两次 get_causal_store 返回同一对象 ──────────

def test_same_dir_returns_same_object() -> None:
    """get_causal_store(d) 调用两次应返回同一个 Python 对象。"""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "causal"
        a = get_causal_store(d)
        b = get_causal_store(d)
        assert a is b, "同目录两次调用应返回同一实例"


# ── 测试 ②：写端写入后，长驻读端立即可见 ──────────────────────

def test_live_visibility_after_write() -> None:
    """长驻实例构造后，经写端（另一处 get_causal_store）写入新疤，
    长驻实例 to_context_block 立即可见。"""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "causal"
        # 模拟 ISA 启动时构造的长驻读端
        resident = get_causal_store(d)
        # 先写一条旧疤
        _make_store_with_scar(d, "old_scar")
        assert "old_scar" in resident.to_context_block()

        # 写端（模拟 ios_causal）通过工厂写入新疤
        _make_store_with_scar(d, "new_scar_live")

        # 长驻实例内存中应立即可见
        visible = any(
            m.action_signature == "new_scar_live"
            for m in resident._memories.values()
        )
        assert visible, "长驻实例应通过共享内存看到新疤"
        assert "new_scar_live" in resident.to_context_block()


# ── 测试 ③：跨实例重载数据持久化正常 ──────────────────────────

def test_cross_instance_persistence() -> None:
    """新构造的实例应能从磁盘加载之前写入的数据。"""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "causal"
        # 第一个实例写入
        s1 = get_causal_store(d)
        _make_store_with_scar(d, "persist_scar")

        # 清空缓存，模拟新进程
        _singleton_cache.clear()

        # 第二个实例从磁盘加载
        s2 = get_causal_store(d)
        assert s2 is not s1, "缓存已清空应创建新实例"
        loaded = any(
            m.action_signature == "persist_scar"
            for m in s2._memories.values()
        )
        assert loaded, "新实例应从磁盘加载到之前的疤痕"


# ── 测试 ④：并发调用工厂 100 次无重复实例 ─────────────────────

def test_concurrent_factory_no_duplicates() -> None:
    """多线程并发调用 get_causal_store，同目录只产生一个实例。"""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "causal"
        results: list = []

        def _grab() -> None:
            store = get_causal_store(d)
            results.append(id(store))

        threads = [threading.Thread(target=_grab) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        unique_ids = set(results)
        assert len(unique_ids) == 1, (
            f"100次并发调用应只产生1个实例，实际产生了 {len(unique_ids)} 个"
        )
