"""
CompactionController 测试 — T-CC-19
===================================

4 tests:
1. test_initialization — 配置创建、验证、默认值
2. test_processing_flow — process() 完整流程：不压缩、截断、摘要、激进
3. test_concurrency — 多线程并发process()安全
4. test_config_hot_reload — 配置热重载
"""
import os
import sys
import time
import threading
import tempfile
import unittest
from pathlib import Path

# Ensure openllm package is importable
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.message import Message
from openllm.compaction_control import (
    CompactionConfig,
    CompactionController,
    CompactionStrategy,
    CompactionStats,
    estimate_tokens,
)


def _make_msg(role: str = "user", content: str = "hello", n: int = 1) -> Message:
    """Helper to create a Message."""
    return Message(role=role, content=content * n)


def _make_conversation(n_messages: int = 10, content_len: int = 200) -> list[Message]:
    """Helper to create a multi-message conversation."""
    msgs = [Message(role="system", content="You are a helpful assistant.")]
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * content_len
        msgs.append(Message(role=role, content=content))
    return msgs


class TestCompactionInitialization(unittest.TestCase):
    """Test 1: 配置创建、验证、默认值。"""

    def test_default_config(self):
        """默认配置应能正常创建。"""
        config = CompactionConfig()
        self.assertEqual(config.max_tokens, 8192)
        self.assertAlmostEqual(config.caution_ratio, 0.70)
        self.assertAlmostEqual(config.critical_ratio, 0.85)
        self.assertEqual(config.keep_recent, 5)
        self.assertTrue(config.keep_system)

    def test_custom_config(self):
        """自定义配置应正确赋值。"""
        config = CompactionConfig(max_tokens=4096, keep_recent=3, caution_ratio=0.60)
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.keep_recent, 3)
        self.assertAlmostEqual(config.caution_ratio, 0.60)

    def test_invalid_max_tokens(self):
        """max_tokens <= 0 应抛异常。"""
        with self.assertRaises(ValueError):
            CompactionConfig(max_tokens=0)
        with self.assertRaises(ValueError):
            CompactionConfig(max_tokens=-1)

    def test_invalid_ratio_order(self):
        """caution >= critical 应抛异常。"""
        with self.assertRaises(ValueError):
            CompactionConfig(caution_ratio=0.9, critical_ratio=0.8)
        with self.assertRaises(ValueError):
            CompactionConfig(caution_ratio=0.5, critical_ratio=0.5)

    def test_controller_creation(self):
        """CompactionController 应能正常初始化。"""
        config = CompactionConfig(max_tokens=1000)
        controller = CompactionController(config)
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller.config)
        self.assertEqual(controller.config.max_tokens, 1000)

    def test_invalid_config_type(self):
        """非 CompactionConfig 类型应抛 TypeError。"""
        with self.assertRaises(TypeError):
            CompactionController("not a config")

    def test_stats_initial(self):
        """初始统计应为零。"""
        controller = CompactionController(CompactionConfig())
        stats = controller.stats()
        self.assertEqual(stats["total_processed"], 0)
        self.assertEqual(stats["total_compressions"], 0)

    def test_estimate_tokens_empty(self):
        """空字符串token估算应为0。"""
        self.assertEqual(estimate_tokens(""), 0)

    def test_estimate_tokens_english(self):
        """英文token估算。"""
        # 4 chars ≈ 1 token
        tokens = estimate_tokens("abcd")
        self.assertEqual(tokens, 1)

    def test_estimate_tokens_chinese(self):
        """中文token估算。"""
        # 2 chars ≈ 1 token
        tokens = estimate_tokens("你好世界")
        self.assertEqual(tokens, 2)

    def test_custom_token_estimator(self):
        """自定义token估算函数。"""
        config = CompactionConfig(
            max_tokens=1000,
            token_estimator=lambda t: len(t.split()),
        )
        tokens = config.estimate_tokens("hello world foo")
        self.assertEqual(tokens, 3)


class TestCompactionProcessingFlow(unittest.TestCase):
    """Test 2: process() 完整流程。"""

    def test_under_threshold_passthrough(self):
        """低于阈值时应原样返回。"""
        config = CompactionConfig(max_tokens=100000)
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=5, content_len=50)
        result = controller.process(messages)
        self.assertEqual(len(result), len(messages))
        self.assertIs(result, messages)  # same object, no copy

    def test_empty_messages(self):
        """空消息列表应返回空。"""
        controller = CompactionController(CompactionConfig())
        result = controller.process([])
        self.assertEqual(result, [])

    def test_truncate_strategy(self):
        """触及caution阈值应触发truncate。"""
        # max=1000, caution=0.50, critical=0.80
        # Each msg: "Message N: " + 200*x ≈ 210 chars ≈ 52 tokens (EN)
        # Need total between 500-800 tokens → ~10-12 non-system msgs + 1 system
        # 12 messages (1 system + 11 user/assistant) ≈ 624 tokens → truncate
        config = CompactionConfig(max_tokens=1000, caution_ratio=0.50, critical_ratio=0.80, keep_recent=3)
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=11, content_len=200)
        result = controller.process(messages)
        # truncate后应减少消息数
        self.assertLess(len(result), len(messages))
        # 应保留最近3条非system
        non_system = [m for m in result if m.role != "system"]
        self.assertLessEqual(len(non_system), 3 + 1)  # keep_recent=3
        # 统计应有记录
        stats = controller.stats()
        self.assertEqual(stats["total_compressions"], 1)
        self.assertEqual(stats["last_strategy"], "truncate")

    def test_summarize_strategy(self):
        """触及critical阈值应触发summarize。"""
        # max=500, caution=0.50, critical=0.80
        # Need total between 400-500 tokens → ~8-9 non-system msgs + 1 system
        # 9 messages (1 system + 8) ≈ 468 tokens → summarize
        config = CompactionConfig(
            max_tokens=500, caution_ratio=0.50, critical_ratio=0.80,
            keep_recent=2
        )
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=8, content_len=200)
        result = controller.process(messages)
        self.assertLess(len(result), len(messages))
        # 应有摘要消息（system role with type=compaction_summary）
        summaries = [
            m for m in result
            if m.role == "system" and m.metadata and m.metadata.get("type") == "compaction_summary"
        ]
        self.assertGreater(len(summaries), 0, "Should have at least one summary message")
        stats = controller.stats()
        self.assertEqual(stats["last_strategy"], "summarize")

    def test_aggressive_strategy(self):
        """超出max_tokens应触发aggressive。"""
        config = CompactionConfig(
            max_tokens=200, caution_ratio=0.50, critical_ratio=0.80,
            keep_recent=2
        )
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=30, content_len=200)
        result = controller.process(messages)
        # aggressive: system + marker + 2 recent
        system_msgs = [m for m in result if m.role == "system"]
        self.assertGreaterEqual(len(system_msgs), 1)
        stats = controller.stats()
        self.assertEqual(stats["last_strategy"], "aggressive")

    def test_stats_compression_ratio(self):
        """压缩率计算正确。"""
        config = CompactionConfig(max_tokens=500, keep_recent=2)
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=20, content_len=200)
        controller.process(messages)
        stats = controller.stats()
        self.assertGreater(stats["tokens_before"], 0)
        self.assertGreater(stats["tokens_after"], 0)
        self.assertLessEqual(stats["compression_ratio"], 1.0)

    def test_stats_reset(self):
        """reset_stats 应清零统计。"""
        config = CompactionConfig(max_tokens=500, keep_recent=2)
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=20, content_len=200)
        controller.process(messages)
        self.assertGreater(controller.stats()["total_compressions"], 0)
        controller.reset_stats()
        self.assertEqual(controller.stats()["total_compressions"], 0)

    def test_preserves_system_messages(self):
        """所有策略应保留system消息。"""
        config = CompactionConfig(
            max_tokens=200, caution_ratio=0.30, critical_ratio=0.50,
            keep_recent=2, keep_system=True
        )
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=20, content_len=200)
        result = controller.process(messages)
        # 至少保留原始的1条system消息
        system_count = sum(1 for m in result if m.role == "system")
        self.assertGreaterEqual(system_count, 1)


class TestCompactionConcurrency(unittest.TestCase):
    """Test 3: 多线程并发process()安全。"""

    def test_concurrent_process(self):
        """多个线程同时调用process()不应崩溃。"""
        config = CompactionConfig(
            max_tokens=500, caution_ratio=0.50, critical_ratio=0.80,
            keep_recent=3
        )
        controller = CompactionController(config)
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(10):
                    messages = _make_conversation(n_messages=20, content_len=150)
                    result = controller.process(messages)
                    # 结果必须是list
                    assert isinstance(result, list)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [], f"Errors in threads: {errors}")
        # 统计应正确累计
        stats = controller.stats()
        self.assertEqual(stats["total_processed"], 50)  # 5 threads * 10 iterations

    def test_concurrent_config_update(self):
        """config热重载不应干扰正在处理的线程。"""
        config = CompactionConfig(max_tokens=1000, keep_recent=5)
        controller = CompactionController(config)
        errors = []

        def processor():
            try:
                for _ in range(20):
                    messages = _make_conversation(n_messages=15, content_len=150)
                    result = controller.process(messages)
                    assert isinstance(result, list)
            except Exception as e:
                errors.append(str(e))

        def reloader():
            for i in range(10):
                try:
                    controller.update_config(keep_recent=3 + (i % 5))
                except Exception as e:
                    errors.append(str(e))

        t1 = threading.Thread(target=processor)
        t2 = threading.Thread(target=reloader)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertEqual(errors, [], f"Errors: {errors}")


class TestCompactionConfigHotReload(unittest.TestCase):
    """Test 4: 配置热重载。"""

    def test_hot_reload_single_field(self):
        """更新单个字段。"""
        config = CompactionConfig(max_tokens=8192, keep_recent=5)
        controller = CompactionController(config)
        controller.update_config(keep_recent=10)
        self.assertEqual(controller.config.keep_recent, 10)
        # max_tokens未变
        self.assertEqual(controller.config.max_tokens, 8192)

    def test_hot_reload_multiple_fields(self):
        """同时更新多个字段。"""
        config = CompactionConfig(max_tokens=8192)
        controller = CompactionController(config)
        controller.update_config(max_tokens=4096, keep_recent=3, caution_ratio=0.60)
        c = controller.config
        self.assertEqual(c.max_tokens, 4096)
        self.assertEqual(c.keep_recent, 3)
        self.assertAlmostEqual(c.caution_ratio, 0.60)

    def test_hot_reload_invalid_raises(self):
        """无效配置应抛异常，原配置不变。"""
        config = CompactionConfig(max_tokens=8192)
        controller = CompactionController(config)
        with self.assertRaises(ValueError):
            controller.update_config(max_tokens=-1)
        # 原配置不变
        self.assertEqual(controller.config.max_tokens, 8192)

    def test_hot_reload_invalid_field(self):
        """不存在的字段应抛 AttributeError。"""
        controller = CompactionController(CompactionConfig())
        with self.assertRaises(AttributeError):
            controller.update_config(nonexistent_field=42)

    def test_hot_reload_affects_processing(self):
        """配置热重载应影响后续处理行为。"""
        config = CompactionConfig(
            max_tokens=500, caution_ratio=0.50, critical_ratio=0.80,
            keep_recent=2
        )
        controller = CompactionController(config)
        messages = _make_conversation(n_messages=20, content_len=200)

        # 第一次处理
        result1 = controller.process(messages)
        count1 = len(result1)

        # 热重载：增大keep_recent
        controller.update_config(keep_recent=5)

        # 第二次处理（同样输入）
        result2 = controller.process(messages)
        count2 = len(result2)

        # keep_recent增大后，保留的消息应更多
        self.assertGreaterEqual(count2, count1)

    def test_hot_reload_persist_state(self):
        """持久化统计状态到文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "compaction_state.json"
            config = CompactionConfig(
                max_tokens=500, keep_recent=2, state_path=state_path
            )
            controller = CompactionController(config)
            messages = _make_conversation(n_messages=20, content_len=200)
            controller.process(messages)
            controller.persist_state()

            self.assertTrue(state_path.exists())
            import json
            data = json.loads(state_path.read_text())
            self.assertEqual(data["version"], 1)
            self.assertIn("stats", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
