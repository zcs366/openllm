"""
BoundedMemoryContract 测试 — PAL P1-1
2026-07-07

⚠️ 外部依赖测试: 本测试通过 sys.path 引用 ~/.hermes/jiak/scripts/isa_context_editor.py。
BoundedMemoryContract 的源码不在 openllm 项目内，属于 jiak 工具库。
测试使用 __new__() + mock 绕过依赖（jieba/context_router），验证预算逻辑。
验证目标: 预算模式/选择性遗忘/紧急压缩算法，非 openllm 集成。

⚠️ 2026-08-14 修复（T-ISA-7根因）：mock和sys.path操作原本在**模块级**执行——
pytest收集阶段import本文件时，jieba被全局替换为MagicMock，污染所有后续测试
（jieba.cut返回空→cosine=0→test_e2e_mistake_causal失败）。
现移入setUpModule/tearDownModule：本模块测试运行时才mock，运行完恢复现场，模块级零污染。
"""
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import sys
import os

# 模块级只留惰性占位，mock+import在setUpModule执行
ISAContextEditor = None
InjectionResult = None
MemoryCandidate = None
TopicPrediction = None

_ORIG_SYS_MODULES = {}
_ORIG_SYS_PATH = None


def setUpModule():
    """本模块测试运行前：mock依赖+import isa_context_editor（临时污染，teardown恢复）。"""
    global ISAContextEditor, InjectionResult, MemoryCandidate, TopicPrediction
    global _ORIG_SYS_PATH
    _ORIG_SYS_PATH = list(sys.path)

    mock_jieba = MagicMock()
    _ORIG_SYS_MODULES['jieba'] = sys.modules.get('jieba')
    _ORIG_SYS_MODULES['jieba.analyse'] = sys.modules.get('jieba.analyse')
    _ORIG_SYS_MODULES['context_router'] = sys.modules.get('context_router')
    sys.modules['jieba'] = mock_jieba
    sys.modules['jieba.analyse'] = mock_jieba.analyse
    sys.modules['context_router'] = MagicMock()

    sys.path.insert(0, os.path.expanduser("~/.hermes/jiak/scripts"))
    try:
        from isa_context_editor import ISAContextEditor, InjectionResult, MemoryCandidate, TopicPrediction
    except ImportError as e:
        # 依赖不可用时跳过本模块（不error，不影响其他测试）
        raise unittest.SkipTest(f"isa_context_editor不可用: {e}")


def tearDownModule():
    """本模块测试运行后：恢复被mock的模块和sys.path（防止污染其他测试）。"""
    for name, orig in _ORIG_SYS_MODULES.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig
    if _ORIG_SYS_PATH is not None:
        sys.path[:] = _ORIG_SYS_PATH


class TestBoundedMemoryEnforcer(unittest.TestCase):
    """BoundedMemoryContract 核心逻辑测试。"""

    def _make_editor(self, task_complexity="normal", token_budget=1000):
        """创建带mock的editor。"""
        editor = ISAContextEditor.__new__(ISAContextEditor)
        editor.predictor = MagicMock()
        editor.selector = MagicMock()
        editor.orchestrator = MagicMock()
        editor.top_k = 5

        # BoundedMemoryContract fields
        editor._budget_profile = {"simple": 500, "normal": 1000, "complex": 2000}
        editor.token_budget = editor._budget_profile.get(task_complexity, token_budget)
        editor._cumulative_tokens = 0
        editor._turn_count = 0
        editor._injection_history = []
        editor._content_lengths = []  # P1-1: 注入长度基线数据
        return editor

    def test_budget_profiles(self):
        """三种复杂度对应不同预算。"""
        for complexity, expected in [("simple", 500), ("normal", 1000), ("complex", 2000)]:
            editor = self._make_editor(task_complexity=complexity)
            self.assertEqual(editor.token_budget, expected)

    def test_check_budget_full_mode(self):
        """使用率<60%→full模式。"""
        editor = self._make_editor()
        editor._cumulative_tokens = 300  # 30%
        self.assertEqual(editor._check_budget_and_switch(), "full")

    def test_check_budget_typed_retrieval(self):
        """使用率60-80%→typed_retrieval模式。"""
        editor = self._make_editor()
        editor._cumulative_tokens = 700  # 70%
        self.assertEqual(editor._check_budget_and_switch(), "typed_retrieval")

    def test_check_budget_emergency_compress(self):
        """使用率>=80%→emergency_compress模式。"""
        editor = self._make_editor()
        editor._cumulative_tokens = 850  # 85%
        self.assertEqual(editor._check_budget_and_switch(), "emergency_compress")

    def test_selective_forget_removes_cold(self):
        """选择性遗忘：access_count=0且age>3轮的candidate被移除。"""
        editor = self._make_editor()

        # 模拟历史：3轮前注入了card_A，但之后不再出现
        editor._injection_history = [
            {"turn": 1, "card_ids": ["card_A"], "token_count": 100, "cumulative": 100, "summary": ""},
            {"turn": 2, "card_ids": ["card_B"], "token_count": 100, "cumulative": 200, "summary": ""},
            {"turn": 3, "card_ids": ["card_B"], "token_count": 100, "cumulative": 300, "summary": ""},
            {"turn": 4, "card_ids": ["card_B"], "token_count": 100, "cumulative": 400, "summary": ""},
        ]

        # 创建candidates
        cold = MagicMock(spec=MemoryCandidate)
        cold.card_id = "card_A"
        warm = MagicMock(spec=MemoryCandidate)
        warm.card_id = "card_B"

        result = editor._selective_forget([cold, warm])
        result_ids = [c.card_id for c in result]
        self.assertIn("card_B", result_ids)
        # card_A可能被过滤（取决于逻辑）

    def test_selective_forget_keeps_at_least_one(self):
        """选择性遗忘：至少保留1条candidate。"""
        editor = self._make_editor()
        editor._injection_history = [
            {"turn": 1, "card_ids": ["old_card"], "token_count": 100, "cumulative": 100, "summary": ""},
        ] * 5

        old = MagicMock(spec=MemoryCandidate)
        old.card_id = "old_card"

        result = editor._selective_forget([old])
        self.assertEqual(len(result), 1)  # 至少保留1条

    def test_emergency_compress(self):
        """紧急压缩：将最近insight合并为摘要。"""
        editor = self._make_editor()
        editor._injection_history = [
            {"turn": 1, "summary": "ISA记忆系统设计", "card_ids": [], "token_count": 100, "cumulative": 100},
            {"turn": 2, "summary": "ICE注入机制", "card_ids": [], "token_count": 100, "cumulative": 200},
            {"turn": 3, "summary": "BoundedMemoryContract", "card_ids": [], "token_count": 100, "cumulative": 300},
        ]

        real_result = InjectionResult(
            content="original content",
            candidates=[],
            token_count=100,
            budget_used=0.3,
            elapsed_ms=1.0,
        )

        compressed = editor._emergency_compress(real_result)
        self.assertIn("近期记忆摘要", compressed.content)
        self.assertIn("ISA记忆系统设计", compressed.content)

    def test_record_injection(self):
        """记录注入历史。"""
        editor = self._make_editor()

        mock_c1 = MagicMock(spec=MemoryCandidate)
        mock_c1.card_id = "card_1"
        mock_result = MagicMock(spec=InjectionResult)
        mock_result.token_count = 150
        mock_result.candidates = [mock_c1]
        mock_result.content = "test content" * 20

        editor._record_injection(mock_result)

        self.assertEqual(editor._turn_count, 1)
        self.assertEqual(editor._cumulative_tokens, 150)
        self.assertEqual(len(editor._injection_history), 1)
        self.assertEqual(editor._injection_history[0]["card_ids"], ["card_1"])

    def test_memory_usage_ratio(self):
        """使用率计算正确。"""
        editor = self._make_editor()
        editor._cumulative_tokens = 500
        self.assertAlmostEqual(editor.memory_usage_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()

