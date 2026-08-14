"""
Tests for CompactionStrategy base class and RollingWindowStrategy.

Three test cases:
  1. base_class_interface  — ABC cannot be instantiated, abstract methods enforced
  2. rolling_window        — RollingWindowStrategy compresses correctly
  3. empty_message_boundary — empty / near-empty message lists handled gracefully
"""
import pytest

from openllm.compaction_strategy import CompactionStrategy, RollingWindowStrategy
from openllm.message import Message


# ── helpers ───────────────────────────────────────────────────

def _msg(content: str, role: str = "user") -> Message:
    """Shorthand to build a Message with deterministic content."""
    return Message(role=role, content=content)


# ── 1. Base class interface ──────────────────────────────────

class TestBaseClassInterface:
    """CompactionStrategy is abstract — concrete usage requires subclassing."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError, match="abstract method"):
            CompactionStrategy()  # type: ignore[abstract]

    def test_subclass_must_implement_methods(self):
        """A subclass that forgets one method is still abstract."""
        class Partial(CompactionStrategy):
            def should_compress(self, messages, token_budget):
                return False
            # missing compress → still abstract

        with pytest.raises(TypeError, match="abstract method"):
            Partial()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class Dummy(CompactionStrategy):
            def should_compress(self, messages, token_budget):
                return False
            def compress(self, messages, token_budget):
                return messages

        d = Dummy()
        msgs = [_msg("hello")]
        assert d.should_compress(msgs, 1000) is False
        assert d.compress(msgs, 1000) == msgs

    def test_estimate_tokens(self):
        """_estimate_tokens uses len(content)//4 heuristic."""
        msg = _msg("a" * 100)  # 100 chars → 25 tokens
        assert CompactionStrategy._estimate_tokens([msg]) == 25


# ── 2. RollingWindowStrategy ─────────────────────────────────

class TestRollingWindow:
    """RollingWindowStrategy keeps recent messages within budget."""

    def setup_method(self):
        self.strategy = RollingWindowStrategy(min_keep=2)

    def test_no_compression_when_under_budget(self):
        msgs = [_msg("short")]
        result = self.strategy.compress(msgs, token_budget=9999)
        assert len(result) == 1
        assert result[0].content == "short"

    def test_compresses_old_messages(self):
        # 5 messages, each ~100 tokens (400 chars) → ~500 total
        msgs = [_msg(f"message-{i} " + "x" * 390) for i in range(5)]
        # Budget for ~2 messages (200 tokens) → should keep last 2
        result = self.strategy.compress(msgs, token_budget=200)
        assert len(result) <= 3  # min_keep=2, so at most a few
        # Most recent messages should be preserved
        assert result[-1].content.startswith("message-4")

    def test_min_keep_respected(self):
        """Even with tiny budget, min_keep messages survive."""
        msgs = [_msg(f"msg-{i}") for i in range(10)]
        result = self.strategy.compress(msgs, token_budget=5)
        assert len(result) >= 2  # min_keep=2

    def test_should_compress_true_when_over(self):
        msgs = [_msg("x" * 4000)]  # ~1000 tokens
        assert self.strategy.should_compress(msgs, token_budget=100) is True

    def test_should_compress_false_when_under(self):
        msgs = [_msg("short")]  # ~1 token
        assert self.strategy.should_compress(msgs, token_budget=100) is False

    def test_should_compress_false_empty(self):
        assert self.strategy.should_compress([], token_budget=100) is False

    def test_zero_budget_returns_min_keep(self):
        msgs = [_msg(f"m-{i}") for i in range(5)]
        result = self.strategy.compress(msgs, token_budget=0)
        assert len(result) == 2  # min_keep

    def test_negative_budget_returns_min_keep(self):
        msgs = [_msg("a"), _msg("b"), _msg("c")]
        result = self.strategy.compress(msgs, token_budget=-10)
        assert len(result) == 2

    def test_messages_order_preserved(self):
        """Output maintains original chronological order."""
        msgs = [_msg("first"), _msg("second"), _msg("third"), _msg("fourth")]
        result = self.strategy.compress(msgs, token_budget=20)
        # Verify ordering: indices should be monotonically increasing
        timestamps = [m.timestamp for m in result]
        assert timestamps == sorted(timestamps)

    def test_custom_min_keep(self):
        s = RollingWindowStrategy(min_keep=1)
        msgs = [_msg(f"m-{i}") for i in range(10)]
        result = s.compress(msgs, token_budget=5)
        assert len(result) >= 1


# ── 3. Empty message boundary ────────────────────────────────

class TestEmptyBoundary:
    """Edge cases around empty / minimal message lists."""

    def test_empty_list_should_not_compress(self):
        s = RollingWindowStrategy()
        assert s.should_compress([], token_budget=100) is False

    def test_empty_list_compress_returns_empty(self):
        s = RollingWindowStrategy()
        result = s.compress([], token_budget=100)
        assert result == []

    def test_single_message_within_budget(self):
        s = RollingWindowStrategy()
        msg = _msg("only one")
        result = s.compress([msg], token_budget=100)
        assert len(result) == 1
        assert result[0].content == "only one"

    def test_single_message_over_budget(self):
        s = RollingWindowStrategy(min_keep=1)
        msg = _msg("x" * 10000)  # ~2500 tokens
        result = s.compress([msg], token_budget=10)
        # min_keep=1, so the single message survives even if over budget
        assert len(result) == 1

    def test_empty_content_messages(self):
        """Messages with empty content are valid (role-only messages)."""
        s = RollingWindowStrategy()
        msg = Message(role="assistant", content="")
        result = s.compress([msg], token_budget=100)
        assert len(result) == 1

    def test_min_keep_zero(self):
        """min_keep=0 means no forced retention."""
        s = RollingWindowStrategy(min_keep=0)
        msgs = [_msg("old"), _msg("new")]
        result = s.compress(msgs, token_budget=0)
        assert result == []

    def test_constructor_rejects_negative_min_keep(self):
        with pytest.raises(ValueError, match="min_keep must be >= 0"):
            RollingWindowStrategy(min_keep=-1)
