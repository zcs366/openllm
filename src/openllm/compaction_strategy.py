"""
CompactionStrategy — compression strategy base class
====================================================

Strategy pattern for different context compression approaches.

Core ABC:
  - CompactionStrategy: abstract base with compress/should_compress

Concrete implementations:
  - RollingWindowStrategy: keeps the most recent N messages

Integration point with ContextPressureMonitor (context_pressure.py):
  ContextPressureMonitor.should_compress() decides WHEN to compress;
  CompactionStrategy subclasses decide HOW to compress.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from .message import Message

logger = logging.getLogger("openllm.compaction_strategy")


class CompactionStrategy(ABC):
    """Abstract base class for context compression strategies.

    Subclasses implement two core operations:
      should_compress — determine whether compression is needed
      compress        — perform the actual compression

    Design rationale:
      This class intentionally does NOT inherit from ContextEngine.
      ContextEngine handles persist/restore/search; CompactionStrategy
      handles only the *how* of compression. The PressureMonitor calls
      should_compress() to decide *when*, then delegates to compress().
    """

    @abstractmethod
    def should_compress(self, messages: List[Message], token_budget: int) -> bool:
        """Determine whether the message list exceeds the token budget.

        Parameters
        ----------
        messages:
            Current conversation messages, oldest first.
        token_budget:
            Maximum allowed total token count.

        Returns
        -------
        True if compression is warranted.
        """
        ...

    @abstractmethod
    def compress(self, messages: List[Message], token_budget: int) -> List[Message]:
        """Compress messages to fit within the token budget.

        Parameters
        ----------
        messages:
            Current conversation messages, oldest first.
        token_budget:
            Maximum allowed total token count after compression.

        Returns
        -------
        A new list of messages that fits within the budget.
        Implementations may drop, merge, or summarize messages.
        """
        ...

    # ── helper ────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(messages: List[Message]) -> int:
        """Rough token estimate: ~4 chars per token (English average).

        This is an approximation; production code should use a real
        tokenizer.  The factor of 4 is a widely-used heuristic that
        works well enough for budget comparison.
        """
        return sum(len(m.content) for m in messages) // 4


class RollingWindowStrategy(CompactionStrategy):
    """Keep only the most recent messages that fit within the budget.

    Parameters
    ----------
    min_keep:
        Minimum number of recent messages to always retain (default 2).
        Prevents the window from collapsing to zero on tiny budgets.

    Algorithm:
        1. If total tokens <= budget, return all messages unchanged.
        2. Walk backward from the most recent message, accumulating
           tokens until the budget is full.
        3. If fewer than min_keep messages remain, keep min_keep anyway
           (accept over-budget gracefully — the alternative is losing
           the entire conversation).
    """

    def __init__(self, min_keep: int = 2) -> None:
        if min_keep < 0:
            raise ValueError(f"min_keep must be >= 0, got {min_keep}")
        self._min_keep = min_keep

    def should_compress(self, messages: List[Message], token_budget: int) -> bool:
        if not messages:
            return False
        if token_budget <= 0:
            return True
        return self._estimate_tokens(messages) > token_budget

    def compress(self, messages: List[Message], token_budget: int) -> List[Message]:
        if not messages:
            return []
        if token_budget <= 0:
            return messages[-self._min_keep:] if self._min_keep else []

        total = self._estimate_tokens(messages)
        if total <= token_budget:
            return list(messages)  # shallow copy — safe to mutate caller side

        # Walk backward, keeping as many recent messages as fit.
        kept: List[Message] = []
        running_tokens = 0
        for msg in reversed(messages):
            msg_tokens = len(msg.content) // 4
            if running_tokens + msg_tokens > token_budget and kept:
                break
            kept.append(msg)
            running_tokens += msg_tokens

        kept.reverse()

        # Enforce minimum retention
        if len(kept) < self._min_keep:
            kept = list(messages[-self._min_keep:])

        logger.debug(
            "[RollingWindow] compressed %d → %d messages (≈%d → ≈%d tokens)",
            len(messages), len(kept), total, running_tokens,
        )
        return kept
