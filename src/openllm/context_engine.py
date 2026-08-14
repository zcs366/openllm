"""ContextEngine ABC — the contract for all context management implementations.

Every context engine must implement four core operations:
  compress  — reduce context window usage
  persist   — save context state to durable storage
  restore   — reload context state from durable storage
  search    — retrieve relevant context entries

Concrete implementations (SQLite-backed, in-memory, distributed, etc.)
subclass this ABC and fill in the abstract methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextEntry:
    """A single piece of context stored in the engine."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search hit returned by the engine."""

    entry: ContextEntry
    score: float  # relevance score, higher is better


class ContextEngine(ABC):
    """Abstract base class defining the contract for context engines.

    Lifecycle:
        1. *persist* / *restore*  — durable state across sessions
        2. *compress*             — shrink context when window fills
        3. *search*               — pull relevant context on demand

    Subclasses MUST implement every abstract method; no defaults are
    provided because each backend has different storage / retrieval
    semantics.
    """

    # ── compress ──────────────────────────────────────────────
    @abstractmethod
    def compress(self, entries: list[ContextEntry], target_tokens: int) -> list[ContextEntry]:
        """Compress *entries* until the total token count ≤ *target_tokens*.

        Parameters
        ----------
        entries:
            Current context window contents, oldest first.
        target_tokens:
            Desired maximum total token count after compression.

        Returns
        -------
        A new (or mutated) list of entries that fits within the budget.
        Implementations may drop, summarize, or merge entries.
        """

    # ── persist ───────────────────────────────────────────────
    @abstractmethod
    def persist(self, entries: list[ContextEntry]) -> int:
        """Persist *entries* to durable storage.

        Returns
        -------
        Number of entries successfully persisted.
        """

    # ── restore ───────────────────────────────────────────────
    @abstractmethod
    def restore(self) -> list[ContextEntry]:
        """Restore previously-persisted context entries.

        Returns
        -------
        The restored entries, or an empty list if nothing is stored.
        """

    # ── search ────────────────────────────────────────────────
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search stored context for entries relevant to *query*.

        Parameters
        ----------
        query:
            Natural-language or keyword query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        A list of :class:`SearchResult`, best-matching first.
        """
