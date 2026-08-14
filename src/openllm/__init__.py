"""
openLLM — AI为自己设计的身体。
五体：isa(神) · iai(大脑) · ios(治理) · isn(技能) · iko(面)
一脑：章鱼I·左右脑LLM对弈
一循环：iax·10阶段心跳主循环

v0.1 · 2026-07-01
"""

# Context engineering modules (2026-07-27)
from .context_engine import ContextEngine, ContextEntry, SearchResult
from .message import Message
from .compaction_strategy import CompactionStrategy, RollingWindowStrategy
from .compaction_control import CompactionController, CompactionConfig
from .persistence import DiskPersistence
from .embedding import EmbeddingEngine
from .tool_index import ToolIndex
from .tool_search import tool_search_handler, register_tools
from .memory_api import MemoryStore
