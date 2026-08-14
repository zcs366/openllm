"""
CompactionController — 上下文压缩控制核心
========================================

核心职责：监控消息列表的token使用量，当超出阈值时触发压缩策略。
纯规则引擎，零LLM调用。

设计原则：
  - CompactionConfig 声明式配置，支持热重载
  - process() 是唯一公开处理接口，输入输出都是 list[Message]
  - 压缩策略分三级：truncate / summarize / aggressive
  - 统计信息通过 stats() 暴露，便于外部监控

压缩触发链：
  1. 计算总token数（简单估算：1 token ≈ 4 chars 英文 / 2 chars 中文）
  2. 比对 config.max_tokens 阈值
  3. 低于阈值 → 原样返回
  4. 触及 caution 阈值 → truncate 策略（丢弃最旧消息）
  5. 触及 critical 阈值 → summarize 策略（合并旧消息摘要）
  6. 超出 max_tokens → aggressive 策略（仅保留 system + 最近N条）

上下文格言：
  上下文窗口是Agent的短期记忆。压缩不是删除记忆，而是把散装记忆打包。
"""
from __future__ import annotations

import copy
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .message import Message

logger = logging.getLogger("openllm.compaction_control")


# ── 估算常量 ────────────────────────────────────────────
# 粗略token估算：英文~4chars/token，中文~2chars/token
CHARS_PER_TOKEN_EN = 4.0
CHARS_PER_TOKEN_CN = 2.0


def estimate_tokens(text: str) -> int:
    """估算文本的token数（粗略，零依赖）。

    策略：统计中文字符占比，加权计算。
    """
    if not text:
        return 0
    total_chars = len(text)
    if total_chars == 0:
        return 0
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    cn_ratio = cn_chars / total_chars
    # 加权：CN部分用2 chars/token，其余用4 chars/token
    effective_chars_per_token = (
        cn_ratio * CHARS_PER_TOKEN_CN + (1 - cn_ratio) * CHARS_PER_TOKEN_EN
    )
    return max(1, math.ceil(total_chars / effective_chars_per_token))


# ── 压缩策略 ─────────────────────────────────────────────

class CompactionStrategy(Enum):
    """三级压缩策略。"""
    NONE = "none"              # 不压缩
    TRUNCATE = "truncate"      # 丢弃最旧消息
    SUMMARIZE = "summarize"    # 合并旧消息为摘要
    AGGRESSIVE = "aggressive"  # 极限压缩：仅保留system+最近N条


@dataclass
class CompactionConfig:
    """压缩控制配置——声明式，支持热重载。

    Attributes:
        max_tokens: token上限，超出触发压缩
        caution_ratio: caution阈值比例（0-1），触发 truncate
        critical_ratio: critical阈值比例（0-1），触发 summarize
        keep_recent: 压缩时至少保留最近N条消息
        keep_system: 始终保留system消息
        summary_max_tokens: summarize策略产出的最大token数
        token_estimator: 自定义token估算函数（可选）
        state_path: 持久化路径（可选）
    """
    max_tokens: int = 8192
    caution_ratio: float = 0.70
    critical_ratio: float = 0.85
    keep_recent: int = 5
    keep_system: bool = True
    summary_max_tokens: int = 500
    token_estimator: Optional[Callable[[str], int]] = None
    state_path: Optional[Path] = None

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not 0 < self.caution_ratio < 1:
            raise ValueError(f"caution_ratio must be in (0,1), got {self.caution_ratio}")
        if not self.caution_ratio < self.critical_ratio <= 1:
            raise ValueError(
                f"Must have caution_ratio < critical_ratio <= 1, "
                f"got {self.caution_ratio} < {self.critical_ratio}"
            )

    def estimate_tokens(self, text: str) -> int:
        """使用配置的估算函数或默认估算。"""
        if self.token_estimator:
            return self.token_estimator(text)
        return estimate_tokens(text)


@dataclass
class CompactionStats:
    """压缩统计信息。"""
    total_processed: int = 0
    total_compressions: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    messages_before: int = 0
    messages_after: int = 0
    last_strategy: str = "none"
    last_compression_at: float = 0.0
    error_count: int = 0

    @property
    def compression_ratio(self) -> float:
        """压缩率：tokens_after / tokens_before。"""
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_after / self.tokens_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "total_compressions": self.total_compressions,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "compression_ratio": round(self.compression_ratio, 4),
            "last_strategy": self.last_strategy,
            "last_compression_at": self.last_compression_at,
            "error_count": self.error_count,
        }


class CompactionController:
    """上下文压缩控制核心——监控token使用，触发压缩。

    用法：
        controller = CompactionController(CompactionConfig(max_tokens=8192))
        compressed = controller.process(messages)
    """

    def __init__(self, config: CompactionConfig):
        """初始化压缩控制器。

        Args:
            config: 压缩配置
        """
        if not isinstance(config, CompactionConfig):
            raise TypeError(f"Expected CompactionConfig, got {type(config).__name__}")
        self._config = config
        self._stats = CompactionStats()
        self._lock = threading.Lock()

    @property
    def config(self) -> CompactionConfig:
        """当前配置（只读副本）。"""
        return copy.deepcopy(self._config)

    def update_config(self, **kwargs: Any) -> None:
        """热重载配置——更新指定字段。

        先验证新值，再原子应用。线程安全。
        """
        with self._lock:
            # Validate before applying
            test_values = {}
            for key, value in kwargs.items():
                if not hasattr(self._config, key):
                    raise AttributeError(f"CompactionConfig has no field '{key}'")
                test_values[key] = value

            # Build a test config with the new values applied
            current = self._config
            merged = CompactionConfig(
                max_tokens=test_values.get("max_tokens", current.max_tokens),
                caution_ratio=test_values.get("caution_ratio", current.caution_ratio),
                critical_ratio=test_values.get("critical_ratio", current.critical_ratio),
                keep_recent=test_values.get("keep_recent", current.keep_recent),
                keep_system=test_values.get("keep_system", current.keep_system),
                summary_max_tokens=test_values.get("summary_max_tokens", current.summary_max_tokens),
            )
            # Validation passed — apply
            for key, value in kwargs.items():
                setattr(self._config, key, value)
        logger.info(f"[Compaction] Config hot-reloaded: {kwargs}")

    def process(self, messages: List[Message]) -> List[Message]:
        """处理消息列表——监控token，必要时压缩。

        核心接口。输入输出都是 list[Message]。

        Args:
            messages: 当前消息列表（从旧到新）

        Returns:
            可能压缩后的消息列表
        """
        if not messages:
            return []

        with self._lock:
            self._stats.total_processed += 1

        # 1. 估算总token数
        total_tokens = self._count_tokens(messages)
        max_tokens = self._config.max_tokens

        # 2. 未超阈值 → 原样返回
        if total_tokens <= max_tokens * self._config.caution_ratio:
            logger.debug(
                f"[Compaction] Under threshold: {total_tokens}/{max_tokens} "
                f"({total_tokens/max_tokens:.1%})"
            )
            return messages

        # 3. 选择策略
        strategy = self._select_strategy(total_tokens, max_tokens)

        # 4. 执行压缩
        compressed = self._apply_strategy(messages, strategy, total_tokens)

        # 5. 更新统计
        compressed_tokens = self._count_tokens(compressed)
        with self._lock:
            self._stats.total_compressions += 1
            self._stats.tokens_before = total_tokens
            self._stats.tokens_after = compressed_tokens
            self._stats.messages_before = len(messages)
            self._stats.messages_after = len(compressed)
            self._stats.last_strategy = strategy.value
            self._stats.last_compression_at = time.time()

        logger.info(
            f"[Compaction] Applied {strategy.value}: "
            f"{total_tokens}→{compressed_tokens} tokens, "
            f"{len(messages)}→{len(compressed)} messages"
        )
        return compressed

    def stats(self) -> Dict[str, Any]:
        """获取压缩统计信息。"""
        with self._lock:
            return self._stats.to_dict()

    def reset_stats(self) -> None:
        """重置统计信息。"""
        with self._lock:
            self._stats = CompactionStats()

    # ── 内部方法 ──────────────────────────────────────────

    def _count_tokens(self, messages: List[Message]) -> int:
        """计算消息列表总token数。"""
        return sum(
            self._config.estimate_tokens(m.content) for m in messages
        )

    def _select_strategy(self, total_tokens: int, max_tokens: int) -> CompactionStrategy:
        """根据token使用率选择压缩策略。"""
        ratio = total_tokens / max_tokens
        if ratio >= 1.0:
            return CompactionStrategy.AGGRESSIVE
        elif ratio >= self._config.critical_ratio:
            return CompactionStrategy.SUMMARIZE
        else:
            return CompactionStrategy.TRUNCATE

    def _apply_strategy(
        self,
        messages: List[Message],
        strategy: CompactionStrategy,
        total_tokens: int,
    ) -> List[Message]:
        """执行具体压缩策略。"""
        if strategy == CompactionStrategy.TRUNCATE:
            return self._truncate(messages)
        elif strategy == CompactionStrategy.SUMMARIZE:
            return self._summarize(messages)
        elif strategy == CompactionStrategy.AGGRESSIVE:
            return self._aggressive(messages)
        return messages

    def _truncate(self, messages: List[Message]) -> List[Message]:
        """截断策略：丢弃最旧消息，保留最近N条 + system消息。"""
        keep = self._config.keep_recent
        if self._config.keep_system:
            system_msgs = [m for m in messages if m.role == "system"]
            non_system = [m for m in messages if m.role != "system"]
            recent = non_system[-keep:] if len(non_system) > keep else non_system
            return system_msgs + recent
        else:
            return messages[-keep:] if len(messages) > keep else messages

    def _summarize(self, messages: List[Message]) -> List[Message]:
        """摘要策略：将旧消息合并为一条摘要 + 保留最近消息。

        注意：这是一个纯规则摘要（拼接+截断），不调用LLM。
        真正的LLM摘要应在外部实现后通过 token_estimator 回调。
        """
        keep = self._config.keep_recent
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if len(non_system) <= keep:
            return messages

        old_messages = non_system[:-keep]
        recent_messages = non_system[-keep:]

        # 构造摘要：提取旧消息的关键内容
        summary_parts = []
        for m in old_messages:
            # 取每条消息的前100字符作为摘要片段
            snippet = m.content[:100]
            if len(m.content) > 100:
                snippet += "..."
            summary_parts.append(f"[{m.role}] {snippet}")

        summary_text = "[上下文摘要] " + " | ".join(summary_parts)

        # 截断到 max_tokens
        max_tokens = self._config.summary_max_tokens
        est = self._config.estimate_tokens(summary_text)
        if est > max_tokens:
            # 按比例截断
            char_limit = int(len(summary_text) * max_tokens / est)
            summary_text = summary_text[:char_limit]

        summary_msg = Message(
            role="system",
            content=summary_text,
            metadata={"type": "compaction_summary", "compressed_from": len(old_messages)},
        )

        return system_msgs + [summary_msg] + recent_messages

    def _aggressive(self, messages: List[Message]) -> List[Message]:
        """激进策略：仅保留system消息 + 最近N条。"""
        keep = self._config.keep_recent
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        recent = non_system[-keep:] if len(non_system) > keep else non_system

        # 添加标记
        if messages and len(non_system) > keep:
            marker = Message(
                role="system",
                content=f"[上下文已被激进压缩] 原始{len(messages)}条消息，仅保留最近{keep}条",
                metadata={"type": "compaction_marker", "original_count": len(messages)},
            )
            return system_msgs + [marker] + recent

        return system_msgs + recent

    def persist_state(self) -> None:
        """持久化当前统计状态。"""
        if self._config.state_path is None:
            return
        try:
            path = self._config.state_path
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "updated_at": time.time(),
                "stats": self.stats(),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"[Compaction] State persisted to {path}")
        except OSError as e:
            logger.warning(f"[Compaction] State persist failed: {e}")
            with self._lock:
                self._stats.error_count += 1
