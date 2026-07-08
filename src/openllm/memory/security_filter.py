"""
ISA+IAI 输入侧安全分层（Layer 1）

将检索到的 RetrievalItem 按 trust_level 分三层处理：
  SAFE      → 直接注入上下文
  CAUTION   → 加安全前缀后注入
  BLOCKED   → 丢弃 + 审计日志

设计原则：
  - 纯规则，零 LLM 调用
  - trust_level 来源于 immune.py / causal_memory.py 的 TrustLevel 枚举
  - 审计链完整，blocked 条目写入日志供追溯
"""

import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

logger = logging.getLogger("openllm.security_filter")


# ── 安全层级 ──────────────────────────────────────────

class SecurityLayer(str, Enum):
    """过滤后的安全层级"""
    SAFE = "safe"
    CAUTION = "caution"
    BLOCKED = "blocked"


# ── 数据结构 ──────────────────────────────────────────

@dataclass(frozen=True)
class RetrievalItem:
    """检索结果条目（输入）"""
    content: str
    source: str
    trust_level: str          # trusted / internal / untrusted / unknown
    relevance_score: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FilteredResult:
    """过滤后的条目（输出）"""
    item: RetrievalItem
    layer: SecurityLayer
    reason: str
    prefix: str = ""


@dataclass(frozen=True)
class SecurityFilterReport:
    """一次过滤的完整报告"""
    total_items: int
    safe_count: int
    caution_count: int
    blocked_count: int
    blocked_items: list
    filtered_results: list


# ── 默认阈值映射 ─────────────────────────────────────

DEFAULT_TRUST_THRESHOLDS: Dict[str, SecurityLayer] = {
    "trusted":  SecurityLayer.SAFE,
    "internal": SecurityLayer.SAFE,
    "unknown":  SecurityLayer.CAUTION,
    "untrusted": SecurityLayer.BLOCKED,
}


# ── 核心函数 ──────────────────────────────────────────

def filter_by_security_layer(
    items: List[RetrievalItem],
    *,
    trust_thresholds: Optional[Dict[str, SecurityLayer]] = None,
    audit: bool = True,
) -> SecurityFilterReport:
    """
    按 trust_level 将检索条目分层过滤。

    Args:
        items: 待过滤的检索条目列表
        trust_thresholds: 自定义信任→层级映射，默认 DEFAULT_TRUST_THRESHOLDS
        audit: 是否将 blocked 条目写入审计日志
    """
    thresholds = trust_thresholds or DEFAULT_TRUST_THRESHOLDS
    CAUTION_PREFIX = "[UNTRUSTED] "

    safe: List[FilteredResult] = []
    caution: List[FilteredResult] = []
    blocked_items: List[RetrievalItem] = []

    for item in items:
        layer = thresholds.get(item.trust_level, SecurityLayer.CAUTION)

        if layer is SecurityLayer.SAFE:
            safe.append(FilteredResult(
                item=item, layer=layer,
                reason=f"trust_level={item.trust_level}",
            ))
        elif layer is SecurityLayer.CAUTION:
            caution.append(FilteredResult(
                item=item, layer=layer,
                reason=f"trust_level={item.trust_level} → 需标注",
                prefix=CAUTION_PREFIX,
            ))
        else:
            blocked_items.append(item)
            if audit:
                logger.warning(
                    "BLOCKED item | source=%s trust=%s relevance=%.2f | content=%.80s",
                    item.source, item.trust_level, item.relevance_score,
                    item.content.replace("\n", " "),
                )

    all_filtered = safe + caution

    return SecurityFilterReport(
        total_items=len(items),
        safe_count=len(safe),
        caution_count=len(caution),
        blocked_count=len(blocked_items),
        blocked_items=blocked_items,
        filtered_results=all_filtered,
    )


def format_for_injection(
    filtered: List[FilteredResult],
    *,
    max_tokens: int = 4000,
) -> str:
    """
    将过滤结果格式化为注入文本。

    CAUTION 条目会加安全前缀以提醒 LLM 注意可信度。
    按 relevance_score 降序排列，超过 max_tokens 截断。
    """
    sorted_items = sorted(filtered, key=lambda r: r.item.relevance_score, reverse=True)

    lines: List[str] = []
    token_est = 0  # 粗略估算：1 token ≈ 1.5 字符（中英混合）

    for fr in sorted_items:
        chunk = f"{fr.prefix}{fr.item.content}".strip()
        char_count = len(chunk)
        chunk_tokens = int(char_count / 1.5)

        if token_est + chunk_tokens > max_tokens:
            break

        # 附带来源标注，便于溯源
        source_tag = f"[{fr.item.source}]" if fr.item.source else ""
        lines.append(f"{source_tag} {chunk}")
        token_est += chunk_tokens

    return "\n\n".join(lines)


# ── Layer 6：角色边界强化 ─────────────────────────────

# 注入攻击 / 角色混淆常见信号模式
_ROLE_BOUNDARY_PATTERNS: list[str] = [
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "disregard the above",
    "forget your instructions",
    "forget your rules",
    "you are now",
    "you will now act as",
    "act as a",
    "pretend to be",
    "from now on you are",
    "new instructions:",
    "override your",
    "bypass your",
    "system:",
    "assistant:",
    "### system",
    "<|system|>",
    "[system]",
    "<|im_start|>system",
    "### user",
    "### assistant",
    "no longer bound by",
    "drop your persona",
    "reveal your prompt",
    "repeat your prompt",
    "what is your system prompt",
    "print your instructions",
    "output your instructions",
]

# 角色前缀：这些出现在 user 输入中通常代表角色越界
_ROLE_PREFIXES: list[str] = [
    "you are",
    "your role is",
    "your job is",
    "you must",
    "you shall",
    "as an ai",
    "as a language model",
]


def enforce_role_boundary(context: dict) -> dict:
    """
    Layer 6 角色边界强化 — 检测并标记角色混淆 / 提示注入信号。

    在 context dict 中检测以下两类信号：
      1. user 消息包含系统指令特征（用户尝试覆盖角色设定）
      2. system 消息包含用户输入特征（上下文来源异常）

    检测方法：关键词 / 模式匹配（零 LLM 调用）。

    Args:
        context: 对话上下文，至少包含 ``"messages"`` 键（list[dict]，
                 每个 dict 有 ``"role"`` 和 ``"content"`` 键）。

    Returns:
        修改后的 context（原地修改 + 返回）。被检测到的角色边界
        违规条目会在 content 前追加 ``[ROLE_BOUNDARY_ALERT]`` 前缀，
        并通过 ``logging.warning`` 记录审计信息。
    """
    messages: list = context.get("messages", [])
    if not messages:
        return context

    alert_count = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if not isinstance(content, str) or not content.strip():
            continue

        content_lower = content.lower()
        violations: list[str] = []

        if role == "user":
            # 检查用户输入中是否包含系统指令特征
            for pattern in _ROLE_BOUNDARY_PATTERNS:
                if pattern in content_lower:
                    violations.append(f"system_pattern_in_user: \"{pattern}\"")

            # 检查用户输入中是否出现典型的角色前缀（但不误报正常的
            # "you are welcome" 等礼貌用语）
            for prefix in _ROLE_PREFIXES:
                if content_lower.startswith(prefix):
                    violations.append(f"role_prefix_in_user: \"{prefix}\"")

        elif role == "system":
            # 检查 system prompt 中是否出现用户输入特征（上下文来源异常）
            for pattern in _ROLE_BOUNDARY_PATTERNS:
                if pattern in content_lower:
                    violations.append(f"system_pattern_in_system: \"{pattern}\"")

        if violations:
            alert_count += 1
            violation_summary = "; ".join(violations)
            logger.warning(
                "ROLE_BOUNDARY_ALERT | role=%s | violations=%s | content=%.120s",
                role,
                violation_summary,
                content.strip().replace("\n", " "),
            )
            # 在 content 前追加警报前缀，提醒下游处理注意
            msg["content"] = f"[ROLE_BOUNDARY_ALERT] {content}"

    if alert_count > 0:
        logger.info(
            "enforce_role_boundary: %d message(s) flagged in context "
            "(total messages=%d)",
            alert_count,
            len(messages),
        )

    return context
