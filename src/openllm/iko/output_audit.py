"""
Output Audit — IKO 输出审计链
=============================

为 IKO（输出体）的每一次输出建立不可变审计记录，形成链式哈希审计链。
每条审计条目包含：
- output_id: 输出唯一标识
- intent: 输出意图描述
- content_hash: 输出内容的 sha256[:16] 摘要
- decision_source: 决策来源（哪个六体/模块触发了此输出）
- risk_level: 风险等级（0-1 浮点数）
- confidence: 输出置信度（0-1 浮点数）
- prev_hash: 前一条审计条目的哈希（链式链接）
- timestamp: 时间戳
- reasoning_chain_hash: 推理链哈希（赫淮斯托斯约束：必须有此字段）

链式链接遵循 protocol.py 的 prev_hash 模式：genesis → hash₁ → hash₂ → ...

阿瑞斯约束：审计链必须能被外部工具独立验证。

用法：
    chain = OutputAuditChain()
    entry = chain.append(
        output_id="out-001",
        intent="回答用户关于Python的问题",
        content=b"Python is a programming language",
        decision_source="IKO",
        risk_level=0.1,
        confidence=0.95,
        reasoning_chain_hash=compute_reasoning_hash("step1", "step2"),
    )
    assert chain.verify()
    provenance = chain.get_provenance("out-001")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional


def _sha256_truncate(data: bytes, length: int = 16) -> str:
    """计算 sha256 并截取前 length 个十六进制字符。

    Args:
        data: 要哈希的字节数据。
        length: 截取的十六进制字符数，默认 16。

    Returns:
        截断的哈希字符串。
    """
    return hashlib.sha256(data).hexdigest()[:length]


def compute_entry_hash(entry: OutputAuditEntry) -> str:
    """计算审计条目的完整哈希（用于 prev_hash 链接）。

    Args:
        entry: 审计条目。

    Returns:
        该条目的 sha256 哈希（完整 64 字符）。
    """
    content = json.dumps({
        "output_id": entry.output_id,
        "intent": entry.intent,
        "content_hash": entry.content_hash,
        "decision_source": entry.decision_source,
        "risk_level": entry.risk_level,
        "confidence": entry.confidence,
        "prev_hash": entry.prev_hash,
        "timestamp": entry.timestamp,
        "reasoning_chain_hash": entry.reasoning_chain_hash,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


@dataclass(frozen=True)
class OutputAuditEntry:
    """不可变审计条目——每次 IKO 输出的审计快照。

    frozen=True 保证条目创建后不可修改，确保审计不可变性。
    reasoning_chain_hash 字段满足赫淮斯托斯约束：记录推理过程的哈希。

    Attributes:
        output_id: 输出唯一标识符。
        intent: 输出的意图/目的描述。
        content_hash: 输出内容的 sha256[:16] 摘要。
        decision_source: 决策来源标识（如 "IKO", "IOS:decision.result"）。
        risk_level: 风险等级，0.0（无风险）到 1.0（高风险）。
        confidence: 置信度，0.0（最低）到 1.0（最高）。
        prev_hash: 前一条审计条目的哈希，首条为 "genesis"。
        timestamp: 创建时间戳（Unix epoch seconds）。
        reasoning_chain_hash: 推理链哈希，记录生成此输出的推理过程。
    """

    output_id: str
    intent: str
    content_hash: str
    decision_source: str
    risk_level: float
    confidence: float
    prev_hash: str
    timestamp: float
    reasoning_chain_hash: str


class OutputAuditChain:
    """输出审计链——管理有序的审计条目链。

    维护一条通过 prev_hash 链接的不可变审计链。
    每次 append 自动计算条目哈希并链接到前一条。
    verify() 可独立验证链的完整性。

    Attributes:
        entries: 链中的所有审计条目（有序）。
        genesis_hash: 链的起始哈希，固定为 "genesis"。
    """

    genesis_hash: str = "genesis"

    def __init__(self) -> None:
        """初始化空审计链。"""
        self._entries: list[OutputAuditEntry] = []
        self._last_hash: str = self.genesis_hash
        self._index: dict[str, OutputAuditEntry] = {}

    @property
    def entries(self) -> list[OutputAuditEntry]:
        """返回链中所有审计条目的只读副本。"""
        return list(self._entries)

    def append(
        self,
        output_id: str,
        intent: str,
        content: bytes,
        decision_source: str,
        risk_level: float,
        confidence: float,
        reasoning_chain_hash: str,
    ) -> OutputAuditEntry:
        """向审计链追加一条新条目。

        自动生成 content_hash（sha256[:16]）和 prev_hash（引用前一条的完整哈希）。
        首条条目的 prev_hash 为 "genesis"。

        Args:
            output_id: 输出唯一标识。
            intent: 输出意图。
            content: 输出原始内容（用于计算 content_hash）。
            decision_source: 决策来源。
            risk_level: 风险等级 0.0-1.0。
            confidence: 置信度 0.0-1.0。
            reasoning_chain_hash: 推理链哈希。

        Returns:
            创建的审计条目。

        Raises:
            ValueError: 如果 output_id 已存在（防重复）。
        """
        if output_id in self._index:
            raise ValueError(f"Duplicate output_id: {output_id}")

        content_hash = _sha256_truncate(content)
        timestamp = time.time()

        entry = OutputAuditEntry(
            output_id=output_id,
            intent=intent,
            content_hash=content_hash,
            decision_source=decision_source,
            risk_level=risk_level,
            confidence=confidence,
            prev_hash=self._last_hash,
            timestamp=timestamp,
            reasoning_chain_hash=reasoning_chain_hash,
        )

        # 计算条目完整哈希，更新链
        self._last_hash = compute_entry_hash(entry)
        self._entries.append(entry)
        self._index[output_id] = entry

        return entry

    def verify(self) -> bool:
        """验证整条审计链的完整性。

        检查：
        1. 每条条目的 prev_hash 是否等于前一条的完整哈希
        2. 首条条目的 prev_hash 是否为 "genesis"
        3. 每条条目的 content_hash 格式正确（16 字符十六进制）

        Returns:
            链完整返回 True，否则 False。
        """
        if not self._entries:
            return True

        expected_prev = self.genesis_hash
        for entry in self._entries:
            # 验证 prev_hash 链接
            if entry.prev_hash != expected_prev:
                return False

            # 验证 content_hash 格式（16 字符十六进制）
            if len(entry.content_hash) != 16:
                return False
            try:
                int(entry.content_hash, 16)
            except ValueError:
                return False

            # 计算当前条目的哈希，作为下一条的 expected_prev
            expected_prev = compute_entry_hash(entry)

        return True

    def get_provenance(self, output_id: str) -> list[OutputAuditEntry]:
        """获取指定输出的完整溯源链。

        从 output_id 开始，沿 prev_hash 反向追溯，返回从 genesis 到
        该条目的所有审计条目（正序）。

        Args:
            output_id: 要查询的输出标识。

        Returns:
            从 genesis 到指定 output_id 的审计条目列表（正序）。
            如果 output_id 不存在，返回空列表。

        Raises:
            KeyError: 如果 output_id 不在链中。
        """
        if output_id not in self._index:
            raise KeyError(f"output_id not found: {output_id}")

        # 沿 prev_hash 反向追溯
        chain: list[OutputAuditEntry] = []
        current = self._index[output_id]
        chain.append(current)

        while current.prev_hash != self.genesis_hash:
            # 找到 prev_hash 对应的条目
            found = False
            for entry in self._entries:
                if compute_entry_hash(entry) == current.prev_hash:
                    chain.append(entry)
                    current = entry
                    found = True
                    break
            if not found:
                break

        # 反转得到正序（genesis → ... → target）
        chain.reverse()
        return chain

    def __len__(self) -> int:
        """返回链中条目数。"""
        return len(self._entries)

    def __repr__(self) -> str:
        """返回链的字符串表示。"""
        return f"OutputAuditChain(entries={len(self._entries)}, last_hash={self._last_hash[:8]}...)"
