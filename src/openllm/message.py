"""
Message — 核心消息数据结构
==========================

openLLM中所有上下文管理的基础数据结构。

字段：
- role: 消息角色（user/assistant/system/tool）
- content: 消息内容
- metadata: 附加元数据（可选）
- timestamp: 创建时间戳（Unix秒）

方法：
- from_dict(d: dict) -> Message：从字典反序列化
- to_dict() -> dict：序列化为字典
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Message:
    """openLLM核心消息结构——上下文管理的基础单元。"""

    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """验证字段合法性。"""
        valid_roles = {"user", "assistant", "system", "tool"}
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role: {self.role!r}. Must be one of {valid_roles}"
            )
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, got {type(self.content).__name__}")

    # ── 序列化 ──

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    # ── 反序列化 ──

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """从字典创建Message。"""
        if "role" not in d:
            raise KeyError("Missing required key: 'role'")
        if "content" not in d:
            raise KeyError("Missing required key: 'content'")
        return cls(
            role=d["role"],
            content=d["content"],
            metadata=d.get("metadata"),
            timestamp=d.get("timestamp", time.time()),
        )
