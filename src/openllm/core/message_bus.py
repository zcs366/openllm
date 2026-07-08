"""
MessageQueue — 六体消息总线
============================

六体通过publish/subscribe通信，不再直接调用函数。
每条血管=一种MessageType+handler对。

用法：
    bus = MessageQueue()
    bus.subscribe(MessageType.MEMORY_WRITE, isa_handler)
    bus.publish(MessageType.MEMORY_WRITE, source="IOS", payload={...})
"""

import logging
from collections import defaultdict
from typing import Callable, Optional

from ..protocol import MessageType, MessageEnvelope, BodyName, Protocol

logger = logging.getLogger(__name__)


class MessageQueue:
    """六体消息总线——publish/subscribe模式。"""

    def __init__(self, protocol: Optional[Protocol] = None):
        self._protocol = protocol or Protocol()
        self._handlers: dict[MessageType, list[Callable]] = defaultdict(list)
        self._stats: dict[str, int] = defaultdict(int)

    def subscribe(self, message_type: MessageType, handler: Callable):
        """注册消息处理器。"""
        self._handlers[message_type].append(handler)
        logger.debug(f"订阅 {message_type.value} → {handler.__qualname__}")

    def publish(
        self,
        message_type: MessageType,
        source: BodyName,
        target: BodyName,
        payload: dict,
        correlation_id: str = "",
    ) -> MessageEnvelope:
        """发布消息——构建信封、路由到handler、记录。"""
        envelope = self._protocol.send(
            source=source,
            target=target,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
        )

        # 路由到handler
        handlers = self._handlers.get(message_type, [])
        if not handlers:
            logger.debug(f"无handler: {message_type.value}")
        for handler in handlers:
            try:
                handler(envelope)
                self._stats[message_type.value] += 1
            except Exception as e:
                logger.error(f"handler异常 {message_type.value}: {e}")

        return envelope

    def get_stats(self) -> dict:
        """获取总线统计。"""
        return {
            "total_published": sum(self._stats.values()),
            "by_type": dict(self._stats),
            "handler_count": {mt.value: len(h) for mt, h in self._handlers.items()},
            "chain_valid": self._protocol.verify_chain(),
        }
