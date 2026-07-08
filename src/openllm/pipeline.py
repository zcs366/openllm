"""
MessageQueue Pipeline — 六体消息队列调度器
==========================================

赫淮斯托斯修正：pipeline.py刚性耦合会炸。
替代方案：六体独立进程+消息队列通信。

每个六体通过Protocol发送/接收消息，调度器负责路由。
不再有10阶段硬编码流水线——每个体独立运行，消息驱动。

v2.0：增加路由+死信+优先级+中间件+重试。
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional, Union

from openllm.protocol import Protocol, MessageEnvelope, MessageType, BodyName

logger = logging.getLogger("openllm.pipeline")


class Priority(IntEnum):
    """消息优先级。"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class DeadLetter:
    """死信——处理失败的消息。"""
    envelope: MessageEnvelope
    error: str
    attempts: int
    timestamp: float
    handler_name: str = ""


class MessageQueue:
    """消息队列引擎——六体间异步通信。

    v2.0特性：
    - 路由：按MessageType分发到不同handler
    - 死信队列：处理失败的消息不丢弃，进入死信等待重试
    - 优先级：CRITICAL消息优先处理
    - 中间件：消息处理前后可插入钩子
    - 重试：失败消息可配置重试次数

    用法：
        mq = MessageQueue(protocol)
        mq.subscribe(BodyName.IOS, handler=ios_process)
        mq.add_route(MessageType.HEARTBEAT_PING, handler=handle_ping)
        mq.publish(source=BodyName.IAX, target=BodyName.IOS,
                   msg_type=MessageType.HEARTBEAT_PING, payload={"alive": True})
        mq.process_pending()
    """

    MAX_RETRIES = 3

    def __init__(self, protocol: Protocol):
        self._protocol = protocol
        self._queues: dict[str, list[MessageEnvelope]] = defaultdict(list)
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._routes: dict[str, Any] = {}
        self._dead_letters: list[DeadLetter] = []
        self._middlewares: list[Callable] = []
        self._processed_count = 0
        self._error_count = 0
        self._retry_count = 0

    def subscribe(self, body: BodyName, handler: Callable[[MessageEnvelope], Optional[dict]]):
        """按六体订阅——处理器处理该六体的所有消息。"""
        self._handlers[body.value].append(handler)

    def add_route(self, msg_type: MessageType, handler: Callable[[MessageEnvelope], Optional[dict]]):
        """按消息类型路由——特定消息类型由特定handler处理。"""
        self._routes[msg_type.value] = handler

    def add_middleware(self, middleware: Callable[[MessageEnvelope], MessageEnvelope]):
        """添加中间件——消息处理前的钩子。"""
        self._middlewares.append(middleware)

    def publish(
        self,
        source: BodyName,
        target: BodyName,
        msg_type: MessageType,
        payload: dict,
        correlation_id: str = "",
        priority: Priority = Priority.NORMAL,
    ) -> MessageEnvelope:
        """发布消息——优先级高的插到队列前面。"""
        envelope = self._protocol.send(
            source=source,
            target=target,
            message_type=msg_type,
            payload=payload,
            correlation_id=correlation_id,
        )

        # 中间件处理
        for mw in self._middlewares:
            envelope = mw(envelope)

        # 优先级排序：CRITICAL插到队首
        queue = self._queues[target.value]
        if priority >= Priority.HIGH:
            queue.insert(0, envelope)
        else:
            queue.append(envelope)

        return envelope

    def process_pending(self) -> int:
        """处理所有待处理消息。返回处理数量。"""
        total = 0
        for body_name, messages in list(self._queues.items()):
            while messages:
                envelope = messages.pop(0)
                handled = self._dispatch(envelope)
                if handled:
                    total += 1
        return total

    def _dispatch(self, envelope: MessageEnvelope) -> bool:
        """分发消息到正确的handler。优先路由匹配，其次body订阅。"""
        # 优先：按MessageType路由
        route_handler = self._routes.get(envelope.message_type.value)
        if route_handler:
            return self._safe_call(route_handler, envelope, f"route:{envelope.message_type.value}")

        # 其次：按目标六体订阅（全部handler都调用）
        body_handlers = self._handlers.get(envelope.target, [])
        if body_handlers:
            any_success = False
            for handler in body_handlers:
                if self._safe_call(handler, envelope, f"sub:{envelope.target}"):
                    any_success = True
            return any_success

        # 无handler→死信
        self._dead_letters.append(DeadLetter(
            envelope=envelope,
            error="no_handler",
            attempts=0,
            timestamp=time.time(),
        ))
        return False

    def _safe_call(self, handler: Callable, envelope: MessageEnvelope, handler_name: str) -> bool:
        """安全调用handler——捕获异常，失败进入死信。"""
        try:
            handler(envelope)
            self._processed_count += 1
            return True
        except Exception as e:
            self._error_count += 1
            self._dead_letters.append(DeadLetter(
                envelope=envelope,
                error=str(e),
                attempts=1,
                timestamp=time.time(),
                handler_name=handler_name,
            ))
            logger.warning(f"Handler {handler_name} failed: {e}")
            return False

    def retry_dead_letters(self) -> int:
        """重试死信——最多重试MAX_RETRIES次。"""
        retried = 0
        surviving = []
        for dl in self._dead_letters:
            if dl.attempts < self.MAX_RETRIES:
                dl.attempts += 1
                self._retry_count += 1
                # 重新入队到对应目标
                self._queues[dl.envelope.target].insert(0, dl.envelope)
                retried += 1
            else:
                surviving.append(dl)
        self._dead_letters = surviving
        return retried

    def get_dead_letters(self) -> list[DeadLetter]:
        return list(self._dead_letters)

    def get_statistics(self) -> dict:
        return {
            "total_published": self._processed_count + sum(len(q) for q in self._queues.values()),
            "total_processed": self._processed_count,
            "total_errors": self._error_count,
            "total_retries": self._retry_count,
            "dead_letters": len(self._dead_letters),
            "pending": sum(len(q) for q in self._queues.values()),
            "subscribers": {k: len(v) for k, v in self._handlers.items()},
            "routes": list(self._routes.keys()),
            "middlewares": len(self._middlewares),
        }
