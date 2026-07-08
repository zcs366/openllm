"""
Protocol — 六体通信协议
========================

定义六体架构（IAX+IAI+ISA+IOS+ISN+IKO）之间的消息传递协议。

核心设计：
- MessageEnvelope不可变信封（version+MessageType+source+target+payload+HMAC签名）
- 19种MessageType覆盖六体全部消息类型
- hash链审计——每条消息包含前一条的hash
- HMAC签名——防止消息在传输中被篡改

赫淮斯托斯修正：协议独立于六体实现，防止pipeline.py刚性耦合。
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── 默认HMAC密钥模板（实例化时从环境变量读取） ──
_DEFAULT_HMAC_KEY_TEMPLATE = "openllm-protocol-default-key-change-in-production"


class MessageType(Enum):
    """19种消息类型——覆盖六体全部通信模式。"""

    # IAX心跳
    HEARTBEAT_PING = "heartbeat.ping"          # 心跳探测
    HEARTBEAT_PONG = "heartbeat.pong"          # 心跳响应
    HEARTBEAT_ALERT = "heartbeat.alert"        # 心跳异常告警

    # IAI感知
    SEARCH_REQUEST = "search.request"          # 搜索请求
    SEARCH_RESULT = "search.result"            # 搜索结果
    PERCEPTION_REPORT = "perception.report"    # 感知报告

    # ISA记忆
    MEMORY_QUERY = "memory.query"              # 记忆查询
    MEMORY_RESULT = "memory.result"            # 记忆结果
    MEMORY_WRITE = "memory.write"              # 记忆写入
    PROVENANCE_RECORD = "provenance.record"    # 溯源记录

    # IOS决策
    DECISION_REQUEST = "decision.request"      # 决策请求
    DECISION_RESULT = "decision.result"        # 决策结果
    REJECTION_NOTICE = "rejection.notice"      # 拒绝通知
    GOVERNANCE_EVENT = "governance.event"      # 治理事件

    # ISN执行
    TOOL_INVOCATION = "tool.invocation"        # 工具调用
    TOOL_RESULT = "tool.result"                # 工具结果

    # IKO输出
    OUTPUT_RENDER = "output.render"            # 输出渲染
    OBSERVABILITY_LOG = "observability.log"    # 可观测日志

    # 系统
    ERROR = "error"                            # 错误消息
    SHUTDOWN = "shutdown"                      # 关机通知


@dataclass(frozen=True)
class MessageEnvelope:
    """不可变消息信封——六体间通信的基本单位。

    每条消息都是不可变的、带HMAC签名的、包含前一条消息hash的。
    这确保了通信链路的完整性和可审计性。
    """
    version: str                    # 协议版本
    message_type: MessageType       # 消息类型
    source: str                     # 发送方（六体名称）
    target: str                     # 接收方（六体名称）
    payload: dict[str, Any]         # 消息载荷
    timestamp: float                # 发送时间
    message_id: str                 # 唯一消息ID
    prev_hash: str                  # 前一条消息的hash（链式审计）
    hmac_signature: str = ""        # HMAC签名
    correlation_id: str = ""        # 关联ID（追踪请求-响应对）

    def compute_content_hash(self) -> str:
        """计算消息内容hash——用于链式审计。"""
        content = json.dumps({
            "version": self.version,
            "message_type": self.message_type.value,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def compute_hmac(self, key: Optional[bytes] = None) -> str:
        """计算HMAC签名——防篡改。"""
        if key is None:
            key = _DEFAULT_HMAC_KEY_TEMPLATE.encode()
        content = self.compute_content_hash()
        return hmac.new(key, content.encode(), hashlib.sha256).hexdigest()

    def verify_hmac(self, key: Optional[bytes] = None) -> bool:
        """验证HMAC签名。"""
        expected = self.compute_hmac(key)
        return hmac.compare_digest(self.hmac_signature, expected)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "version": self.version,
            "message_type": self.message_type.value,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "prev_hash": self.prev_hash,
            "hmac_signature": self.hmac_signature,
            "correlation_id": self.correlation_id,
            "_content_hash": self.compute_content_hash(),
        }


# ── 六体名称常量 ──

class BodyName(Enum):
    """六体名称。"""
    IAX = "IAX"   # 心跳
    IAI = "IAI"   # 感知
    ISA = "ISA"   # 记忆
    IOS = "IOS"   # 决策
    ISN = "ISN"   # 执行
    IKO = "IKO"   # 输出
    SYSTEM = "SYSTEM"  # 系统级


class Protocol:
    """六体通信协议引擎。

    用法：
        protocol = Protocol()
        envelope = protocol.send(
            source=BodyName.IAX,
            target=BodyName.IOS,
            message_type=MessageType.HEARTBEAT_PING,
            payload={"status": "alive"},
        )
        # 验证
        assert envelope.verify_hmac()
    """

    PROTOCOL_VERSION = "1.0.0"

    def __init__(self, hmac_key: Optional[bytes] = None):
        if hmac_key is not None:
            self._hmac_key = hmac_key
        else:
            env_key = os.environ.get("OPENLLM_HMAC_KEY")
            self._hmac_key = env_key.encode() if env_key else _DEFAULT_HMAC_KEY_TEMPLATE.encode()
        self._send_last_hash = "genesis"    # 发送链
        self._recv_last_hash = "genesis"    # 接收链（独立追踪）
        self._message_log: list[MessageEnvelope] = []
        self._message_count = 0

    def send(
        self,
        source: BodyName,
        target: BodyName,
        message_type: MessageType,
        payload: dict[str, Any],
        correlation_id: str = "",
    ) -> MessageEnvelope:
        """发送一条消息——构建信封、签名、记录。"""
        self._message_count += 1
        now = time.time()
        msg_id = f"msg-{self._message_count}-{int(now*1000)}"

        envelope = MessageEnvelope(
            version=self.PROTOCOL_VERSION,
            message_type=message_type,
            source=source.value,
            target=target.value,
            payload=payload,
            timestamp=now,
            message_id=msg_id,
            prev_hash=self._send_last_hash,
            correlation_id=correlation_id,
        )

        # 计算签名
        signature = envelope.compute_hmac(self._hmac_key)
        envelope = MessageEnvelope(
            version=envelope.version,
            message_type=envelope.message_type,
            source=envelope.source,
            target=envelope.target,
            payload=envelope.payload,
            timestamp=envelope.timestamp,
            message_id=envelope.message_id,
            prev_hash=envelope.prev_hash,
            hmac_signature=signature,
            correlation_id=envelope.correlation_id,
        )

        # 更新发送链
        self._send_last_hash = envelope.compute_content_hash()
        self._message_log.append(envelope)

        return envelope

    def receive(self, envelope: MessageEnvelope) -> bool:
        """接收并验证一条消息。"""
        # 验证HMAC
        if not envelope.verify_hmac(self._hmac_key):
            return False

        # 更新接收链（独立于发送链）
        self._recv_last_hash = envelope.compute_content_hash()
        return True

    def get_log(
        self,
        source: Optional[BodyName] = None,
        target: Optional[BodyName] = None,
        message_type: Optional[MessageType] = None,
    ) -> list[MessageEnvelope]:
        """查询消息日志。"""
        results = self._message_log
        if source:
            results = [m for m in results if m.source == source.value]
        if target:
            results = [m for m in results if m.target == target.value]
        if message_type:
            results = [m for m in results if m.message_type == message_type]
        return results

    def verify_chain(self) -> bool:
        """验证整条消息链的完整性。"""
        if not self._message_log:
            return True

        prev = "genesis"
        for msg in self._message_log:
            if msg.prev_hash != prev:
                return False
            if not msg.verify_hmac(self._hmac_key):
                return False
            prev = msg.compute_content_hash()
        return True

    def get_statistics(self) -> dict:
        """获取协议统计。"""
        type_counts = {}
        for m in self._message_log:
            type_counts[m.message_type.value] = type_counts.get(m.message_type.value, 0) + 1

        return {
            "total_messages": self._message_count,
            "chain_valid": self.verify_chain(),
            "type_distribution": type_counts,
        }
