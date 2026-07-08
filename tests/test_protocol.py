"""
Tests for Protocol — 六体通信协议测试
=====================================

12个测试用例覆盖：
1. 基本消息发送
2. HMAC签名验证
3. 消息链连续性
4. 篡改检测
5. 19种消息类型
6. 六体名称
7. 查询和过滤
8. 链完整性验证
9. 统计功能
10. correlation_id追踪
11. 创世hash
12. 多消息链
"""

import pytest
from openllm.protocol import (
    Protocol,
    MessageEnvelope,
    MessageType,
    BodyName,
)


class TestProtocol:
    """六体通信协议测试。"""

    def test_basic_send(self):
        """T1: 基本消息发送——构建信封并签名。"""
        proto = Protocol()
        envelope = proto.send(
            source=BodyName.IAX,
            target=BodyName.IOS,
            message_type=MessageType.HEARTBEAT_PING,
            payload={"status": "alive"},
        )
        assert envelope.source == "IAX"
        assert envelope.target == "IOS"
        assert envelope.message_type == MessageType.HEARTBEAT_PING
        assert envelope.hmac_signature != ""

    def test_hmac_verification(self):
        """T2: HMAC签名验证——正确签名通过，篡改失败。"""
        proto = Protocol()
        envelope = proto.send(
            source=BodyName.IAX,
            target=BodyName.IOS,
            message_type=MessageType.HEARTBEAT_PING,
            payload={"status": "alive"},
        )
        # 正确签名验证通过
        assert envelope.verify_hmac() is True

        # 篡改payload后验证失败
        tampered = MessageEnvelope(
            version=envelope.version,
            message_type=envelope.message_type,
            source=envelope.source,
            target=envelope.target,
            payload={"status": "dead"},  # 篡改
            timestamp=envelope.timestamp,
            message_id=envelope.message_id,
            prev_hash=envelope.prev_hash,
            hmac_signature=envelope.hmac_signature,  # 旧签名
        )
        assert tampered.verify_hmac() is False

    def test_message_chain_continuity(self):
        """T3: 消息链连续性——prev_hash指向前一条。"""
        proto = Protocol()
        msg1 = proto.send(
            source=BodyName.IAX,
            target=BodyName.IOS,
            message_type=MessageType.HEARTBEAT_PING,
            payload={"seq": 1},
        )
        msg2 = proto.send(
            source=BodyName.IOS,
            target=BodyName.IAX,
            message_type=MessageType.HEARTBEAT_PONG,
            payload={"seq": 2},
        )
        # msg2的prev_hash应该是msg1的content_hash
        assert msg2.prev_hash == msg1.compute_content_hash()

    def test_tamper_detection(self):
        """T4: 篡改检测——receive验证失败。"""
        proto = Protocol()
        envelope = proto.send(
            source=BodyName.IAX,
            target=BodyName.IOS,
            message_type=MessageType.HEARTBEAT_PING,
            payload={"data": "original"},
        )

        # 正常接收
        assert proto.receive(envelope) is True

        # 篡改后接收
        tampered = MessageEnvelope(
            version=envelope.version,
            message_type=envelope.message_type,
            source=envelope.source,
            target=envelope.target,
            payload={"data": "tampered"},
            timestamp=envelope.timestamp,
            message_id=envelope.message_id,
            prev_hash=envelope.prev_hash,
            hmac_signature=envelope.hmac_signature,
        )
        assert proto.receive(tampered) is False

    def test_all_message_types(self):
        """T5: 19种消息类型——全部可创建。"""
        proto = Protocol()
        types = list(MessageType)
        assert len(types) == 20

        for msg_type in types:
            envelope = proto.send(
                source=BodyName.SYSTEM,
                target=BodyName.SYSTEM,
                message_type=msg_type,
                payload={"test": msg_type.value},
            )
            assert envelope.message_type == msg_type

    def test_all_body_names(self):
        """T6: 六体名称——7个（含SYSTEM）。"""
        names = list(BodyName)
        assert len(names) == 7
        expected = {"IAX", "IAI", "ISA", "IOS", "ISN", "IKO", "SYSTEM"}
        assert {n.value for n in names} == expected

    def test_query_and_filter(self):
        """T7: 查询和过滤功能。"""
        proto = Protocol()
        proto.send(source=BodyName.IAX, target=BodyName.IOS, message_type=MessageType.HEARTBEAT_PING, payload={})
        proto.send(source=BodyName.ISA, target=BodyName.IOS, message_type=MessageType.MEMORY_QUERY, payload={})
        proto.send(source=BodyName.IAX, target=BodyName.ISA, message_type=MessageType.HEARTBEAT_PING, payload={})

        # 按source过滤
        iax_msgs = proto.get_log(source=BodyName.IAX)
        assert len(iax_msgs) == 2

        # 按target过滤
        ios_msgs = proto.get_log(target=BodyName.IOS)
        assert len(ios_msgs) == 2

        # 按类型过滤
        ping_msgs = proto.get_log(message_type=MessageType.HEARTBEAT_PING)
        assert len(ping_msgs) == 2

    def test_chain_integrity_verification(self):
        """T8: 链完整性验证——10条消息全部通过。"""
        proto = Protocol()
        for i in range(10):
            proto.send(
                source=BodyName.IAX,
                target=BodyName.IOS,
                message_type=MessageType.HEARTBEAT_PING,
                payload={"seq": i},
            )
        assert proto.verify_chain() is True

    def test_statistics(self):
        """T9: 统计功能。"""
        proto = Protocol()
        proto.send(source=BodyName.IAX, target=BodyName.IOS, message_type=MessageType.HEARTBEAT_PING, payload={})
        proto.send(source=BodyName.ISA, target=BodyName.IOS, message_type=MessageType.MEMORY_QUERY, payload={})

        stats = proto.get_statistics()
        assert stats["total_messages"] == 2
        assert stats["chain_valid"] is True
        assert "heartbeat.ping" in stats["type_distribution"]

    def test_correlation_id(self):
        """T10: correlation_id追踪——请求-响应配对。"""
        proto = Protocol()
        req = proto.send(
            source=BodyName.IAX,
            target=BodyName.IOS,
            message_type=MessageType.DECISION_REQUEST,
            payload={"question": "approve?"},
            correlation_id="corr-001",
        )
        resp = proto.send(
            source=BodyName.IOS,
            target=BodyName.IAX,
            message_type=MessageType.DECISION_RESULT,
            payload={"decision": "approve"},
            correlation_id="corr-001",
        )
        assert req.correlation_id == resp.correlation_id == "corr-001"

    def test_genesis_hash(self):
        """T11: 创世hash——第一条消息的prev_hash是'genesis'。"""
        proto = Protocol()
        msg = proto.send(
            source=BodyName.IAX,
            target=BodyName.IOS,
            message_type=MessageType.HEARTBEAT_PING,
            payload={},
        )
        assert msg.prev_hash == "genesis"

    def test_multi_message_chain(self):
        """T12: 多消息链——50条消息链完整。"""
        proto = Protocol()
        for i in range(50):
            proto.send(
                source=BodyName.IAX,
                target=BodyName.IOS,
                message_type=MessageType.HEARTBEAT_PING,
                payload={"seq": i},
            )
        assert proto.verify_chain() is True
        stats = proto.get_statistics()
        assert stats["total_messages"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
