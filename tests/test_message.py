"""
Tests for Message — 核心消息数据结构测试
=========================================

3个测试用例覆盖：
1. 构造与验证（valid/invalid role, content type）
2. 序列化（to_dict）
3. 反序列化（from_dict）及往返一致性
"""

import time
import pytest
from openllm.message import Message


class TestMessageConstruction:
    """消息构造与字段验证。"""

    def test_basic_construction(self):
        """基本构造：role+content，timestamp自动填充。"""
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}

    def test_all_valid_roles(self):
        """所有合法角色均可构造。"""
        for role in ("user", "assistant", "system", "tool"):
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_invalid_role_raises(self):
        """非法角色抛出ValueError。"""
        with pytest.raises(ValueError, match="Invalid role"):
            Message(role="admin", content="hello")

    def test_content_must_be_str(self):
        """content非str抛出TypeError。"""
        with pytest.raises(TypeError, match="content must be str"):
            Message(role="user", content=123)

    def test_metadata_optional(self):
        """metadata可传入自定义字典。"""
        msg = Message(role="user", content="hi", metadata={"key": "val"})
        assert msg.metadata == {"key": "val"}


class TestMessageSerialization:
    """序列化 to_dict。"""

    def test_to_dict_minimal(self):
        """最小字段序列化（无metadata）。"""
        msg = Message(role="user", content="hi", timestamp=1000.0)
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hi", "timestamp": 1000.0}
        assert "metadata" not in d

    def test_to_dict_with_metadata(self):
        """带metadata序列化。"""
        msg = Message(role="assistant", content="ok", metadata={"tok": 42})
        d = msg.to_dict()
        assert d["metadata"] == {"tok": 42}


class TestMessageDeserialization:
    """反序列化 from_dict 及往返一致性。"""

    def test_from_dict_minimal(self):
        """最少字段反序列化。"""
        d = {"role": "user", "content": "hi"}
        msg = Message.from_dict(d)
        assert msg.role == "user"
        assert msg.content == "hi"
        assert msg.metadata is None

    def test_from_dict_full(self):
        """完整字段反序列化。"""
        d = {"role": "assistant", "content": "ok", "metadata": {"x": 1}, "timestamp": 999.0}
        msg = Message.from_dict(d)
        assert msg.role == "assistant"
        assert msg.metadata == {"x": 1}
        assert msg.timestamp == 999.0

    def test_from_dict_missing_role_raises(self):
        """缺少role抛KeyError。"""
        with pytest.raises(KeyError):
            Message.from_dict({"content": "hi"})

    def test_from_dict_missing_content_raises(self):
        """缺少content抛KeyError。"""
        with pytest.raises(KeyError):
            Message.from_dict({"role": "user"})

    def test_roundtrip(self):
        """序列化→反序列化往返一致性。"""
        msg = Message(role="user", content="hello", metadata={"k": "v"}, timestamp=123.4)
        restored = Message.from_dict(msg.to_dict())
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.metadata == msg.metadata
        assert restored.timestamp == msg.timestamp
