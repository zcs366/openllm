"""
OpenLLM Phase 1 测试 — Provider + Tools + Engine + CLI。
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from openllm.core.provider import (
    DeepSeekProvider, ModelConfig, Message, ModelResponse,
    create_provider,
)
from openllm.tools.executor import (
    ToolRegistry, ToolResult, create_default_tools,
    tool_read_file, tool_write_file, tool_shell, tool_search_files,
)
from openllm.core.engine import OpenLLMEngine, AgentConfig


# ── Provider 测试 ────────────────────────────────

class TestModelConfig:
    def test_defaults(self):
        c = ModelConfig()
        assert c.provider == "deepseek"
        assert c.model == "deepseek-chat"

    def test_env_key(self):
        os.environ["DEEPSEEK_API_KEY"] = "test-key-123"
        c = ModelConfig()
        assert c.api_key == "test-key-123"
        del os.environ["DEEPSEEK_API_KEY"]


class TestMessage:
    def test_create(self):
        m = Message(role="user", content="你好")
        assert m.role == "user"
        assert m.content == "你好"


class TestProvider:
    def test_create_deepseek(self):
        p = create_provider("deepseek", api_key="test")
        assert p is not None

    def test_mock_chat(self):
        """模拟API响应——不真调API。"""
        p = create_provider("deepseek", api_key="test")
        # 不调用真实API，只验证对象构造
        assert p.config.model == "deepseek-chat"

    def test_build_messages(self):
        p = create_provider("deepseek", api_key="test")
        msgs = p._build_messages(
            [Message(role="user", content="hi")],
            system="You are helpful.",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


# ── Tools 测试 ────────────────────────────────────

class TestToolReadFile:
    def test_read_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3")
            path = f.name
        try:
            result = tool_read_file(path)
            assert "line1" in result
            assert "line2" in result
        finally:
            os.unlink(path)

    def test_read_nonexistent(self):
        result = tool_read_file("/tmp/nonexistent_xyz123.txt")
        assert "不存在" in result


class TestToolWriteFile:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            result = tool_write_file(str(path), "hello world")
            assert "已写入" in result
            assert path.read_text() == "hello world"


class TestToolShell:
    def test_echo(self):
        result = tool_shell("echo hello")
        assert "hello" in result

    def test_timeout(self):
        result = tool_shell("sleep 5", timeout=1)
        assert "超时" in result


class TestToolSearch:
    def test_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "test.py").write_text("def hello(): pass")
            result = tool_search_files("hello", str(p))
            assert "hello" in result


class TestToolRegistry:
    def test_register_and_execute(self):
        r = ToolRegistry()
        r.register("echo", lambda x: x, "回显")
        result = r.execute("echo", x="hi")
        assert result.success
        assert result.output == "hi"

    def test_list_tools(self):
        r = create_default_tools()
        tools = r.list_tools()
        assert len(tools) >= 4
        names = [t["name"] for t in tools]
        assert "read_file" in names
        assert "shell" in names

    def test_unknown_tool(self):
        r = ToolRegistry()
        result = r.execute("nonexistent")
        assert not result.success
        assert "未知工具" in result.error


# ── Engine 测试 ───────────────────────────────────

class TestAgentConfig:
    def test_defaults(self):
        c = AgentConfig()
        assert c.name == "OpenLLM"
        assert c.model == "deepseek-chat"


class TestOpenLLMEngine:
    def _engine_without_keys(self, monkeypatch, tmp):
        """构造"确实没有可用 key"的引擎（2026-09-16 修）。

        原用例假设「没有 key 就是未连接」——但现实已多 provider + keyvault 取 key，
        本机 keyvault 里有 gateway/mimo，引擎会真的连上（测试前提失效）。
        显式钉住三个来源：① engine_utils.get_api_key（函数内 import，须 patch 定义处）
        ② 环境变量 ③ provider 显式指定，避开 ~/.openllm/config.json 的 default_provider 解析。
        """
        from openllm.core import engine_utils
        monkeypatch.setattr(engine_utils, "get_api_key", lambda *a, **k: "")
        for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "MIMO_API_KEY",
                    "ALIBABA_PLAN_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        config = AgentConfig(capsule_dir=tmp, provider="deepseek")
        return OpenLLMEngine(config)

    def test_init_no_api(self, monkeypatch):
        """无可用 key 时仍可初始化（connected=False）。"""
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine_without_keys(monkeypatch, tmp)
            assert not engine.connected
            assert engine.config.name == "OpenLLM"

    def test_wake(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(capsule_dir=tmp)
            engine = OpenLLMEngine(config)
            msg = engine.wake()
            assert "苏醒" in msg or "已苏醒" in msg

    def test_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(capsule_dir=tmp)
            engine = OpenLLMEngine(config)
            s = engine.status()
            assert s["name"] == "OpenLLM"
            assert "state" in s
            assert "capsules" in s

    def test_chat_no_api(self, monkeypatch):
        """无可用 key 时 chat 返回友好提示（走 [未连接API] 分支）。"""
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine_without_keys(monkeypatch, tmp)
            resp = engine.chat("你好")
            assert "未连接" in resp or "API" in resp

    def test_execute_tool_with_security(self):
        """工具执行经过安全检查。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(capsule_dir=tmp)
            engine = OpenLLMEngine(config)
            # shell需要L2权限（默认有），应允许
            result = engine.execute_tool("shell", command="echo test")
            assert result.success

    def test_sleep_cycle(self):
        """sleep→写胶囊→文件存在。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(capsule_dir=tmp)
            engine = OpenLLMEngine(config)
            engine.wake()
            msg = engine.sleep()
            assert "已保存" in msg
            # 检查文件
            v06 = list(Path(tmp).glob("v06_*.json"))
            assert len(v06) >= 1

    def test_full_wake_sleep_wake_cycle(self):
        """两次wake→sleep→wake：记忆恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            # Session 1: 写入一些有意义的内容
            from openllm.isa.capsule import MemoryOS, TextCapsule, DeltaCapsule
            mos = MemoryOS(Path(tmp))
            text = TextCapsule(
                session_id="s1",
                decisions=[{"summary": "测试了完整循环"}],
                insights=["Agent需要六维躯体"],
            )
            delta = DeltaCapsule(
                session_id="s1",
                vector=np.random.randn(256).astype(np.float32) * 0.01,
            )
            mos.write(text, delta)

            # Session 2: 使用同一个capsule目录
            config = AgentConfig(capsule_dir=tmp)
            engine2 = OpenLLMEngine(config)
            msg = engine2.wake()
            assert "记忆" in msg


# ── Security + Tools 集成 ────────────────────────

class TestSecurityToolsIntegration:
    def test_read_allowed(self):
        from openllm.security.gate import SecurityFoundation
        sf = SecurityFoundation()
        ok, _ = sf.check_action("read_file")
        assert ok

    def test_write_allowed_at_l2(self):
        from openllm.security.gate import SecurityFoundation
        sf = SecurityFoundation()
        ok, _ = sf.check_action("write_file")
        assert ok

    def test_immutable_denied(self):
        from openllm.security.gate import SecurityFoundation
        sf = SecurityFoundation()
        ok, reason = sf.check_action("disable_security")
        assert not ok
        assert "安全基座" in reason
