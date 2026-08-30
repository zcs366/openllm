"""test_tool_calling.py — 工具循环断裂（P0）修复测试。

覆盖四个主断点：
  1. DeepSeekProvider.chat 收到 tools/tool_choice 且 payload 含 tools 字段
  2. _sync_chat 解析 OpenAI 原生 tool_calls（content 为 null 时也正常）
  3. _stream_chat 按 index 合并分片 tool_calls（OpenAI 流式格式）
  4. 引擎 chat() 收到结构化 tool_calls → 执行工具 → 结果注入 history → 继续循环；
     无结构化 tool_calls 时回落现有正则路径（向后兼容）

全部 mock，不发真实网络请求。
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openllm.core.provider import (
    DeepSeekProvider, ModelConfig, ModelResponse, Message,
)
from openllm.core.engine import AgentConfig, OpenLLMEngine


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """家目录重定向到 tmp_path + 清空API key环境变量。

    防止 _record_model_trace 写真实 ~/.io-s/traces/、checkpoint 写真实
    ~/.openllm/auto_checkpoint.json。
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) + p.lstrip("~"))
    for var in (
        "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY", "MIMO_API_KEY", "ALIBABA_PLAN_API_KEY",
        "OPENLLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# ── 工具声明 ─────────────────────────────────────

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取文件内容",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}]


class _FakeResp:
    """同步响应桩。"""
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeStreamResp:
    """流式响应桩：iter_lines 产出 SSE 行。"""
    def __init__(self, lines: list[str]):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for l in self._lines:
            yield l.encode("utf-8")


# ═══ A. Provider 层 ═══

class TestProviderPayload:
    def test_chat_payload_includes_tools_and_tool_choice(self, monkeypatch):
        provider = DeepSeekProvider(ModelConfig(api_key="test-key"))
        captured = {}

        def fake_post(url, json=None, **kw):
            captured["payload"] = json
            return _FakeResp({
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "model": "m", "usage": {},
            })

        monkeypatch.setattr(provider._session, "post", fake_post)
        resp = provider.chat(
            messages=[Message(role="user", content="读取文件")],
            tools=_TOOLS, tool_choice="auto",
        )
        assert captured["payload"]["tools"] == _TOOLS
        assert captured["payload"]["tool_choice"] == "auto"
        assert resp.content == "hi"

    def test_chat_payload_omits_tools_when_absent(self, monkeypatch):
        provider = DeepSeekProvider(ModelConfig(api_key="test-key"))
        captured = {}

        def fake_post(url, json=None, **kw):
            captured["payload"] = json
            return _FakeResp({
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "model": "m", "usage": {},
            })

        monkeypatch.setattr(provider._session, "post", fake_post)
        provider.chat(messages=[Message(role="user", content="你好")])
        assert "tools" not in captured["payload"]
        assert "tool_choice" not in captured["payload"]


class TestSyncToolCalls:
    def test_sync_chat_parses_tool_calls(self, monkeypatch):
        """同步响应：message.tool_calls 解析进 ModelResponse；content=null 时为空串。"""
        provider = DeepSeekProvider(ModelConfig(api_key="test-key"))

        def fake_post(url, json=None, **kw):
            return _FakeResp({
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "model": "m", "usage": {},
            })

        monkeypatch.setattr(provider._session, "post", fake_post)
        resp = provider.chat(
            messages=[Message(role="user", content="读取文件")],
            tools=_TOOLS, tool_choice="auto",
        )
        assert resp.content == ""
        assert resp.finish_reason == "tool_calls"
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp/x"}

    def test_sync_chat_no_tool_calls_field(self, monkeypatch):
        """普通响应：tool_calls 为 None。"""
        provider = DeepSeekProvider(ModelConfig(api_key="test-key"))

        def fake_post(url, json=None, **kw):
            return _FakeResp({
                "choices": [{"message": {"content": "普通回答"}, "finish_reason": "stop"}],
                "model": "m", "usage": {},
            })

        monkeypatch.setattr(provider._session, "post", fake_post)
        resp = provider.chat(messages=[Message(role="user", content="你好")])
        assert resp.tool_calls is None


class TestStreamToolCalls:
    def test_stream_chat_accumulates_fragmented_tool_calls(self, monkeypatch):
        """流式：分片 tool_calls（index/name/arguments 增量）按 index 合并还原。"""
        provider = DeepSeekProvider(ModelConfig(api_key="test-key"))
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":null,'
            '"tool_calls":[{"index":0,"id":"call_9","type":"function",'
            '"function":{"name":"read_file","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"{\\"path\\": \\""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"/tmp/hello.txt\\"}"}}]}}]}',
            'data: [DONE]',
        ]
        monkeypatch.setattr(
            provider._session, "post",
            lambda url, json=None, **kw: _FakeStreamResp(lines),
        )

        tokens = []
        resp = provider.chat(
            messages=[Message(role="user", content="读取文件")],
            stream=True, on_token=lambda t: tokens.append(t),
            tools=_TOOLS, tool_choice="auto",
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc["id"] == "call_9"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        # arguments 增量片段拼接后是合法JSON
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp/hello.txt"}

    def test_stream_chat_content_and_tool_calls_coexist(self, monkeypatch):
        """流式：内容token与tool_calls同时出现时两者都保留。"""
        provider = DeepSeekProvider(ModelConfig(api_key="test-key"))
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":"我先看看"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
            '"type":"function","function":{"name":"list_dir","arguments":"{\\"path\\": \\".\\"}"}}]}}]}',
            'data: [DONE]',
        ]
        monkeypatch.setattr(
            provider._session, "post",
            lambda url, json=None, **kw: _FakeStreamResp(lines),
        )
        resp = provider.chat(
            messages=[Message(role="user", content="列出目录")],
            stream=True, on_token=lambda t: None,
            tools=_TOOLS, tool_choice="auto",
        )
        assert resp.content == "我先看看"
        assert resp.tool_calls[0]["function"]["name"] == "list_dir"
        assert json.loads(resp.tool_calls[0]["function"]["arguments"]) == {"path": "."}


# ═══ B. 引擎层 ═══

class _ScriptedProvider:
    """脚本化provider：按顺序返回预设ModelResponse，记录每次调用。"""
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages=None, stream=False, on_token=None, **kwargs):
        self.calls.append({
            "messages": messages, "stream": stream,
            "tools": kwargs.get("tools"), "tool_choice": kwargs.get("tool_choice"),
        })
        return self.responses.pop(0)


def _make_engine(tmp_path) -> OpenLLMEngine:
    return OpenLLMEngine(AgentConfig(
        capsule_dir=str(tmp_path), enable_io_s_checkpoint=False,
    ))


class TestEngineToolLoop:
    def test_structured_tool_calls_executed_and_loop_continues(self, tmp_path, _isolate_home):
        """模型返回结构化tool_calls → 引擎执行工具 → 注入history → 继续循环拿到最终回答。"""
        target = tmp_path / "hello.txt"
        target.write_text("world", encoding="utf-8")

        provider = _ScriptedProvider([
            ModelResponse(
                content="我先读取文件。",
                tool_calls=[{
                    "id": "call_1", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": str(target)})},
                }],
            ),
            ModelResponse(content="文件内容为 world。"),
        ])
        engine = _make_engine(tmp_path)
        engine.provider = provider

        resp = engine.chat("读取hello.txt的内容", stream=False)

        assert "world" in resp
        assert len(provider.calls) == 2  # 第一轮触发工具，第二轮基于结果回答
        # 工具结果已注入history
        history_texts = " ".join(m.content for m in engine._history)
        assert "[工具结果: read_file]" in history_texts
        # 工具执行结果内容出现在history里
        assert "world" in history_texts

    def test_multiple_structured_tool_calls_in_one_response(self, tmp_path, _isolate_home):
        """一次响应带多个tool_calls → 逐个执行。"""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("AAA", encoding="utf-8")
        f2.write_text("BBB", encoding="utf-8")

        provider = _ScriptedProvider([
            ModelResponse(
                content="",
                tool_calls=[
                    {"id": "c1", "type": "function",
                     "function": {"name": "read_file", "arguments": json.dumps({"path": str(f1)})}},
                    {"id": "c2", "type": "function",
                     "function": {"name": "read_file", "arguments": json.dumps({"path": str(f2)})}},
                ],
            ),
            ModelResponse(content="两个文件都读完了。"),
        ])
        engine = _make_engine(tmp_path)
        engine.provider = provider

        resp = engine.chat("读两个文件", stream=False)
        assert "两个文件都读完了" in resp
        history_texts = " ".join(m.content for m in engine._history)
        assert history_texts.count("[工具结果: read_file]") == 2
        assert "AAA" in history_texts and "BBB" in history_texts

    def test_regex_path_still_works_as_fallback(self, tmp_path, _isolate_home):
        """无结构化tool_calls时回落正则路径（向后兼容文本调用习惯）。"""
        target = tmp_path / "x.txt"
        target.write_text("regex-content", encoding="utf-8")

        provider = _ScriptedProvider([
            ModelResponse(content=f'```python\nread_file("{target}")\n```'),
            ModelResponse(content="正则路径已处理完毕。"),
        ])
        engine = _make_engine(tmp_path)
        engine.provider = provider

        resp = engine.chat("读取x.txt", stream=False)
        assert "正则路径已处理完毕" in resp
        history_texts = " ".join(m.content for m in engine._history)
        assert "[工具结果: read_file]" in history_texts
        assert "regex-content" in history_texts

    def test_no_tool_calls_no_tool_execution(self, tmp_path, _isolate_home):
        """纯聊天响应：不执行任何工具，单轮调用。"""
        provider = _ScriptedProvider([ModelResponse(content="你好，我是OpenLLM。")])
        engine = _make_engine(tmp_path)
        engine.provider = provider

        resp = engine.chat("你好", stream=False)
        assert resp == "你好，我是OpenLLM。"
        assert len(provider.calls) == 1
        history_texts = " ".join(m.content for m in engine._history)
        assert "[工具结果:" not in history_texts


class TestEngineToolsSchema:
    def test_call_model_passes_tools_schema(self, tmp_path, monkeypatch, _isolate_home):
        """引擎 _call_model 把工具注册表转为OpenAI风格tools传给DeepSeekProvider。"""
        engine = _make_engine(tmp_path)
        provider = DeepSeekProvider(ModelConfig(api_key="test-key"))
        captured = {}

        def fake_post(url, json=None, **kw):
            captured["payload"] = json
            return _FakeResp({
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "model": "m", "usage": {},
            })

        monkeypatch.setattr(provider._session, "post", fake_post)
        engine.provider = provider
        engine._history.append(Message(role="user", content="hi"))

        out = engine._call_model(stream=False)
        assert out == "hi"
        payload = captured["payload"]
        assert payload["tool_choice"] == "auto"
        assert "tools" in payload
        names = [t["function"]["name"] for t in payload["tools"]]
        assert "read_file" in names and "shell" in names
        # 每个工具都是标准结构
        for t in payload["tools"]:
            assert t["type"] == "function"
            assert set(t["function"].keys()) == {"name", "description", "parameters"}
        # 核心工具的默认参数声明存在（write_file含content）
        wf = next(t for t in payload["tools"] if t["function"]["name"] == "write_file")
        assert "content" in wf["function"]["parameters"]["properties"]

    def test_call_model_skips_tools_for_non_openai_provider(self, tmp_path, _isolate_home):
        """Anthropic/Gemini协议不同——非DeepSeekProvider不传tools，防止TypeError。"""
        engine = _make_engine(tmp_path)
        provider = _ScriptedProvider([ModelResponse(content="ok")])
        engine.provider = provider
        engine._history.append(Message(role="user", content="hi"))

        out = engine._call_model(stream=False)
        assert out == "ok"
        assert provider.calls[0]["tools"] is None
        assert provider.calls[0]["tool_choice"] is None

    def test_build_tools_schema_covers_all_registered_tools(self, tmp_path, _isolate_home):
        """注册表里每个工具都有schema（注册的或默认生成的）。"""
        engine = _make_engine(tmp_path)
        schema = engine._build_tools_schema()
        registered = engine.tools.list_tools()
        assert len(schema) == len(registered)
        schema_names = {t["function"]["name"] for t in schema}
        assert schema_names == {t["name"] for t in registered}
        for t in schema:
            assert t["function"]["parameters"]["type"] == "object"
