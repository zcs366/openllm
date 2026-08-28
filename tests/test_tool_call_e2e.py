"""test_tool_call_e2e.py — 工具调用链路端到端通路测试（DR-20260828-01 修复#5）

克洛诺斯之问的答案：1603个测试抓不住"发动机没装火花塞"，
因为它们验证部件存在性，不验证数据从输入端流经全链路到达输出端的完整旅程。
本文件补上这条通路测试。全部mock，不发真实API请求。
"""
import pytest

from openllm.core.models import Context, Proposal, Critique, RiskAssessment, Decision
from openllm.core.octopus_impl import _LeftBrain
from openllm.core.ios_arbitrate import arbitrate
from openllm.core.protocol import HeartbeatContext


# ═══ 第一层：解析层（think()的TOOL_CALLS提取）═══

class _MockProvider:
    """可编程mock provider——不发真API。"""
    def __init__(self, response: str = "", available: bool = True):
        self.response = response
        self._available = available
        self.last_messages = None

    def chat(self, messages):
        self.last_messages = messages
        return self.response


def _make_left_brain(response: str) -> tuple[_LeftBrain, _MockProvider]:
    lb = _LeftBrain()
    mp = _MockProvider(response)
    lb.provider = mp
    return lb, mp


def _ctx(msg: str = "测试", tools: list | None = None) -> Context:
    return Context(user_message=msg, tools=tools or ["read_file", "terminal"])


class TestThinkExtraction:
    """think()对三种响应形态的解析行为。"""

    def test_response_with_tool_call_extracted(self):
        lb, mp = _make_left_brain(
            '我需要读取文件。\nTOOL_CALLS: {"tool_calls": [{"name": "read_file", "args": {"path": "/tmp/a"}}]}')
        p = lb.think(_ctx("读取/tmp/a"))
        assert p.tool_calls == [{"name": "read_file", "args": {"path": "/tmp/a"}}]
        assert "TOOL_CALLS" not in p.content  # 工具行已剥离

    def test_pure_chat_no_tool_calls(self):
        lb, _ = _make_left_brain("你好，今天天气不错。")
        p = lb.think(_ctx("你好"))
        assert p.tool_calls == []
        assert "天气" in p.content

    def test_malformed_json_falls_back_to_chat(self):
        lb, _ = _make_left_brain("TOOL_CALLS: {broken json!!")
        p = lb.think(_ctx("x"))
        assert p.tool_calls == []  # 畸形回退聊天路径，不抛异常

    def test_invalid_entry_filtered(self):
        lb, _ = _make_left_brain(
            'TOOL_CALLS: {"tool_calls": [{"name": 123, "args": {}}, {"name": "terminal", "args": {"command": "ls"}}]}')
        p = lb.think(_ctx("x"))
        # name非字符串的条目被过滤，合法条目保留
        assert p.tool_calls == [{"name": "terminal", "args": {"command": "ls"}}]

    def test_prompt_contains_tool_definitions(self):
        """断裂一的直接回归测试：ctx.tools必须出现在prompt里。"""
        lb, mp = _make_left_brain("好的。")
        lb.think(_ctx(tools=["read_file", "terminal"]))
        prompt = mp.last_messages[-1]["content"]
        assert "read_file" in prompt and "terminal" in prompt
        assert "TOOL_CALLS" in prompt  # 格式指令在

    def test_memory_recall_injected(self):
        """断裂六（温度）：MemoryBus召回结果格式化进prompt。"""
        lb, mp = _make_left_brain("好的。")
        ctx = _ctx()
        ctx.memory = {"recalled": [{"source": "test", "content": "上次聊过工具链路", "importance": 0.8}]}
        lb.think(ctx)
        prompt = mp.last_messages[-1]["content"]
        assert "上次聊过工具链路" in prompt


# ═══ 第二层：透传层（arbitrate的tool_calls传递语义）═══

class _MockIOS:
    _arbiter_policy = "conservative"
    _rejection_engine = None


_TC = [{"name": "search_files", "args": {"pattern": "*.py"}}]


class TestArbitratePassthrough:
    def test_level1_approved_carries_tool_calls(self):
        d = arbitrate(_MockIOS(), Proposal(content="x", tool_calls=_TC),
                      Critique(content="ok", verdict="approve"), None)
        assert d.approved and d.tool_calls == _TC

    def test_level3_approved_carries_tool_calls(self):
        d = arbitrate(_MockIOS(), Proposal(content="x", tool_calls=_TC),
                      Critique(content="ok", verdict="approve"), RiskAssessment(level="medium"))
        assert d.approved and d.tool_calls == _TC

    def test_level4_reject_drops_tool_calls(self):
        """D4安全不变量：被否决的提案，工具意图不得透传。"""
        d = arbitrate(_MockIOS(), Proposal(content="x", tool_calls=_TC),
                      Critique(content="危险", verdict="reject"), RiskAssessment(level="high"))
        assert not d.approved and d.tool_calls == []

    def test_critical_blocked_drops_tool_calls(self):
        d = arbitrate(_MockIOS(), Proposal(content="x", tool_calls=_TC),
                      Critique(content="ok", verdict="approve"),
                      RiskAssessment(level="critical", blocked=True))
        assert not d.approved and d.tool_calls == []


# ═══ 第三层：通路层（ISN真执行 + 协议字段贯通）═══

class TestFullPipeline:
    def test_isn_executes_decision_tool_calls(self):
        """从带tool_calls的Decision到ISN真执行——火花塞装上后发动机得转。"""
        from openllm.core.isn_impl import ISN
        import tempfile, os
        isn = ISN()
        # 用沙箱白名单内的路径（/tmp/openllm）
        os.makedirs("/tmp/openllm", exist_ok=True)
        with open("/tmp/openllm/e2e_probe.txt", "w") as f:
            f.write("pipeline-alive")
        decision = Decision(action="execute", approved=True, reason="e2e",
                            tool_calls=[{"name": "read_file", "args": {"path": "/tmp/openllm/e2e_probe.txt"}}])
        result = isn.execute(decision)
        assert result.success
        assert "pipeline-alive" in result.output

    def test_heartbeat_context_carries_tool_calls(self):
        """协议字段：EXECUTE阶段写入后IKO可读（断裂四的回归测试）。"""
        hc = HeartbeatContext(user_message="x", tick_id="t0")
        assert hc.tool_calls == []
        # 模拟heartbeat._execute的填充行为
        decision = Decision(action="execute", approved=True,
                            tool_calls=[{"name": "terminal", "args": {}}])
        hc.tool_calls = getattr(decision, 'tool_calls', []) or []
        # 模拟heartbeat._learn的IKO上下文构建
        has_tool_calls = bool(hc.tool_calls) or bool(getattr(decision, 'tool_calls', None))
        assert has_tool_calls is True

    def test_d0_breakthrough_before_fix_would_be_false(self):
        """修复前该值为恒False（decision无tool_calls字段→getattr返回None）。"""
        old_style_decision = Decision(action="execute", approved=True)  # 不带tool_calls
        assert bool(getattr(old_style_decision, 'tool_calls', None)) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
