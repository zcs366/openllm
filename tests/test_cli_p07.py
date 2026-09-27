"""
P0.7 自问自答止血刀：输入卫生 + 回声标注（非拦截） + 引用语义。

T1: 空引号 "" → vacuous 拦截
T2: 纯标点 "，。？！" → vacuous 拦截
T3: 纯空白 "   " → vacuous 拦截
T4: 正常短输入 "你好" → 不拦截
T5: 含可见内容的引号 '"这段话有用"' → 不拦截
T6: 回声——用户原样粘贴 assistant 回复 → _echo_annotated 加前缀（不拦截）
T7: 回声——输入与 assistant 回复相似度 < 阈值 → _echo_annotated 原样返回
T8: 回声——空 history → _echo_annotated 原样返回
T9: 回声——输入太短(< 8字) → _echo_annotated 原样返回
T10: default() 集成——vacuous 输入 → print + return + 不入 LLM
T11: default() 集成——echo 输入 → 不拦截 + annotate + 照常进 LLM + 落账 raw
T12: default() 集成——正常输入 → 照常进 LLM + 入账
T13: 引用条款在 system prompt 中（中性版）
"""
import re
import sys
import importlib
from types import SimpleNamespace

import pytest

# ── 公共工具：stub 快路径所需 import ──

def _stub_fast_path_imports(monkeypatch):
    """Mock 快路径所需 import，防止真实模块加载。"""
    import types as _t

    fake_isa = _t.ModuleType("openllm.core.isa_impl")
    fake_isa.FULL_TOOLS = []
    fake_isa.BASE_TOOLS = []
    fake_isa.TOOL_DESCRIPTIONS = {}
    monkeypatch.setitem(sys.modules, "openllm.core.isa_impl", fake_isa)

    fake_models = _t.ModuleType("openllm.core.models")
    fake_models.Decision = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "openllm.core.models", fake_models)

    fake_octopus = _t.ModuleType("openllm.iai.octopus")
    fake_octopus._LeftBrain = type("_LB", (), {
        "_TOOLCALL_RE": re.compile(
            r"^\s*\*{0,2}\s*TOOL_CALLS:\s*(\{.*\})\s*\*{0,2}\s*$", re.MULTILINE)
    })()
    monkeypatch.setitem(sys.modules, "openllm.iai.octopus", fake_octopus)

    fake_main_loop = _t.ModuleType("openllm.core.main_loop")
    fake_main_loop.Agent = type("Agent", (), {})
    monkeypatch.setitem(sys.modules, "openllm.core.main_loop", fake_main_loop)


def _stub_main(monkeypatch):
    """确保 main 模块加载到最新代码。"""
    _stub_fast_path_imports(monkeypatch)
    mod = sys.modules.get("openllm.cli.main")
    if mod is not None:
        importlib.reload(mod)


def _build_shell(provider, monkeypatch, isn_exec=None, extract_fn=None):
    """构造带 stub agent 的 AgentShell。"""
    _stub_main(monkeypatch)
    from openllm.cli.main import AgentShell

    shell = AgentShell.__new__(AgentShell)

    _left = SimpleNamespace(
        provider=provider,
        _extract_tool_calls=extract_fn,
    )
    _right = SimpleNamespace(provider=provider)
    _octopus = SimpleNamespace(left=_left, right=_right)

    if isn_exec is not None:
        _isn = SimpleNamespace(execute=isn_exec)
    else:
        _isn = SimpleNamespace()

    _session = SimpleNamespace(active_turn=None, turns=[])
    _agent = SimpleNamespace(octopus=_octopus, isn=_isn, session=_session)
    shell.agent = _agent
    shell._history = []
    return shell


# ============================================================
# T1-T5: 输入卫生 (_is_vacuous_input)
# ============================================================

class TestVacuousInput:
    def test_empty_double_quotes(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input('""') is True

    def test_empty_single_quotes(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("''") is True

    def test_empty_chinese_quotes(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("「」") is True

    def test_punctuation_only(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("，。？！") is True

    def test_punctuation_and_whitespace(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("  ？！，。  ") is True

    def test_whitespace_only(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("   ") is True

    def test_single_char_not_vacuous(self):
        """单字输入 >= 2 字符（含空白 strip 后）不算空壳。"""
        from openllm.cli.main import _is_vacuous_input
        # "a" 只有 1 个可见字符，应该 vacuous
        assert _is_vacuous_input("a") is True
        # "ab" 有 2 个，不算
        assert _is_vacuous_input("ab") is False

    def test_normal_short_input(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("你好") is False

    def test_quoted_content_not_vacuous(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input('"这段话有用"') is False

    def test_mixed_quotes_and_content(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("'分析一下这个'") is False

    def test_empty_angle_quotes(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("《》") is True

    def test_empty_brackets(self):
        from openllm.cli.main import _is_vacuous_input
        assert _is_vacuous_input("（）") is True
        assert _is_vacuous_input("[]") is True
        assert _is_vacuous_input("{}") is True


# ============================================================
# T6-T9: 回声检测 (_is_echo_of_recent + _echo_annotated)
# ============================================================

class TestEchoDetection:
    def _hist(self, *msgs):
        """构造 history 列表。偶数为 user，奇数为 assistant。"""
        h = []
        for i, msg in enumerate(msgs):
            role = "assistant" if i % 2 == 1 else "user"
            h.append({"role": role, "content": msg})
        return h

    def test_exact_echo_detected(self):
        from openllm.cli.main import _is_echo_of_recent
        hist = self._hist(
            "你好", "老搭档，我把宾语错认了",
            "又一个问题", "签名制度是这样的",
        )
        assert _is_echo_of_recent("老搭档，我把宾语错认了", hist) is True

    def test_partial_echo_below_threshold(self):
        from openllm.cli.main import _is_echo_of_recent
        hist = self._hist(
            "你好", "签名制度是这样的定义的",
        )
        # 输入与 assistant 消息差异较大
        assert _is_echo_of_recent("签名制度是什么", hist) is False

    def test_empty_history_no_echo(self):
        from openllm.cli.main import _is_echo_of_recent
        assert _is_echo_of_recent("随便什么", []) is False

    def test_short_input_no_echo(self):
        """太短的输入 (< 8 字) 不判回声。"""
        from openllm.cli.main import _is_echo_of_recent
        hist = self._hist("你好", "签名制度是这样的")
        assert _is_echo_of_recent("签名", hist) is False

    def test_long_echo_detected(self):
        """用户粘贴一段长 assistant 回复。"""
        from openllm.cli.main import _is_echo_of_recent
        long_reply = "签名制度是 openLLM 中用于标识消息来源的机制，每个 turn 都有一个唯一签名。"
        hist = self._hist("问一下", long_reply)
        assert _is_echo_of_recent(long_reply, hist) is True

    def test_echo_with_noise_still_detected(self):
        """用户在回声前后加了少量文字，相似度仍然够。"""
        from openllm.cli.main import _is_echo_of_recent
        assistant_reply = "签名制度是这样的：每个 turn 的消息都带有签名标识"
        hist = self._hist("问", assistant_reply)
        # 用户加了前缀但核心内容一样
        echo_input = "嗯" + assistant_reply
        assert _is_echo_of_recent(echo_input, hist) is True

    def test_echo_annotated_adds_prefix(self):
        """T7c: echo 时 _echo_annotated 返回带 _ECHO_NOTE 前缀的文本。"""
        from openllm.cli.main import _echo_annotated, _ECHO_NOTE
        assistant_reply = "签名制度是 openLLM 中用于标识消息来源的机制，每个 turn 都有一个唯一签名。"
        hist = self._hist("问一下", assistant_reply)
        result = _echo_annotated(assistant_reply, hist)
        assert result.startswith(_ECHO_NOTE), "echo 应加 _ECHO_NOTE 前缀"
        assert assistant_reply in result, "原句应保留"

    def test_echo_annotated_passthrough_non_echo(self):
        """非回声输入 → _echo_annotated 原样返回。"""
        from openllm.cli.main import _echo_annotated
        hist = self._hist("问一下", "签名制度是这样的")
        result = _echo_annotated("完全不同的新问题", hist)
        assert result == "完全不同的新问题"

    def test_echo_annotated_empty_history(self):
        """空历史 → _echo_annotated 原样返回。"""
        from openllm.cli.main import _echo_annotated
        result = _echo_annotated("随便什么", [])
        assert result == "随便什么"


# ============================================================
# T10-T12: default() 集成
# ============================================================

class TestDefaultIntegration:
    def test_vacuous_blocks_llm(self, capsys, monkeypatch):
        """vacuous 输入 → print 消息 + return + 不入 LLM + 不入 hist。"""
        from openllm.cli.main import _VACUOUS_MSG
        call_log = []
        shell = _build_shell(
            SimpleNamespace(chat=lambda m: (call_log.append(m), "ok")[-1]),
            monkeypatch,
        )
        shell.default('""')
        assert len(call_log) == 0, "vacuous input should not call LLM"
        assert len(shell._hist()) == 0, "vacuous input should not enter history"
        out = capsys.readouterr().out
        assert "没收到内容" in out

    def test_echo_passes_through_with_annotation(self, capsys, monkeypatch):
        """T11: echo 输入 → 不拦截 + _echo_annotated 加前缀进 LLM + 落账 raw 原句。"""
        from openllm.cli.main import _ECHO_NOTE

        long_reply = "签名制度是这样的：每个 turn 的消息都带有签名标识，用于追踪对话来源"

        captured_messages = []

        def fake_chat(messages):
            captured_messages.append(messages)
            return long_reply

        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "unused"),
            monkeypatch,
        )
        # 先预设 history：让 assistant 的上一条回复 = long_reply，
        # 这样 echo 检测才能匹配。同时让 provider.chat 返回 long_reply。
        shell.agent.octopus.left.provider.chat = lambda m: long_reply
        shell.default("你好")
        hist_before = len(shell._hist())
        assert hist_before == 2
        # 确认 history 里 assistant 的回复是 long_reply
        assert shell._hist()[-1]["content"] == long_reply

        # 再替换 fake_chat 以捕获 echo 轮的 messages
        captured_messages.clear()
        shell.agent.octopus.left.provider.chat = fake_chat

        # 然后输入与 assistant 回复完全相同（触发回声检测）
        shell.default(long_reply)

        # echo 不拦截：LLM 应被调用（captured_messages 里有1条记录）
        assert len(captured_messages) == 1, "echo input should pass through to LLM"
        # 最后一条 user content 应含 _ECHO_NOTE 前缀
        last_user_msg = captured_messages[0][-1]
        assert last_user_msg["role"] == "user"
        assert last_user_msg["content"].startswith(_ECHO_NOTE), \
            "user content to LLM should have _ECHO_NOTE prefix"
        assert long_reply in last_user_msg["content"], \
            "original text should be preserved after prefix"

        # 落账卫生：history 里存的是 raw 原句，不含标注
        hist_user_entries = [m for m in shell._hist() if m["role"] == "user"]
        last_hist_user = hist_user_entries[-1]
        assert last_hist_user["content"] == long_reply, \
            "history should store raw line, not annotated text"
        assert _ECHO_NOTE not in last_hist_user["content"], \
            "annotation must NOT leak into history"

    def test_normal_input_proceeds(self, capsys, monkeypatch):
        """正常输入 → 照常进 LLM + 入账。"""
        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "normal reply"),
            monkeypatch,
        )
        shell.default("你好世界")
        assert len(shell._hist()) == 2, "normal input should pair in history"
        assert shell._hist()[0]["content"] == "你好世界"
        assert shell._hist()[1]["content"] == "normal reply"


# ============================================================
# T13: 引用条款在 system prompt 中（中性版）
# ============================================================

class TestQuoteClause:
    def test_quote_clause_in_system_prompt(self, monkeypatch):
        """fast path system prompt 包含中性版引用规则。"""
        _stub_main(monkeypatch)
        from openllm.cli.main import AgentShell

        import inspect
        src = inspect.getsource(AgentShell.default)
        assert "引用规则" in src, "system prompt should contain quote clause"
        # 中性版：不再说"不是你的发言"，改为"分析这段材料"
        assert "分析这段材料" in src, \
            "quote clause should instruct model to analyze the material"
        assert "不是你的发言" not in src, \
            "old non-neutral clause should be removed"


# ============================================================
# P0.7 回潮检查：default() 不含旧版回声拦截调用
# ============================================================

class TestP07NoRegression:
    def test_default_no_is_echo_of_recent_call(self):
        """default() 体内不存在 'if _is_echo_of_recent(line' 形态的拦截调用。

        P0.7 的合法用法是 _echo_annotated（注释不拦截）。
        若有人重新加回拦截，此钉即红。
        """
        import inspect
        from openllm.cli.main import AgentShell
        src = inspect.getsource(AgentShell.default)
        # 拦截模式：if _is_echo_of_recent(line, ...)
        assert "if _is_echo_of_recent(line" not in src, \
            "default() 不应直接调用 _is_echo_of_recent 拦截——应用 _echo_annotated"


# ============================================================
# T14-T17: _clean_output 反斜切回归钉子（军师 2026-09-25 实弹）
# ============================================================

class TestCleanOutputAntiTruncation:
    """验证 _clean_output 从旧版「反向扫描截断保尾段」改为「逐行过滤」后的行为。

    四条实弹回归钉 + 兜底验证，来自军师 2026-09-25 亲测真案。
    """

    @staticmethod
    def _clean(raw: str) -> str:
        """通过 Agent 实例调用 _clean_output，保持与生产链路一致。"""
        from openllm.core.main_loop import Agent
        a = Agent.__new__(Agent)
        return a._clean_output(raw)

    def test_markdown_with_separator_preserved(self):
        """T14: 含 --- 分隔线的 markdown 长文——旧版会切到尾段，新版应全保。"""
        raw = ("第一段正文内容，描述一个完整的技术方案。\n"
               "包含详细的实现步骤和注意事项。\n"
               "---\n"
               "第二段收尾总结。")
        result = self._clean(raw)
        assert "第一段" in result, f"首段被截断: {result!r}"
        assert "---" in result, f"分隔线被删: {result!r}"
        assert "收尾总结" in result, f"尾段丢失: {result!r}"

    def test_double_separator_preserved(self):
        """T15: 双 --- 切分的三段文——旧版只剩尾行，新版应全保。"""
        raw = ("段落一开头\n"
               "---\n"
               "段落二中间\n"
               "---\n"
               "段落三结尾")
        result = self._clean(raw)
        assert "段落一" in result
        assert "段落二" in result
        assert "段落三" in result

    def test_source_citation_preserved(self):
        """T16: 含 '来源: xx' 的正文引用行——旧版会把前面全切，新版应保留。"""
        raw = ("这篇论文提出了新的方法。\n"
               "来源: Smith et al. 2024\n"
               "该方法在实验中表现优异。")
        result = self._clean(raw)
        assert "提出了新的方法" in result, f"引用前文被截: {result!r}"
        assert "来源: Smith" in result, f"引用行被删: {result!r}"

    def test_system_injection_marker_still_filtered(self):
        """T17: 行首 [上下文压缩] 注入行——仍应被剔除。"""
        raw = ("这是正文。\n"
               "[上下文压缩] 内部标记行\n"
               "正文续写。")
        result = self._clean(raw)
        assert "这是正文" in result
        assert "上下文压缩" not in result
        assert "正文续写" in result

    def test_clean_text_unchanged(self):
        """T18: 干净无标记长文——逐字符不变。"""
        raw = "这是一段完全干净的文本，不含任何系统标记。"
        assert self._clean(raw) == raw

    def test_all_noise_falls_back_to_raw(self):
        """T19: 全噪声输入——回退返回 raw.strip()。"""
        raw = "  [上下文压缩] 标记一\n  [isa_ice] 标记二  "
        result = self._clean(raw)
        assert result == raw.strip()

    def test_empty_input(self):
        """空字符串 → 空字符串。"""
        assert self._clean("") == ""

    def test_none_like_input(self):
        """纯空白 → raw.strip() 兜底。"""
        raw = "   \n  \n  "
        result = self._clean(raw)
        # 全空行被过滤后为空，应返回 raw.strip()
        assert result == raw.strip()

    def test_chinese_common_words_not_filtered(self):
        """旧版黑名单的裸中文词（铁律/语义场/不要总结等）不应被新版过滤。"""
        raw = ("这段话提到铁律的重要性。\n"
               "语义场的概念很有意思。\n"
               "不要总结，直接回答。")
        result = self._clean(raw)
        assert "铁律" in result
        assert "语义场" in result
        assert "不要总结" in result

    def test_symbols_not_filtered(self):
        """旧版黑名单的符号（emoji/制表符等）不应被新版过滤。"""
        raw = "🔴 这是一个重要提示\n🗜️ 压缩标记\n🐙 章鱼相关"
        result = self._clean(raw)
        assert "🔴" in result
        assert "🗜️" in result
        assert "🐙" in result
