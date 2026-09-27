#!/usr/bin/env python3
"""DR-20260915 陷阱测试——「说了话没反应」事件的接骨钉子。

每一颗钉子都断言 **bug 的触发条件**，而不是只断言修好之后的行为。
拆骨反证时应大面积变红（见报告「反证法」一节）。

覆盖：
  刀1  provider 把服务端 error 体吞成 KeyError
  刀2  timeout=120 对吐心跳的慢响应无效（静默 900 秒）
  刀3  CLI 噪声过滤把多段回复砍到只剩最后一段
  刀4  /exit 不退出
  刀6  空回复返回 '' 不给解释
"""
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ══════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════

class _FakeResp:
    """伪造 requests 响应（只实现 chat() 用到的三个方法）。"""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _bare_provider(endpoint="http://127.0.0.1:1/v1/chat/completions"):
    """绕过 __init__（config/密钥库）造一个最小 provider。"""
    from openllm.core.provider_impl import LLMProvider

    p = LLMProvider.__new__(LLMProvider)
    p.model = "test-model"
    p.endpoint = endpoint
    p.api_key = "test-key"
    p._available = True
    p._last_usage = {}
    return p


class _DeadHTTPServer:
    """收下 TCP 连接但永不回一个字节——模拟「服务端吐心跳/排队」的慢响应。"""

    def __init__(self):
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self._srv.settimeout(0.3)
        self.port = self._srv.getsockname()[1]
        self._conns = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
                self._conns.append(conn)      # 只收不回
            except (socket.timeout, OSError):
                continue

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    def close(self):
        self._stop.set()
        for c in self._conns:
            try:
                c.close()
            except OSError:
                pass
        try:
            self._srv.close()
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════
# 刀1 · 服务端把 error 体塞进 HTTP 200，不得吞成 KeyError
# ══════════════════════════════════════════════════════════════════

class TestServerErrorPassthrough:

    def test_http200_with_error_body_surfaces_server_message(self, monkeypatch):
        """陷阱形态：DeepSeek 排队超时的真实响应——HTTP 200 + error 体。

        旧行为：data["choices"] → KeyError → except 兜成 "[LLM错误] 'choices'"，
        服务端原话（含 '900-second timeout'）彻底丢失。
        """
        import requests
        payload = {"error": {"message": "We were unable to start processing "
                                        "your request within the 900-second timeout limit.",
                             "type": "server_error"}}
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp(payload))

        p = _bare_provider()
        out = p.chat([{"role": "user", "content": "hi"}])

        assert "900-second timeout" in out, f"服务端原话必须直达用户，实际={out!r}"
        assert "'choices'" not in out, "不得再退化成 KeyError 文本"

    def test_error_as_plain_string_still_surfaced(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: _FakeResp({"error": "rate limited"}))
        out = _bare_provider().chat([{"role": "user", "content": "hi"}])
        assert "rate limited" in out

    def test_response_without_choices_is_explained(self, monkeypatch):
        """陷阱形态：合法 JSON 但没有 choices，且没有 error 字段。"""
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp({"foo": "bar"}))
        out = _bare_provider().chat([{"role": "user", "content": "hi"}])
        assert "choices" in out and "foo" in out, f"应报出原始返回，实际={out!r}"

    def test_empty_choices_list_is_explained(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp({"choices": []}))
        out = _bare_provider().chat([{"role": "user", "content": "hi"}])
        assert "choices" in out


# ══════════════════════════════════════════════════════════════════
# 刀2 · 超时必须真的超时
# ══════════════════════════════════════════════════════════════════

class TestRealTimeout:

    def test_dead_server_times_out_instead_of_hanging(self):
        """陷阱形态：服务端收连接不回数据。

        旧行为：timeout=120 是单值（connect+read 同值），且 read timeout 按字节
        间隔重置——只要能收到心跳就永不超时，实测静默 900 秒。
        """
        from openllm.core.provider_impl import LLMProvider

        srv = _DeadHTTPServer()
        # 用 getattr 取值：若被拆骨/回退，类上根本没有这个常量——此时不能死在
        # setup 上（那是假钉子），必须让 chat() 真跑起来、真挂住，再由 dt<20 抓住。
        old_read = getattr(LLMProvider, "READ_TIMEOUT", None)
        try:
            LLMProvider.READ_TIMEOUT = 1.5
            p = _bare_provider(srv.url)
            t0 = time.time()
            out = p.chat([{"role": "user", "content": "hi"}])
            dt = time.time() - t0
        finally:
            if old_read is None:
                delattr(LLMProvider, "READ_TIMEOUT")
            else:
                LLMProvider.READ_TIMEOUT = old_read
            srv.close()

        assert dt < 20, f"必须在读超时后立即返回，实际耗时 {dt:.1f}s（旧行为=静默挂死）"
        assert "错误" in out and "Timeout" in out, f"应报超时，实际={out!r}"

    def test_timeout_is_connect_read_tuple(self, monkeypatch):
        """超时必须是 (connect, read) 元组——单值对慢响应无效。"""
        import requests

        seen = {}

        def _spy(url, **kw):
            seen.update(kw)
            return _FakeResp({"choices": [{"message": {"content": "ok"},
                                           "finish_reason": "stop"}]})

        monkeypatch.setattr(requests, "post", _spy)
        _bare_provider().chat([{"role": "user", "content": "hi"}])

        assert isinstance(seen.get("timeout"), tuple), \
            f"timeout 应为 (connect, read) 元组，实际={seen.get('timeout')!r}"
        assert len(seen["timeout"]) == 2


# ══════════════════════════════════════════════════════════════════
# 刀6 · 空回复要解释
# ══════════════════════════════════════════════════════════════════

class TestEmptyReplyExplanation:

    def test_empty_content_gives_explanation_not_empty_string(self, monkeypatch):
        """陷阱形态：推理占满 max_tokens → content=''。

        旧行为：返回 ''，CLI 只显示 "(无输出)"，用户无法判断是模型坏了还是 token 用尽。
        """
        import requests
        payload = {"choices": [{"message": {"content": "", "reasoning_content": ""},
                                "finish_reason": "length"}]}
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp(payload))

        out = _bare_provider().chat([{"role": "user", "content": "hi"}])

        assert out != "", "不得返回空串"
        assert "max_tokens" in out and "length" in out, f"应解释原因，实际={out!r}"

    def test_reasoning_content_still_used_as_fallback(self, monkeypatch):
        """不要误伤：content 空但 reasoning_content 有内容 → 仍取 reasoning。"""
        import requests
        payload = {"choices": [{"message": {"content": "",
                                            "reasoning_content": "理由文本"},
                                "finish_reason": "stop"}]}
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp(payload))
        assert _bare_provider().chat([{"role": "user", "content": "hi"}]) == "理由文本"

    def test_normal_reply_untouched(self, monkeypatch):
        import requests
        payload = {"choices": [{"message": {"content": "正常回复"}, "finish_reason": "stop"}]}
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResp(payload))
        assert _bare_provider().chat([{"role": "user", "content": "hi"}]) == "正常回复"


# ══════════════════════════════════════════════════════════════════
# 刀3 · 多段回复不得被砍
# ══════════════════════════════════════════════════════════════════

class TestReplyRendering:

    def _render(self, s):
        from openllm.cli.main import _render_reply
        return _render_reply(s)

    def test_multiline_reply_survives_whole(self):
        """陷阱形态：多段回复中有一段提到「搜索」——旧过滤会把它和它之前的内容全删。"""
        reply = ("老搭档，我在的。\n\n"
                 "今天六体都醒着：ISA 记着上次的接骨手术。\n\n"
                 "说起来，你要是想让我去搜索点东西，直接说就行。\n\n"
                 "——还是要先把手上的运输安全材料收个尾？")
        assert self._render(reply) == reply, "多段回复必须原样保留"

    def test_last_paragraph_is_not_the_only_thing_left(self):
        """真正咬住 bug 的断言：输出不能只剩最后一段。"""
        reply = "第一段：开场。\n\n第二段：正文核心。\n\n第三段：收尾。"
        out = self._render(reply)
        for seg in ("第一段：开场。", "第二段：正文核心。", "第三段：收尾。"):
            assert seg in out, f"丢了 {seg!r}，实际={out!r}"

    def test_common_words_do_not_delete_lines(self):
        """黑名单曾含 搜索/写入/洞察/来源/关键/意识/openLLM 等常用词。"""
        reply = "我可以用搜索找资料。\n我会写入文件。\n我给出关键结论。\n我是 openLLM。"
        out = self._render(reply)
        assert out == reply, f"常用词行不得被删，实际={out!r}"

    def test_single_line_reply_survives(self):
        assert self._render("只有一行。") == "只有一行。"

    def test_empty_and_none(self):
        assert self._render("") == ""
        assert self._render(None) == ""
        assert self._render("   \n  ") == ""

    def test_fenced_block_unwrapped(self):
        """沿用原意：整段被 ``` 包裹时剥掉围栏。"""
        assert self._render('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert self._render('```\n正文\n```') == "正文"

    def test_inner_fence_not_stripped(self):
        """正文里夹代码块时不得把外层正文也吃掉。"""
        reply = "看这段：\n\n```python\nprint(1)\n```\n\n就这些。"
        assert self._render(reply) == reply

    def test_default_prints_whole_multiline_reply_end_to_end(self, monkeypatch, capsys):
        """行为级钉子——对**老代码同样有效**（老过滤逻辑内联在 default() 里）。

        前面几条测的是新抽出的接缝 _render_reply；若拆骨回 HEAD，那些会因
        ImportError 而红（红在 import，不算行为反证）。这一条直接驱动真实的
        AgentShell.default() 并抓 stdout：无论清理逻辑在哪一层，只要它把多段
        回复砍成最后一段，这里就红。
        """
        from types import SimpleNamespace
        from openllm.cli.main import AgentShell

        reply = ("第一段：开场。\n\n"
                 "第二段：正文核心，这里提到搜索这个词。\n\n"
                 "第三段：收尾。")

        class _Provider:
            def chat(self, messages):
                return reply

        shell = AgentShell.__new__(AgentShell)          # 跳过重的 __init__
        side = SimpleNamespace(provider=_Provider())
        # 用 setattr 塞替身：Pyright 会因 agent 声明为 Agent 而报类型不符，
        # 但这里要的正是「不构建真 Agent 也能驱动 default()」。
        setattr(shell, "agent", SimpleNamespace(octopus=SimpleNamespace(left=side, right=side)))

        shell.default("你好")                            # 短句 → 快路径
        out = capsys.readouterr().out

        for seg in ("第一段：开场。", "第二段：正文核心，这里提到搜索这个词。", "第三段：收尾。"):
            assert seg in out, f"CLI 漏印了 {seg!r}；实际输出=\n{out}"


# ══════════════════════════════════════════════════════════════════
# 刀4 · /exit 必须真的退出
# ══════════════════════════════════════════════════════════════════

class TestExitCommand:

    def _shell(self):
        """跳过 AgentShell.__init__（会构建整个 Agent），只测命令分发。"""
        from openllm.cli.main import AgentShell
        return AgentShell.__new__(AgentShell)

    @pytest.mark.parametrize("line", ["/exit", "/quit", "/q", "/EXIT", "  /exit  "])
    def test_exit_returns_truthy_to_cmdloop(self, line):
        """陷阱形态：cmdloop 靠 onecmd 的真值终止——旧代码这条链是断的。"""
        shell = self._shell()
        assert shell.onecmd(line) is True, f"{line!r} 必须返回 True 才能终止循环"

    def test_non_exit_command_does_not_terminate(self, monkeypatch):
        shell = self._shell()
        monkeypatch.setattr(shell, "_handle_command", lambda c: None)
        assert not shell.onecmd("/status"), "普通命令不得终止循环"

    def test_plain_text_does_not_terminate(self, monkeypatch):
        shell = self._shell()
        monkeypatch.setattr(shell, "default", lambda line: None)
        assert not shell.onecmd("你好")


# ══════════════════════════════════════════════════════════════════
# 刀7 · IOS 治理审计不得寄生在 Hermes 家目录
# ══════════════════════════════════════════════════════════════════

class TestAuditDbSelfContained:
    """陷阱形态：HOME 下没有 ~/.hermes。

    旧行为：硬编码 ~/.hermes/hermes.db，目录不存在 → sqlite3.OperationalError:
    unable to open database file，六体起不来（实测于 HOME=/tmp/dsh 沙箱）。
    """

    def test_constructs_when_hermes_home_absent(self, monkeypatch, tmp_path):
        from openllm.core.governance_engine import GovernanceAuditLog

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("OPENLLM_AUDIT_DB", raising=False)

        assert not (tmp_path / ".hermes").exists(), "前置条件：沙箱里没有 ~/.hermes"

        log = GovernanceAuditLog()                     # 旧代码在这里崩
        try:
            row = log.append("test_event", action="sandbox")
            assert row > 0
            assert str(tmp_path) in log._db_path, f"库应落在本 HOME 下，实际={log._db_path}"
        finally:
            log._conn.close()

    def test_never_lands_under_dot_hermes(self, monkeypatch, tmp_path):
        from openllm.core.governance_engine import GovernanceAuditLog

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("OPENLLM_AUDIT_DB", raising=False)
        log = GovernanceAuditLog()
        try:
            # 2026-09-26 军师修：原断言 ".hermes" not in db_path 在 Hermes 本机
            # （TMPDIR=~/.hermes/cache/scratch）误报。真意图=跟随重定向HOME、
            # 不寄生真实 ~/.hermes——判「db 在 $HOME/.openllm 下」等价且更准。
            db = Path(log._db_path).resolve()
            assert db.is_relative_to((tmp_path / ".openllm").resolve()), \
                f"IOS 审计应落重定向HOME的.openllm下，实际={log._db_path}"
        finally:
            log._conn.close()

    def test_env_override_respected_and_parent_created(self, monkeypatch, tmp_path):
        from openllm.core.governance_engine import GovernanceAuditLog

        target = tmp_path / "deep" / "nested" / "audit.db"
        monkeypatch.setenv("OPENLLM_AUDIT_DB", str(target))

        log = GovernanceAuditLog()
        try:
            assert Path(log._db_path) == target
            assert target.parent.exists(), "父目录应被自动创建"
        finally:
            log._conn.close()

    def test_explicit_db_path_wins(self, monkeypatch, tmp_path):
        from openllm.core.governance_engine import GovernanceAuditLog

        monkeypatch.setenv("OPENLLM_AUDIT_DB", str(tmp_path / "from_env.db"))
        explicit = tmp_path / "explicit.db"
        log = GovernanceAuditLog(db_path=str(explicit))
        try:
            assert Path(log._db_path) == explicit
        finally:
            log._conn.close()


# ══════════════════════════════════════════════════════════════════
# 追加刀 · IOS 审计目录的测试隔离（防「测试写进生产目录」）
# ══════════════════════════════════════════════════════════════════

class TestIosAuditIsolation:
    """陷阱形态：默认审计目录指向 ~/.hermes/jiak/governance_audit/。

    现场证据：该目录累积 427 个文件，其中 test-001.jsonl(1.6M)/test-002.jsonl(705K)
    的 mtime 正是全量回归的时刻——tests/test_governance.py 的 session id 就叫
    test-001/test-002。与 skill 里「测试会污染用户真实数据」同源。

    2026-09-26 军师修：断言判据从子串 ".hermes" 收紧为「不得落进生产目录
    ~/.hermes/jiak/」。原因：Hermes 本机把 TMPDIR 指到 ~/.hermes/cache/scratch，
    pytest 夹具把审计目录重定向进去后子串误报——scratch 是合法临时区（72h清理），
    真正要防的只有生产 jiak 目录。意图不变，误报消除。
    """

    _PROD_AUDIT = Path.home() / ".hermes" / "jiak"

    @classmethod
    def _assert_not_production(cls, path, what: str) -> None:
        resolved = Path(str(path)).resolve()
        prod = cls._PROD_AUDIT.resolve()
        assert not (resolved == prod or resolved.is_relative_to(prod)), \
            f"{what}泄漏到生产Hermes目录: {path}"

    def test_module_defaults_pinned_away_from_hermes(self):
        from openllm.ios import audit as audit_mod
        from openllm.ios import stateful_audit as sa_mod

        self._assert_not_production(audit_mod.DEFAULT_AUDIT_DIR, "AuditChain 默认目录")
        self._assert_not_production(sa_mod.DEFAULT_AUDIT_DIR, "StatefulAuditTrail 默认目录")

    def test_constructing_objects_lands_in_sandbox(self):
        from openllm.ios.audit import AuditChain
        from openllm.ios.stateful_audit import StatefulAuditTrail

        chain, trail = AuditChain(), StatefulAuditTrail()
        for obj in (chain, trail):
            self._assert_not_production(obj.audit_dir, "实例目录")
        assert chain.audit_dir == trail.audit_dir, "两者应落到同一测试沙箱"

    def test_write_target_would_be_sandbox_not_hermes(self):
        """咬住 bug 的核心断言：**真写入时的落点**也必须在沙箱，不在生产Hermes。

        比「构造后文件没变」狠——构造本来就不写文件，那条断言在 bug 态下也会绿
        （属守护型）。这条直接算写入路径：bug 态下 _session_path 会落在
        ~/.hermes/jiak/governance_audit/dr20260915-probe.jsonl，必然红。
        """
        from openllm.ios.audit import AuditChain

        chain = AuditChain()
        target = chain._session_path("dr20260915-probe")
        self._assert_not_production(target, "写入落点")
        assert str(target).startswith(str(chain.audit_dir)), \
            f"落点应挂在实例目录下: target={target} dir={chain.audit_dir}"

    def test_real_hermes_audit_dir_unchanged_after_constructing(self):
        """守护型断言：构造审计对象不得在生产目录留下痕迹（构造本不该写文件）。"""
        real = Path.home() / ".hermes" / "jiak" / "governance_audit"

        def snapshot():
            if not real.exists():
                return {}
            return {p.name: p.stat().st_size for p in real.glob("*.jsonl")}

        before = snapshot()
        from openllm.ios.audit import AuditChain
        from openllm.ios.stateful_audit import StatefulAuditTrail

        AuditChain()
        StatefulAuditTrail()
        assert snapshot() == before, "构造审计对象时写进了真实 Hermes 目录"
