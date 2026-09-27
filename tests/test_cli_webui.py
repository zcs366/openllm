"""DR-20260927-02 Session Web UI 测试。

T1 会话落盘生命周期：start → append ×2 → list → load
T2 路径穿越防护：非法 session id → load 返回 None
T3 坏行容错：断电半行 → 跳过不炸
T4 /clear 轮换：新会话新文件
T5 HTTP 端点：/ 返回 HTML、/api/sessions、/api/session/<id>、404
T6 TUI 接线：AgentShell 每轮 default() 后会话文件增长两条
"""
import io
import json
import re
import sys
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from openllm.cli import webui


@pytest.fixture()
def sdir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_SESSIONS_DIR", str(tmp_path / "sessions"))
    return tmp_path / "sessions"


# ══════════════════════════════════════════════════════
# T1 落盘生命周期
# ══════════════════════════════════════════════════════

class TestLifecycle:

    def test_start_append_list_load(self, sdir):
        p = webui.start_session_file(model="mimo-v2.5")
        webui.append_msg(p, "user", "你好，介绍一下六体")
        webui.append_msg(p, "assistant", "openLLM 六体：IAI·IAX·ISA·IOS·ISN·IKO")

        sessions = webui.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["title"].startswith("你好")
        assert "mimo-v2.5" in sessions[0]["meta"]

        sid = sessions[0]["id"]
        msgs = webui.load_session(sid)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "六体" in msgs[1]["content"]

    def test_append_none_noop(self, sdir):
        """path=None（TUI 裸壳/落盘失败）→ 静默跳过。"""
        webui.append_msg(None, "user", "x")  # 不应抛

    def test_newest_first(self, sdir):
        p1 = webui.start_session_file()
        webui.append_msg(p1, "user", "旧的")
        time.sleep(0.02)
        p2 = webui.start_session_file()
        webui.append_msg(p2, "user", "新的")
        ids = [s["id"] for s in webui.list_sessions()]
        assert ids.index(p2.stem) < ids.index(p1.stem)


# ══════════════════════════════════════════════════════
# T2 路径穿越防护
# ══════════════════════════════════════════════════════

class TestSidSafety:

    @pytest.mark.parametrize("bad", [
        "../../etc/passwd", "..%2F..%2Fetc", "a/b", "a\\b",
        "", "  ", ".hidden", "id with space",
    ])
    def test_bad_sid_rejected(self, sdir, bad):
        assert webui.load_session(bad) is None


# ══════════════════════════════════════════════════════
# T3 坏行容错（断电半行）
# ══════════════════════════════════════════════════════

class TestCorruptLine:

    def test_partial_line_skipped(self, sdir):
        p = webui.start_session_file()
        webui.append_msg(p, "user", "完整的一条")
        with open(p, "a", encoding="utf-8") as f:
            f.write('{"type":"msg","role":"assistant","content":"断电半行')  # 无\n无闭合
        sid = p.stem
        msgs = webui.load_session(sid)
        assert len(msgs) == 1  # 半行被跳过，完整消息健在


# ══════════════════════════════════════════════════════
# T4 /clear 轮换语义（start_session_file 生成新 id）
# ══════════════════════════════════════════════════════

class TestRotation:

    def test_two_files_distinct_ids(self, sdir):
        p1 = webui.start_session_file()
        p2 = webui.start_session_file()
        assert p1 != p2


# ══════════════════════════════════════════════════════
# T5 HTTP 端点
# ══════════════════════════════════════════════════════

@pytest.fixture()
def server(sdir):
    srv = webui.ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


class TestHttp:

    def test_index_html(self, server):
        with urllib.request.urlopen(server + "/") as r:
            body = r.read().decode()
        assert r.status == 200
        assert "openLLM" in body and "/api/sessions" in body

    def test_api_sessions_and_session(self, server, sdir):
        p = webui.start_session_file(model="test-model")
        webui.append_msg(p, "user", "端到端问题")
        webui.append_msg(p, "assistant", "端到端回答")
        sid = p.stem

        with urllib.request.urlopen(server + "/api/sessions") as r:
            sessions = json.loads(r.read())
        assert any(s["id"] == sid for s in sessions)

        with urllib.request.urlopen(f"{server}/api/session/{sid}") as r:
            data = json.loads(r.read())
        assert data["id"] == sid
        assert len(data["messages"]) == 2

    def test_api_404_unknown_and_bad_sid(self, server):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(server + "/api/session/no-such-id")
        assert e.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(server + "/api/session/..%2Fetc%2Fpasswd")
        assert e.value.code == 404


# ══════════════════════════════════════════════════════
# T6 TUI 接线：default() 每轮落两条
# ══════════════════════════════════════════════════════

def _stub_fast(monkeypatch):
    import types as _t
    fake_isa = _t.ModuleType("openllm.core.isa_impl")
    fake_isa.FULL_TOOLS = []
    fake_isa.BASE_TOOLS = []
    fake_isa.TOOL_DESCRIPTIONS = {}
    monkeypatch.setitem(sys.modules, "openllm.core.isa_impl", fake_isa)
    fake_models = _t.ModuleType("openllm.core.models")
    fake_models.Decision = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "openllm.core.models", fake_models)
    fake_oct = _t.ModuleType("openllm.iai.octopus")
    fake_oct._LeftBrain = type("_LB", (), {
        "_TOOLCALL_RE": re.compile(r"^$")})()
    monkeypatch.setitem(sys.modules, "openllm.iai.octopus", fake_oct)
    fake_ml = _t.ModuleType("openllm.core.main_loop")
    fake_ml.Agent = type("Agent", (), {})
    monkeypatch.setitem(sys.modules, "openllm.core.main_loop", fake_ml)


class TestTuiWiring:

    def test_default_writes_session_file(self, sdir, monkeypatch):
        _stub_fast(monkeypatch)
        from openllm.cli.main import AgentShell

        class _P:
            model = "stub-model"
            _available = False
            def chat(self, messages):
                return "回复正文"

        shell = AgentShell.__new__(AgentShell)
        side = SimpleNamespace(provider=_P(), _extract_tool_calls=None)
        shell.agent = SimpleNamespace(
            octopus=SimpleNamespace(left=side, right=side),
            isn=SimpleNamespace(),
            session=SimpleNamespace(active_turn=None, turns=[]))
        shell._history = []
        shell._session_file = webui.start_session_file(model="stub-model")

        shell.default("测试输入一")

        msgs = webui.load_session(shell._session_file.stem)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "测试输入一"
        assert "回复正文" in msgs[1]["content"]
