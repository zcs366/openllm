"""DR-20260927-07 DSH 只追加事件日志测试。

不变式钉子：
D1 append-only + seq 单调从0
D2 语义事件齐全（start/turn/message/tool/end）
D3 坏行容错（断电半行跳过）
D4 路径穿越防护（非法 sid → None）
D5 replay_session → user/assistant 对
D6 rebuild_history → 完整重建包（history 可直接喂 provider）
D7 list_dsh_sessions 列表（新→旧、标题=首条用户输入）
"""
import json

import pytest

from openllm.cli import dsh


@pytest.fixture()
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_DSH_DIR", str(tmp_path / "dsh"))
    return tmp_path / "dsh"


class TestAppendOnly:

    def test_seq_monotonic_and_shape(self, ddir):
        # session/start 由构造器写入占 seq 0
        lg = dsh.DshLogger(model="m1")
        assert lg.append("user/message", content="a") == 1
        assert lg.append("assistant/message", content="b") == 2
        assert lg.append("note", text="c") == 3
        lg.close()

        events = dsh.load_events(lg.session_id)
        assert [e["seq"] for e in events] == [0, 1, 2, 3]
        assert events[0]["type"] == "session/start"

    def test_start_is_first_event(self, ddir):
        lg = dsh.DshLogger(model="m1")
        lg.user_message("hello")
        lg.assistant_message("hi")
        lg.close()
        events = dsh.load_events(lg.session_id)
        assert events[0]["type"] == "session/start"
        assert events[0]["model"] == "m1"
        assert [e["seq"] for e in events] == [0, 1, 2]


class TestSemanticEvents:

    def test_full_turn(self, ddir):
        lg = dsh.DshLogger(model="m2")
        lg.turn_start("看下方案")
        lg.user_message("看下方案")
        lg.tool_calls([{"name": "read_file", "args": {"path": "x"}}])
        lg.assistant_message("结论：可行")
        lg.turn_end(duration_ms=1234.5, status="ok", tool_rounds=1)
        lg.end("exit")
        lg.close()

        events = dsh.load_events(lg.session_id)
        types = [e["type"] for e in events]
        assert types == ["session/start", "turn/start", "user/message",
                         "tool/call", "assistant/message", "turn/end",
                         "session/end"]
        assert events[-1]["reason"] == "exit"
        assert events[4]["content"] == "结论：可行"


class TestRobustness:

    def test_corrupt_line_skipped(self, ddir):
        lg = dsh.DshLogger(model="m3")
        lg.user_message("完整")
        lg.close()
        with open(lg.path, "a", encoding="utf-8") as f:
            f.write('{"seq":99,"type":"note","tex')  # 断电半行
        events = dsh.load_events(lg.session_id)
        # 半行被跳过；session/start + user/message 两条健在
        assert [e["type"] for e in events] == ["session/start",
                                               "user/message"]
        assert events[1]["content"] == "完整"

    @pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "", "x-y-z",
                                     "20260927-999999-zzzz"])
    def test_sid_traversal_blocked(self, ddir, bad):
        assert dsh.load_events(bad) is None
        assert dsh.rebuild_history(bad) is None


class TestReplay:

    def _make_session(self):
        lg = dsh.DshLogger(model="mimo-v2.5")
        lg.user_message("聊聊通史")
        lg.tool_calls([{"name": "read_file", "args": {"path": "骨架.md"}}])
        lg.assistant_message("通史两条轴：正当性与央地")
        lg.user_message("继续")
        lg.assistant_message("秦：郡县制破封建")
        lg.turn_end(status="ok")
        lg.end("exit")
        lg.close()
        return lg.session_id

    def test_replay_gives_pairs(self, ddir):
        sid = self._make_session()
        msgs = dsh.replay_session(sid)
        assert [m["role"] for m in msgs] == ["user", "assistant", "user",
                                             "assistant"]
        assert msgs[0]["content"] == "聊聊通史"

    def test_rebuild_full_package(self, ddir):
        sid = self._make_session()
        pack = dsh.rebuild_history(sid)
        assert pack["model"] == "mimo-v2.5"
        assert pack["ended"] is not None
        assert len(pack["turns"]) == 2
        assert pack["turns"][0]["tools"][0]["name"] == "read_file"
        # history 可直接喂 provider：形状合法
        assert all(set(m) == {"role", "content"} for m in pack["history"])
        assert pack["history"][0]["role"] == "user"

    def test_list_sessions(self, ddir):
        sid = self._make_session()
        lst = dsh.list_dsh_sessions()
        assert any(s["id"] == sid for s in lst)
        me = [s for s in lst if s["id"] == sid][0]
        assert "通史" in me["title"]
        assert me["events"] >= 7
