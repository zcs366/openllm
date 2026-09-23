"""DR-20260917-03/04 的钉子测试。

诊断来源（2026-09-17 品尝实测，openLLM 读 /mnt/i/hermes/output/818具神智能研究）：
用户说"你先读读，熟悉熟悉！给出你的看法"，openLLM 连续三问三不知——
  ① 命令被[拦截]：「ls -la /mnt/i/hermes/output/ …」里的 hermes 含子串 rm；
  ② 路径被[沙箱拒绝]：只读白名单没开，连 output 都进不去；
  ③ 三句之后反问"我们刚刚在谈什么"：CLI 每轮 messages 从零新建，对话历史不存在。

刀①② 的修复（只读白名单 + 词元级危险命令）已在 DR-20260917-01/02 落地，
本文件的守位由 tests/test_dr20260917_taster.py 承担。本文件钉的是同一夜暴露的
**残余两类病**：

刀③（终端误拒三连）：旧实现取整条命令的**首词**定 op，一刀切作用于命令内
所有路径。于是 `cd <只读目录> && ls` 因首词 cd 被判写意图而误拒；只读路径
出现在任何写意图命令里都会被按写检查掉。
  病断言：test_cd_into_readonly_dir_allowed / test_mixed_cmd_read_path_allowed
           （旧代码此处必红——这就是病本身）
  安全回归：test_write_to_readonly_still_denied / test_redirect_to_readonly_denied
           / test_dangerous_cmd_still_caught（修法不得放松写作语义）

刀④（会话失忆）：快路径 messages 每轮从零新建；心跳 prompt 也不含历史。
  病断言：test_left_brain_prompt_carries_history / test_fast_path_carries_history
  接线：test_execute_tick_stores_history / test_run_once_accepts_history

测试纪律：终端用例只在 tmp_path 造的目录里跑，不碰真实只读白名单与记忆库。
"""
import inspect
from types import SimpleNamespace

import pytest


# ═══════════════════════════════════════════════════════
# 工具：造一个只挂沙箱、不跑 __init__ 的 ISN
#   ISN.__init__ 会 bridge.scan() + 载 ToolRegistry + 读 HOME，
#   对本文件的断言全是噪声且慢；_terminal/_search_files 只依赖 self.sandbox。
# ═══════════════════════════════════════════════════════

def _isn(rw=None, ro=None):
    from openllm.core.isn_impl import ISN
    from openllm.core.sandbox import Sandbox
    obj = ISN.__new__(ISN)
    obj.sandbox = Sandbox(allowed_paths=rw, readonly_paths=ro)
    return obj


# ═══════════════════════════════════════════════════════
# 刀③-a：命令段意图判定
# ═══════════════════════════════════════════════════════

class TestSegmentOp:
    @staticmethod
    def _op(seg):
        from openllm.core.isn_impl import ISN
        return ISN._segment_op(seg)

    def test_readonly_commands_are_read(self):
        for seg in ("ls -la /mnt/i/openllm",
                    "cat /etc/hosts",
                    'grep -n foo "/tmp/openllm/a.txt"',
                    "head -50 /x/y.md"):
            assert self._op(seg) == "read", seg

    def test_cd_is_neutral(self):
        """cd 只换当前目录，不改文件系统内容 → 按读检查。"""
        assert self._op('cd "/mnt/i/hermes/output/818具神智能研究"') == "read"
        assert self._op("cd /mnt/i/openllm") == "read"

    def test_write_commands_are_write(self):
        for seg in ("tee /tmp/openllm/x",
                    "mkdir -p /tmp/openllm/x",
                    "python3 -c pass",
                    "sed -i s/a/b/ /tmp/openllm/f",
                    "cp /tmp/openllm/a /tmp/openllm/b"):
            assert self._op(seg) == "write", seg

    def test_git_read_subcommand_relaxed(self):
        assert self._op("git status") == "read"
        assert self._op("git log --oneline -5") == "read"
        # 值感知：-C 的下一个词元是路径不是子命令，不能被当成子命令
        assert self._op("git -C /mnt/i/openllm status") == "read"
        assert self._op("git -C /mnt/i/openllm push") == "write"

    def test_empty_segment_is_read(self):
        assert self._op("   ") == "read"


# ═══════════════════════════════════════════════════════
# 刀③-b：终端闸门（病断言 + 安全回归）
# ═══════════════════════════════════════════════════════

class TestTerminalGate:
    def test_cd_into_readonly_dir_allowed(self, tmp_path):
        """旧代码必红：首词 cd 被判写意图 → 只读目录被按写检查→误拒。"""
        ro = tmp_path / "ro"
        ro.mkdir()
        (ro / "x.md").write_text("hi", encoding="utf-8")
        isn = _isn(rw=[str(tmp_path / "rw")], ro=[str(ro)])

        out = isn._terminal(f'cd "{ro}" && ls')

        assert "[沙箱拒绝]" not in out, out
        assert "x.md" in out, out

    def test_mixed_cmd_read_path_allowed(self, tmp_path):
        """旧代码必红：混合命令里只读路径被按写检查→误拒。"""
        ro = tmp_path / "ro"
        ro.mkdir()
        (ro / "x.md").write_text("hi", encoding="utf-8")
        rw = tmp_path / "rw"
        rw.mkdir()
        isn = _isn(rw=[str(rw)], ro=[str(ro)])

        out = isn._terminal(f'mkdir -p "{rw}/sub" && ls "{ro}"')

        assert "[沙箱拒绝]" not in out, out
        assert "x.md" in out, out
        assert (rw / "sub").is_dir()

    def test_readonly_dir_listable_and_readable(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        (ro / "材料.md").write_text("研究资料", encoding="utf-8")
        isn = _isn(rw=[str(tmp_path / "rw")], ro=[str(ro)])

        assert "材料.md" in isn._terminal(f'ls "{ro}"')
        assert "研究资料" in isn._terminal(f'cat "{ro}/材料.md"')

    # ── 安全回归：修法不得放松写语义 ──

    def test_write_to_readonly_still_denied(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        isn = _isn(rw=[str(tmp_path / "rw")], ro=[str(ro)])

        out = isn._terminal(f'tee "{ro}/evil.txt"')

        assert "[沙箱拒绝]" in out, out
        assert not (ro / "evil.txt").exists()

    def test_redirect_to_readonly_denied(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        isn = _isn(rw=[str(tmp_path / "rw")], ro=[str(ro)])

        out = isn._terminal(f'echo x > "{ro}/evil.txt"')

        assert "[沙箱拒绝]" in out, out
        assert not (ro / "evil.txt").exists()

    def test_outside_rw_ro_still_denied(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        isn = _isn(rw=[str(tmp_path / "rw")], ro=[str(ro)])
        assert "[沙箱拒绝]" in isn._terminal("ls /etc")

    def test_dangerous_cmd_still_caught(self, tmp_path):
        isn = _isn(rw=[str(tmp_path)], ro=[])
        assert "[拦截]" in isn._terminal("rm -rf /tmp/openllm/x")
        assert "[拦截]" in isn._terminal("ls /tmp; rm -rf /tmp/openllm/x")
        assert "[拦截]" in isn._terminal("sudo ls")

    def test_hermes_substring_not_dangerous(self, tmp_path):
        """hermes 含子串 rm —— 旧子串匹配把它整条命令判成危险命令。"""
        from openllm.core.isn_impl import ISN
        assert ISN._find_dangerous_cmd(
            "ls -la /mnt/i/hermes/output/818具神智能研究") is None
        # 参数位的 rm 是搜索模式，不是命令位
        assert ISN._find_dangerous_cmd("grep rm /tmp/openllm/x") is None


# ═══════════════════════════════════════════════════════
# 刀③-c：search_files 覆盖只读资料库
# ═══════════════════════════════════════════════════════

class TestSearchFiles:
    def test_covers_readonly_root(self, tmp_path):
        """旧实现只搜 ~ 且 maxdepth=4 —— 研究资料库里的文件必然搜不到。"""
        ro = tmp_path / "ro"
        (ro / "deep" / "er").mkdir(parents=True)
        (ro / "deep" / "er" / "findme.md").write_text("x", encoding="utf-8")
        isn = _isn(rw=[str(tmp_path / "rw")], ro=[str(ro)])

        assert "findme.md" in isn._search_files("findme.md")

    def test_missing_returns_placeholder(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        isn = _isn(rw=[str(tmp_path / "rw")], ro=[str(ro)])
        assert isn._search_files("绝不存在的文件名zzz") == "[未找到]"


# ═══════════════════════════════════════════════════════
# 刀④：会话历史
# ═══════════════════════════════════════════════════════

class TestConversationHistory:
    def test_context_history_defaults_empty(self):
        from openllm.core.models import Context
        assert Context(user_message="hi").history == []

    def test_left_brain_prompt_carries_history(self):
        """旧代码必红：think() 的 prompt 里没有历史，模型不知道前文。"""
        from openllm.core.models import Context
        from openllm.iai.octopus import _LeftBrain

        lb = _LeftBrain.__new__(_LeftBrain)     # 跳过 __init__（不建真 provider）
        lb.provider = SimpleNamespace(model="stub", _available=True)
        captured = {}

        def _fake_chat(messages):
            captured["messages"] = messages
            return "我读完了，看法是……"

        lb._chat = _fake_chat
        ctx = Context(
            user_message="你的看法呢？",
            history=[
                {"role": "user", "content": "读读818具神智能研究"},
                {"role": "assistant", "content": "我读完了，先说结论"},
            ],
        )

        lb.think(ctx)

        prompt = captured["messages"][1]["content"]
        assert "读读818具神智能研究" in prompt, prompt
        assert "我读完了，先说结论" in prompt, prompt
        assert "你的看法呢？" in prompt, prompt

    def test_right_brain_prompt_carries_history(self):
        from openllm.core.models import Context, Proposal
        from openllm.iai.octopus import _RightBrain

        rb = _RightBrain.__new__(_RightBrain)
        rb.provider = SimpleNamespace(model="stub", _available=True)
        captured = {}

        def _fake_chat(messages):
            captured["m"] = messages
            return "approve"

        rb.provider.chat = _fake_chat

        ctx = Context(
            user_message="继续",
            history=[{"role": "user", "content": "读读818具神智能研究"}],
        )
        rb.review(ctx, Proposal(content="读完了", confidence=0.8))

        prompt = captured["m"][1]["content"]
        assert "读读818具神智能研究" in prompt, prompt

    def test_no_history_prompt_unchanged(self):
        """首轮（history 为空）行为与修前一致——不注入空块。"""
        from openllm.core.models import Context
        from openllm.iai.octopus import _LeftBrain

        lb = _LeftBrain.__new__(_LeftBrain)
        lb.provider = SimpleNamespace(model="stub", _available=True)
        captured = {}

        def _fake_chat(messages):
            captured["m"] = messages
            return "好"

        lb._chat = _fake_chat

        lb.think(Context(user_message="你好"))

        prompt = captured["m"][1]["content"]
        assert "本会话此前的对话" not in prompt

    def test_execute_tick_stores_history(self, monkeypatch):
        """run_once(history=...) → _pending_history → _perceive 读走。"""
        import openllm.iax.agent_heartbeat as hb
        from openllm.core import main_loop

        monkeypatch.setattr(hb, "execute_tick", lambda agent, msg: None)
        a = main_loop.Agent.__new__(main_loop.Agent)
        a._pending_history = []

        hist = [{"role": "user", "content": "上一句"}]
        a._execute_tick(main_loop.Message(text="hi"), hist)

        assert a._pending_history == hist
        # 深拷贝：外部改动不得回灌进 agent
        hist.append({"role": "assistant", "content": "x"})
        assert a._pending_history == [{"role": "user", "content": "上一句"}]

    def test_run_once_accepts_history(self):
        from openllm.core.main_loop import Agent
        assert "history" in inspect.signature(Agent.run_once).parameters

    def test_fast_path_carries_history(self):
        """旧代码必红：快路径 messages 每轮从零新建，历史全丢。"""
        from openllm.cli.main import AgentShell

        sh = AgentShell.__new__(AgentShell)
        sh._history = [
            {"role": "user", "content": "读读818具神智能研究"},
            {"role": "assistant", "content": "读完了"},
        ]
        captured = {}

        class _Provider:
            model = "stub"
            _available = True

            def chat(self, messages):
                captured["messages"] = messages
                return "我上一句说的是读完了"

        sh.agent = SimpleNamespace(
            octopus=SimpleNamespace(left=SimpleNamespace(
                provider=_Provider(),
                _extract_tool_calls=lambda _r: [],
            )),
        )

        sh.default("你的看法呢？")

        contents = [m["content"] for m in captured["messages"]]
        assert "读读818具神智能研究" in contents, contents
        assert "读完了" in contents, contents
        assert contents[-1] == "你的看法呢？"
        # 本轮问答落账
        assert sh._history[-2:] == [
            {"role": "user", "content": "你的看法呢？"},
            {"role": "assistant", "content": "我上一句说的是读完了"},
        ]

    def test_search_path_carries_identity_and_history(self, monkeypatch):
        """第三处直调点（_do_search）：原来既无 system 身份（暴露底层模型名），
        也无历史（搜完即忘）。两侧都要钉。"""
        import subprocess

        from openllm.cli.main import AgentShell

        class _Completed:
            stdout = "搜索结果：紫竹林在宜宾"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())

        sh = AgentShell.__new__(AgentShell)
        sh._history = [{"role": "user", "content": "上一轮的上下文"}]
        captured = {}

        class _Provider:
            model = "stub"
            _available = True

            def chat(self, messages):
                captured["m"] = messages
                return "紫竹林在宜宾"

        sh.agent = SimpleNamespace(octopus=SimpleNamespace(
            left=SimpleNamespace(provider=_Provider())))

        sh._do_search("紫竹林在哪")

        msgs = captured["m"]
        assert msgs[0]["role"] == "system", msgs
        assert "openLLM" in msgs[0]["content"], msgs
        assert any(m["content"] == "上一轮的上下文" for m in msgs), msgs
        assert msgs[-1]["role"] == "user"
        # 搜索问答同样入账
        assert sh._history[-2:] == [
            {"role": "user", "content": "紫竹林在哪"},
            {"role": "assistant", "content": "紫竹林在宜宾"},
        ]
