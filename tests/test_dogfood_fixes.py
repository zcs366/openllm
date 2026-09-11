"""品尝师首驾修复回归测试（2026-09-10）。

钉住六处病灶修复，防止回退：
  P0-1 嘴漏思维链      → provider content/reasoning 分流
  P0-2 记忆唤醒断裂    → capsule 按时间而非文件名字典序选最新
  P0-3 "请记住X"失效   → memory_write/read/search 工具挂载并接通MemoryOS
  P1-1 挂名工具        → 注册表里的工具必须在安全闸有映射（否则默认拒绝）
  P1-2 iam_harness缺失 → 包加载注入sys.modules
  P1-3 config被无视    → default_provider 生效
"""
import json
import time
import tempfile
from pathlib import Path

import pytest


# ── P0-2：胶囊必须按时间选最新，不能按文件名字典序 ──────────────

class TestCapsuleRecency:
    def test_read_picks_newest_by_timestamp_not_filename(self):
        from openllm.memory.capsule import MemoryOS, TextCapsule

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # 陷阱：旧文件名字典序更大（zzz > aaa），旧逻辑会永远选中它
            old = TextCapsule(session_id="zzz_old", insights=["旧记忆"])
            old.timestamp = time.time() - 86400
            (d / f"v06_{old.session_id}.json").write_text(
                json.dumps(old.to_dict(), ensure_ascii=False), encoding="utf-8")

            new = TextCapsule(session_id="aaa_new", insights=["新记忆"])
            new.timestamp = time.time()
            (d / f"v06_{new.session_id}.json").write_text(
                json.dumps(new.to_dict(), ensure_ascii=False), encoding="utf-8")

            os_ = MemoryOS(d)
            os_.read()

            assert os_.text is not None
            assert os_.text.session_id == "aaa_new", (
                f"唤醒读到了 {os_.text.session_id}——回退到字典序了"
            )
            assert os_.text.insights == ["新记忆"]

    def test_read_with_explicit_session_id(self):
        """session_id 参数以前被完全无视，现在必须精确读取。"""
        from openllm.memory.capsule import MemoryOS, TextCapsule

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for sid in ("s1", "s2"):
                c = TextCapsule(session_id=sid, insights=[f"{sid}的记忆"])
                c.timestamp = time.time()
                (d / f"v06_{sid}.json").write_text(
                    json.dumps(c.to_dict(), ensure_ascii=False), encoding="utf-8")

            os_ = MemoryOS(d)
            os_.read(session_id="s1")
            assert os_.text.session_id == "s1"


# ── P1-1 / P0-3：注册表与安全闸必须同步 ────────────────────────

class TestGateCoverage:
    def test_every_registered_tool_has_gate_mapping(self):
        """每个注册工具都要在 PermissionGate.ACTION_MAP 里——否则用户一调就被
        '未知操作，默认拒绝' 拦掉（list_dir 曾因此挂名）。"""
        from openllm.tools import create_default_tools
        from openllm.security.gate import SecurityFoundation

        names = [t["name"] for t in create_default_tools().list_tools()]
        action_map = SecurityFoundation.ACTION_MAP if hasattr(
            SecurityFoundation, "ACTION_MAP") else None
        if action_map is None:
            from openllm.security.gate import PermissionGate
            action_map = PermissionGate.ACTION_MAP
        missing = [n for n in names if n not in action_map]
        assert not missing, f"这些工具在安全闸里没有映射，调用会被默认拒绝: {missing}"

    def test_list_dir_and_python_exec_pass_gate(self):
        from openllm.security.gate import (
            PermissionGate, AuditLog, PermissionLevel,
        )

        with tempfile.TemporaryDirectory() as tmp:
            gate = PermissionGate(AuditLog(Path(tmp) / "audit.jsonl"),
                                  PermissionLevel.LOCAL_WRITE)
            for tool in ("list_dir", "python_exec", "memory_read", "memory_write"):
                ok, reason = gate.check(tool)
                assert ok, f"{tool} 被闸门拒绝: {reason}"

    def test_unknown_tool_still_denied(self):
        """放行挂名工具不能让默认拒绝失效。"""
        from openllm.security.gate import (
            PermissionGate, AuditLog, PermissionLevel,
        )

        with tempfile.TemporaryDirectory() as tmp:
            gate = PermissionGate(AuditLog(Path(tmp) / "audit.jsonl"),
                                  PermissionLevel.LOCAL_WRITE)
            ok, _ = gate.check("rm_rf_everything")
            assert not ok


# ── P0-3：记忆工具端到端接通 ─────────────────────────────────

class TestMemoryTools:
    """记忆工具绑定在引擎自己的 MemoryOS 实例上（而非工具工厂里新建一个），
    否则工具写的和引擎注入读的是两份缓存，写进去也看不见。"""

    @staticmethod
    def _engine(tmp):
        from openllm.core.engine import OpenLLMEngine, AgentConfig
        import os
        os.environ.setdefault("OPENLLM_SECURITY_LEVEL", "3")
        return OpenLLMEngine(AgentConfig(capsule_dir=tmp, provider="ollama",
                                         model="nonexistent-model"))

    @staticmethod
    def _sandbox_home(tmp):
        """引擎的 MemoryOS 默认落在 ~/.openllm/output/memory，测试必须改HOME，
        否则会污染（并读到）用户真实记忆库。"""
        import os
        home = Path(tmp) / "fakehome"
        (home / ".openllm").mkdir(parents=True, exist_ok=True)
        prev = os.environ.get("HOME")
        os.environ["HOME"] = str(home)
        return prev

    def test_memory_tools_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            prev = self._sandbox_home(tmp)
            try:
                engine = self._engine(tmp)
                names = [t["name"] for t in engine.tools.list_tools()]
                assert {"memory_write", "memory_read", "memory_search"} <= set(names), (
                    f"记忆工具没进系统提示的工具清单: {sorted(names)}"
                )
            finally:
                import os
                if prev is not None:
                    os.environ["HOME"] = prev

    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            prev = self._sandbox_home(tmp)
            try:
                self._roundtrip(tmp)
            finally:
                import os
                if prev is not None:
                    os.environ["HOME"] = prev

    def _roundtrip(self, tmp):
            engine = self._engine(tmp)
            names = [t["name"] for t in engine.tools.list_tools()]
            assert {"memory_write", "memory_read", "memory_search"} <= set(names)

            r = engine.execute_tool(
                "memory_write", key="老搭档的口味", content="军师喝美式不加糖")
            assert r.success, f"memory_write 失败: {r.error}"
            assert "老搭档的口味" in r.output

            r2 = engine.execute_tool("memory_read", key="老搭档的口味")
            assert r2.success, f"memory_read 失败: {r2.error}"
            assert "美式" in r2.output, f"读回内容不对: {r2.output}"

            r3 = engine.execute_tool("memory_search", query="美式")
            assert r3.success
            assert "老搭档的口味" in r3.output

    def test_memory_survives_engine_restart(self):
        """跨会话记忆：新引擎实例必须读得到（P0-3 的核心诉求）。"""
        with tempfile.TemporaryDirectory() as tmp:
            import os
            home = Path(tmp) / "fakehome"
            (home / ".openllm").mkdir(parents=True)
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = str(home)
            try:
                e1 = self._engine(tmp)
                assert e1.execute_tool(
                    "memory_write", key="重启测试", content="这句话要活过重启").success

                e2 = self._engine(tmp)  # 模拟重启
                r = e2.execute_tool("memory_read", key="重启测试")
                assert r.success and "活过重启" in r.output, (
                    f"重启后记忆丢了: {r.output}"
                )
            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home


# ── P0-1：思维链不得混进回答 ──────────────────────────────────

class TestReasoningSeparation:
    class _FakeResp:
        def __init__(self, payload, stream_lines=None):
            self._payload = payload
            self._lines = stream_lines or []

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

        def iter_lines(self):
            for chunk in self._lines:
                yield chunk.encode("utf-8")

    def _provider(self):
        from openllm.core.provider import create_provider
        return create_provider("deepseek", model="m", api_key="k",
                               endpoint="http://127.0.0.1:9/v1/chat/completions")

    def test_sync_path_keeps_reasoning_out_of_content(self):
        p = self._provider()
        payload = {"choices": [{"message": {
            "content": "我是军师。",
            "reasoning_content": "（内心：他问我身份，我要不要自报家门……第3稿）",
        }, "finish_reason": "stop"}], "usage": {}, "model": "m"}
        p._session.post = lambda *a, **k: self._FakeResp(payload)

        resp = p.chat([], stream=False)
        assert resp.content == "我是军师。"
        assert "内心" not in resp.content
        assert "内心" in resp.reasoning

    def test_stream_path_keeps_reasoning_out_of_content(self):
        p = self._provider()
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"让我想想…"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"再想一层…"}}]}',
            'data: {"choices":[{"delta":{"content":"结论："}}]}',
            'data: {"choices":[{"delta":{"content":"四个字。"}}]}',
            'data: [DONE]',
        ]
        p._session.post = lambda *a, **k: self._FakeResp({}, lines)

        seen = []
        resp = p.chat([], stream=True, on_token=seen.append)
        assert resp.content == "结论：四个字。"
        assert "让我想想" not in "".join(seen), "思考过程被打到用户的终端上了"
        assert "让我想想" in resp.reasoning

    def test_inline_think_tag_stripped(self):
        p = self._provider()
        payload = {"choices": [{"message": {
            "content": "<think>偷偷想</think>答案在这。",
        }, "finish_reason": "stop"}], "usage": {}, "model": "m"}
        p._session.post = lambda *a, **k: self._FakeResp(payload)
        resp = p.chat([], stream=False)
        assert resp.content.strip() == "答案在这。"
        assert "偷偷想" not in resp.content


# ── P1-2：iam_harness 必须真的装上 ───────────────────────────

class TestIamHarnessLoad:
    def test_iam_harness_package_loads(self):
        from openllm.identity.iam_integration import IamIntegration, IAM_HARNESS_PATH
        if not IAM_HARNESS_PATH.exists():
            pytest.skip("本机未安装 iam_harness")

        iam = IamIntegration(auto_load=True)
        assert iam.is_loaded(), "iam_harness 未加载——身份约束层带病运行"
        assert iam.get_version()


# ── P1-3：config.json 的 default_provider 必须生效 ────────────

class TestConfigDefaultProvider:
    def test_default_provider_from_config_is_honored(self):
        import os
        from openllm.core.engine import OpenLLMEngine, AgentConfig

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "fakehome"
            (home / ".openllm").mkdir(parents=True)
            (home / ".openllm" / "config.json").write_text(json.dumps({
                "default_provider": "ollama",
                "providers": {"ollama": {
                    "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
                    "model": "qwen2.5:7b", "api_key": "ollama"}},
            }), encoding="utf-8")

            old_home = os.environ.get("HOME")
            os.environ["HOME"] = str(home)
            try:
                engine = OpenLLMEngine(AgentConfig(capsule_dir=tmp))
                assert engine.config.provider == "ollama", (
                    f"config.json 的 default_provider 被无视了，仍是 {engine.config.provider}"
                )
                assert engine.config.model == "qwen2.5:7b"
            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home

    def test_explicit_provider_wins_over_config(self):
        """显式传参优先级最高（品尝师第一驱就是靠它绕过了当时的bug）。"""
        import os
        from openllm.core.engine import OpenLLMEngine, AgentConfig

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "fakehome"
            (home / ".openllm").mkdir(parents=True)
            (home / ".openllm" / "config.json").write_text(json.dumps({
                "default_provider": "ollama",
                "providers": {"ollama": {"endpoint": "", "model": "x"}},
            }), encoding="utf-8")

            old_home = os.environ.get("HOME")
            os.environ["HOME"] = str(home)
            try:
                engine = OpenLLMEngine(AgentConfig(
                    capsule_dir=tmp, provider="deepseek", model="deepseek-chat"))
                assert engine.config.provider == "deepseek"
            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home


# ── 版本号单一真源 ────────────────────────────────────────────

class TestVersionSingleSource:
    def test_soul_engine_pyproject_agree(self):
        import re
        from openllm.identity.soul import Soul

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
        assert m, "pyproject.toml 没有 version"
        assert Soul().version == m.group(1), (
            f"身份版本 {Soul().version} 与包版本 {m.group(1)} 不一致——"
            "系统提示与banner曾各说各话"
        )
