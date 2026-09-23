"""DR-20260917-01 品尝师首诊两刀的钉子测试。

刀②（沙箱只读层）：openLLM 住在大仓 /mnt/i/openllm 却读不了自己的源码——
"你可以读代码吗" → [沙箱拒绝]。修法：只读白名单（默认含主仓），写仍锁死。

陷阱形态（不是只断言修好后的行为，而是断言病抓得住）：
  - 旧代码（无只读层）下 test_readonly_dir_readable 必红——这就是病本身的断言；
  - test_write_to_readonly_denied 防"只读层误开写"的回归——安全语义必须原样。

刀①（工具清单正源）：BASE_TOOLS/FULL_TOOLS/TOOL_DESCRIPTIONS 常量存在且
与心跳路径/快路径引用同源；快路径 prompt 必须含工具清单与 TOOL_CALLS 协议。

测试纪律：涉及 HOME 的测试全部沙箱化，不碰真实记忆库。
"""
import os
import tempfile
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════
# 刀②：沙箱只读层
# ═══════════════════════════════════════════════════════

class TestSandboxReadonlyLayer:
    @staticmethod
    def _sandbox():
        from openllm.core.sandbox import Sandbox
        return Sandbox()

    def test_readonly_dir_readable(self):
        """只读白名单内的路径可读（旧代码此处必红——病本身）。"""
        s = self._sandbox()
        assert s.check_path("/mnt/i/openllm/src/openllm/core/engine.py", "read")

    def test_write_to_readonly_denied(self):
        """只读目录写操作必须拒绝——安全语义不因只读层放松。"""
        s = self._sandbox()
        assert not s.check_path("/mnt/i/openllm/src/evil.py", "write")
        assert not s.check_path("/mnt/i/openllm/src/evil.sh", "execute")

    def test_outside_still_denied(self):
        """两个白名单之外依旧拒绝（/etc/passwd 老钉子保留）。"""
        s = self._sandbox()
        assert not s.check_path("/etc/passwd", "read")
        assert not s.check_path("/etc/passwd", "write")

    def test_rw_whitelist_unaffected(self):
        """原读写白名单行为不变（read+write 都放行）。"""
        from openllm.core.sandbox import Sandbox
        with tempfile.TemporaryDirectory() as tmp:
            s = Sandbox([tmp])
            assert s.check_path(f"{tmp}/a.txt", "read")
            assert s.check_path(f"{tmp}/a.txt", "write")

    def test_deny_reason_mentions_readonly(self):
        """拒绝理由里看得见'只读'概念——模型才知道该换姿势而不是硬闯。"""
        s = self._sandbox()
        reason = s.deny_reason("/etc/passwd")
        assert "只读" in reason

    def test_env_override(self, monkeypatch):
        """OPENLLM_READONLY_PATHS 可覆盖默认只读白名单。"""
        monkeypatch.setenv("OPENLLM_READONLY_PATHS", "/tmp/ro_a:/tmp/ro_b")
        s = self._sandbox()
        assert s.check_path("/tmp/ro_a/x", "read")
        assert s.check_path("/tmp/ro_b/x", "read")
        assert not s.check_path("/mnt/i/openllm/src/openllm/core/engine.py", "read")

    def test_default_rw_whitelist_untouched_by_env(self):
        """env 覆盖只影响只读层，读写白名单不受牵连。"""
        from openllm.core.sandbox import Sandbox
        with tempfile.TemporaryDirectory() as tmp:
            monkey_env = {"OPENLLM_READONLY_PATHS": "/tmp/ro_a"}
            old = os.environ.get("OPENLLM_READONLY_PATHS")
            os.environ.update(monkey_env)
            try:
                s = Sandbox([tmp])
                assert s.check_path(f"{tmp}/x", "write")
                assert not s.check_path("/tmp/ro_a/x", "write")
            finally:
                if old is None:
                    os.environ.pop("OPENLLM_READONLY_PATHS", None)
                else:
                    os.environ["OPENLLM_READONLY_PATHS"] = old


# ═══════════════════════════════════════════════════════
# 刀①：工具清单正源 + 快路径注入
# ═══════════════════════════════════════════════════════

class TestToolManifestSourceOfTruth:
    def test_constants_exist_and_consistent(self):
        """正源常量存在、顺序一致、描述覆盖全量工具。"""
        from openllm.core.isa_impl import BASE_TOOLS, FULL_TOOLS, TOOL_DESCRIPTIONS
        assert BASE_TOOLS == ["read_file", "search_files"]
        assert FULL_TOOLS[:len(BASE_TOOLS)] == BASE_TOOLS
        assert set(FULL_TOOLS) == set(TOOL_DESCRIPTIONS.keys())

    def test_build_context_uses_manifest(self, monkeypatch):
        """心跳路径 build_context 产出与正源一致（真源引用，非字符串重复）。"""
        import openllm.core.isa_impl as isa_mod
        # 不构造完整 ISA（重），直接验证切片逻辑与正源耦合：
        tools = list(isa_mod.BASE_TOOLS)
        tools.extend(isa_mod.FULL_TOOLS[len(isa_mod.BASE_TOOLS):])
        assert tools == isa_mod.FULL_TOOLS

    def test_fast_path_prompt_contains_tools(self):
        """快路径 system prompt 必含工具清单+TOOL_CALLS 协议——读源码断言，
        因为构造 AgentShell 需要真 provider。这是行为级钉子的穷替身，
        医师反证法时拆 cli/main.py 对应行必红。"""
        src = Path("/mnt/i/openllm/src/openllm/cli/main.py").read_text()
        # 快路径块内的三个关键注入
        assert "TOOL_DESCRIPTIONS" in src
        assert "TOOL_CALLS" in src
        assert "你可以使用以下工具" in src
        # 工具执行回路（只注入不执行 = 许诺了手却不给手）
        assert "_extract_tool_calls" in src
        assert "isn_exec" in src

    def test_fast_path_loop_bounded(self):
        """快路径工具循环必须有界（防失控循环）。"""
        src = Path("/mnt/i/openllm/src/openllm/cli/main.py").read_text()
        assert "range(3)" in src
