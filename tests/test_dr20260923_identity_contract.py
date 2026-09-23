"""test_dr20260923_identity_contract.py — 身份契约钉子（DR-20260923 残留处置A）。

病灶（0923品尝实测）：7B 答"我叫 Taster0923，是张成市…的创造者"——
会话名当身份、创造者揽到自己头上。根因：system prompt 无硬身份契约，
identity 参数缺什么模型就编什么。

钉子（对旧代码同样有效）：
  - 契约必须是 system prompt 的固定首段，identity="" 时依然在场
    （旧码 parts=[identity] → 空 identity 时无契约 → 红）
  - 契约必须含三要素：创造者归属（张成市是创造者）、代号非身份、底层模型非名字
"""
from openllm.core.engine_utils import build_system_prompt

CONTRACT_KEY = "身份契约"


class TestIdentityContract:
    def test_contract_present_with_empty_identity(self):
        """identity 缺失时契约仍在场——固定段，非 identity 衍生（旧码红）。"""
        prompt = build_system_prompt([], "", {})
        assert CONTRACT_KEY in prompt, "system prompt 无身份契约段（空identity时）"
        assert prompt.index(CONTRACT_KEY) == 0 or prompt.startswith(CONTRACT_KEY) or \
            prompt.split("\n", 1)[0].endswith(CONTRACT_KEY) or CONTRACT_KEY in prompt.split("\n")[0], \
            "契约必须在首行/首段（优先级位）"

    def test_contract_three_elements(self):
        """三要素齐全：创造者归属/代号非身份/底层模型非名字。"""
        prompt = build_system_prompt([], "一段普通的identity文本", {})
        assert "张成市是创造者" in prompt, "缺创造者归属——7B会再把创造者揽到自己头上"
        assert "会话名" in prompt and "openLLM 自称" in prompt, "缺代号非身份约束"
        assert "底层模型" in prompt, "缺底层模型约束"

    def test_contract_beats_conflicting_memory_claim(self):
        """冲突裁决条款在场：记忆把创造者记成自己时以契约为准。"""
        prompt = build_system_prompt([], "", {})
        assert "以本契约为准" in prompt, "缺冲突裁决条款——wake念错记忆时无法翻案"
