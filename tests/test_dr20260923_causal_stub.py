"""test_dr20260923_causal_stub.py — 因果stub升级钉子（陷阱形态 · 医师刀DR-20260923）。

病灶：engine.chat() 后的因果自动写入 success=len(response)>10 恒真、
prediction 为占位符"模型会基于记忆和知识回答" → 真实库373条 delta_magnitude
全 0.0 —— 因果记忆成空壳聊天日志（0923品尝师实测定案）。

钉子（断言 bug 触发条件本身，两条对旧代码同样有效）：
  1. 验证管线 failed_steps>0 → 写盘记录 actual_success 必须 False 且 delta_magnitude>0
     （旧码恒 True/0.0 → 红）
  2. prediction 不得是占位符，必须含本轮路径判断与验证步数
     （旧码恒为占位符 → 红）
  3. 验证通过 → success=True、delta=0、actual 带验证票数（防止矫枉过正把成功也写失败）
  4. provider._available：DeepSeek族补缺（旧码无此属性 → AttributeError → 红）

HOME 隔离靠 conftest 全局打桩 + 本文件 autouse fixture（照抄 test_tool_calling 姿势），
写入落点=DEFAULT_STORE_DIR（conftest 重定向后的假家目录），绝不碰 ~/.openllm 真实库。
"""
import json
import os
from pathlib import Path

import pytest

from openllm.core.engine import AgentConfig, OpenLLMEngine

PLACEHOLDER = "模型会基于记忆和知识回答"
MARK_FAIL = "DR0923因果钉子失败场景XYZQQ"
MARK_OK = "DR0923因果钉子成功场景ABCMM"


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """家目录重定向 + 清空API key环境变量（照抄 test_tool_calling._isolate_home）。"""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) + p.lstrip("~"))
    for var in (
        "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY", "MIMO_API_KEY", "ALIBABA_PLAN_API_KEY",
        "OPENLLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _find_records(marker):
    """按 action_signature 里的标记找写盘的因果记录。"""
    from openllm.isa.causal_memory import DEFAULT_STORE_DIR
    store = Path(DEFAULT_STORE_DIR)
    hits = []
    if not store.exists():
        return hits
    for f in store.glob("auto-*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if marker in (d.get("action_signature") or ""):
            hits.append(d)
    return hits


class _Report:
    def __init__(self, final_response, passed, failed):
        self.final_response = final_response
        self.passed_steps = passed
        self.failed_steps = failed
        self.total_steps = passed + failed


def _drive_chat(monkeypatch, tmp_path, marker, passed, failed):
    """驱动真实 engine.chat()：假 _call_model + 假验证管线，返回写盘记录列表。"""
    eng = OpenLLMEngine(AgentConfig(
        capsule_dir=str(tmp_path / "caps"),
        enable_io_s_checkpoint=False,
    ))
    # connected 判定依赖 provider——不注入则 chat 在 [未连接API] 处早退，
    # 因果写入块根本执行不到（首跑实测的坑，附 trap 断言钉住早退路径）
    from types import SimpleNamespace
    eng.provider = SimpleNamespace(_available=True, chat=lambda *a, **k: None)
    monkeypatch.setattr(
        eng, "_call_model",
        lambda stream=True: f"这是一段足够长的模型回复，内容正常。{marker}",
    )
    import openllm.core.verification_pipeline as vp_mod

    class _FakeVP:
        def __init__(self, engine):
            pass

        def run(self, resp):
            return _Report(resp, passed, failed)

    monkeypatch.setattr(vp_mod, "VerificationPipeline", _FakeVP)
    resp = eng.chat(marker, stream=False)
    # trap：chat 必须走完主路径到因果写入（早退路径会带 [未连接API] 前缀）
    assert marker in (resp or ""), f"chat未走完主路径(早退?): {resp!r}"
    # writer.record 在 chat 内部同步完成，直接读盘
    return _find_records(marker)


class TestCausalStub:
    def test_verify_fail_writes_failure_record(self, monkeypatch, tmp_path):
        """陷阱形态：验证失败必须落 success=False + delta>0（旧码恒True/0 → 红）。"""
        hits = _drive_chat(monkeypatch, tmp_path, MARK_FAIL, passed=1, failed=2)
        assert hits, f"因果记录未写盘: {MARK_FAIL}"
        rec = hits[-1]
        assert rec["actual_success"] is False, (
            f"验证失败(2步)但 actual_success={rec['actual_success']} —— 恒真病复发")
        assert rec["delta_magnitude"] > 0, (
            f"失败记录 delta_magnitude={rec['delta_magnitude']} —— 空壳病复发")

    def test_prediction_not_placeholder(self, monkeypatch, tmp_path):
        """陷阱形态：prediction 必须是本轮路径判断（旧码恒为占位符 → 红）。"""
        hits = _drive_chat(monkeypatch, tmp_path, MARK_OK, passed=3, failed=0)
        assert hits, f"因果记录未写盘: {MARK_OK}"
        rec = hits[-1]
        assert PLACEHOLDER not in (rec.get("prediction") or ""), (
            f"prediction 仍是占位符: {rec.get('prediction')}")
        assert "验证管线" in (rec.get("prediction") or ""), (
            f"prediction 缺验证管线判断: {rec.get('prediction')}")

    def test_verify_pass_still_success_with_tally(self, monkeypatch, tmp_path):
        """防矫枉过正：验证全过 → success=True、delta=0、actual 带票数。"""
        hits = _drive_chat(monkeypatch, tmp_path, MARK_OK + "2", passed=3, failed=0)
        assert hits, "因果记录未写盘"
        rec = hits[-1]
        assert rec["actual_success"] is True
        assert rec["delta_magnitude"] == 0.0
        assert "3/3" in (rec.get("actual_result") or ""), (
            f"actual 缺验证票数: {rec.get('actual_result')}")


class TestProviderAvailable:
    def test_deepseek_family_exposes_available(self):
        """DR-20260923: 本族 provider 必须有 _available，否则 CLI 恒降级 BASE_TOOLS。"""
        from openllm.core.provider import DeepSeekProvider, ModelConfig, OllamaProvider

        keyed = DeepSeekProvider(ModelConfig(api_key="k", endpoint="https://api.example.com/v1"))
        assert getattr(keyed, "_available", None) is True

        local = OllamaProvider(ModelConfig(api_key="", endpoint="http://localhost:11434/api/chat"))
        assert getattr(local, "_available", None) is True, "ollama localhost 免key应可用"

        remote_nokey = DeepSeekProvider(ModelConfig(api_key="", endpoint="https://api.example.com/v1"))
        assert getattr(remote_nokey, "_available", None) is False, "无key远程必须不可用"
