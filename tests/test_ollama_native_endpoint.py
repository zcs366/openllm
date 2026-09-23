"""P0-1/P0-2 回归钉子 —— "苏醒即失语" 与 "错误体不透传"（2026-09-16 医师接骨）。

■ P0-1 病象：wake() 注入 system prompt 后，system + 27 工具 schema ≈ 4581 tokens
  > ollama 服务侧默认 num_ctx(4096) → 每次 chat 必 400。
  实测根因：/v1/chat/completions（OpenAI 兼容）**忽略请求级 num_ctx**
  （8192/32768 均 400），原生 /api/chat **认**（8192/16384 均 200）。

■ P0-2 病象：HTTP 400 被降格成 content 返回，服务端原话
  （"exceeds the available context size"）全程不露面。

■ 钉子设计纪律：新抽的 _extract_error_detail 无法用"拆回 HEAD"反证（HEAD 里没这个
  函数），因此除单元级断言外，必须有一条**行为级钉子**——
  TestErrorDetailPropagation::test_sync_chat_surfaces_detail 直接驱动真实
  _sync_chat() 并断言服务端原话出现在返回内容里，老代码会红。
"""
import json
import sys
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openllm.core.provider import (  # noqa: E402
    DeepSeekProvider, ModelConfig, create_provider,
)


# ── 测试替身：模拟 requests.Response ──────────────────────────

class FakeResp:
    """最小 Response 替身：status_code / json() / text / raise_for_status()。"""

    def __init__(self, status: int, payload=None, raw: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = raw or (json.dumps(payload, ensure_ascii=False) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Client Error: Bad Request for url: http://fake",
                response=self,
            )


def _ollama_400_body(tokens: int = 4581, n_ctx: int = 4096):
    """ollama 真实错误体形态：error.message 是**再套一层的 JSON 字符串**。"""
    inner = json.dumps({
        "error": {"code": 400,
                  "message": f"request ({tokens} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it",
                  "type": "exceed_context_size_error",
                  "n_prompt_tokens": tokens, "n_ctx": n_ctx},
    })
    return {"error": {"message": inner, "type": "invalid_request_error"}}


# ── P0-1：ollama 必须走原生端点 ───────────────────────────────

class TestOllamaNativeEndpoint:
    def test_returns_native_endpoint(self):
        """陷阱形态：若退回 DeepSeekProvider 打 /v1/chat/completions，本断言必红。"""
        p = create_provider("ollama", model="ilm-v6:latest")
        assert p.config.endpoint.endswith("/api/chat"), f"实际: {p.config.endpoint}"
        assert "/v1/" not in p.config.endpoint

    def test_endpoint_normalization_keeps_host(self):
        p = create_provider("ollama", model="m",
                            endpoint="http://127.0.0.1:11500/v1/chat/completions")
        assert p.config.endpoint == "http://127.0.0.1:11500/api/chat"

    def test_native_payload_carries_num_ctx_above_default(self):
        """陷阱形态：num_ctx 必须 > 4096（服务侧默认），否则苏醒后仍会 400。"""
        p = create_provider("ollama", model="m")
        body = p._native_payload({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100, "temperature": 0.5,
        })
        assert body["options"]["num_ctx"] > 4096, body["options"]
        assert body["options"]["num_predict"] == 100
        assert "max_tokens" not in body  # OpenAI 键不得泄漏到原生端点

    def test_native_payload_passes_tools_through(self):
        p = create_provider("ollama", model="m")
        body = p._native_payload({"model": "m", "messages": [], "tools": [{"type": "function"}]})
        assert body.get("tools") == [{"type": "function"}]

    def test_still_isinstance_deepseek(self):
        """继承的意义：engine 用 isinstance(provider, DeepSeekProvider) 决定是否下发 tools。"""
        p = create_provider("ollama", model="m")
        assert isinstance(p, DeepSeekProvider)


# ── P0-2：错误体透传 ──────────────────────────────────────────

class TestErrorDetailPropagation:
    def test_nested_ollama_error_body_unwrapped(self):
        """ollama 把 error.message 再套一层 JSON 字符串，必须剥开。"""
        detail = DeepSeekProvider._extract_error_detail(FakeResp(400, _ollama_400_body()))
        assert "exceeds the available context size" in detail
        assert "4581" in detail

    def test_openai_style_error(self):
        detail = DeepSeekProvider._extract_error_detail(
            FakeResp(401, {"error": {"message": "Invalid API key"}}))
        assert "Invalid API key" in detail

    def test_plain_string_error(self):
        detail = DeepSeekProvider._extract_error_detail(FakeResp(500, {"error": "boom"}))
        assert "boom" in detail

    def test_non_json_body(self):
        detail = DeepSeekProvider._extract_error_detail(
            FakeResp(502, raw="<html>Bad Gateway</html>"))
        assert "Bad Gateway" in detail

    def test_no_response(self):
        assert DeepSeekProvider._extract_error_detail(None) == "(无响应体)"

    def test_sync_chat_surfaces_detail(self, monkeypatch):
        """★ 行为级钉子（对老代码同样有效）。

        老代码在此返回 "[API错误] 400 Client Error: Bad Request for url: ..."，
        不含服务端原话 → 本断言红。这正是 P0-1 被误判成"模型不行"的原因。
        """
        p = DeepSeekProvider(ModelConfig(provider="deepseek", api_key="k", model="m"))
        monkeypatch.setattr(p._session, "post",
                            lambda *a, **kw: FakeResp(400, _ollama_400_body()))
        r = p._sync_chat({"model": "m", "messages": []}, time.time())
        assert "exceeds the available context size" in r.content, r.content
        assert "4581" in r.content
        assert "400" in r.content          # 状态码也要在

    def test_stream_chat_surfaces_detail(self, monkeypatch):
        """流式路径同样不许把错误降格成一句 status。"""
        p = DeepSeekProvider(ModelConfig(provider="deepseek", api_key="k", model="m"))
        monkeypatch.setattr(p._session, "post",
                            lambda *a, **kw: FakeResp(400, _ollama_400_body()))
        r = p._stream_chat({"model": "m", "messages": []}, lambda t: None, time.time())
        assert "exceeds the available context size" in r.content, r.content


# ── 端到端（需本地 ollama，不可达则跳过）──────────────────────

def _ollama_alive() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_alive(), reason="本地 ollama 未启动")
class TestOllamaEndToEnd:
    def test_native_endpoint_accepts_oversized_wake_payload(self):
        """真跑：构造超过服务侧默认 4096 的请求，原生端点必须 200。

        这是"苏醒即失语"的最小复现形态——老代码（/v1 端点）在此必 400。

        注意：本用例只验**请求被接受**（num_predict=1，不生成长文），
        否则本地 7B 处理 8000-token prompt 会拖到分钟级（实测 12000 字符
        prompt 会 read timeout）。prompt 取"刚越过 4096"的水位即可。
        """
        p = create_provider("ollama", model="ilm-v6:latest")
        big = "记忆注入" * 2000           # 8000 汉字 ≈ 5000+ tokens，越过 4096
        body = p._native_payload({
            "model": "ilm-v6:latest",
            "messages": [{"role": "system", "content": big},
                         {"role": "user", "content": "说一句话"}],
            "max_tokens": 1,              # 只验接受请求，不验生成
            "stream": False,
        })
        assert body["options"]["num_ctx"] > 4096
        resp = p._session.post(p.config.endpoint, json=body, timeout=300)
        assert resp.status_code == 200, resp.text[:300]
