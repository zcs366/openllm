"""
provider 弹性刀测试（2026-09-25）——网络瞬时故障重试 + 错误人话化。

隔离策略：monkeypatch requests.post，绝不打外网。
timeout 常量 patch 为极小值防止真实阻塞。
"""
import pytest
import requests.exceptions as req_exc
from unittest.mock import MagicMock, call


# ══════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════

class _FakeResp:
    """伪造 requests.Response。"""
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code >= 400:
            raise req_exc.HTTPError(f"{self.status_code} Error",
                                    response=self)
    def json(self):
        return self._payload


def _bare_provider():
    """绕过 __init__ 造最小 provider。"""
    from openllm.core.provider_impl import LLMProvider
    p = LLMProvider.__new__(LLMProvider)
    p.model = "test-model"
    p.endpoint = "http://127.0.0.1:1/v1/chat/completions"
    p.api_key = "test-key"
    p._available = True
    p._last_usage = {}
    return p


def _success_resp():
    return _FakeResp({
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


def _ssl_error():
    return req_exc.SSLError("SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]')")


def _connect_error():
    return req_exc.ConnectionError("Connection refused")


# ══════════════════════════════════════════════════════════════════
# T1 · SSLError × 2 后成功 → chat 返回正文，post 共 3 次
# ══════════════════════════════════════════════════════════════════

class TestRetryTransientSuccess:
    def test_ssl_error_x2_then_success(self, monkeypatch):
        call_count = [0]
        def _mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise _ssl_error()
            return _success_resp()

        monkeypatch.setattr("requests.post", _mock_post)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert call_count[0] == 3, f"应调 3 次，实际 {call_count[0]}"

    def test_connection_error_x1_then_success(self, monkeypatch):
        call_count = [0]
        def _mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise _connect_error()
            return _success_resp()

        monkeypatch.setattr("requests.post", _mock_post)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert call_count[0] == 2


# ══════════════════════════════════════════════════════════════════
# T2 · 全败 SSLError×3 → 返回串含 "[LLM错误]" + "网络没连通" + "SSLError"
#      不含 "HTTPSConnectionPool"（技术原话不上脸）
# ══════════════════════════════════════════════════════════════════

class TestRetryAllFail:
    def test_ssl_error_all_fail_human_message(self, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **k: (_ for _ in ()).throw(_ssl_error()))
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert result.startswith("[LLM错误]"), f"必须以 [LLM错误] 开头: {result}"
        assert "网络没连通" in result, f"必须有人话: {result}"
        assert "SSLError" in result, f"必须含异常类型: {result}"
        assert "HTTPSConnectionPool" not in result, f"技术原话不得上脸: {result}"
        assert "已重试" in result

    def test_timeout_all_fail_human_message(self, monkeypatch):
        def _raise_timeout(*a, **k):
            raise req_exc.Timeout("Read timed out")
        monkeypatch.setattr("requests.post", _raise_timeout)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert "响应超时" in result
        assert "Timeout" in result
        assert "已重试" in result


# ══════════════════════════════════════════════════════════════════
# T3 · 400 业务错误 → 不重试，post 仅 1 次，"服务端拒绝" 原路径不变
# ══════════════════════════════════════════════════════════════════

class TestBusinessErrorNoRetry:
    def test_400_not_retried(self, monkeypatch):
        call_count = [0]
        def _mock_post(*args, **kwargs):
            call_count[0] += 1
            return _FakeResp({"error": {"message": "Bad request"}}, status_code=400)

        monkeypatch.setattr("requests.post", _mock_post)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        # raise_for_status 抛 HTTPError → 通用异常处理，不重试
        assert "[LLM错误]" in result
        assert "已重试" in result
        assert call_count[0] == 1, f"400 不得重试，实际 {call_count[0]} 次"

    def test_server_error_200_with_error_body_no_retry(self, monkeypatch):
        """DR-20260915 回归：服务端 200+error 体不应触发重试。"""
        call_count = [0]
        def _mock_post(*args, **kwargs):
            call_count[0] += 1
            return _FakeResp({"error": {"message": "900-second timeout"}})

        monkeypatch.setattr("requests.post", _mock_post)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert "900-second timeout" in result
        assert call_count[0] == 1, f"200+error 不得重试，实际 {call_count[0]} 次"


# ══════════════════════════════════════════════════════════════════
# T4 · 429 一次后 200 → 重试 1 次成功
# ══════════════════════════════════════════════════════════════════

class TestRateLimitRetry:
    def test_429_then_success(self, monkeypatch):
        call_count = [0]
        def _mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                resp = _FakeResp({"error": {"message": "Rate limited"}}, status_code=429)
                resp.raise_for_status()  # 手动触发以确保正确异常
                return resp
            return _success_resp()

        # 用一个抛异常的版本
        def _mock_post_with_raise(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise req_exc.HTTPError("429", response=MagicMock(status_code=429))
            return _success_resp()

        monkeypatch.setattr("requests.post", _mock_post_with_raise)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert call_count[0] == 2, f"429 应重试 1 次共 2 次，实际 {call_count[0]}"


# ══════════════════════════════════════════════════════════════════
# T5 · 开关 OFF → SSLError 直接失败仅调 1 次
# ══════════════════════════════════════════════════════════════════

class TestRetrySwitchOff:
    def test_switch_off_single_attempt(self, monkeypatch):
        monkeypatch.setenv("OPENLLM_PROVIDER_RETRY", "0")
        call_count = [0]
        def _mock_post(*args, **kwargs):
            call_count[0] += 1
            raise _ssl_error()

        monkeypatch.setattr("requests.post", _mock_post)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert call_count[0] == 1, f"开关 OFF 应仅 1 次，实际 {call_count[0]}"
        assert result.startswith("[LLM错误]")

    def test_switch_off_value_on(self, monkeypatch):
        """OPENLLM_PROVIDER_RETRY=on → 仍启用重试。"""
        monkeypatch.setenv("OPENLLM_PROVIDER_RETRY", "on")
        call_count = [0]
        def _mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise _ssl_error()
            return _success_resp()

        monkeypatch.setattr("requests.post", _mock_post)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert call_count[0] == 3


# ══════════════════════════════════════════════════════════════════
# T6 · 日志含完整原始异常（caplog 断言 str(e) 里的 host 在 log 里）
# ══════════════════════════════════════════════════════════════════

class TestLoggingFullException:
    def test_log_contains_original_exception_detail(self, monkeypatch, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="openllm.provider")

        def _raise_with_host(*a, **k):
            raise req_exc.SSLError(
                "HTTPSConnectionPool(host='api.deepseek.com', port=443): "
                "Max retries exceeded"
            )

        monkeypatch.setattr("requests.post", _raise_with_host)
        monkeypatch.setattr("time.sleep", lambda s: None)
        p = _bare_provider()
        result = p.chat([{"role": "user", "content": "hi"}])

        # 用户脸上的输出不含技术原话
        assert "deepseek.com" not in result
        # 但日志里必须有完整信息（可审计）
        assert "deepseek.com" in caplog.text, f"日志应含 host: {caplog.text}"
        assert "Max retries exceeded" in caplog.text, f"日志应含原始消息: {caplog.text}"
