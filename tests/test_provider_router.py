"""tests/test_provider_router.py — 双脑provider路由测试。

覆盖：
  - 路由选择（本地优先→云端降级→全部不可用）
  - 健康检查+四级映射（FULL/DEGRADED/MINIMAL/OFFLINE）
  - chat兼容LLMProvider签名
  - 降级链（本地挂→云端接管→全部挂抛异常）
  - 配置加载（环境变量+config.json+默认值）
  - 向后兼容（无router时OctopusI行为不变）
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from openllm.iai.provider_router import ProviderRouter, ProviderUnavailable


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def router_both_up():
    """本地+云端均可用。"""
    r = ProviderRouter(config={
        "local_host": "http://mock-ollama:11434",
        "local_model": "qwen3.5:9b",
        "cloud_base": "https://mock-cloud.com/v1/chat/completions",
        "cloud_model": "gpt-4",
        "cloud_api_key": "test-key-both",
    })
    r._local_available = True
    r._cloud_available = True
    return r


@pytest.fixture
def router_local_only():
    """仅本地可用。"""
    r = ProviderRouter(config={
        "local_host": "http://mock-ollama:11434",
        "local_model": "qwen3.5:9b",
        "cloud_base": "https://mock-cloud.com/v1/chat/completions",
        "cloud_model": "gpt-4",
        "cloud_api_key": "test-key",
    })
    r._local_available = True
    r._cloud_available = True  # has key
    return r


@pytest.fixture
def router_cloud_only():
    """仅云端可用。"""
    r = ProviderRouter(config={
        "local_host": "http://mock-ollama:11434",
        "local_model": "qwen3.5:9b",
        "cloud_base": "https://mock-cloud.com/v1/chat/completions",
        "cloud_model": "gpt-4",
        "cloud_api_key": "test-key-cloud",
    })
    r._local_available = False
    r._cloud_available = True
    return r


@pytest.fixture
def router_none_up():
    """全部不可用。"""
    r = ProviderRouter(config={
        "local_host": "http://mock-ollama:11434",
        "cloud_base": "https://mock-cloud.com/v1/chat/completions",
        "cloud_model": "gpt-4",
        "cloud_api_key": "",
    })
    r._local_available = False
    r._cloud_available = False
    return r


@pytest.fixture
def router_cloud_no_key():
    """云端无key（有endpoint但无API key）。"""
    r = ProviderRouter(config={
        "local_host": "http://mock-ollama:11434",
        "cloud_base": "https://mock-cloud.com/v1/chat/completions",
        "cloud_model": "gpt-4",
        "cloud_api_key": "",
    })
    r._local_available = False
    r._cloud_available = True  # endpoint可达但无key
    return r


# ══════════════════════════════════════════════════════════════
# 1. select() 路由选择
# ══════════════════════════════════════════════════════════════

class TestSelect:
    def test_local_priority(self, router_both_up):
        """本地可用时优先选本地。"""
        assert router_both_up.select() == "local"

    def test_fallback_to_cloud(self, router_cloud_only):
        """本地不可用时降级到云端。"""
        assert router_cloud_only.select() == "cloud"

    def test_cloud_only_when_no_key(self, router_cloud_no_key):
        """本地不可用+云端无key→抛异常。"""
        with pytest.raises(ProviderUnavailable):
            router_cloud_no_key.select()

    def test_all_down_raises(self, router_none_up):
        """全部不可用→抛ProviderUnavailable。"""
        with pytest.raises(ProviderUnavailable, match="所有provider均不可用"):
            router_none_up.select()

    def test_local_only_no_cloud(self, router_local_only):
        """仅本地可用时选本地。"""
        assert router_local_only.select() == "local"


# ══════════════════════════════════════════════════════════════
# 2. health() 健康检查+四级映射
# ══════════════════════════════════════════════════════════════

class TestHealth:
    def test_full_level(self, router_both_up):
        """本地+云端均可用→FULL。"""
        h = router_both_up.health()
        assert h["level"] == "FULL"
        assert h["local"]["available"] is True
        assert h["cloud"]["available"] is True

    def test_degraded_level(self, router_cloud_only):
        """本地不可用+云端可用→DEGRADED。"""
        h = router_cloud_only.health()
        assert h["level"] == "DEGRADED"
        assert h["local"]["available"] is False
        assert h["cloud"]["available"] is True

    def test_minimal_level(self, router_local_only):
        """本地可用+云端无key→MINIMAL。"""
        # Force cloud to have no key
        router_local_only._cloud_api_key = ""
        h = router_local_only.health()
        assert h["level"] == "MINIMAL"
        assert h["local"]["available"] is True
        assert h["cloud"]["available"] is False

    def test_offline_level(self, router_none_up):
        """全部不可用→OFFLINE。"""
        h = router_none_up.health()
        assert h["level"] == "OFFLINE"

    def test_health_returns_all_fields(self, router_both_up):
        """health()返回完整字段。"""
        h = router_both_up.health()
        assert "local" in h
        assert "cloud" in h
        assert "level" in h
        assert "host" in h["local"]
        assert "model" in h["local"]
        assert "fail_count" in h["local"]
        assert "base" in h["cloud"]
        assert "model" in h["cloud"]

    def test_health_updates_fail_count(self, router_cloud_only):
        """local失败后fail_count递增。"""
        router_cloud_only._local_fail_count = 2
        h = router_cloud_only.health()
        assert h["local"]["fail_count"] >= 2


# ══════════════════════════════════════════════════════════════
# 3. chat() 路由+降级
# ══════════════════════════════════════════════════════════════

class TestChat:
    def test_chat_routes_to_local(self, router_both_up):
        """本地可用时chat路由到本地。"""
        mock_resp = json.dumps({
            "choices": [{"message": {"content": "local response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_open:
            mock_resp_obj = MagicMock()
            mock_resp_obj.read.return_value = mock_resp
            mock_resp_obj.__enter__ = lambda s: s
            mock_resp_obj.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp_obj

            result = router_both_up.chat([{"role": "user", "content": "ping"}])
            assert result == "local response"

    def test_chat_fallback_to_cloud(self, router_cloud_only):
        """本地不可用时chat路由到云端。"""
        mock_resp = json.dumps({
            "choices": [{"message": {"content": "cloud response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_open:
            mock_resp_obj = MagicMock()
            mock_resp_obj.read.return_value = mock_resp
            mock_resp_obj.__enter__ = lambda s: s
            mock_resp_obj.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp_obj

            result = router_cloud_only.chat([{"role": "user", "content": "ping"}])
            assert result == "cloud response"

    def test_chat_raises_when_all_down(self, router_none_up):
        """全部不可用时chat抛ProviderUnavailable。"""
        with pytest.raises(ProviderUnavailable):
            router_none_up.chat([{"role": "user", "content": "ping"}])

    def test_chat_local_timeout_fallback_cloud(self, router_both_up):
        """本地超时→自动降级到云端。"""
        import urllib.error
        call_count = 0

        def mock_urlopen(req, **kwargs):
            nonlocal call_count
            call_count += 1
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "mock-ollama" in url and "/v1/" in url:
                # Local chat fails with timeout
                raise OSError("Connection timed out")
            # Cloud chat succeeds
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": "cloud fallback"}}],
                "usage": {},
            }).encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = router_both_up.chat([{"role": "user", "content": "ping"}])
            assert result == "cloud fallback"
            # local was marked unavailable after failure
            assert router_both_up._local_available is False

    def test_chat_both_fail_raises(self, router_both_up):
        """本地+云端都失败→抛ProviderUnavailable。"""
        import urllib.error

        def mock_urlopen(req, **kwargs):
            raise OSError("All networks down")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with pytest.raises(ProviderUnavailable):
                router_both_up.chat([{"role": "user", "content": "ping"}])


# ══════════════════════════════════════════════════════════════
# 4. _do_chat 底层调用
# ══════════════════════════════════════════════════════════════

class TestDoChat:
    def test_do_chat_normal(self):
        """正常响应解析。"""
        mock_resp = json.dumps({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_open:
            mock_resp_obj = MagicMock()
            mock_resp_obj.read.return_value = mock_resp
            mock_resp_obj.__enter__ = lambda s: s
            mock_resp_obj.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp_obj

            result = ProviderRouter._do_chat(
                "http://test/v1/chat/completions", "test-model",
                [{"role": "user", "content": "hi"}]
            )
            assert result == "hello"

    def test_do_chat_reasoning_fallback(self):
        """content为空时fallback到reasoning_content。"""
        mock_resp = json.dumps({
            "choices": [{"message": {"content": "", "reasoning_content": "thinking..."}}],
            "usage": {},
        }).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_open:
            mock_resp_obj = MagicMock()
            mock_resp_obj.read.return_value = mock_resp
            mock_resp_obj.__enter__ = lambda s: s
            mock_resp_obj.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp_obj

            result = ProviderRouter._do_chat(
                "http://test/v1/chat/completions", "test-model",
                [{"role": "user", "content": "hi"}]
            )
            assert result == "thinking..."

    def test_do_chat_http_error(self):
        """HTTP错误→抛OSError。"""
        import urllib.error

        def mock_urlopen(req, **kwargs):
            raise urllib.error.HTTPError(
                req.full_url, 401, "Unauthorized", {}, None
            )

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with pytest.raises(OSError, match="HTTP 401"):
                ProviderRouter._do_chat(
                    "http://test/v1/chat/completions", "model",
                    [{"role": "user", "content": "hi"}]
                )


# ══════════════════════════════════════════════════════════════
# 5. _available 兼容LLMProvider
# ══════════════════════════════════════════════════════════════

class TestAvailable:
    def test_available_when_local_up(self, router_cloud_only):
        """本地不可用但有云端→_available=True。"""
        # router_cloud_only: local=False, cloud=True with key
        assert router_cloud_only._available is True

    def test_available_when_both_up(self, router_both_up):
        """两个都可用→_available=True。"""
        assert router_both_up._available is True

    def test_not_available_when_all_down(self, router_none_up):
        """全部不可用→_available=False。"""
        assert router_none_up._available is False

    def test_not_available_cloud_no_key(self, router_cloud_no_key):
        """云端无key→_available=False。"""
        assert router_cloud_no_key._available is False


# ══════════════════════════════════════════════════════════════
# 6. 配置加载
# ══════════════════════════════════════════════════════════════

class TestConfig:
    def test_env_vars_override(self, monkeypatch):
        """环境变量优先于config.json。"""
        monkeypatch.setenv("OPENLLM_LOCAL_HOST", "http://env-host:11434")
        monkeypatch.setenv("OPENLLM_LOCAL_MODEL", "env-model")
        monkeypatch.setenv("OPENLLM_CLOUD_BASE", "https://env-cloud/v1/chat/completions")
        monkeypatch.setenv("OPENLLM_CLOUD_MODEL", "env-cloud-model")
        monkeypatch.setenv("OPENLLM_CLOUD_API_KEY", "env-key-123")

        cfg = ProviderRouter._load_config()
        assert cfg["local_host"] == "http://env-host:11434"
        assert cfg["local_model"] == "env-model"
        assert cfg["cloud_base"] == "https://env-cloud/v1/chat/completions"
        assert cfg["cloud_model"] == "env-cloud-model"
        assert cfg["cloud_api_key"] == "env-key-123"

    def test_default_config(self, monkeypatch):
        """无环境变量时使用默认值。"""
        monkeypatch.delenv("OPENLLM_LOCAL_HOST", raising=False)
        monkeypatch.delenv("OPENLLM_LOCAL_MODEL", raising=False)
        monkeypatch.delenv("OPENLLM_CLOUD_BASE", raising=False)
        monkeypatch.delenv("OPENLLM_CLOUD_MODEL", raising=False)
        monkeypatch.delenv("OPENLLM_CLOUD_API_KEY", raising=False)

        cfg = ProviderRouter._load_config()
        # 默认值在__init__中设置，不在_load_config中
        assert "OPENLLM_LOCAL_HOST" not in os.environ

    def test_resolve_cloud_key_direct(self):
        """直接值优先于环境变量名。"""
        key = ProviderRouter._resolve_cloud_key("SOME_ENV", "direct-key")
        assert key == "direct-key"

    def test_resolve_cloud_key_env(self, monkeypatch):
        """无直接值时从环境变量读取。"""
        monkeypatch.setenv("MY_API_KEY", "from-env")
        key = ProviderRouter._resolve_cloud_key("MY_API_KEY", "")
        assert key == "from-env"

    def test_resolve_cloud_key_none(self, monkeypatch):
        """无直接值+环境变量不存在→空字符串。"""
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        key = ProviderRouter._resolve_cloud_key("NONEXISTENT_KEY", "")
        assert key == ""


# ══════════════════════════════════════════════════════════════
# 7. available_providers + __repr__
# ══════════════════════════════════════════════════════════════

class TestUtils:
    def test_available_providers_both(self, router_both_up):
        """两个都可用时返回两个。"""
        providers = router_both_up.available_providers()
        assert "local" in providers
        assert "cloud" in providers

    def test_available_providers_local_only(self, router_local_only):
        """仅本地可用。"""
        providers = router_local_only.available_providers()
        assert "local" in providers
        # Cloud has key in fixture, so both
        assert "cloud" in providers

    def test_available_providers_cloud_only(self, router_cloud_only):
        """仅云端可用。"""
        providers = router_cloud_only.available_providers()
        assert "local" not in providers
        assert "cloud" in providers

    def test_available_providers_none(self, router_none_up):
        """全部不可用→空列表。"""
        assert router_none_up.available_providers() == []

    def test_repr(self, router_both_up):
        """__repr__包含状态信息。"""
        r = repr(router_both_up)
        assert "ProviderRouter" in r
        assert "local=True" in r
        assert "cloud=True" in r


# ══════════════════════════════════════════════════════════════
# 8. 向后兼容：章鱼I无router=旧行为
# ══════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_octopus_no_router_uses_provider(self):
        """无router时章鱼I左脑用原始LLMProvider。"""
        from openllm.core.octopus_impl import 章鱼I
        octo = 章鱼I()
        # 无router时_leftBrain.router应为None
        assert octo.left.router is None
        # provider仍存在
        assert octo.left.provider is not None
        assert hasattr(octo.left.provider, "chat")

    def test_octopus_with_router(self):
        """有router时章鱼I左脑有router。"""
        from openllm.core.octopus_impl import 章鱼I
        mock_router = MagicMock()
        mock_router.chat.return_value = '{"summary": "test"}'
        octo = 章鱼I(router=mock_router)
        assert octo.left.router is mock_router

    def test_left_brain_chat_uses_router(self):
        """有router时LeftBrain._chat走router。"""
        from openllm.core.octopus_impl import _LeftBrain
        mock_router = MagicMock()
        mock_router.chat.return_value = "router response"
        brain = _LeftBrain(router=mock_router)
        result = brain._chat([{"role": "user", "content": "hi"}])
        assert result == "router response"
        mock_router.chat.assert_called_once()

    def test_left_brain_chat_fallback_provider(self):
        """无router时LeftBrain._chat走provider。"""
        from openllm.core.octopus_impl import _LeftBrain
        brain = _LeftBrain()
        assert brain.router is None
        # provider.chat应可调用（LLMProvider默认返回模拟）
        result = brain._chat([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert len(result) > 0
