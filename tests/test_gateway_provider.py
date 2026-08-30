"""
OpenLLM Gateway Provider 测试 — 为 gateway provider 新增代码补零测试覆盖缺口。

被测（未提交改动，4 个文件）：
  1. src/openllm/core/provider.py      — create_provider("gateway") 分支
  2. src/openllm/core/engine_utils.py  — get_api_key / get_endpoint 的 gateway 分支
  3. src/openllm/core/provider_impl.py — LLMProvider 的 127.0.0.1 key 文件回退
  4. src/openllm/core/engine.py        — OpenLLMEngine._init_provider 的 gateway 分支

隔离策略：autouse fixture 把 Path.home 重定向到 tmp_path，并删除
OPENLLM_GATEWAY_KEY / DEEPSEEK_API_KEY 等环境变量。
绝不触碰真实 ~/one-api/.gateway_key、真实 ~/.openllm/config.json，
测试全程不构造真实网络请求。
"""

import json
from pathlib import Path

import pytest

from openllm.core.engine import AgentConfig, OpenLLMEngine
from openllm.core.engine_utils import get_api_key, get_endpoint
from openllm.core.provider import DeepSeekProvider, create_provider
from openllm.core.provider_impl import LLMProvider

GATEWAY_DEFAULT_ENDPOINT = "http://127.0.0.1:13000/v1/chat/completions"
GATEWAY_DEFAULT_MODEL = "gemini-3.6-flash"
GATEWAY_KEY_FILE_REL = Path("one-api") / ".gateway_key"
CONFIG_JSON_REL = Path(".openllm") / "config.json"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """家目录重定向到 tmp_path + 清空相关环境变量（测试之间互不污染）。

    注意：ModelConfig.__post_init__ 对未知 provider 会 fallback 到
    DEEPSEEK_API_KEY 环境变量，因此必须一并删除，否则"全缺失"场景不可复现。
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in (
        "OPENLLM_GATEWAY_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "MIMO_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _write_gateway_key(home: Path, content: str = "file-key-123") -> Path:
    """在 fake home 下写 ~/one-api/.gateway_key。"""
    p = home / GATEWAY_KEY_FILE_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _write_openllm_config(home: Path, cfg: dict) -> Path:
    """在 fake home 下写 ~/.openllm/config.json。"""
    p = home / CONFIG_JSON_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# ── A. create_provider("gateway") 分支（provider.py） ─────────────

class TestCreateProviderGateway:
    def test_defaults_from_key_file(self, _isolated_home):
        """默认路径：key 来自 key 文件，endpoint/model 用默认值，trust_env=False。"""
        _write_gateway_key(_isolated_home)
        provider = create_provider("gateway")
        assert isinstance(provider, DeepSeekProvider)
        assert provider.config.endpoint == GATEWAY_DEFAULT_ENDPOINT
        assert provider.config.model == GATEWAY_DEFAULT_MODEL
        assert provider.config.api_key == "file-key-123"
        assert provider._session.trust_env is False

    def test_key_file_content_stripped(self, _isolated_home):
        """key 文件内容被 strip 处理。"""
        _write_gateway_key(_isolated_home, content="  padded-key \n")
        provider = create_provider("gateway")
        assert provider.config.api_key == "padded-key"

    def test_kwargs_override_endpoint_and_model(self, _isolated_home):
        """kwargs 显式传入 endpoint/model 时优先。"""
        _write_gateway_key(_isolated_home)
        custom_endpoint = "http://127.0.0.1:19999/custom/v1/chat/completions"
        provider = create_provider(
            "gateway", endpoint=custom_endpoint, model="custom-model-x",
        )
        assert provider.config.endpoint == custom_endpoint
        assert provider.config.model == "custom-model-x"
        assert provider.config.api_key == "file-key-123"  # key 仍来自文件

    def test_kwargs_api_key_beats_key_file(self, _isolated_home):
        """三级解析①：kwargs 的 api_key > key 文件。"""
        _write_gateway_key(_isolated_home, content="file-key")
        provider = create_provider("gateway", api_key="kw-key")
        assert provider.config.api_key == "kw-key"

    def test_key_file_beats_env(self, _isolated_home, monkeypatch):
        """三级解析②：env 有但文件也有 → 用文件的。"""
        _write_gateway_key(_isolated_home, content="file-key")
        monkeypatch.setenv("OPENLLM_GATEWAY_KEY", "env-key")
        provider = create_provider("gateway")
        assert provider.config.api_key == "file-key"

    def test_env_api_key_when_file_missing(self, _isolated_home, monkeypatch):
        """三级解析③：文件缺失时用 OPENLLM_GATEWAY_KEY 环境变量。"""
        monkeypatch.setenv("OPENLLM_GATEWAY_KEY", "env-key")
        provider = create_provider("gateway")
        assert provider.config.api_key == "env-key"

    def test_missing_all_keys_raises(self, _isolated_home):
        """三级 key 全缺失 → ValueError。"""
        with pytest.raises(ValueError, match="gateway需要api_key"):
            create_provider("gateway")


# ── B. engine_utils 的 gateway 分支（engine_utils.py） ────────────

class TestEngineUtilsGateway:
    def test_get_api_key_prefers_key_file(self, _isolated_home, monkeypatch):
        """get_api_key("gateway")：文件优先于 env。"""
        _write_gateway_key(_isolated_home, content="file-key")
        monkeypatch.setenv("OPENLLM_GATEWAY_KEY", "env-key")
        assert get_api_key("gateway") == "file-key"

    def test_get_api_key_env_fallback(self, _isolated_home, monkeypatch):
        """get_api_key("gateway")：文件缺失时 fallback 到 env。"""
        monkeypatch.setenv("OPENLLM_GATEWAY_KEY", "env-key")
        assert get_api_key("gateway") == "env-key"

    def test_get_api_key_empty_when_missing(self, _isolated_home):
        """get_api_key("gateway")：文件与 env 都缺失 → 空串。"""
        assert get_api_key("gateway") == ""

    def test_get_endpoint_returns_default(self, _isolated_home):
        """get_endpoint("gateway")：返回硬编码默认值，不读 config.json。"""
        _write_openllm_config(_isolated_home, {
            "providers": {"gateway": {"endpoint": "http://127.0.0.1:9999/x"}},
        })
        assert get_endpoint("gateway") == GATEWAY_DEFAULT_ENDPOINT


# ── C. LLMProvider 的 127.0.0.1 key 回退（provider_impl.py） ───────

class TestLLMProviderGatewayFallback:
    @staticmethod
    def _cfg(endpoint: str, api_key: str = "") -> dict:
        return {
            "default_provider": "deepseek",
            "providers": {"deepseek": {"endpoint": endpoint, "api_key": api_key}},
        }

    def test_reads_key_for_local_endpoint(self, _isolated_home):
        """endpoint 含 127.0.0.1 且 api_key 为空 → 自动读 key 文件并 _available=True。"""
        _write_gateway_key(_isolated_home, content="gw-key")
        _write_openllm_config(_isolated_home, self._cfg(GATEWAY_DEFAULT_ENDPOINT))
        p = LLMProvider()
        assert p.api_key == "gw-key"
        assert p._available is True

    def test_skips_file_for_remote_endpoint(self, _isolated_home):
        """endpoint 不含 127.0.0.1 → 不读 key 文件。"""
        _write_gateway_key(_isolated_home, content="gw-key")
        _write_openllm_config(
            _isolated_home, self._cfg("https://api.deepseek.com/v1/chat/completions"),
        )
        p = LLMProvider()
        assert p.api_key == ""
        assert p._available is False

    def test_local_endpoint_without_key_file_keeps_unavailable(self, _isolated_home):
        """endpoint 含 127.0.0.1 但 key 文件缺失 → api_key 仍空、_available=False。"""
        _write_openllm_config(_isolated_home, self._cfg(GATEWAY_DEFAULT_ENDPOINT))
        p = LLMProvider()
        assert p.api_key == ""
        assert p._available is False

    def test_existing_api_key_not_overridden(self, _isolated_home):
        """config.json 已有 api_key → 不回退读文件。"""
        _write_gateway_key(_isolated_home, content="file-key")
        _write_openllm_config(_isolated_home, self._cfg(GATEWAY_DEFAULT_ENDPOINT, api_key="cfg-key"))
        p = LLMProvider()
        assert p.api_key == "cfg-key"
        assert p._available is True


# ── D. OpenLLMEngine._init_provider 的 gateway 分支（engine.py） ───

class TestOpenLLMEngineGatewayConfig:
    def test_config_json_endpoint_model_and_override(self, _isolated_home):
        """config.json 存在：endpoint/model 被读取，cfg_model 覆盖 self.config.model。"""
        _write_gateway_key(_isolated_home, content="gw-key")
        _write_openllm_config(_isolated_home, {
            "providers": {"gateway": {
                "endpoint": GATEWAY_DEFAULT_ENDPOINT,
                "model": GATEWAY_DEFAULT_MODEL,
            }},
        })
        engine = OpenLLMEngine(AgentConfig(
            provider="gateway", model="initial-model",
            capsule_dir=str(_isolated_home),
        ))
        assert engine.connected
        assert engine.config.model == GATEWAY_DEFAULT_MODEL       # cfg_model 覆盖
        assert engine.provider.config.model == GATEWAY_DEFAULT_MODEL
        assert engine.provider.config.endpoint == GATEWAY_DEFAULT_ENDPOINT
        assert engine.provider.config.api_key == "gw-key"

    def test_config_json_api_key_fallback(self, _isolated_home):
        """key 文件缺失但 config.json 提供 api_key → provider 可用。"""
        _write_openllm_config(_isolated_home, {
            "providers": {"gateway": {
                "api_key": "cfg-key",
                "endpoint": GATEWAY_DEFAULT_ENDPOINT,
                "model": GATEWAY_DEFAULT_MODEL,
            }},
        })
        engine = OpenLLMEngine(AgentConfig(
            provider="gateway", model="initial-model",
            capsule_dir=str(_isolated_home),
        ))
        assert engine.connected
        assert engine.provider.config.api_key == "cfg-key"

    def test_no_config_json_uses_default_endpoint(self, _isolated_home):
        """config.json 不存在 → endpoint 用默认值，model 不被覆盖。"""
        _write_gateway_key(_isolated_home, content="gw-key")
        engine = OpenLLMEngine(AgentConfig(
            provider="gateway", model="my-model",
            capsule_dir=str(_isolated_home),
        ))
        assert engine.connected
        assert engine.provider.config.endpoint == GATEWAY_DEFAULT_ENDPOINT
        assert engine.config.model == "my-model"                  # 未被覆盖
        assert engine.provider.config.model == "my-model"
