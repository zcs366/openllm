"""ProviderRouter — 双脑provider路由。

本地Ollama（qwen3.5:9b，个性/主脑/断网独立）+ 云端API（教师/副脑）。
可切换+降级链+health四级对接+chat兼容LLMProvider签名。

配置优先级：环境变量 > ~/.openllm/config.json > 默认值。
API key从环境变量读，不硬编码。
"""
import json
import time
from pathlib import Path
from typing import Any


class ProviderUnavailable(Exception):
    """所有provider均不可用。"""


class ProviderRouter:
    """双脑provider路由：本地优先→云端降级→全部不可用抛异常。

    兼容LLMProvider的chat(messages)签名：_LeftBrain可直接用router.chat()。
    """

    # 降级链默认值
    _DEFAULT_LOCAL_HOST = "http://localhost:11434"
    _DEFAULT_LOCAL_MODEL = "qwen3.5:9b"
    _DEFAULT_CLOUD_BASE = "https://api.deepseek.com/v1/chat/completions"
    _DEFAULT_CLOUD_MODEL = "deepseek-chat"

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: 可选配置字典。None时从环境变量+config.json加载。
        """
        self._config = config if config is not None else self._load_config()

        # 本地provider配置
        self._local_host = self._config.get(
            "local_host", self._DEFAULT_LOCAL_HOST
        )
        self._local_model = self._config.get(
            "local_model", self._DEFAULT_LOCAL_MODEL
        )
        self._local_endpoint = f"{self._local_host}/v1/chat/completions"
        self._local_available = False

        # 云端provider配置
        self._cloud_base = self._config.get(
            "cloud_base", self._DEFAULT_CLOUD_BASE
        )
        self._cloud_model = self._config.get(
            "cloud_model", self._DEFAULT_CLOUD_MODEL
        )
        self._cloud_api_key = self._resolve_cloud_key(
            self._config.get("cloud_api_key_env", "OPENLLM_CLOUD_API_KEY"),
            self._config.get("cloud_api_key", ""),
        )
        self._cloud_available = bool(self._cloud_api_key)

        # 统计
        self._last_usage: dict = {}
        self._local_fail_count: int = 0

        # 初始健康检查
        self._check_local_health()

    # ── 配置加载 ────────────────────────────────────────────

    @staticmethod
    def _load_config() -> dict:
        """从环境变量和~/.openllm/config.json加载配置。

        环境变量优先于config.json。
        """
        cfg: dict = {}

        # config.json（兜底）
        config_path = Path.home() / ".openllm" / "config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    raw = json.load(f)
                # 提取providers中default_provider的配置
                providers = raw.get("providers", {})
                default = raw.get("default_provider", "deepseek")
                if default in providers:
                    p = providers[default]
                    cfg["cloud_base"] = p.get("endpoint", "")
                    cfg["cloud_model"] = p.get("model", "")
                    cfg["cloud_api_key"] = p.get("api_key", "")
            except (json.JSONDecodeError, OSError):
                pass

        # 环境变量覆盖config.json
        env_map = {
            "OPENLLM_LOCAL_HOST": "local_host",
            "OPENLLM_LOCAL_MODEL": "local_model",
            "OPENLLM_CLOUD_BASE": "cloud_base",
            "OPENLLM_CLOUD_MODEL": "cloud_model",
            "OPENLLM_CLOUD_API_KEY_ENV": "cloud_api_key_env",
            "OPENLLM_CLOUD_API_KEY": "cloud_api_key",
        }
        import os
        for env_key, cfg_key in env_map.items():
            val = os.environ.get(env_key, "")
            if val:
                cfg[cfg_key] = val

        return cfg

    @staticmethod
    def _resolve_cloud_key(env_name: str, direct_key: str) -> str:
        """解析云端API key：直接值 > 环境变量名读取。"""
        if direct_key:
            return direct_key
        import os
        return os.environ.get(env_name, "")

    # ── 健康检查 ────────────────────────────────────────────

    def _check_local_health(self) -> bool:
        """本地Ollama健康检查。POST /api/tags 简单探测。"""
        import urllib.request
        import urllib.error

        try:
            tags_url = f"{self._local_host}/api/tags"
            req = urllib.request.Request(tags_url, method="POST",
                                         data=b"{}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self._local_available = resp.status < 400
                self._local_fail_count = 0
        except (OSError, ValueError):
            self._local_available = False
            self._local_fail_count += 1

        return self._local_available

    def _check_cloud_health(self) -> bool:
        """云端API健康检查。GET base URL（不消耗API配额）。"""
        import urllib.request
        import urllib.error

        try:
            base = self._cloud_base
            # 去掉/chat/completions路径做轻量探测
            probe_url = base.split("/v1/")[0] + "/v1/models" if "/v1/" in base else base
            req = urllib.request.Request(probe_url)
            if self._cloud_api_key:
                req.add_header("Authorization", f"Bearer {self._cloud_api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._cloud_available = resp.status < 400
        except (OSError, ValueError):
            self._cloud_available = False

        return self._cloud_available

    def health(self) -> dict:
        """健康状态报告（使用缓存状态）。

        Returns:
            {
                "local": {"available": bool, "host": str, "model": str, "fail_count": int},
                "cloud": {"available": bool, "base": str, "model": str},
                "level": "FULL"|"DEGRADED"|"MINIMAL"|"OFFLINE",
            }
        """
        cloud_ok = self._cloud_available and bool(self._cloud_api_key)

        if self._local_available and cloud_ok:
            level = "FULL"
        elif not self._local_available and cloud_ok:
            level = "DEGRADED"
        elif self._local_available and not cloud_ok:
            level = "MINIMAL"
        else:
            level = "OFFLINE"

        return {
            "local": {
                "available": self._local_available,
                "host": self._local_host,
                "model": self._local_model,
                "fail_count": self._local_fail_count,
            },
            "cloud": {
                "available": cloud_ok,
                "base": self._cloud_base,
                "model": self._cloud_model,
            },
            "level": level,
        }

    def refresh_health(self) -> dict:
        """实时健康探测（HTTP调用）+ 更新缓存 + 返回报告。"""
        self._check_local_health()
        self._check_cloud_health()
        return self.health()

    # ── 路由选择 ────────────────────────────────────────────

    def select(self) -> str:
        """选择活跃provider。本地优先→云端降级→抛异常。

        Returns:
            "local" 或 "cloud"

        Raises:
            ProviderUnavailable: 所有provider均不可用。
        """
        if self._local_available:
            return "local"
        if self._cloud_available and self._cloud_api_key:
            return "cloud"
        raise ProviderUnavailable("所有provider均不可用")

    # ── chat（兼容LLMProvider.chat签名）────────────────────

    def chat(self, messages: list[dict]) -> str:
        """调LLM·返回文本。兼容LLMProvider.chat(messages)签名。

        路由策略：local优先→local失败fallback cloud→全部失败报错。
        """
        target = self.select()

        if target == "local":
            return self._chat_local(messages)
        return self._chat_cloud(messages)

    def _chat_local(self, messages: list[dict]) -> str:
        """本地Ollama chat。失败时fallback到云端。"""
        try:
            return self._do_chat(
                self._local_endpoint, self._local_model,
                messages, api_key="", timeout=120,
            )
        except OSError:
            # 本地超时/不可达→降级到云端
            self._local_available = False
            self._local_fail_count += 1
            if self._cloud_available and self._cloud_api_key:
                try:
                    return self._chat_cloud(messages)
                except OSError:
                    raise ProviderUnavailable(
                        "本地和云端均不可用"
                    ) from None
            raise ProviderUnavailable("本地不可用且无云端fallback")

    def _chat_cloud(self, messages: list[dict]) -> str:
        """云端API chat。"""
        return self._do_chat(
            self._cloud_base, self._cloud_model,
            messages, api_key=self._cloud_api_key, timeout=120,
        )

    @staticmethod
    def _do_chat(
        endpoint: str, model: str, messages: list[dict],
        api_key: str = "", timeout: int = 120,
    ) -> str:
        """底层HTTP调用（本地/云端共用）。"""
        import urllib.request
        import urllib.error

        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "lm-studio":
            headers["Authorization"] = f"Bearer {api_key}"

        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.7,
        }).encode("utf-8")

        req = urllib.request.Request(endpoint, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                usage = data.get("usage", {})
                # 提取回复内容
                choice = data["choices"][0]["message"]
                content = choice.get("content", "")
                # 推理模型: content可能为空(reasoning吃掉全部token)
                if not content and choice.get("reasoning_content"):
                    content = choice["reasoning_content"]
                return content
        except urllib.error.HTTPError as e:
            raise OSError(f"HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise OSError(f"网络错误: {e.reason}") from e

    # ── 工具方法 ────────────────────────────────────────────

    @property
    def _available(self) -> bool:
        """兼容LLMProvider._available。任一provider可用即为True。"""
        return self._local_available or (
            self._cloud_available and bool(self._cloud_api_key)
        )

    def available_providers(self) -> list[str]:
        """返回当前可用的provider列表。"""
        providers = []
        if self._local_available:
            providers.append("local")
        if self._cloud_available and self._cloud_api_key:
            providers.append("cloud")
        return providers

    def __repr__(self) -> str:
        return (
            f"ProviderRouter(local={self._local_available}, "
            f"cloud={self._cloud_available})"
        )
