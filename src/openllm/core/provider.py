"""
OpenLLM Model Provider - 模型接入层。

支持多Provider架构（对标CX）：
  - deepseek: DeepSeek API (默认)
  - openai: OpenAI API / 兼容接口
  - local: 本地模型（2080 22GB可跑7B量化）

Phase 1: DeepSeek API 优先。
"""
import json
import os
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Generator
import requests

logger = logging.getLogger("openllm.provider")


# ── 配置 ────────────────────────────────────────────

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"  # 非推理模型，结构化输出友好
DEFAULT_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7


class ProviderType(Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    """模型配置。"""
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    api_key: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self):
        # 从环境变量读API key（按provider类型选择）
        if not self.api_key:
            key_map = {
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }
            env_var = key_map.get(self.provider, "DEEPSEEK_API_KEY")
            self.api_key = os.environ.get(env_var, "")


@dataclass
class Message:
    """单条消息。"""
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ModelResponse:
    """模型响应。"""
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: str = "stop"


# ── Provider 实现 ───────────────────────────────────

class DeepSeekProvider:
    """DeepSeek API Provider。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })

    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> ModelResponse:
        """发送对话请求。"""
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(messages, system),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }

        t0 = time.time()

        if stream and on_token:
            return self._stream_chat(payload, on_token, t0)
        else:
            return self._sync_chat(payload, t0)

    def _build_messages(self, messages: list[Message], system: Optional[str] = None) -> list[dict]:
        """构建消息列表。"""
        result = []
        if system:
            result.append({"role": "system", "content": system})
        for m in messages:
            result.append({"role": m.role, "content": m.content})
        return result

    def _sync_chat(self, payload: dict, t0: float) -> ModelResponse:
        """同步调用。"""
        try:
            resp = self._session.post(
                self.config.endpoint,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            response = ModelResponse(
                content=choice["message"]["content"],
                model=data.get("model", self.config.model),
                usage=usage,
                latency_ms=(time.time() - t0) * 1000,
                finish_reason=choice.get("finish_reason", "stop"),
            )
            # 成本跟踪：写入model trace
            self._record_model_trace(response, usage)
            return response
        except requests.exceptions.RequestException as e:
            return ModelResponse(
                content=f"[API错误] {e}",
                model="error",
                latency_ms=(time.time() - t0) * 1000,
            )

    def _stream_chat(
        self,
        payload: dict,
        on_token: Callable[[str], None],
        t0: float,
    ) -> ModelResponse:
        """流式调用——逐token回调。"""
        full_content = ""
        try:
            resp = self._session.post(
                self.config.endpoint,
                json=payload,
                timeout=120,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            full_content += token
                            on_token(token)
                    except (json.JSONDecodeError, KeyError):
                        continue
            response = ModelResponse(
                content=full_content,
                model=self.config.model,
                latency_ms=(time.time() - t0) * 1000,
            )
            # 成本跟踪：流式模式无usage数据，只记录latency
            self._record_model_trace(response, {})
            return response
        except requests.exceptions.RequestException as e:
            return ModelResponse(
                content=f"[流式错误] {e}",
                model="error",
                latency_ms=(time.time() - t0) * 1000,
            )

    def _record_model_trace(self, response: ModelResponse, usage: dict):
        """记录模型调用成本数据。"""
        try:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            # DeepSeek 定价估算: 输入$0.14/M, 输出$0.28/M
            cost_usd = (input_tokens * 0.14 + output_tokens * 0.28) / 1_000_000
            # 写入 ~/.io-s/traces/models.jsonl
            trace_dir = Path.home() / ".io-s" / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "turn_id": f"t{int(time.time())}",
                "model": response.model,
                "provider": self.config.provider,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": response.latency_ms,
                "cost_usd": round(cost_usd, 8),
                "timestamp": time.time(),
            }
            cost_path = trace_dir / "models.jsonl"
            with open(cost_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"cost跟踪跳过: {e}")


# ── Anthropic (Claude) Provider ─────────────────────

class AnthropicProvider:
    """Anthropic Claude API Provider。
    API格式与OpenAI不同：独立endpoint、x-api-key头、system独立字段。
    """

    ENDPOINT = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, config: ModelConfig):
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({
            "x-api-key": config.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        })

    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> ModelResponse:
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            payload["system"] = system

        t0 = time.time()
        try:
            if stream and on_token:
                return self._stream_chat(payload, on_token, t0)
            else:
                return self._sync_chat(payload, t0)
        except Exception as e:
            return ModelResponse(
                content=f"[Anthropic错误] {e}", model="error",
                latency_ms=(time.time() - t0) * 1000,
            )

    def _sync_chat(self, payload: dict, t0: float) -> ModelResponse:
        resp = self._session.post(self.ENDPOINT, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data["content"][0]["text"]
        except (KeyError, IndexError) as e:
            return ModelResponse(
                content=f"[Anthropic响应格式异常] {e}: {json.dumps(data)[:200]}",
                model="error", latency_ms=(time.time() - t0) * 1000,
            )
        usage = data.get("usage", {})
        return ModelResponse(
            content=content, model=data.get("model", self.config.model),
            usage={"prompt_tokens": usage.get("input_tokens", 0),
                   "completion_tokens": usage.get("output_tokens", 0)},
            latency_ms=(time.time() - t0) * 1000,
            finish_reason=data.get("stop_reason", "end_turn"),
        )

    def _stream_chat(self, payload: dict, on_token: Callable, t0: float) -> ModelResponse:
        payload["stream"] = True
        full = ""
        resp = self._session.post(self.ENDPOINT, json=payload, timeout=120, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if data.get("type") == "content_block_delta":
                    token = data.get("delta", {}).get("text", "")
                    if token:
                        full += token
                        on_token(token)
            except (json.JSONDecodeError, KeyError):
                continue
        return ModelResponse(
            content=full, model=self.config.model,
            latency_ms=(time.time() - t0) * 1000,
        )


# ── Gemini (Google) Provider ───────────────────────

class GeminiProvider:
    """Google Gemini API Provider。
    API格式：URL路径带key，响应结构不同。
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: ModelConfig):
        self.config = config
        self._session = requests.Session()
        self._api_key = config.api_key

    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> ModelResponse:
        # Gemini: system_instruction是独立字段
        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        endpoint = f"{self.BASE_URL}/models/{self.config.model}:generateContent?key={self._api_key}"
        stream_endpoint = f"{self.BASE_URL}/models/{self.config.model}:streamGenerateContent?key={self._api_key}&alt=sse"

        t0 = time.time()
        try:
            if stream and on_token:
                return self._stream_chat(payload, stream_endpoint, on_token, t0)
            else:
                return self._sync_chat(payload, endpoint, t0)
        except Exception as e:
            return ModelResponse(
                content=f"[Gemini错误] {e}", model="error",
                latency_ms=(time.time() - t0) * 1000,
            )

    def _sync_chat(self, payload: dict, endpoint: str, t0: float) -> ModelResponse:
        resp = self._session.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        try:
            content = candidates[0]["content"]["parts"][0]["text"] if candidates else ""
        except (KeyError, IndexError) as e:
            return ModelResponse(
                content=f"[Gemini响应格式异常] {e}: {json.dumps(data)[:200]}",
                model="error", latency_ms=(time.time() - t0) * 1000,
            )
        usage = data.get("usageMetadata", {})
        return ModelResponse(
            content=content, model=self.config.model,
            usage={"prompt_tokens": usage.get("promptTokenCount", 0),
                   "completion_tokens": usage.get("candidatesTokenCount", 0)},
            latency_ms=(time.time() - t0) * 1000,
        )

    def _stream_chat(self, payload: dict, endpoint: str, on_token: Callable, t0: float) -> ModelResponse:
        full = ""
        resp = self._session.post(endpoint, json=payload, timeout=120, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            try:
                data = json.loads(data_str)
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        token = part.get("text", "")
                        if token:
                            full += token
                            on_token(token)
            except (json.JSONDecodeError, KeyError):
                continue
        return ModelResponse(
            content=full, model=self.config.model,
            latency_ms=(time.time() - t0) * 1000,
        )


# ── Provider 工厂 ───────────────────────────────────

def create_provider(provider_type: str = DEFAULT_PROVIDER, **kwargs) -> Any:
    """创建Provider实例。支持：deepseek/openai/anthropic/gemini/ollama/local。"""
    config = ModelConfig(provider=provider_type, **kwargs)

    if provider_type == "anthropic":
        config.model = kwargs.get("model", "claude-sonnet-4-20250514")
        config.api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not config.api_key:
            raise ValueError("Anthropic需要ANTHROPIC_API_KEY环境变量")
        logger.info(f"Anthropic provider: {config.model}")
        return AnthropicProvider(config)

    if provider_type == "gemini":
        config.model = kwargs.get("model", "gemini-2.0-flash")
        config.api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        if not config.api_key:
            raise ValueError("Gemini需要GEMINI_API_KEY环境变量")
        logger.info(f"Gemini provider: {config.model}")
        return GeminiProvider(config)

    if provider_type == "ollama":
        config.endpoint = kwargs.get("endpoint", "http://localhost:11434/v1/chat/completions")
        config.api_key = kwargs.get("api_key", "ollama")
        config.model = kwargs.get("model", "qwen3.5:9b")
        logger.info(f"Ollama provider: {config.model} @ {config.endpoint}")
        return DeepSeekProvider(config)

    if provider_type == "deepseek":
        return DeepSeekProvider(config)
    elif provider_type == "openai":
        config.endpoint = kwargs.get("endpoint", "https://api.openai.com/v1/chat/completions")
        return DeepSeekProvider(config)  # 同接口
    else:
        raise ValueError(f"不支持的Provider: {provider_type}。支持: deepseek/openai/anthropic/gemini/ollama")
