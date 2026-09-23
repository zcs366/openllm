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
import re
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
    GATEWAY = "gateway"


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
class ChatMessage:
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
    tool_calls: Optional[list] = None  # OpenAI原生tool_calls（[{id,type,function:{name,arguments}}]）
    reasoning: str = ""  # 推理模型的思考过程（2026-09-10品尝师修复：与content分离，不混入回答）


# ── Provider 实现 ───────────────────────────────────

class DeepSeekProvider:
    """DeepSeek API Provider。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        # DR-20260923（品尝师医师刀）: _available 语义对齐 provider_impl.LLMProvider。
        # CLI快路径读 getattr(provider, "_available", False)——本族provider此前无此
        # 属性 → 恒 False → 真实可用的 provider 也被降级成 BASE_TOOLS（同病四灶：
        # Anthropic/Gemini 同批补）。localhost/127.0.0.1 免 key 也算可用（ollama本地通道）。
        _ep = config.endpoint or ""
        self._available = bool(config.api_key) or "localhost" in _ep or "127.0.0.1" in _ep
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })

    def chat(
        self,
        messages: list[ChatMessage],
        system: Optional[str] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> ModelResponse:
        """发送对话请求。

        tools: OpenAI风格工具声明列表，如
          [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        tool_choice: "auto" 等。同步/流式共用同一payload。
        """
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(messages, system),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        t0 = time.time()

        if stream and on_token:
            return self._stream_chat(payload, on_token, t0)
        else:
            return self._sync_chat(payload, t0)

    @staticmethod
    def _split_reasoning_tag(text: str) -> tuple[str, str]:
        """剥离 <think>...</think> 标签包裹的思考过程（2026-09-10品尝师修复P0-1嘴漏）。

        mimo等推理模型有时把思考直接以<think>标签内联在content里。
        Returns: (answer, reasoning)
        """
        if "</think>" not in text:
            return text, ""
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if not m:
            return text, ""
        reasoning = m.group(1).strip()
        answer = (text[:m.start()] + text[m.end():]).strip()
        return answer, reasoning

    @staticmethod
    def _extract_error_detail(resp) -> str:
        """从失败响应体里挖出服务端原话（2026-09-16 医师接骨 · P0-2）。

        旧行为：`raise_for_status()` 抛出的异常 str 只有
        "400 Client Error: Bad Request for url: ..."，服务端真正的原因
        （如 "request (4581 tokens) exceeds the available context size (4096
        tokens)"）藏在 body 里，**全程不露面**。后果不是"少了一条日志"——而是
        P0-1「苏醒即失语」被误判成"模型不行/网络抖动"，真实根因被错误信息掩埋。

        兼容三类形态：{"error": {"message": ...}} / {"error": "str"} /
        ollama 那种把 error.message 再套一层 JSON 字符串的形态。
        """
        if resp is None:
            return "(无响应体)"
        try:
            data = resp.json()
        except Exception:
            return (getattr(resp, "text", "") or "")[:400] or "(空响应体)"
        if not isinstance(data, dict):
            return str(data)[:400]
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or json.dumps(err, ensure_ascii=False)
        elif err:
            msg = str(err)
        else:
            msg = json.dumps(data, ensure_ascii=False)
        # ollama /v1 把内层错误再序列化成字符串，剥一层
        try:
            inner = json.loads(msg)
            if isinstance(inner, dict) and inner.get("error"):
                e2 = inner["error"]
                msg = e2.get("message") if isinstance(e2, dict) else str(e2)
        except Exception:
            pass
        return str(msg)[:400]

    def _build_messages(self, messages: list[ChatMessage], system: Optional[str] = None) -> list[dict]:
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
            message = choice.get("message", {})
            usage = data.get("usage", {})
            # P0修复：解析OpenAI原生tool_calls（纯工具调用时content可能为null）
            tool_calls = message.get("tool_calls") or None
            # 2026-09-10品尝师修复P0-1：content与reasoning分离
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            answer, inline_think = self._split_reasoning_tag(content)
            if not answer.strip() and reasoning.strip():
                answer = reasoning  # 仅在content确实为空时兜底（推理token耗尽场景）
            response = ModelResponse(
                content=answer,
                model=data.get("model", self.config.model),
                usage=usage,
                latency_ms=(time.time() - t0) * 1000,
                finish_reason=choice.get("finish_reason", "stop"),
                tool_calls=tool_calls,
                reasoning=(inline_think + "\n" + reasoning).strip(),
            )
            # 成本跟踪：写入model trace
            self._record_model_trace(response, usage)
            return response
        except requests.exceptions.HTTPError as e:
            # 2026-09-16 医师接骨（P0-2）：4xx/5xx 必须带上服务端原话。
            detail = self._extract_error_detail(getattr(e, "response", None))
            code = getattr(getattr(e, "response", None), "status_code", "?")
            return ModelResponse(
                content=f"[API错误 {code}] {detail}",
                model="error",
                latency_ms=(time.time() - t0) * 1000,
            )
        except requests.exceptions.RequestException as e:
            return ModelResponse(
                content=f"[API错误] {type(e).__name__}: {e}",
                model="error",
                latency_ms=(time.time() - t0) * 1000,
            )

    def _stream_chat(
        self,
        payload: dict,
        on_token: Callable[[str], None],
        t0: float,
    ) -> ModelResponse:
        """流式调用——逐token回调。

        流式tool_calls是分片传输的：每个delta携带index、function.name、
        function.arguments增量片段，按index合并后还原完整结构。
        """
        full_content = ""
        full_reasoning = ""  # 2026-09-10品尝师修复P0-1：reasoning单独累积，不混入content
        # 流式tool_calls分片累积：index -> 已合并的tool_call dict
        tool_calls_acc: dict[int, dict] = {}
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
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        # 2026-09-10品尝师修复P0-1嘴漏：content与reasoning_content分流。
                        # 旧逻辑：token为空就抓reasoning_content，导致思考过程整段混入回答。
                        token = delta.get("content", "") or ""
                        r_token = delta.get("reasoning_content", "") or ""
                        if r_token:
                            full_reasoning += r_token
                            # 思考过程不进on_token（不打到终端），只累积供内省
                        if token:
                            full_content += token
                            on_token(token)
                        # P0修复：流式tool_calls按index合并
                        for tc in (delta.get("tool_calls") or []):
                            idx = tc.get("index", 0)
                            slot = tool_calls_acc.setdefault(
                                idx,
                                {"id": "", "type": "function",
                                 "function": {"name": "", "arguments": ""}},
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            if tc.get("type"):
                                slot["type"] = tc["type"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            tool_calls = (
                [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
                if tool_calls_acc else None
            )
            # 2026-09-10品尝师修复P0-1：content为空且只有reasoning时（推理token耗尽），
            # 才用reasoning兜底；并剥离内联<think>标签。
            answer, inline_think = self._split_reasoning_tag(full_content)
            if not answer.strip() and full_reasoning.strip():
                answer = full_reasoning
            reasoning_all = (inline_think + "\n" + full_reasoning).strip()
            response = ModelResponse(
                content=answer,
                model=self.config.model,
                latency_ms=(time.time() - t0) * 1000,
                tool_calls=tool_calls,
                reasoning=reasoning_all,
            )
            # 成本跟踪：流式模式无usage数据，只记录latency
            self._record_model_trace(response, {})
            return response
        except requests.exceptions.HTTPError as e:
            # 2026-09-16 医师接骨（P0-2）：流式同样不许把错误降格成一句 status。
            detail = self._extract_error_detail(getattr(e, "response", None))
            code = getattr(getattr(e, "response", None), "status_code", "?")
            return ModelResponse(
                content=f"[流式错误 {code}] {detail}",
                model="error",
                latency_ms=(time.time() - t0) * 1000,
            )
        except requests.exceptions.RequestException as e:
            return ModelResponse(
                content=f"[流式错误] {type(e).__name__}: {e}",
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


class OllamaProvider(DeepSeekProvider):
    """Ollama Provider —— 走**原生 /api/chat**（2026-09-16 医师接骨 · P0-1）。

    ■ 为什么不用 /v1/chat/completions（OpenAI 兼容端点）
      该端点**忽略请求级 num_ctx**。实测（system 10488 字符 ≈ 5735 tokens）：

          /v1        + options.num_ctx=8192    → 400 exceed_context_size_error
          /v1        + options.num_ctx=32768   → 400（同样被忽略）
          /api/chat  + options.num_ctx=8192    → 200 ✅
          /api/chat  + options.num_ctx=16384   → 200 ✅

      而服务侧默认 num_ctx = 4096（模型自身 context_length = 32768，默认值陷阱）。

    ■ 病象「苏醒即失语」
      wake() 注入 system prompt(2621字符) 后，system + 27 个工具 schema
      ≈ 4581 tokens > 4096 → **每一次 chat 都 400**。唤醒越成功，越说不出话。
      开发者跑单测先 chat（无 system）不触发；真实用户开机器先 wake，第一脚就踩。

    ■ 为什么继承 DeepSeekProvider
      engine._call_model 用 `isinstance(provider, DeepSeekProvider)` 决定是否下发
      tools 声明。不继承则该分支失效，工具能力静默消失。
    """

    # 默认上下文窗口：覆盖 system prompt + 全量工具 schema 的常见水位。
    # 可用 OPENLLM_OLLAMA_NUM_CTX 覆盖；服务侧 OLLAMA_CONTEXT_LENGTH 优先。
    DEFAULT_NUM_CTX = int(os.environ.get("OPENLLM_OLLAMA_NUM_CTX", "16384"))

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        self.config.endpoint = self._to_native_endpoint(config.endpoint)
        self.num_ctx = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", str(self.DEFAULT_NUM_CTX)))

    @staticmethod
    def _to_native_endpoint(endpoint: str) -> str:
        """OpenAI 兼容端点 → 原生端点（保留 host，换路径）。"""
        ep = (endpoint or "").rstrip("/")
        if not ep:
            return "http://localhost:11434/api/chat"
        if "/v1/chat/completions" in ep:
            return ep.split("/v1/chat/completions")[0] + "/api/chat"
        if ep.endswith("/v1"):
            return ep[:-3] + "/api/chat"
        if ep.endswith("/api/chat"):
            return ep
        return ep + "/api/chat"

    def _native_payload(self, payload: dict) -> dict:
        """OpenAI 风格 payload → 原生 payload（核心：补 options.num_ctx）。"""
        opts = {
            "num_ctx": self.num_ctx,
            "temperature": payload.get("temperature", 0.7),
        }
        if payload.get("max_tokens"):
            # 原生端点用 num_predict（OpenAI 叫 max_tokens）
            opts["num_predict"] = payload["max_tokens"]
        out = {
            "model": payload.get("model"),
            "messages": payload.get("messages", []),
            "stream": payload.get("stream", False),
            "options": opts,
        }
        if payload.get("tools"):
            out["tools"] = payload["tools"]
        return out

    @staticmethod
    def _native_usage(data: dict) -> dict:
        return {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }

    def _sync_chat(self, payload: dict, t0: float) -> ModelResponse:
        try:
            resp = self._session.post(
                self.config.endpoint, json=self._native_payload(payload), timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") or {}
            answer, inline_think = self._split_reasoning_tag(msg.get("content") or "")
            usage = self._native_usage(data)
            response = ModelResponse(
                content=answer,
                model=data.get("model", self.config.model),
                usage=usage,
                latency_ms=(time.time() - t0) * 1000,
                finish_reason="stop" if data.get("done", True) else "length",
                tool_calls=msg.get("tool_calls") or None,
                reasoning=inline_think,
            )
            self._record_model_trace(response, usage)
            return response
        except requests.exceptions.HTTPError as e:
            detail = self._extract_error_detail(getattr(e, "response", None))
            code = getattr(getattr(e, "response", None), "status_code", "?")
            return ModelResponse(content=f"[API错误 {code}] {detail}", model="error",
                                 latency_ms=(time.time() - t0) * 1000)
        except requests.exceptions.RequestException as e:
            return ModelResponse(content=f"[API错误] {type(e).__name__}: {e}", model="error",
                                 latency_ms=(time.time() - t0) * 1000)

    def _stream_chat(self, payload: dict, on_token: Callable[[str], None], t0: float) -> ModelResponse:
        """原生流式：NDJSON 逐行（不是 SSE 的 `data: ` 前缀）。"""
        full_content = ""
        usage: dict = {}
        try:
            resp = self._session.post(
                self.config.endpoint, json=self._native_payload(payload),
                timeout=120, stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                tok = (data.get("message") or {}).get("content") or ""
                if tok:
                    full_content += tok
                    on_token(tok)
                if data.get("done"):
                    usage = self._native_usage(data)
                    break
            answer, inline_think = self._split_reasoning_tag(full_content)
            response = ModelResponse(
                content=answer,
                model=self.config.model,
                usage=usage,
                latency_ms=(time.time() - t0) * 1000,
                finish_reason="stop",
                reasoning=inline_think,
            )
            self._record_model_trace(response, usage)
            return response
        except requests.exceptions.HTTPError as e:
            detail = self._extract_error_detail(getattr(e, "response", None))
            code = getattr(getattr(e, "response", None), "status_code", "?")
            return ModelResponse(content=f"[流式错误 {code}] {detail}", model="error",
                                 latency_ms=(time.time() - t0) * 1000)
        except requests.exceptions.RequestException as e:
            return ModelResponse(content=f"[流式错误] {type(e).__name__}: {e}", model="error",
                                 latency_ms=(time.time() - t0) * 1000)


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
        self._available = bool(config.api_key)  # DR-20260923 同批补缺（CLI恒降级病）
        self._session.headers.update({
            "x-api-key": config.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        })

    def chat(
        self,
        messages: list[ChatMessage],
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
        self._available = bool(config.api_key)  # DR-20260923 同批补缺（CLI恒降级病）

    def chat(
        self,
        messages: list[ChatMessage],
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
        # 2026-09-16 医师接骨（P0-1）：改走**原生端点**。
        # 原实现复用 DeepSeekProvider 打 /v1/chat/completions，而该端点忽略
        # 请求级 num_ctx（实测 8192/32768 均 400）→ wake 注入 system 后
        # system+tools 超服务侧默认 4096，每次 chat 必 400（"苏醒即失语"）。
        # OllamaProvider 继承 DeepSeekProvider，保留 engine 的 tools 下发分支。
        config.endpoint = kwargs.get("endpoint", "http://localhost:11434/v1/chat/completions")
        config.api_key = kwargs.get("api_key", "ollama")
        config.model = kwargs.get("model", "qwen3.5:9b")
        provider = OllamaProvider(config)
        logger.info(
            f"Ollama provider: {config.model} @ {provider.config.endpoint} "
            f"(num_ctx={provider.num_ctx})"
        )
        return provider

    if provider_type == "mimo":
        config.endpoint = kwargs.get("endpoint", "https://token-plan-cn.xiaomimimo.com/v1/chat/completions")
        config.api_key = kwargs.get("api_key", "")
        config.model = kwargs.get("model", "MiMo-v2.5")
        if not config.api_key:
            raise ValueError("mimo需要api_key参数")
        logger.info(f"mimo provider: {config.model} @ {config.endpoint}")
        return DeepSeekProvider(config)

    if provider_type == "qwen":
        config.endpoint = kwargs.get("endpoint", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        config.api_key = kwargs.get("api_key", "")
        config.model = kwargs.get("model", "qwen-max")
        if not config.api_key:
            raise ValueError("qwen需要api_key参数")
        logger.info(f"qwen provider: {config.model} @ {config.endpoint}")
        return DeepSeekProvider(config)  # DashScope兼容OpenAI接口

    if provider_type == "gateway":
        config.endpoint = kwargs.get("endpoint") or "http://127.0.0.1:13000/v1/chat/completions"
        # 三级 key 获取：kwargs > ~/one-api/.gateway_key > env
        if not config.api_key:
            gw_key_path = Path.home() / "one-api" / ".gateway_key"
            if gw_key_path.exists():
                config.api_key = gw_key_path.read_text().strip()
        if not config.api_key:
            config.api_key = os.environ.get("OPENLLM_GATEWAY_KEY", "")
        if not config.api_key:
            raise ValueError("gateway需要api_key或~/one-api/.gateway_key文件")
        config.model = kwargs.get("model", "gemini-3.6-flash")
        logger.info(f"gateway provider: {config.model} @ {config.endpoint}")
        provider = DeepSeekProvider(config)
        # 禁用代理：网关是本地服务，不应走 Clash/系统代理
        provider._session.trust_env = False
        return provider

    if provider_type == "deepseek":
        return DeepSeekProvider(config)
    elif provider_type == "openai":
        config.endpoint = kwargs.get("endpoint", "https://api.openai.com/v1/chat/completions")
        return DeepSeekProvider(config)  # 同接口
    else:
        raise ValueError(f"不支持的Provider: {provider_type}。支持: deepseek/openai/anthropic/gemini/ollama")


# Backward compatibility alias
Message = ChatMessage
