"""extracted from main_loop.py"""
import json, os, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .models import *

class LLMProvider:
    """最简单的LLM调用封装（支持多provider）"""

    # ── 超时（2026-09-15 医师接骨）──
    # 原 timeout=120 对「服务端吐心跳的慢响应」无效：requests 的 read timeout 是
    # 「两个字节之间的间隔」语义，服务端每吐一个字节就重置计时器。实测 DeepSeek
    # 排队 900 秒才回，客户端全程静默（用户看到的是「说完话没有任何反应」）。
    # 拆成 (connect, read) 元组才是真超时；read 可用环境变量调小以便测试。
    CONNECT_TIMEOUT = float(os.environ.get("OPENLLM_LLM_CONNECT_TIMEOUT", "10"))
    READ_TIMEOUT = float(os.environ.get("OPENLLM_LLM_READ_TIMEOUT", "90"))

    def __init__(self, model: Optional[str] = None, provider_name: Optional[str] = None):
        config = self._load_config()
        
        # 从config读取default_provider（或显式指定的 provider_name）
        default_provider = provider_name or config.get("default_provider", "deepseek")
        provider_cfg = config.get("providers", {}).get(default_provider, {})
        
        self.model = model or provider_cfg.get("model", "deepseek-chat")
        self.endpoint = provider_cfg.get("endpoint", "https://api.deepseek.com/v1/chat/completions")
        # 正确取key：环境变量 → 密钥库(keyvault) → config明文（2026-09-10 迁keyring后）
        from ..security.keyvault import resolve_api_key
        self.api_key, _key_source = resolve_api_key(default_provider, provider_cfg)
        self._available = bool(self.api_key) or "localhost" in self.endpoint
        # Gateway fallback: config.json 里 key 为空时读文件
        if not self.api_key and self.endpoint and "127.0.0.1" in self.endpoint:
            gw_key = Path.home() / "one-api" / ".gateway_key"
            if gw_key.exists():
                self.api_key = gw_key.read_text().strip()
                self._available = bool(self.api_key)
        self._last_usage: dict = {}
    
    def _load_config(self) -> dict:
        """从config.json加载配置"""
        config_path = Path.home() / ".openllm" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def chat(self, messages: list[dict]) -> str:
        """调LLM·返回文本。"""
        if not self._available:
            user_msg = messages[-1]["content"] if messages else ""
            self._last_usage = {}
            return f"[模拟LLM] 已收到: {user_msg[:50]}"
        
        try:
            import requests
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key != "lm-studio":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            resp = requests.post(
                self.endpoint,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 8192,
                    "temperature": 0.7,
                },
                timeout=(self.CONNECT_TIMEOUT, self.READ_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()

            # ── 服务端会把 error 体塞进 HTTP 200（2026-09-15 医师接骨）──
            # 实测 DeepSeek 排队超时返回的就是 200 + {"error":{"message":"..."}}。
            # 旧代码直接取 data["choices"] → KeyError → 被下面 except 兜成
            # "[LLM错误] 'choices'"，把服务端原话换成了最没用的一条信息。
            if isinstance(data, dict) and data.get("error"):
                err = data["error"]
                msg = err.get("message") if isinstance(err, dict) else str(err)
                self._last_usage = {}
                return f"[LLM错误] 服务端拒绝: {msg}"
            if not isinstance(data, dict) or not data.get("choices"):
                self._last_usage = {}
                return f"[LLM错误] 响应缺少 choices 字段: {str(data)[:300]}"

            usage = data.get("usage", {})
            prompt_details = usage.get("prompt_tokens_details", {}) or {}
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cached_tokens": prompt_details.get("cached_tokens", 0),
            }
            choice = data["choices"][0]["message"]
            content = choice.get("content", "")
            # 推理模型: content可能为空(reasoning吃掉全部token)
            # 仅在content确实为空时fallback到reasoning_content
            if not content and choice.get("reasoning_content"):
                content = choice["reasoning_content"]
            # ── 空回复要给出解释，不能返回 ''（2026-09-15 接骨）──
            # 旧行为：返回 '' → CLI 打印 "(无输出)"，用户无法判断是模型坏了还是
            # token 用尽了。实测 mimo/glm/oneapi 都会出现这种空 content。
            if not content:
                finish = data["choices"][0].get("finish_reason", "")
                return (f"[LLM空回复] 模型未产出正文（finish_reason={finish}）。"
                        f"多为推理占满 max_tokens——请调大 max_tokens 或换模型。")
            return content
        except Exception as e:
            return f"[LLM错误] {type(e).__name__}: {e}"
