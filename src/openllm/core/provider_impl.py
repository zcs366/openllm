"""extracted from main_loop.py"""
import json, os, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .models import *

class LLMProvider:
    """最简单的LLM调用封装（支持多provider）"""
    
    def __init__(self, model: str = None):
        config = self._load_config()
        
        # 从config读取default_provider
        default_provider = config.get("default_provider", "deepseek")
        provider_cfg = config.get("providers", {}).get(default_provider, {})
        
        self.model = model or provider_cfg.get("model", "deepseek-chat")
        self.endpoint = provider_cfg.get("endpoint", "https://api.deepseek.com/v1/chat/completions")
        self.api_key = provider_cfg.get("api_key", "")
        self._available = bool(self.api_key) or "localhost" in self.endpoint
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
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            choice = data["choices"][0]["message"]
            content = choice.get("content", "")
            # 推理模型: content可能为空(reasoning吃掉全部token)
            # 仅在content确实为空时fallback到reasoning_content
            if not content and choice.get("reasoning_content"):
                content = choice["reasoning_content"]
            return content
        except Exception as e:
            return f"[LLM错误] {e}"
