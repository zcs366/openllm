"""extracted from main_loop.py"""
import json, os, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .models import *
class LLMProvider:
    """最简单的LLM调用封装"""
    
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        # 优先从config读·其次环境变量
        self.api_key = self._load_key()
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"
        self._available = bool(self.api_key)
        self._last_usage: dict = {}  # 最近一次调用的token使用量
    
    def _load_key(self) -> str:
        """从config.json加载API key"""
        config_path = Path.home() / ".openllm" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                return cfg.get("providers", {}).get("deepseek", {}).get("api_key", "")
            except:
                pass
        return os.environ.get("DEEPSEEK_API_KEY", "")
    
    def chat(self, messages: list[dict]) -> str:
        """调LLM·返回文本。M0: 返回模拟响应。"""
        if not self._available:
            # M0降级：返回模拟响应
            user_msg = messages[-1]["content"] if messages else ""
            self._last_usage = {}
            return f"[模拟LLM] 已收到: {user_msg[:50]}"
        
        # TODO: Phase 2接入真实API
        try:
            import requests
            resp = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM错误] {e}"


# ═══════════════════════════════════════════════════════
# 五体（Stub · M0阶段用简单实现）
# ═══════════════════════════════════════════════════════
