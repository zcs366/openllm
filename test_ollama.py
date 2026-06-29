#!/usr/bin/env python3
"""测试 Ollama provider 连接"""
import sys
sys.path.insert(0, 'src')

from openllm.core.provider import create_provider, Message, ProviderType

print("=== Ollama Provider 测试 ===")

# 1. 测试 provider 创建
p = create_provider('ollama', model='qwen3.5:9b')
print(f"Provider: {p.config.provider}")
print(f"Model: {p.config.model}")
print(f"Endpoint: {p.config.endpoint}")
print(f"API key set: {bool(p.config.api_key)}")
print(f"ProviderType.OLLAMA: {ProviderType.OLLAMA.value}")

# 2. 测试对话
print("\n发送消息...")
resp = p.chat([Message(role='user', content='说一个字')], stream=False)
print(f"Response: {resp.content[:100]}")
print(f"Latency: {resp.latency_ms:.0f}ms")
print(f"Usage: {resp.usage}")

print("\n✅ Ollama provider 测试通过!")
