#!/usr/bin/env python3
"""测试 OpenLLM 完整引擎 + Ollama"""
import sys
sys.path.insert(0, 'src')

from openllm.core.engine import OpenLLMEngine, AgentConfig

print("=== OpenLLM Engine + Ollama 测试 ===")

# 创建引擎
config = AgentConfig(
    name='OpenLLM',
    provider='ollama',
    model='qwen3.5:9b',
)
engine = OpenLLMEngine(config)

# 唤醒
wake_msg = engine.wake()
print(wake_msg)
print(f'\nconnected: {engine.connected}')
print(f'tools: {len(engine.tools.list_tools())}')

# 测试对话
print("\n发送消息...")
response = engine.chat('你好，用一句话介绍你自己', stream=False)
print(f'\nResponse: {response[:300]}')

# 状态
status = engine.status()
print(f'\n状态: {status}')

print('\n✅ 完整引擎测试通过!')
