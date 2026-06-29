#!/usr/bin/env python3
"""测试章鱼搜索工具"""
import sys
sys.path.insert(0, 'src')

from openllm.tools.octopus import tool_octopus_search, tool_octopus_self_model

print("=== 章鱼搜索测试 ===")
result = tool_octopus_search('OpenLLM', limit=3)
print(result[:500])
print("---")
result2 = tool_octopus_self_model()
print(result2[:300])
print("\n✅ 测试通过!")
