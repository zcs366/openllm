"""
isn ExternalToolBridge — 发现→桥接→使用→学习。

理念：造人不造工具。openLLM的手能握任何外部工具。

流程：
  1. scan: 自动发现PATH中的可执行工具
  2. bridge: 返回工具的调用接口描述
  3. use: 通过terminal执行外部工具
  4. learn: 记录工具使用经验和技巧

默认扫描：
  - hermes: 多平台Agent
  - codex: 代码Agent
  - python3: 脚本执行
  - git: 版本控制
  - curl: HTTP请求
"""

import shutil
from pathlib import Path
from typing import Optional


class ExternalToolBridge:
    """
    外部工具桥接器。
    
    发现PATH中的可执行工具，提供统一的调用接口。
    """
    
    # 默认扫描的工具列表
    DEFAULT_TOOLS = ["hermes", "codex", "python3", "git", "curl", "node", "jq"]
    
    def __init__(self):
        self.discovered: dict[str, dict] = {}  # {tool_name: {path, verified, ...}}
        self.usage_log: list[dict] = []  # 使用日志
    
    def scan(self, tools: Optional[list[str]] = None) -> list[str]:
        """
        自动发现PATH中的可执行工具。
        
        Args:
            tools: 要扫描的工具列表（默认使用DEFAULT_TOOLS）
        
        Returns:
            发现的工具名列表
        """
        if tools is None:
            tools = self.DEFAULT_TOOLS
        
        found = []
        for name in tools:
            path = shutil.which(name)
            if path:
                self.discovered[name] = {
                    "path": path,
                    "verified": True,
                    "first_seen": __import__("time").time(),
                }
                found.append(name)
        
        return found
    
    def bridge(self, tool_name: str) -> dict:
        """
        桥接：返回工具的调用接口描述。
        
        Args:
            tool_name: 工具名称
        
        Returns:
            工具信息字典
        """
        if tool_name not in self.discovered:
            return {
                "available": False,
                "name": tool_name,
                "reason": "未发现",
            }
        
        return {
            "available": True,
            "name": tool_name,
            "path": self.discovered[tool_name]["path"],
            "how_to_use": f"通过terminal调用: {tool_name} <args>",
        }
    
    def list_available(self) -> list[dict]:
        """
        列出所有可用外部工具。
        
        Returns:
            工具信息列表
        """
        return [self.bridge(name) for name in self.discovered]
    
    def record_usage(self, tool_name: str, command: str, result: str, success: bool):
        """
        记录工具使用。
        
        Args:
            tool_name: 工具名称
            command: 执行的命令
            result: 执行结果
            success: 是否成功
        """
        import time
        self.usage_log.append({
            "tool": tool_name,
            "command": command[:200],
            "result": result[:200],
            "success": success,
            "timestamp": time.time(),
        })
    
    def summary(self) -> dict:
        """状态摘要。"""
        return {
            "discovered_count": len(self.discovered),
            "tools": list(self.discovered.keys()),
            "usage_count": len(self.usage_log),
        }


# ═══════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== ExternalToolBridge 测试 ===\n")
    
    bridge = ExternalToolBridge()
    
    # 测试1: scan
    found = bridge.scan()
    print(f"✅ 测试1: scan发现 {len(found)} 个工具: {found}")
    
    # 测试2: python3和git应该存在
    assert "python3" in bridge.discovered
    assert "git" in bridge.discovered
    print(f"✅ 测试2: python3和git存在")
    
    # 测试3: bridge
    info = bridge.bridge("python3")
    assert info["available"] == True
    assert "path" in info
    print(f"✅ 测试3: bridge python3 → {info['path']}")
    
    # 测试4: list_available
    tools = bridge.list_available()
    assert len(tools) > 0
    print(f"✅ 测试4: list_available {len(tools)} 个工具")
    
    # 测试5: 不存在的工具
    info = bridge.bridge("nonexistent_tool")
    assert info["available"] == False
    print(f"✅ 测试5: 不存在的工具返回available=False")
    
    print(f"\n全部 5/5 测试通过 ✅")
