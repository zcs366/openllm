"""ISN Tool Registry Bridge 测试。"""
import pytest
from openllm.tools.executor import ToolRegistry, create_default_tools
from openllm.isn.tool_registry_bridge import ToolRegistryBridge, ToolConfig, BridgeResult


# ── 测试用工具函数 ──

def dummy_tool(message: str = "hello") -> str:
    """测试用工具。"""
    return f"echo: {message}"


def another_tool(x: int, y: int) -> int:
    """另一个测试工具。"""
    return x + y


# ── ToolConfig 测试 ──

class TestToolConfig:
    def test_valid_config(self):
        config = ToolConfig(
            name="test_tool",
            description="测试工具",
            module="test_module",
            function="test_func",
        )
        errors = config.validate()
        assert errors == []
    
    def test_missing_name(self):
        config = ToolConfig(
            name="",
            description="测试工具",
            module="test_module",
            function="test_func",
        )
        errors = config.validate()
        assert "name不能为空" in errors
    
    def test_missing_module(self):
        config = ToolConfig(
            name="test_tool",
            description="测试工具",
            module="",
            function="test_func",
        )
        errors = config.validate()
        assert "module不能为空" in errors
    
    def test_missing_function(self):
        config = ToolConfig(
            name="test_tool",
            description="测试工具",
            module="test_module",
            function="",
        )
        errors = config.validate()
        assert "function不能为空" in errors


# ── ToolRegistryBridge 测试 ──

class TestToolRegistryBridge:
    def setup_method(self):
        self.registry = create_default_tools()
        self.bridge = ToolRegistryBridge(self.registry)
    
    def test_register_from_config_success(self):
        result = self.bridge.register_from_config({
            "name": "dummy_tool",
            "description": "测试工具",
            "module": "openllm.tools.executor",
            "function": "tool_read_file",
            "params": {"path": "str"},
        })
        assert result.success
        assert result.tool_name == "dummy_tool"
        assert "dummy_tool" in [t["name"] for t in self.registry.list_tools()]
    
    def test_register_duplicate(self):
        config = {
            "name": "dummy_tool",
            "description": "测试工具",
            "module": "openllm.tools.executor",
            "function": "tool_read_file",
        }
        self.bridge.register_from_config(config)
        result = self.bridge.register_from_config(config)
        assert not result.success
        assert "已注册" in result.message
    
    def test_register_invalid_config(self):
        result = self.bridge.register_from_config({
            "name": "",
            "description": "测试工具",
            "module": "test_module",
            "function": "test_func",
        })
        assert not result.success
        assert "验证失败" in result.message
    
    def test_register_missing_module(self):
        result = self.bridge.register_from_config({
            "name": "bad_tool",
            "description": "测试工具",
            "module": "nonexistent.module",
            "function": "func",
        })
        assert not result.success
        assert "加载函数失败" in result.message
    
    def test_unregister(self):
        self.bridge.register_from_config({
            "name": "dummy_tool",
            "description": "测试工具",
            "module": "openllm.tools.executor",
            "function": "tool_read_file",
        })
        result = self.bridge.unregister("dummy_tool")
        assert result.success
        assert "dummy_tool" not in [t["name"] for t in self.registry.list_tools()]
    
    def test_unregister_nonexistent(self):
        result = self.bridge.unregister("nonexistent")
        assert not result.success
        assert "未注册" in result.message
    
    def test_list_registered(self):
        self.bridge.register_from_config({
            "name": "dummy_tool",
            "description": "测试工具",
            "module": "openllm.tools.executor",
            "function": "tool_read_file",
        })
        registered = self.bridge.list_registered()
        assert len(registered) == 1
        assert registered[0]["name"] == "dummy_tool"
    
    def test_execute_registered_tool(self):
        self.bridge.register_from_config({
            "name": "dummy_tool",
            "description": "测试工具",
            "module": "openllm.tools.executor",
            "function": "tool_read_file",
            "params": {"path": "str"},
        })
        result = self.registry.execute("dummy_tool", path="/etc/hostname")
        assert result.success
    
    def test_execute_unregistered_tool(self):
        result = self.registry.execute("nonexistent")
        assert not result.success
        assert "未知工具" in result.error
    
    def test_requires_verify_flag(self):
        self.bridge.register_from_config({
            "name": "write_tool",
            "description": "写工具",
            "module": "openllm.tools.executor",
            "function": "tool_write_file",
            "requires_verify": True,
        })
        assert "write_tool" in self.registry._requires_verify
    
    def test_tool_config_dataclass(self):
        config = ToolConfig(
            name="test",
            description="test",
            module="test",
            function="test",
            params={"x": "int"},
            requires_verify=True,
            enabled=False,
        )
        assert config.params == {"x": "int"}
        assert config.requires_verify is True
        assert config.enabled is False
