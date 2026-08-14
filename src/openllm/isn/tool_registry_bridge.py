"""
ISN Tool Registry Bridge — 动态工具注册协议
============================================

连接ISN技能系统与ToolRegistry，实现：
1. 从技能配置动态注册工具
2. 工具验证（函数签名、安全检查）
3. 工具生命周期管理（注册/注销/热更新）
4. 审计日志

用法：
    from openllm.tools.executor import ToolRegistry
    from openllm.isn.tool_registry_bridge import ToolRegistryBridge
    
    registry = ToolRegistry()
    bridge = ToolRegistryBridge(registry)
    
    # 从技能配置注册工具
    bridge.register_from_config({
        "name": "my_tool",
        "description": "我的自定义工具",
        "module": "my_package.tools",
        "function": "my_tool_func",
        "params": {"path": "str", "limit": "int"},
    })
"""

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("openllm.isn.tool_registry_bridge")


@dataclass
class ToolConfig:
    """工具配置schema。"""
    name: str                          # 工具名（唯一）
    description: str                   # 工具描述
    module: str                        # 模块路径（如 "my_package.tools"）
    function: str                      # 函数名
    params: Dict[str, str] = field(default_factory=dict)  # 参数名→类型
    requires_verify: bool = False      # 是否需要验证（写操作）
    enabled: bool = True               # 是否启用
    
    def validate(self) -> List[str]:
        """验证配置，返回错误列表（空=通过）。"""
        errors = []
        if not self.name:
            errors.append("name不能为空")
        if not self.module:
            errors.append("module不能为空")
        if not self.function:
            errors.append("function不能为空")
        return errors


@dataclass
class BridgeResult:
    """桥接操作结果。"""
    success: bool
    tool_name: str = ""
    message: str = ""
    errors: List[str] = field(default_factory=list)


class ToolRegistryBridge:
    """
    ISN ↔ ToolRegistry 动态注册桥。
    
    职责：
    1. 接收ISN技能配置
    2. 验证工具函数
    3. 注册到ToolRegistry
    4. 管理生命周期
    """
    
    def __init__(self, registry):
        """
        Args:
            registry: ToolRegistry实例
        """
        self.registry = registry
        self._registered: Dict[str, ToolConfig] = {}  # name → config
        self._modules: Dict[str, Any] = {}  # module_path → module
        self._functions: Dict[str, Callable] = {}  # tool_name → function
    
    def register_from_config(self, config: dict) -> BridgeResult:
        """
        从配置字典注册工具。
        
        Args:
            config: 工具配置字典，字段见ToolConfig
            
        Returns:
            BridgeResult
        """
        try:
            tool_config = ToolConfig(**config)
        except TypeError as e:
            return BridgeResult(
                success=False,
                message=f"配置解析失败: {e}",
                errors=[str(e)]
            )
        
        return self.register_from_tool_config(tool_config)
    
    def register_from_tool_config(self, tool_config: ToolConfig) -> BridgeResult:
        """
        从ToolConfig注册工具。
        
        Args:
            tool_config: ToolConfig实例
            
        Returns:
            BridgeResult
        """
        # 1. 验证配置
        errors = tool_config.validate()
        if errors:
            return BridgeResult(
                success=False,
                tool_name=tool_config.name,
                message="配置验证失败",
                errors=errors
            )
        
        # 2. 检查是否已注册
        if tool_config.name in self._registered:
            return BridgeResult(
                success=False,
                tool_name=tool_config.name,
                message=f"工具 '{tool_config.name}' 已注册",
                errors=["重复注册"]
            )
        
        # 3. 加载模块和函数
        func = self._load_function(tool_config.module, tool_config.function)
        if func is None:
            return BridgeResult(
                success=False,
                tool_name=tool_config.name,
                message=f"加载函数失败: {tool_config.module}.{tool_config.function}",
                errors=["模块或函数不存在"]
            )
        
        # 4. 验证函数签名
        sig_errors = self._validate_function_signature(func, tool_config.params)
        if sig_errors:
            return BridgeResult(
                success=False,
                tool_name=tool_config.name,
                message="函数签名验证失败",
                errors=sig_errors
            )
        
        # 5. 注册到ToolRegistry
        try:
            self.registry.register(
                name=tool_config.name,
                func=func,
                description=tool_config.description,
                schema={"params": tool_config.params}
            )
        except Exception as e:
            return BridgeResult(
                success=False,
                tool_name=tool_config.name,
                message=f"注册到ToolRegistry失败: {e}",
                errors=[str(e)]
            )
        
        # 6. 更新内部状态
        self._registered[tool_config.name] = tool_config
        self._functions[tool_config.name] = func
        
        # 7. 处理verify标记
        if tool_config.requires_verify:
            self.registry._requires_verify.add(tool_config.name)
        
        logger.info(f"✅ 工具注册成功: {tool_config.name}")
        return BridgeResult(
            success=True,
            tool_name=tool_config.name,
            message=f"工具 '{tool_config.name}' 注册成功"
        )
    
    def unregister(self, tool_name: str) -> BridgeResult:
        """
        注销工具。
        
        Args:
            tool_name: 工具名
            
        Returns:
            BridgeResult
        """
        if tool_name not in self._registered:
            return BridgeResult(
                success=False,
                tool_name=tool_name,
                message=f"工具 '{tool_name}' 未注册"
            )
        
        # 从ToolRegistry移除
        if tool_name in self.registry._tools:
            del self.registry._tools[tool_name]
        if tool_name in self.registry._descriptions:
            del self.registry._descriptions[tool_name]
        if tool_name in self.registry._schemas:
            del self.registry._schemas[tool_name]
        if tool_name in self.registry._requires_verify:
            self.registry._requires_verify.discard(tool_name)
        
        # 更新内部状态
        del self._registered[tool_name]
        if tool_name in self._functions:
            del self._functions[tool_name]
        
        logger.info(f"🗑️ 工具注销: {tool_name}")
        return BridgeResult(
            success=True,
            tool_name=tool_name,
            message=f"工具 '{tool_name}' 已注销"
        )
    
    def list_registered(self) -> List[dict]:
        """列出所有通过bridge注册的工具。"""
        return [
            {
                "name": name,
                "description": config.description,
                "module": config.module,
                "function": config.function,
                "enabled": config.enabled,
                "requires_verify": config.requires_verify,
            }
            for name, config in self._registered.items()
        ]
    
    def reload_tool(self, tool_name: str) -> BridgeResult:
        """
        热更新工具（重新加载模块+重新注册）。
        
        Args:
            tool_name: 工具名
            
        Returns:
            BridgeResult
        """
        if tool_name not in self._registered:
            return BridgeResult(
                success=False,
                tool_name=tool_name,
                message=f"工具 '{tool_name}' 未注册，无法热更新"
            )
        
        config = self._registered[tool_name]
        
        # 先注销
        self.unregister(tool_name)
        
        # 清除模块缓存
        if config.module in self._modules:
            del self._modules[config.module]
        
        # 重新注册
        return self.register_from_tool_config(config)
    
    def _load_function(self, module_path: str, function_name: str) -> Optional[Callable]:
        """加载模块并获取函数。"""
        try:
            # 尝试从缓存加载
            if module_path not in self._modules:
                self._modules[module_path] = importlib.import_module(module_path)
            
            module = self._modules[module_path]
            return getattr(module, function_name, None)
        except Exception as e:
            logger.error(f"加载模块失败: {module_path}.{function_name} — {e}")
            return None
    
    def _validate_function_signature(
        self, func: Callable, expected_params: Dict[str, str]
    ) -> List[str]:
        """验证函数签名是否匹配。"""
        errors = []
        try:
            sig = inspect.signature(func)
            actual_params = {
                name: str(param.annotation)
                for name, param in sig.parameters.items()
                if name not in ('self', 'cls')
            }
            
            # 检查期望的参数是否存在
            for param_name, param_type in expected_params.items():
                if param_name not in actual_params:
                    errors.append(f"函数缺少参数: {param_name}")
        except Exception as e:
            errors.append(f"签名解析失败: {e}")
        
        return errors
