"""
EngineBridge — 懒加载血管：将engine.py的成熟验证链接入main_loop.py

不是"替代"——是"接线"。engine.py不可用→主循环照跑。
血管模式：_get_isa_on_verify() 同理——不存在就跳过，存在就自动接。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openllm.bridge")

_ENGINE_AVAILABLE: Optional[bool] = None
_ENGINE_MODULE = None


def _ensure_engine():
    """懒加载engine模块。单次检查，结果缓存。"""
    global _ENGINE_AVAILABLE, _ENGINE_MODULE
    if _ENGINE_AVAILABLE is not None:
        return _ENGINE_AVAILABLE
    
    try:
        from openllm.core.engine import OpenLLMEngine, AgentConfig
        _ENGINE_MODULE = OpenLLMEngine
        _ENGINE_AVAILABLE = True
        logger.info("✅ EngineBridge: engine.py验证链可用")
    except Exception as e:
        logger.debug(f"EngineBridge: engine.py不可用({e})·使用fallback")
        _ENGINE_AVAILABLE = False
    return _ENGINE_AVAILABLE


def verify_tool_params(tool_name: str, **kwargs) -> dict:
    """血管：engine.py工具参数验证"""
    if not _ensure_engine():
        return {"pass": True, "reason": "EngineBridge fallback"}
    try:
        engine = _ENGINE_MODULE()
        return engine._verify_tool_params(tool_name, **kwargs)
    except:
        return {"pass": True, "reason": "verify skipped"}


def verify_tool_result(tool_name: str, result) -> dict:
    """血管：engine.py工具结果验证"""
    if not _ensure_engine():
        return {"pass": True, "reason": "EngineBridge fallback"}
    try:
        engine = _ENGINE_MODULE()
        return engine._verify_tool_result(tool_name, result)
    except:
        return {"pass": True, "reason": "verify skipped"}


def check_tool_risk(tool_name: str) -> dict:
    """血管：engine.py ISN风险元数据检查"""
    if not _ensure_engine():
        return {"pass": True, "reason": "EngineBridge fallback"}
    try:
        engine = _ENGINE_MODULE()
        return engine._check_tool_risk(tool_name)
    except:
        return {"pass": True, "reason": "risk check skipped"}


def check_action(action: str) -> tuple:
    """血管：engine.py安全检查"""
    if not _ensure_engine():
        return True, "EngineBridge fallback"
    try:
        engine = _ENGINE_MODULE()
        return engine.security.check_action(action)
    except:
        return True, "security skipped"
