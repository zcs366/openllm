"""
Iam Harness v2.3 集成模块 — 将身份约束框架集成到openLLM的ISA系统。

功能：
1. 决策前检索相关原则（retrieve）
2. 格式化为注入文本（format_for_injection）
3. 决策后验证合规性（verify）
4. 冲突检测（check_conflict）

依赖：
- ~/.hermes/iam_harness/ 目录下的模块
- 零外部依赖，独立运行

用法：
    from openllm.identity.iam_integration import IamIntegration
    
    iam = IamIntegration()
    
    # 决策前检索
    principles = iam.retrieve("用户在质疑我的判断")
    
    # 格式化注入文本
    injection_text = iam.format_for_injection(principles)
    
    # 决策后验证
    result = iam.verify(context, decision, principles, verifier_response)
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("openllm.iam_integration")

# ── 路径配置 ──
IAM_HARNESS_PATH = Path.home() / ".hermes" / "iam_harness"


class IamIntegration:
    """
    Iam Harness v2.3 集成类。
    
    将身份约束框架集成到openLLM的ISA系统中。
    懒加载：只在第一次调用时导入iam_harness模块。
    """
    
    def __init__(self, auto_load: bool = True):
        """
        初始化Iam集成。
        
        Args:
            auto_load: 是否自动加载iam_harness模块（默认True）
        """
        self._loaded = False
        self._retrieve = None
        self._format_for_injection = None
        self._verify = None
        self._check_conflict = None
        self._log_decision = None
        self._get_daily_stats = None
        
        if auto_load:
            self._load_iam_harness()
    
    def _load_iam_harness(self) -> bool:
        """
        懒加载iam_harness模块。
        
        Returns:
            bool: 是否加载成功
        """
        if self._loaded:
            return True
        
        if not IAM_HARNESS_PATH.exists():
            logger.warning(f"iam_harness目录不存在: {IAM_HARNESS_PATH}")
            return False
        
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("iam_harness", IAM_HARNESS_PATH.parent / "iam_harness" / "__init__.py")
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                retrieve = _mod.retrieve
                format_for_injection = _mod.format_for_injection
                verify = _mod.verify
                check_conflict = _mod.check_conflict
                log_decision = _mod.log_decision
                get_daily_stats = _mod.get_daily_stats
            
            self._retrieve = retrieve
            self._format_for_injection = format_for_injection
            self._verify = verify
            self._check_conflict = check_conflict
            self._log_decision = log_decision
            self._get_daily_stats = get_daily_stats
            
            self._loaded = True
            logger.info("✅ iam_harness 已加载")
            return True
            
        except Exception as e:
            logger.warning(f"iam_harness 加载失败: {e}")
            return False
    
    def retrieve(self, context: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        检索与上下文相关的原则。
        
        Args:
            context: 决策上下文（用户消息、任务描述等）
            top_n: 返回的原则数量（默认3）
            
        Returns:
            List[Dict]: 相关原则列表
        """
        if not self._loaded:
            logger.warning("iam_harness未加载，返回空列表")
            return []
        
        try:
            return self._retrieve(context, top_n=top_n)
        except Exception as e:
            logger.error(f"原则检索失败: {e}")
            return []
    
    def format_for_injection(self, principles: List[Dict[str, Any]]) -> str:
        """
        将原则格式化为可注入prompt的文本。
        
        Args:
            principles: 原则列表（retrieve的返回值）
            
        Returns:
            str: 格式化后的文本
        """
        if not self._loaded:
            logger.warning("iam_harness未加载，返回空字符串")
            return ""
        
        try:
            return self._format_for_injection(principles)
        except Exception as e:
            logger.error(f"原则格式化失败: {e}")
            return ""
    
    def verify(
        self,
        context: str,
        decision: str,
        principles: List[Dict[str, Any]],
        verifier_response: str = "",
    ) -> Dict[str, Any]:
        """
        验证决策是否符合原则。
        
        Args:
            context: 决策上下文
            decision: 决策内容
            principles: 相关原则列表
            verifier_response: 验证器响应（可选）
            
        Returns:
            Dict: 验证结果
        """
        if not self._loaded:
            logger.warning("iam_harness未加载，返回默认通过")
            return {"pass": True, "reason": "iam_harness未加载"}
        
        try:
            return self._verify(context, decision, principles, verifier_response)
        except Exception as e:
            logger.error(f"原则验证失败: {e}")
            return {"pass": False, "reason": f"验证异常: {e}"}
    
    def check_conflict(self, principles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检查原则之间是否存在冲突。
        
        Args:
            principles: 原则列表
            
        Returns:
            Dict: 冲突检测结果，格式：
                - has_conflict: bool
                - conflicts: list
                - description: str (如果有冲突)
        """
        if not self._loaded:
            logger.warning("iam_harness未加载，返回无冲突")
            return {"has_conflict": False, "conflicts": [], "description": ""}
        
        try:
            result = self._check_conflict(principles)
            if result is None:
                return {"has_conflict": False, "conflicts": [], "description": ""}
            else:
                # result 是 tuple (a, b, description)
                a, b, description = result
                return {
                    "has_conflict": True,
                    "conflicts": [a, b],
                    "description": description,
                }
        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
            return {"has_conflict": False, "conflicts": [], "description": ""}
    
    def log_decision(
        self,
        context: str,
        decision: str,
        principles: List[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> bool:
        """
        记录决策日志。
        
        Args:
            context: 决策上下文
            decision: 决策内容
            principles: 相关原则列表
            result: 验证结果
            
        Returns:
            bool: 是否记录成功
        """
        if not self._loaded:
            logger.warning("iam_harness未加载，跳过日志记录")
            return False
        
        try:
            self._log_decision(context, decision, principles, result)
            return True
        except Exception as e:
            logger.error(f"决策日志记录失败: {e}")
            return False
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """
        获取每日合规统计。
        
        Returns:
            Dict: 统计数据
        """
        if not self._loaded:
            logger.warning("iam_harness未加载，返回空统计")
            return {}
        
        try:
            return self._get_daily_stats()
        except Exception as e:
            logger.error(f"获取每日统计失败: {e}")
            return {}
    
    def is_loaded(self) -> bool:
        """
        检查iam_harness是否已加载。
        
        Returns:
            bool: 是否已加载
        """
        return self._loaded
    
    def get_version(self) -> str:
        """
        获取iam_harness版本。
        
        Returns:
            str: 版本号
        """
        if not self._loaded:
            return "未加载"
        
        try:
            import iam_harness
            return getattr(iam_harness, '__version__', '未知')
        except Exception:
            return "未知"


# ── 便捷函数 ──

def create_iam_integration(auto_load: bool = True) -> IamIntegration:
    """
    创建IamIntegration实例。
    
    Args:
        auto_load: 是否自动加载iam_harness模块
        
    Returns:
        IamIntegration: 实例
    """
    return IamIntegration(auto_load=auto_load)


def retrieve_principles(context: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    便捷函数：检索相关原则。
    
    Args:
        context: 决策上下文
        top_n: 返回的原则数量
        
    Returns:
        List[Dict]: 相关原则列表
    """
    iam = create_iam_integration()
    return iam.retrieve(context, top_n=top_n)


def format_principles_for_injection(principles: List[Dict[str, Any]]) -> str:
    """
    便捷函数：格式化原则为注入文本。
    
    Args:
        principles: 原则列表
        
    Returns:
        str: 格式化后的文本
    """
    iam = create_iam_integration()
    return iam.format_for_injection(principles)


def verify_decision(
    context: str,
    decision: str,
    principles: List[Dict[str, Any]],
    verifier_response: str = "",
) -> Dict[str, Any]:
    """
    便捷函数：验证决策合规性。
    
    Args:
        context: 决策上下文
        decision: 决策内容
        principles: 相关原则列表
        verifier_response: 验证器响应
        
    Returns:
        Dict: 验证结果
    """
    iam = create_iam_integration()
    return iam.verify(context, decision, principles, verifier_response)