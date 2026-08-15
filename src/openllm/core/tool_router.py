"""
tool_router.py — Tool Router：自然语言→结构化工具调用
=====================================================

七神启示2026-08-15裁决：
  Router必须是engine.chat()的必经之路（非旁路）
  输出复用ETAS ActionTrace格式（REQUEST/COMMIT/DENIED）
  设计为可消亡的脚手架——模型原生支持function-calling时可退役

接口契约：
  输入：MiMo自然语言输出（str）
  输出：ToolCall(name, params) 或 None（无工具调用意图）
  审计：每个ToolCall经过GovernanceEngine产生ActionTrace
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger("openllm.tool_router")


class RouterStrategy(str, Enum):
    REGEX = "regex"
    LLM_EXTRACTION = "llm"
    HYBRID = "hybrid"


@dataclass
class ToolCall:
    name: str
    params: dict
    confidence: float = 1.0
    source: str = "regex"


@dataclass
class RouteResult:
    tool_call: Optional[ToolCall] = None
    trace_events: list = field(default_factory=list)
    raw_text: str = ""
    routed: bool = False


# ═══════════════════════════════════════
# 意图模式库——regex路由的核心
# ═══════════════════════════════════════

INTENT_PATTERNS = [
    {
        "name": "hermes_search",
        "patterns": [
            r"(?:搜索|搜一下|查一下|查找|检索|search|look up)\s*(.+?)(?:\s*[。！\.]|$)",
            r"(?:帮我|请|能不能)\s*(?:搜索|搜|查)\s*(.+)",
        ],
        "param_key": "query",
        "extra_params": {"max_results": 5},
    },
    {
        "name": "read_file",
        "patterns": [
            r"(?:读取|打开|查看|read)\s*(.+?\.(?:py|md|json|txt|yaml|yml|toml|csv))",
        ],
        "param_key": "path",
        "extra_params": {},
    },
    {
        "name": "shell",
        "patterns": [
            r"(?:执行|运行|run|execute)\s*(?:命令|command)?\s*(.+?)(?:\s*[。！\.]|$)",
            r"```(?:bash|sh|shell)?\s*\n(.+?)```",
        ],
        "param_key": "command",
        "extra_params": {},
    },
    {
        "name": "write_file",
        "patterns": [
            r"(?:写入|创建|create|write)\s*(?:文件)?\s*(.+?\.(?:py|md|json|txt|yaml|yml))",
        ],
        "param_key": "path",
        "extra_params": {},
    },
    {
        "name": "list_dir",
        "patterns": [
            r"(?:列出|ls|list|查看)\s*(?:目录|文件夹|文件列表)\s*(.+?)(?:\s*[。！\.]|$)",
        ],
        "param_key": "path",
        "extra_params": {},
    },
]


class ToolRouter:
    """
    Tool Router — MiMo自然语言→结构化工具调用
    
    三层架构：
      1. Regex层（确定性、快速）
      2. 启发式层（上下文推断）
      3. LLM提取层（fallback，可选）
    
    每个路由决策都经过GovernanceEngine审计。
    """
    
    def __init__(self, engine=None, strategy: RouterStrategy = RouterStrategy.REGEX):
        self.engine = engine
        self.strategy = strategy
        self._trace_store = None
        
        if engine:
            try:
                from .action_trace import ActionTraceStore
                self._trace_store = ActionTraceStore()
            except ImportError:
                pass
    
    def route(self, model_output: str) -> RouteResult:
        result = RouteResult(raw_text=model_output)
        
        if not model_output or not model_output.strip():
            return result
        
        tool_call = self._regex_match(model_output)
        
        if not tool_call:
            tool_call = self._heuristic_match(model_output)
        
        if not tool_call:
            return result
        
        result.tool_call = tool_call
        result.routed = True
        
        trace_event = self._audit_request(tool_call)
        if trace_event:
            result.trace_events.append(trace_event)
        
        return result
    
    def _regex_match(self, text: str) -> Optional[ToolCall]:
        for intent in INTENT_PATTERNS:
            for pattern in intent["patterns"]:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    param_value = match.group(1).strip()
                    params = {intent["param_key"]: param_value}
                    params.update(intent.get("extra_params", {}))
                    return ToolCall(
                        name=intent["name"],
                        params=params,
                        confidence=0.9,
                        source="regex",
                    )
        return None
    
    def _heuristic_match(self, text: str) -> Optional[ToolCall]:
        text_lower = text.lower()
        search_keywords = ["搜索", "搜", "查", "找", "search", "find", "look"]
        if any(kw in text_lower for kw in search_keywords):
            query = text
            for kw in ["帮我", "请", "能不能", "搜索", "搜一下", "查一下", "查找", "检索"]:
                query = query.replace(kw, "")
            query = query.strip()
            if query and len(query) > 2:
                return ToolCall(
                    name="hermes_search",
                    params={"query": query[:80], "max_results": 5},
                    confidence=0.7,
                    source="heuristic",
                )
        return None
    
    def _audit_request(self, tool_call: ToolCall) -> Optional[object]:
        if not self.engine:
            return None
        try:
            from .action_trace import make_trace_event, EventPhase
            event = make_trace_event(
                phase=EventPhase.REQUEST,
                action_name=tool_call.name,
                params=tool_call.params,
            )
            if hasattr(self.engine, 'security'):
                allowed, reason = self.engine.security.check_action(tool_call.name)
                if not allowed:
                    event.phase = EventPhase.DENIED
                    event.denial_reason = reason
                    logger.info(f"Router审计拒绝: {tool_call.name} — {reason}")
            return event
        except Exception as e:
            logger.debug(f"Router审计跳过: {e}")
            return None
    
    def execute_and_commit(self, tool_call: ToolCall) -> tuple:
        if not self.engine:
            return None, None
        try:
            result = self.engine.execute_tool(tool_call.name, **tool_call.params)
            from .action_trace import make_trace_event, EventPhase
            event = make_trace_event(
                phase=EventPhase.COMMIT if result.success else EventPhase.DENIED,
                action_name=tool_call.name,
                params=tool_call.params,
                result_summary=result.output[:200] if result.output else None,
            )
            return result, event
        except Exception as e:
            logger.error(f"Router执行失败: {e}")
            return None, None
