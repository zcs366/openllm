"""
openLLM Failure Signature Tracker — Phase 8/9 填充

填充 main_loop.py 中 learn_causal() 和 evolve() 的空壳。
在 engine.py 的 execute_tool_with_hindsight() 和 REFLECT 阶段中集成。

核心机制:
  Phase 8 (learn_causal):
    - 从失败中提取failure signature（失败模式分类）
    - 检测重复失败模式（同一signature出现3次=系统性问题）

  Phase 9 (evolve):
    - 根据failure patterns提出harness修改建议
    - 输出：修改提案（不自动执行，需人类批准）
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class FailureCategory(Enum):
    """失败模式分类"""
    TOOL_PARAM = auto()       # 工具参数错误
    TOOL_TIMEOUT = auto()     # 工具执行超时
    TOOL_PERMISSION = auto()  # 工具权限不足
    LLM_HALLUCINATION = auto()  # LLM幻觉（生成不存在的工具调用）
    LLM_FORMAT = auto()       # LLM输出格式错误
    CONTEXT_OVERFLOW = auto()  # 上下文溢出
    ROUTE_WRONG = auto()      # 路由决策错误（该跳的没跳，不该跳的跳了）
    USER_CORRECTION = auto()  # 用户纠正（Agent做了错事被纠正）
    UNKNOWN = auto()          # 未知失败


# ── Harness七层映射（HTIR-Step1）────────────────────────

class HarnessLayer(Enum):
    """Agent Harness七层（论文HarnessFix arXiv:2606.06324）"""
    EXECUTION = "execution_environment"
    TOOL_INTERFACE = "tool_interface"
    CONTEXT_MEMORY = "context_and_memory"
    LIFECYCLE = "lifecycle_orchestration"
    OBSERVABILITY = "observability"
    VERIFICATION = "verification"
    GOVERNANCE = "governance"


# FailureCategory → HarnessLayer 映射表
# 注: EXECUTION层(execution_environment)无直接映射——
# 执行环境崩溃(sandbox/OOM/segfault)属于基础设施层，不在agent harness七层范围内。
CATEGORY_TO_LAYER = {
    FailureCategory.TOOL_PARAM: HarnessLayer.TOOL_INTERFACE,
    FailureCategory.TOOL_TIMEOUT: HarnessLayer.LIFECYCLE,
    FailureCategory.TOOL_PERMISSION: HarnessLayer.GOVERNANCE,
    FailureCategory.LLM_HALLUCINATION: HarnessLayer.CONTEXT_MEMORY,
    FailureCategory.LLM_FORMAT: HarnessLayer.VERIFICATION,
    FailureCategory.CONTEXT_OVERFLOW: HarnessLayer.CONTEXT_MEMORY,
    FailureCategory.ROUTE_WRONG: HarnessLayer.LIFECYCLE,
    FailureCategory.USER_CORRECTION: HarnessLayer.OBSERVABILITY,
    FailureCategory.UNKNOWN: HarnessLayer.OBSERVABILITY,
}


@dataclass
class FailureSignature:
    """一次失败的签名"""
    category: FailureCategory
    tool_name: str = ""
    error_pattern: str = ""     # 错误的模式化描述（不含具体值）
    context_hash: str = ""      # 上下文指纹（用于检测重复）
    timestamp: float = field(default_factory=time.time)
    raw_error: str = ""         # 原始错误信息（截断）
    harness_layer: str = ""     # HTIR-Step1: 归因到harness七层
    
    def key(self) -> str:
        """用于去重的key。"""
        return f"{self.category.name}:{self.tool_name}:{self.error_pattern}"


@dataclass
class EvolutionProposal:
    """Phase 9 产出的进化提案"""
    proposal_type: str  # "route_rule" | "tool_fix" | "prompt_patch" | "test_case"
    description: str
    target: str         # 修改目标（文件/函数/规则）
    rationale: str      # 基于哪些failure patterns
    confidence: float   # 0-1
    timestamp: float = field(default_factory=time.time)


class FailureSignatureTracker:
    """
    Failure Signature 提取器 + 进化引擎。
    
    使用方式:
        tracker = FailureSignatureTracker()
        
        # Phase 8: 从失败中学习
        sig = tracker.extract_signature(tool_name, error, context)
        tracker.learn_causal(sig)
        
        # Phase 9: 提出进化提案（每N次失败触发一次）
        if tracker.should_evolve():
            proposals = tracker.evolve()
    """
    
    # 错误模式匹配规则
    ERROR_PATTERNS = {
        FailureCategory.TOOL_PARAM: [
            "缺少必填参数", "参数.*为空", "missing required",
            "invalid parameter", "type error",
        ],
        FailureCategory.TOOL_TIMEOUT: [
            "timeout", "timed out", "超时", "deadline exceeded",
        ],
        FailureCategory.TOOL_PERMISSION: [
            "permission denied", "权限", "forbidden", "unauthorized",
            "ISN风险检查未通过",
        ],
        FailureCategory.LLM_HALLUCINATION: [
            "tool.*not found", "工具.*不存在", "unknown tool",
        ],
        FailureCategory.LLM_FORMAT: [
            "JSON.*decode", "json.*parse", "格式错误",
            "invalid json", "expecting",
        ],
        FailureCategory.CONTEXT_OVERFLOW: [
            "context.*length", "token.*limit", "上下文.*溢出",
            "maximum context",
        ],
    }
    
    # 每个category的进化阈值（出现N次同类失败→触发进化）
    EVOLVE_THRESHOLDS = {
        FailureCategory.TOOL_PARAM: 3,
        FailureCategory.TOOL_TIMEOUT: 2,
        FailureCategory.TOOL_PERMISSION: 2,
        FailureCategory.LLM_HALLUCINATION: 3,
        FailureCategory.LLM_FORMAT: 5,  # 格式错误容忍度高
        FailureCategory.CONTEXT_OVERFLOW: 2,
        FailureCategory.ROUTE_WRONG: 2,
        FailureCategory.USER_CORRECTION: 3,
        FailureCategory.UNKNOWN: 5,
    }
    
    def __init__(self, max_signatures: int = 100):
        self.signatures: list[FailureSignature] = []
        self.max_signatures = max_signatures
        self._failure_counts: dict[str, int] = {}  # signature.key() → count
        self._proposals: list[EvolutionProposal] = []
    
    def classify_error(self, error: str) -> FailureCategory:
        """根据错误文本分类失败类型。"""
        error_lower = error.lower()
        for category, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return category
        return FailureCategory.UNKNOWN
    
    def extract_signature(
        self,
        tool_name: str,
        error: str,
        context: Optional[dict] = None,
    ) -> FailureSignature:
        """
        从一次失败中提取signature。
        
        Args:
            tool_name: 失败的工具名称
            error: 原始错误信息
            context: 当前上下文（可选）
        
        Returns:
            FailureSignature
        """
        category = self.classify_error(error)
        
        # 模式化错误（去除具体值，保留结构）
        error_pattern = self._generalize_error(error)
        
        # 上下文指纹（简单版：user_input的hash）
        ctx_hash = ""
        if context and "user_input" in context:
            ctx_hash = str(hash(context["user_input"][:100]))
        
        return FailureSignature(
            category=category,
            tool_name=tool_name,
            error_pattern=error_pattern,
            context_hash=ctx_hash,
            raw_error=error[:500],
        )
    
    def learn_causal(self, sig: FailureSignature) -> Optional[str]:
        """
        Phase 8: 因果学习。
        
        记录signature，检测重复模式。
        返回：如果触发了进化阈值，返回提示信息。
        """
        self.signatures.append(sig)
        
        # 限制数量（FIFO）
        if len(self.signatures) > self.max_signatures:
            self.signatures = self.signatures[-self.max_signatures:]
        
        # 检测重复
        key = sig.key()
        self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
        count = self._failure_counts[key]
        
        threshold = self.EVOLVE_THRESHOLDS.get(sig.category, 5)
        if count >= threshold:
            return (
                f"⚠️ {sig.category.name} 失败模式已出现 {count} 次 "
                f"(阈值={threshold})，建议触发Phase 9进化"
            )
        
        if count >= 2:
            return (
                f"  {sig.category.name} 重复失败 "
                f"({count}/{threshold})"
            )
        
        return None
    
    def should_evolve(self) -> bool:
        """是否应该触发Phase 9进化。按category级别判断。"""
        category_counts: dict[FailureCategory, int] = {}
        for sig in self.signatures:
            category_counts[sig.category] = category_counts.get(sig.category, 0) + 1
        for category, count in category_counts.items():
            threshold = self.EVOLVE_THRESHOLDS.get(category, 5)
            if count >= threshold:
                return True
        return False
    
    def evolve(self) -> list[EvolutionProposal]:
        """
        Phase 9: 进化提案生成。
        
        根据failure patterns提出harness修改建议。
        """
        proposals = []
        
        # 按category统计
        category_counts: dict[FailureCategory, int] = {}
        for sig in self.signatures:
            category_counts[sig.category] = category_counts.get(sig.category, 0) + 1
        
        for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            threshold = self.EVOLVE_THRESHOLDS.get(category, 5)
            if count < threshold:
                continue
            
            # 生成提案
            proposal = self._generate_proposal(category, count)
            if proposal:
                proposals.append(proposal)
        
        self._proposals.extend(proposals)
        return proposals
    
    def _generate_proposal(
        self, category: FailureCategory, count: int
    ) -> Optional[EvolutionProposal]:
        """为特定失败类型生成进化提案。"""
        
        if category == FailureCategory.TOOL_PARAM:
            return EvolutionProposal(
                proposal_type="tool_fix",
                description=f"工具参数验证增强：{count}次参数错误，建议添加更严格的参数schema校验",
                target="engine.py:_verify_tool_params",
                rationale=f"FailureCategory.TOOL_PARAM出现{count}次",
                confidence=min(0.9, 0.5 + count * 0.1),
            )
        
        if category == FailureCategory.TOOL_TIMEOUT:
            return EvolutionProposal(
                proposal_type="tool_fix",
                description=f"工具超时处理增强：{count}次超时，建议添加超时重试或降级策略",
                target="engine.py:execute_tool",
                rationale=f"FailureCategory.TOOL_TIMEOUT出现{count}次",
                confidence=min(0.85, 0.5 + count * 0.1),
            )
        
        if category == FailureCategory.LLM_FORMAT:
            return EvolutionProposal(
                proposal_type="prompt_patch",
                description=f"LLM输出格式修复：{count}次JSON解析失败，建议强化prompt中的格式要求",
                target="loop.py:PLAN phase",
                rationale=f"FailureCategory.LLM_FORMAT出现{count}次",
                confidence=min(0.8, 0.4 + count * 0.08),
            )
        
        if category == FailureCategory.ROUTE_WRONG:
            return EvolutionProposal(
                proposal_type="route_rule",
                description=f"路由规则修正：{count}次路由决策错误，建议更新router.py规则",
                target="router.py",
                rationale=f"FailureCategory.ROUTE_WRONG出现{count}次",
                confidence=min(0.85, 0.5 + count * 0.1),
            )
        
        if category == FailureCategory.USER_CORRECTION:
            return EvolutionProposal(
                proposal_type="test_case",
                description=f"用户纠正记录：{count}次用户纠正，建议生成回归测试用例",
                target="tests/test_regression.py",
                rationale=f"FailureCategory.USER_CORRECTION出现{count}次",
                confidence=min(0.9, 0.6 + count * 0.1),
            )
        
        return None
    
    def _generalize_error(self, error: str) -> str:
        """将具体错误泛化为模式。"""
        # 去除具体路径、数字、ID
        pattern = error
        pattern = re.sub(r'/[\w/.-]+', '<PATH>', pattern)
        pattern = re.sub(r'\d{4,}', '<NUM>', pattern)
        pattern = re.sub(r'[a-f0-9]{8,}', '<HASH>', pattern)
        pattern = re.sub(r'"[^"]{20,}"', '"<LONG_STR>"', pattern)
        return pattern[:200]
    
    def get_stats(self) -> dict:
        """获取tracker统计。"""
        return {
            "total_signatures": len(self.signatures),
            "unique_patterns": len(self._failure_counts),
            "proposals_generated": len(self._proposals),
            "should_evolve": self.should_evolve(),
            "category_breakdown": {
                cat.name: sum(1 for s in self.signatures if s.category == cat)
                for cat in FailureCategory
                if any(s.category == cat for s in self.signatures)
            },
        }


# ── 便捷函数 ──────────────────────────────────────

def create_tracker() -> FailureSignatureTracker:
    """创建Failure Signature Tracker实例。"""
    return FailureSignatureTracker()
