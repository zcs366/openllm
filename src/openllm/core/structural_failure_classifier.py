"""
结构性失败分类器 — P0-a

区分三种失败：
  structural : 结构性失败 → 触发治理转换
  local      : 局部缺陷   → 记录但不触发治理
  ambiguous  : 待定       → 发GovernanceRequest，等待人工或自动处理

判定公式（鲁班修正版·五人合议通过）:
  composite = freq_score × 0.4 + spread_score × 0.3 + ctx_sim × 0.3
  
  ≥ 0.7 → structural
  ≥ 0.4 → ambiguous (→ GovernanceRequest)
  < 0.4 → local

韩信第三信号: ambiguous分类自动发GovernanceRequest

依赖: failure_tracker.FailureCategory, failure_tracker.FailureSignature
"""

import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .failure_tracker import FailureSignature, FailureCategory


# ── 数据模型 ──────────────────────────────────────────

@dataclass
class GovernanceRequest:
    """韩信第三信号: Agent主动请求治理干预。
    
    当分类器输出'ambiguous'时自动发出。
    记录到governance_requests.jsonl等待处理。
    """
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    failure_signature: Optional[FailureSignature] = None
    reason: str = ""                # 为什么ambiguous
    composite_score: float = 0.0    # 判定分数
    component_scores: dict = field(default_factory=dict)  # 各维度分数
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"         # pending | resolved | dismissed
    resolution: str = ""            # 人工处理结果


@dataclass
class ClassificationResult:
    """分类结果"""
    classification: str  # structural | local | ambiguous
    composite_score: float
    freq_score: float
    spread_score: float
    ctx_sim_score: float
    time_decay_factor: float
    affected_components: list[str]
    similar_signatures: list[str]  # 相似signature的key
    governance_request: Optional[GovernanceRequest] = None  # ambiguous时有值


# ── 分类器 ──────────────────────────────────────────────

class StructuralFailureClassifier:
    """结构性失败分类器。
    
    三维评分:
    1. freq_score (0.4): 时间衰减频率——近期失败权重高
    2. spread_score (0.3): 扩散检测——影响多个tool/component
    3. ctx_sim_score (0.3): 上下文相似度——TF-IDF相似度
    
    阈值:
    - ≥ 0.7 → structural (结构性失败，触发治理转换)
    - ≥ 0.4 → ambiguous (待定，发GovernanceRequest)
    - < 0.4 → local (局部缺陷，记录但不触发)
    """
    
    # 时间衰减半衰期（秒）
    HALF_LIFE_SECONDS = 7 * 24 * 3600  # 7天
    
    # 阈值（五人合议: 鲁班审定）
    # 3次同类失败在同一tool → composite ≈ 0.6-0.65
    # 5次以上或跨组件 → composite ≈ 0.7+
    # 阈值设为0.6让3次同类即可触发structural
    STRUCTURAL_THRESHOLD = 0.6
    AMBIGUOUS_THRESHOLD = 0.35
    
    # mechanism名→FailureCategory枚举映射（M1修复，与GovernanceEngine共享）
    _MECHANISM_TO_CATEGORY = {
        "tool_loop": FailureCategory.TOOL_PARAM,
        "missing_artifact": FailureCategory.TOOL_PARAM,
        "wrong_format": FailureCategory.LLM_FORMAT,
        "dependency_missing": FailureCategory.TOOL_PARAM,
        "timeout": FailureCategory.TOOL_TIMEOUT,
        "logic_error": FailureCategory.LLM_HALLUCINATION,
        "permission": FailureCategory.TOOL_PERMISSION,
        "state_corruption": FailureCategory.TOOL_PARAM,
        "exploration_loop": FailureCategory.ROUTE_WRONG,
        "premature_success": FailureCategory.UNKNOWN,
        "tool_param": FailureCategory.TOOL_PARAM,
        "tool_timeout": FailureCategory.TOOL_TIMEOUT,
        "tool_permission": FailureCategory.TOOL_PERMISSION,
        "llm_hallucination": FailureCategory.LLM_HALLUCINATION,
        "llm_format": FailureCategory.LLM_FORMAT,
        "context_overflow": FailureCategory.CONTEXT_OVERFLOW,
        "route_wrong": FailureCategory.ROUTE_WRONG,
        "user_correction": FailureCategory.USER_CORRECTION,
    }
    
    def __init__(self):
        # 历史失败签名（内存缓存，从磁盘加载）
        self._history: list[FailureSignature] = []
        # 组件统计: tool_name → [signature_keys]
        self._component_index: dict[str, list[str]] = defaultdict(list)
        # 签名key → 签名列表（用于去重和频率计算）
        self._signature_index: dict[str, list[FailureSignature]] = defaultdict(list)
        # 持久化路径
        self._storage_dir = Path.home() / ".openllm" / "output" / "ios"
        self._requests_path = self._storage_dir / "governance_requests.jsonl"
        
        # 加载历史
        self._load_history()
    
    def classify(self, sig: FailureSignature) -> ClassificationResult:
        """分类一次失败。
        
        Args:
            sig: 本次失败的签名
            
        Returns:
            ClassificationResult: 包含分类结果、各维度分数、相关签名
        """
        # ① 时间衰减频率
        freq_score, similar_sigs = self._time_weighted_frequency(sig)
        
        # ② 扩散检测
        spread_score, affected_components = self._spread_score(sig)
        
        # ③ 上下文相似度
        ctx_sim_score = self._context_similarity(sig, similar_sigs)
        
        # ④ 综合评分
        composite = (freq_score * 0.4 + 
                     spread_score * 0.3 + 
                     ctx_sim_score * 0.3)
        
        # ⑤ 分类判定
        if composite >= self.STRUCTURAL_THRESHOLD:
            classification = "structural"
        elif composite >= self.AMBIGUOUS_THRESHOLD:
            classification = "ambiguous"
        else:
            classification = "local"
        
        # ⑥ 韩信第三信号: ambiguous → GovernanceRequest
        gov_request = None
        if classification == "ambiguous":
            gov_request = self._emit_governance_request(sig, composite, {
                "freq": freq_score,
                "spread": spread_score, 
                "ctx_sim": ctx_sim_score,
            })
        
        # ⑦ 记录到历史
        self._record_signature(sig)
        
        return ClassificationResult(
            classification=classification,
            composite_score=round(composite, 3),
            freq_score=round(freq_score, 3),
            spread_score=round(spread_score, 3),
            ctx_sim_score=round(ctx_sim_score, 3),
            time_decay_factor=round(self._time_decay_factor(time.time()), 3),
            affected_components=affected_components,
            similar_signatures=[s.key() for s in similar_sigs[:5]],
            governance_request=gov_request,
        )
    
    # ── 评分函数 ──────────────────────────────────────
    
    def _time_weighted_frequency(
        self, sig: FailureSignature
    ) -> tuple[float, list[FailureSignature]]:
        """时间衰减频率评分。
        
        核心思想: 同一signature的失败，近期出现的权重高于远期。
        半衰期7天: 7天前的失败权重衰减到0.5。
        
        公式: freq = Σ(decay(t_now - t_i)) / max_freq
        其中 decay(Δt) = 2^(-Δt / HALF_LIFE)
        """
        now = time.time()
        key = sig.key()
        
        # 找同key的历史签名
        similar = self._signature_index.get(key, [])
        if not similar:
            # 没有完全匹配的key，用error_pattern模糊匹配
            similar = self._find_similar_by_pattern(sig)
        
        if not similar:
            return 0.0, []
        
        # 计算时间衰减频率
        weighted_count = 0.0
        for s in similar:
            age = now - s.timestamp
            if age < 0:
                age = 0  # 未来时间戳容错
            decay = math.pow(2, -age / self.HALF_LIFE_SECONDS)
            weighted_count += decay
        
        # 归一化: 3次近期失败 = 满分（论文: 同一signature出现3次=系统性问题）
        max_freq = 3.0
        freq = min(weighted_count / max_freq, 1.0)
        
        return freq, similar
    
    def _spread_score(
        self, sig: FailureSignature
    ) -> tuple[float, list[str]]:
        """扩散检测评分。
        
        核心思想: 同一失败模式影响的组件越多，越可能是结构性问题。
        
        公式: spread = min(affected_components / 3, 1.0)
        3个以上不同组件受影响 = 满分
        """
        key = sig.key()
        similar = self._signature_index.get(key, [])
        
        # 统计受影响的不同组件
        components = set()
        for s in similar:
            if s.tool_name:
                components.add(s.tool_name)
        # 当前签名的组件
        if sig.tool_name:
            components.add(sig.tool_name)
        
        # 归一化: 3个以上组件 = 满分
        spread = min(len(components) / 3.0, 1.0)
        
        return spread, list(components)
    
    def _context_similarity(
        self, sig: FailureSignature, similar: list[FailureSignature]
    ) -> float:
        """上下文相似度评分（简化TF-IDF）。
        
        核心思想: 相同错误模式在相似上下文中反复出现→结构性。
        使用词袋模型+Jaccard相似度作为TF-IDF的简化替代。
        
        公式: ctx_sim = max(jaccard(sig.raw, hist.raw) for hist in similar)
        """
        if not similar or not sig.raw_error:
            return 0.0
        
        sig_words = self._tokenize(sig.raw_error)
        if not sig_words:
            return 0.0
        
        max_sim = 0.0
        for s in similar:
            if not s.raw_error:
                continue
            hist_words = self._tokenize(s.raw_error)
            if not hist_words:
                continue
            # Jaccard相似度
            intersection = sig_words & hist_words
            union = sig_words | hist_words
            sim = len(intersection) / len(union) if union else 0.0
            max_sim = max(max_sim, sim)
        
        return max_sim
    
    def _tokenize(self, text: str) -> set[str]:
        """简单分词: 小写 + 空格分割 + 去除短词"""
        return {w for w in text.lower().split() if len(w) > 2}
    
    def _time_decay_factor(self, timestamp: float) -> float:
        """计算当前时间的衰减因子（用于报告）"""
        now = time.time()
        age = now - timestamp
        return math.pow(2, -age / self.HALF_LIFE_SECONDS)
    
    # ── 韩信第三信号 ──────────────────────────────────
    
    def _emit_governance_request(
        self, sig: FailureSignature, composite: float, scores: dict
    ) -> GovernanceRequest:
        """韩信第三信号: ambiguous分类→发GovernanceRequest。
        
        写入governance_requests.jsonl等待处理。
        """
        req = GovernanceRequest(
            failure_signature=sig,
            reason=f"ambiguous分类: composite={composite:.3f}, "
                   f"freq={scores['freq']:.3f}, "
                   f"spread={scores['spread']:.3f}, "
                   f"ctx_sim={scores['ctx_sim']:.3f}",
            composite_score=composite,
            component_scores=scores,
        )
        
        # 持久化
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self._requests_path, "a") as f:
            record = {
                "request_id": req.request_id,
                "failure_key": sig.key(),
                "failure_category": sig.category.name,
                "tool_name": sig.tool_name,
                "error_pattern": sig.error_pattern[:200],
                "reason": req.reason,
                "composite_score": req.composite_score,
                "component_scores": req.component_scores,
                "timestamp": req.timestamp,
                "status": req.status,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return req
    
    # ── 历史管理 ──────────────────────────────────────
    
    def _find_similar_by_pattern(
        self, sig: FailureSignature
    ) -> list[FailureSignature]:
        """当精确key匹配无结果时，用error_pattern模糊匹配"""
        if not sig.error_pattern:
            return []
        
        similar = []
        sig_words = self._tokenize(sig.error_pattern)
        
        for key, sigs in self._signature_index.items():
            for s in sigs:
                if not s.error_pattern:
                    continue
                hist_words = self._tokenize(s.error_pattern)
                overlap = sig_words & hist_words
                if len(overlap) >= 2:  # 至少2个词重叠
                    similar.append(s)
        
        return similar[-20:]  # 最近20条
    
    def _record_signature(self, sig: FailureSignature):
        """记录签名到内存索引"""
        key = sig.key()
        self._history.append(sig)
        self._signature_index[key].append(sig)
        if sig.tool_name:
            self._component_index[sig.tool_name].append(key)
        
        # 限制内存大小
        if len(self._history) > 500:
            self._history = self._history[-500:]
    
    def _load_history(self):
        """从磁盘加载历史失败签名"""
        causal_path = Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl"
        if not causal_path.exists():
            return
        
        try:
            with open(causal_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("result_success") is False:
                            mechanism = entry.get("mechanism", "unknown")
                            category = self._MECHANISM_TO_CATEGORY.get(
                                mechanism.lower(), FailureCategory.UNKNOWN
                            )
                            sig = FailureSignature(
                                category=category,
                                tool_name=entry.get("action", "")[:50],
                                error_pattern=entry.get("actual", "")[:200],
                                timestamp=entry.get("timestamp", 0),
                                raw_error=entry.get("actual", "")[:500],
                            )
                            self._record_signature(sig)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except IOError:
            pass
    
    # ── 查询接口 ──────────────────────────────────────
    
    def get_governance_requests(
        self, status: str = "pending"
    ) -> list[dict]:
        """查询GovernanceRequest"""
        if not self._requests_path.exists():
            return []
        
        requests = []
        try:
            with open(self._requests_path) as f:
                for line in f:
                    try:
                        req = json.loads(line)
                        if req.get("status") == status:
                            requests.append(req)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        
        return requests
    
    def resolve_governance_request(
        self, request_id: str, resolution: str, new_status: str = "resolved"
    ) -> bool:
        """处理GovernanceRequest（人工或自动）"""
        if not self._requests_path.exists():
            return False
        
        lines = self._requests_path.read_text().splitlines()
        updated = False
        new_lines = []
        
        for line in lines:
            try:
                req = json.loads(line)
                if req.get("request_id") == request_id:
                    req["status"] = new_status
                    req["resolution"] = resolution
                    updated = True
                new_lines.append(json.dumps(req, ensure_ascii=False))
            except json.JSONDecodeError:
                new_lines.append(line)
        
        if updated:
            self._requests_path.write_text("\n".join(new_lines) + "\n")
        
        return updated
    
    def get_stats(self) -> dict:
        """获取分类器统计"""
        pending = len(self.get_governance_requests("pending"))
        resolved = len(self.get_governance_requests("resolved"))
        
        return {
            "total_history": len(self._history),
            "unique_patterns": len(self._signature_index),
            "unique_components": len(self._component_index),
            "pending_requests": pending,
            "resolved_requests": resolved,
        }
