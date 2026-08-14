from .degradation_trace import trace_degradation
"""extracted from main_loop.py"""
import json, os, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .models import *
from .provider_impl import LLMProvider
class 章鱼I:
    """推理引擎 + 左右脑"""
    
    # 能力降级等级
    HEALTH = {
        "FULL":     {"confidence_min": 0.7, "tools_all": True},
        "DEGRADED": {"confidence_min": 0.4, "tools_all": False},
        "MINIMAL":  {"confidence_min": 0.2, "tools_all": False},
        "OFFLINE":  {"confidence_min": 0.0, "tools_all": False},
    }
    
    def __init__(self):
        self.left = _LeftBrain()
        self.right = _RightBrain()
        # 触手脑：文件监控 + 全文索引
        from .tentacle import FileWatcherBrain, IndexBrain
        self.tentacles = {
            "file_watcher": FileWatcherBrain(),
            "index": IndexBrain(),
        }
        print("  章鱼I 推理引擎就绪 · 左右脑在线 · 触手脑在线")
    
    def health_check(self) -> str:
        """自检。返回当前健康等级。"""
        if not self.left.provider._available:
            return "MINIMAL"  # 无API=降级
        try:
            _ = self.left.provider.chat([{"role":"user","content":"ping"}])
            return "FULL"
        except:
            return "DEGRADED"
    
    def d0_snapshot(self) -> dict:
        """
        D₀感知简化版·实时认知深度报告
        
        完整版需要IAH扫描注意力头D₀——当前不可用。
        简化版基于：API可用性 + health等级 + 可用工具数 + context使用率
        
        Returns:
            sem_ratio: 语义能力比率（0-1）
            d0_budget: 可用token数
            cognitive_rhythm: 认知节奏（fast/steady/slow）
            health: 当前健康等级
        """
        health = self.health_check()
        
        # sem_ratio 估算：FULL→0.9, DEGRADED→0.5, MINIMAL→0.2
        sem_map = {"FULL": 0.9, "DEGRADED": 0.5, "MINIMAL": 0.2, "OFFLINE": 0.0}
        sem_ratio = sem_map.get(health, 0.3)
        
        # d0_budget: 可用API时=大模型语义容量，降级时=本地规则容量
        d0_budget = 100000 if health == "FULL" else 10000 if health == "DEGRADED" else 1000
        
        # cognitive_rhythm: 有API→fast, 降级→steady, 无API→slow
        rhythm = "fast" if health == "FULL" else "steady" if health == "DEGRADED" else "slow"
        
        return {
            "sem_ratio": sem_ratio,
            "d0_budget": d0_budget,
            "cognitive_rhythm": rhythm,
            "health": health,
            "tools_available": len(self.tentacles) if hasattr(self, 'tentacles') else 0,
            "api_available": self.left.provider._available,
        }
    
    def predict_consequences(self, ctx: Context) -> Prediction:
        """Phase 3: 因果预测"""
        return self.left.predict(ctx)
    
    def reason(self, ctx: Context, prediction: Optional[Prediction] = None, 
               risk: Optional[RiskAssessment] = None) -> tuple[Proposal, Critique]:
        """Phase 5: 双脑推理"""
        proposal = self.left.think(ctx, prediction, risk)
        critique = self.right.review(ctx, proposal)
        return proposal, critique
    
    def compare(self, prediction: Prediction, result: ActionResult) -> CausalDelta:
        """Phase 8: 因果对照"""
        # M0: 简单比较
        match = result.success and "模拟" not in result.output
        return CausalDelta(
            prediction_match=match,
            delta_summary=f"预测={'正确' if match else '错误'}",
            learned=["预测准确性基础检查"],
        )
    
    def learn_causal(self, ctx: Context, prediction: Prediction,
                     result: ActionResult, delta: CausalDelta):
        """Phase 8: 因果学习（实际实现在IOS.learn_causal）"""
        pass


class _LeftBrain:
    """左脑(正手)：提案"""
    
    def __init__(self):
        self.provider = LLMProvider()
    
    def predict(self, ctx: Context) -> Prediction:
        """因果预测（v2·数学+LLM双路径）"""
        # [进化] 数学预测：PredictionEngine纯计算
        math_prediction = {"summary": "", "risk_signals": []}
        try:
            from ..iai.prediction import PredictionEngine
            pe = PredictionEngine()
            result = pe.predict_next({"user_message": ctx.user_message, "tools": ctx.tools})
            if result:
                math_prediction["summary"] = f"[数学] 预测类型={result.get('predicted_type','unknown')} 置信={result.get('confidence', 0):.2f}"
                math_prediction["risk_signals"] = result.get("risk_signals", [])
        except Exception:
            pass
        
        # LLM预测：语义理解
        prompt = f"""你是openLLM的因果预测器。预测以下操作的后果。
用户意图：{ctx.user_message}
请预测：
[IF-SUCCESS] 如果操作成功，会发生什么
[IF-FAILURE] 如果操作失败，会发生什么
[IRREVERSIBLE] 哪些后果无法撤销
请用JSON格式输出：
{{"summary": "一句话预测摘要", "risk_signals": ["风险信号"]}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(resp) if resp.startswith("{") else {"summary": resp[:50]}
        except:
            data = {"summary": resp[:50]}
        
        # 合并数学+LLM预测
        combined_summary = data.get("summary", resp[:50])
        if math_prediction["summary"]:
            combined_summary = f"{math_prediction['summary']} | [LLM] {combined_summary}"
        combined_risks = list(set(data.get("risk_signals", []) + math_prediction["risk_signals"]))
        
        return Prediction(
            summary=combined_summary[:200],
            consequences=[data.get("summary", "")],
            confidence=data.get("confidence", 0.7),
            risk_signals=combined_risks,
        )
    
    def think(self, ctx: Context, prediction: Optional[Prediction] = None,
              risk: Optional[RiskAssessment] = None) -> Proposal:
        """基于上下文提出方案"""
        prompt = f"""基于以下用户消息，给出你的回答。直接回答，不要JSON格式。
用户：{ctx.user_message}"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        # 解析LLM返回的JSON
        try:
            data = json.loads(resp) if resp.startswith("{") else {"content": resp}
        except:
            data = {"content": resp, "confidence": 0.6}
        
        evidence = data.get("evidence", [])
        if prediction:
            evidence.append(f"已预测后果: {prediction.summary[:30]}")
        if risk:
            evidence.append(f"风险等级: {risk.level}")
        
        # 血管三：health_check→confidence校准
        # 无API时confidence下降
        confidence = data.get("confidence", 0.6)
        if not self.provider._available:
            confidence *= 0.8  # 无API时打8折
        
        return Proposal(
            content=data.get("content", resp[:100]),
            confidence=confidence,
            evidence=evidence,
            tool_calls=data.get("tool_calls", []),
            prediction_ref=prediction,
        )


class _RightBrain:
    """右脑(反手)：批判"""
    
    def __init__(self):
        self.provider = LLMProvider()
    
    def review(self, ctx: Context, proposal: Proposal) -> Critique:
        """审查左脑提案"""
        prompt = f"""你是openLLM的右脑。审查左脑的提案。
原始上下文：{ctx.user_message}
左脑提案：{proposal.content}
左脑置信度：{proposal.confidence}
左脑证据：{proposal.evidence}
你的任务：
1. 检查提案是否有逻辑漏洞
2. 检查提案是否有安全隐患
3. 检查证据是否支撑提案
4. 给出裁决：approve（通过）/ reject（否决）/ revise（修改）
请用JSON格式输出：
{{"verdict": "approve|reject|revise", "concerns": ["关心点1"], "suggestions": ["建议1"]}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        # 简单判断：包含reject/否/不行→reject，否则approve
        resp_lower = resp.lower() if resp else ""
        verdict = "reject" if any(w in resp_lower for w in ["reject", "否", "不行", "风险", "不合理"]) else "approve"
        return Critique(
            content=resp[:200] if resp else "",
            verdict=verdict,
            concerns=[],
            suggestions=[],
        )


# IOS已提取到ios_impl.py
from .ios_impl import IOS