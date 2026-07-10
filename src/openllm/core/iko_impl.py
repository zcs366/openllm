"""extracted from main_loop.py"""
import json, os, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .models import *
class IKO:
    """IKO — 输出体七因子管线
    
    七因子：IntentClassifier, SilenceAuditor, OutputRouter,
            OutputAuditChain, OutputFeedbackCollector,
            LambdaCalibrator, ProbingTrainer, SymmetricCodec
    """
    
    def __init__(self):
        self.metrics: list[TickMetrics] = []
        self.log_path = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 七因子组件初始化
        from openllm.iko import (
            IntentClassifier, OutputRouter, SilenceAuditor,
            OutputAuditChain, OutputFeedbackCollector,
            LambdaCalibrator, ProbingTrainer, SymmetricCodec,
        )
        self.classifier = IntentClassifier()
        self.router = OutputRouter()
        self.registry = self.router.registry
        self.auditor = SilenceAuditor()
        self.audit_chain = OutputAuditChain()
        self.feedback_collector = OutputFeedbackCollector()
        self.calibrator = LambdaCalibrator()
        self.probing_trainer = ProbingTrainer()
        self.codec = SymmetricCodec()
        self._output_count = 0
        
        print(f"  IKO 可观测就绪 · 七因子管线 · 日志={self.log_path}")
    
    def trace(self, phase: str, status: str, duration_ms: float = 0.0, detail: str = ""):
        """记录一次观测"""
        m = TickMetrics(
            tick_id=uuid.uuid4().hex[:8],
            phase=phase,
            status=status,
            duration_ms=duration_ms,
            detail=detail[:100],
        )
        self.metrics.append(m)
        # 后端持久化
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(m)) + "\n")
    
    def report(self) -> dict:
        """仪表盘快照"""
        if not self.metrics:
            return {"status": "no_data"}
        last_10 = self.metrics[-10:]
        return {
            "total_ticks": len(self.metrics),
            "ok_rate": sum(1 for m in last_10 if m.status == "ok") / max(len(last_10), 1),
            "avg_duration_ms": sum(m.duration_ms for m in last_10) / len(last_10),
            "last_phase": last_10[-1].phase if last_10 else "",
        }
    
    def process_output(self, raw_output: str, context: dict, decision: dict) -> str:
        """七因子输出管线"""
        # 1. 意图分类
        result = self.classifier.classify(context, decision)
        intent = result.intent
        
        # 2. 沉默审计
        intent = self.auditor.audit(intent, context, reversible=True)
        
        # 3. 路由+渲染
        plan = self.router.route(intent, content={"text": raw_output}, user_prefs={})
        renderer = self.registry.get(plan.renderer_name)
        if renderer:
            output = renderer.render(intent, {"text": raw_output}, {}, 0.8)
        else:
            output = raw_output
        
        # 4. 审计链（赫淮斯托斯约束：reasoning_chain_hash必须是真实hash）
        if output:  # 非空输出才审计
            import hashlib, json as _json
            reasoning_hash = hashlib.sha256(
                _json.dumps({"intent": intent.value, "output": output[:200]}, sort_keys=True).encode()
            ).hexdigest()[:16]
            self.audit_chain.append(
                output_id=f"out-{self._output_count}",
                intent=intent.value,
                content=output.encode(),
                decision_source="IKO",
                risk_level=0.1,
                confidence=0.8,
                reasoning_chain_hash=reasoning_hash,
            )
            self._output_count += 1
        
        # 5. 记录trace
        self.trace("output_process", "ok", detail=f"intent={intent.value}")
        
        return output
    
    def shutdown(self):
        """关闭时的报告"""
        print(f"\n  IKO 关闭报告: {len(self.metrics)} tick, 最后状态={self.metrics[-1].status if self.metrics else 'N/A'}")


# ═══════════════════════════════════════════════════════
# Agent主循环（集成Session/Turn + 10阶段心跳）
# ═══════════════════════════════════════════════════════
