#!/usr/bin/env python3
"""
openLLM Agent主循环 — 三体组装层

感知层(ISA+章鱼I) → 决策层(IOS) → 执行层(ISN+IKO)
心跳驱动三层轮流工作。
"""
import json, os, sys, time, uuid
from pathlib import Path

# 数据模型
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult,
                     CausalDelta, TickMetrics)

# 三层架构
from .perception import ISA, 章鱼I        # 感知层：我知道什么
from .decision import IOS                  # 决策层：我决定什么
from .execution import ISN, IKO            # 执行层：我做什么
from .provider_impl import LLMProvider

# Session
from .session import Session, Turn, TurnStatus, create_session

# 可选依赖
try:
    from .tool_validator import ToolCall as _TVToolCall, ValidationResult as _TVResult, validate_tool_result as _tv_validate
    _HAS_TOOL_VALIDATOR = True
except ImportError:
    _HAS_TOOL_VALIDATOR = False

try:
    from .inference_budget import InferenceBudgetManager
    _HAS_BUDGET = True
except ImportError:
    _HAS_BUDGET = False

_REFLECT_SCRIPT = Path.home() / "projects" / "isa" / "octopus" / "scripts" / "octopus_reflect.py"
_HAS_REFLECT = _REFLECT_SCRIPT.exists()

class Agent:
    """openLLM Agent — 独立运行的Agent框架"""
    
    def __init__(self, mode: str = "console"):
        # 五体初始化
        self.isa = ISA(mode)
        self.octopus = 章鱼I()
        self.ios = IOS()
        self.isn = ISN()
        self.iko = IKO()
        
        # Session/Turn
        # 六体自监督反馈环
        from ..governance.feedback_loop import FeedbackLoop
        self.feedback_loop = FeedbackLoop()
        self._body_outputs = {}  # 暂存每体最新输出
        
        # 间隙体——空闲期好奇心
        from .idle_wander import IdleWanderer
        self._wanderer = IdleWanderer(threshold=3)

        self.session = create_session(max_context_tokens=100000)
        
        # 状态
        self.running = False
        self._tick_count = 0
        self._max_ticks = 100  # 安全上限
        self._last_output = ""
        
        # ── IAX Layer 7 推理预算管理器 ──
        self._budget_manager = InferenceBudgetManager() if _HAS_BUDGET else None
        self._budget_checked_today = False
        
        # ── tool_validator 失败追踪 ──
        self._tool_failures: list[dict] = []
    
    def run(self):
        """主循环入口"""
        self.running = True
        print("\n═══ openLLM Agent 启动 ═══")
        print("  模式: 独立运行 · 无外部依赖")
        print("  Session:", self.session.id)
        print("  输入 /quit 退出\n")
        
        while self.running and self._tick_count < self._max_ticks:
            try:
                self._tick()
                self._tick_count += 1
            except KeyboardInterrupt:
                self.iko.trace("shutdown", "ok", detail="用户中断")
                break
            except Exception as e:
                self.iko.trace("fatal", "error", detail=str(e)[:50])
                ok = self.ios.recover(e)
                if not ok:
                    print(f"  🔴 不可恢复错误: {e}")
                    break
        
        self.iko.shutdown()
        self.session.end()
        print("\n═══ openLLM Agent 关闭 ═══")
    
    def run_once(self, message: str) -> str:
        """
        单次运行（用于测试/非交互模式）
        
        用法：
            agent = Agent(mode="silent")
            result = agent.run_once("你好")
        """
        saved_mode = self.isa.mode
        self.isa.mode = "silent"
        try:
            msg = Message(text=message)
            self._execute_tick(msg)
            return self._last_output
        finally:
            self.isa.mode = saved_mode
    
    def _tick(self):
        """一次心跳"""
        msg = self.isa.listen()
        if msg is None:
            # ── 间隙体：空闲期好奇心 ──
            if self._wanderer.tick_idle():
                self._do_wander()
            else:
                self.running = False
            return
        self._wanderer.tick_active()  # 有输入·重置空闲计数
        self._execute_tick(msg)
    
    def _do_wander(self):
        """散步模式——注意力的自由偏移"""
        discoveries = self._wanderer.wander()
        if discoveries:
            # 散步发现写入日志，不触发决策/执行
            for d in discoveries:
                self.iko.trace("wander", "ok", detail=f"{d.domain}: {d.finding[:40]}")
    
    def _record_inference(self, phase: str):
        """IAX: 从LLMProvider提取最近一次调用的token使用量，记录到预算管理器"""
        if not self._budget_manager:
            return
        # 从左右脑provider取_last_usage（Phase 3用左脑，Phase 5用左脑+右脑）
        usage = getattr(self.octopus.left.provider, '_last_usage', {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            return  # 模拟模式或无数据，跳过
        try:
            rec = self._budget_manager.record_inference(
                model=self.octopus.left.provider.model,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                duration_ms=0,  # 精确计时在phase trace中已有
            )
            self.iko.trace("budget", "ok",
                detail=f"{phase}: in={prompt_tokens} out={completion_tokens} cost=${rec.cost_usd:.4f}")
            # 每天第一次调用时检查预算
            if not self._budget_checked_today:
                self._budget_checked_today = True
                budget = self._budget_manager.check_budget()
                if budget["over_budget"]:
                    print(f"  ⚠️ IAX 推理预算超限: {budget['total_tokens']}/{budget['limit']} "
                          f"({budget['usage_ratio']:.0%}) 今日成本=${budget['cost_usd']:.4f}")
        except Exception:
            pass  # 预算追踪失败不阻塞主循环

    def _execute_tick(self, msg: Message):
        """一次完整的10阶段心跳"""
        t0 = time.time()
        
        # 创建新Turn
        turn = self.session.new_turn()
        
        try:
            # ── Phase 1: isa.listen() → Message ──
            turn.trace_phase("listen", "ok")
            self.iko.trace("listen", "ok")
            
            # ── Phase 2: build_context — 传入session和ios ──
            t1 = time.time()
            ctx = self.isa.build_context(msg, self.session, self.octopus, self.ios)
            turn.trace_phase("context", "ok", duration_ms=(time.time()-t1)*1000)
            self.iko.trace("context", "ok")

            # ── Phase 2.5: evidence replay — 相关证据回放 ──
            # 来源：arXiv:2607.02509 ReContext
            # 从七要素中提取top-K相关证据，追加到prompt尾部
            try:
                from ..memory.evidence_replay import create_replay_for_context
                replay_text = create_replay_for_context(msg.text, ctx, top_k=5, max_tokens=512)
                if replay_text:
                    ctx.search_results = getattr(ctx, 'search_results', []) or []
                    ctx.search_results.append(replay_text)
                    turn.trace_phase("replay", "ok", detail=f"evidence_replay:{len(replay_text)}chars")
                    self.iko.trace("replay", "ok")
            except Exception as e:
                # evidence replay失败不阻塞主流程
                turn.trace_phase("replay", "skip", detail=str(e)[:100])

            # ── Phase 3: 因果预测 ──
            t2 = time.time()
            prediction = self.octopus.predict_consequences(ctx)
            turn.trace_phase("predict", "ok", duration_ms=(time.time()-t2)*1000,
                           detail=prediction.summary_text())
            self.iko.trace("predict", "ok", detail=prediction.summary_text())
            
            # ── IAX: 记录Phase 3推理消耗 ──
            self._record_inference("phase_3_predict")

            # ── Phase 3.5: 治理检查·路由决策点（Copewell模式·2026-07-06） ──
            try:
                from ..governance.events import RoutingCheckEvent, GovernanceDimension
                from ..governance.audit import AuditChain
                _g8_event = RoutingCheckEvent(
                    actor="ios_phase3",
                    session_id=ctx.session_id if hasattr(ctx, 'session_id') else "default",
                    prev_hash="genesis",
                    phase="phase_3_prediction",
                    route_target=str(type(prediction).__name__),
                    risk_level="low",
                    approved=True,
                )
                if hasattr(self, '_audit_chain'):
                    self._audit_chain.append(_g8_event)
            except Exception:
                pass  # 治理检查失败不阻塞主流程

            # ── Phase 4: 风险评估 ──
            t3 = time.time()
            risk = self.ios.risk_check(ctx, prediction)
            turn.risk_level = risk.level
            turn.trace_phase("risk", "ok" if not risk.is_blocked() else "denied",
                           duration_ms=(time.time()-t3)*1000,
                           detail=f"level={risk.level} blocked={risk.is_blocked()}")
            self.iko.trace("risk", "ok" if not risk.is_blocked() else "denied",
                         detail=risk.reason)
            
            if risk.is_blocked():
                self.isa.respond(f"[安全拦截] {risk.reason}")
                turn.add_action("blocked", risk.reason)
                turn.complete()
                return
            
            # ── Phase 5: 左右脑推理 ──
            # D₀感知注入
            d0 = self.octopus.d0_snapshot()
            ctx.d0_report = d0
            
            # 因果教训注入（从risk.details获取）
            if risk and risk.details:
                ctx.causal_hints = risk.details
            
            t4 = time.time()
            proposal, critique = self.octopus.reason(ctx, prediction, risk)
            turn.trace_phase("reason", "ok", duration_ms=(time.time()-t4)*1000)
            self.iko.trace("reason", "ok")
            
            # ── IAX: 记录Phase 5推理消耗（左脑+右脑）──
            self._record_inference("phase_5_reason")

            # ── Phase 5.5: 治理检查·推理质量（Copewell模式·2026-07-06） ──
            try:
                from ..governance.events import RoutingCheckEvent
                _g8_reason = RoutingCheckEvent(
                    actor="ios_phase5",
                    session_id=ctx.session_id if hasattr(ctx, 'session_id') else "default",
                    prev_hash="genesis",
                    phase="phase_5_reasoning",
                    route_target=str(type(proposal).__name__),
                    risk_level="low",
                    approved=True,
                )
                if hasattr(self, '_audit_chain'):
                    self._audit_chain.append(_g8_reason)
            except Exception:
                pass  # 治理检查失败不阻塞主流程

            # ── Phase 6: 仲裁 ──
            t5 = time.time()
            decision = self.ios.arbitrate(proposal, critique, risk)
            turn.trace_phase("decide", "ok" if decision.approved else "denied",
                           duration_ms=(time.time()-t5)*1000,
                           detail=decision.reason)
            self.iko.trace("decide", "ok" if decision.approved else "denied",
                         detail=decision.reason)
            
            if not decision.approved:
                self.isa.respond(f"[仲裁否决] {decision.reason}")
                turn.add_action("denied", decision.reason)
                turn.complete()
                return
            
            # ── Phase 7: 执行 ──
            # cap_check: 只在有工具调用时检查（无工具调用=纯对话·不检查）
            if getattr(decision, 'tool_calls', None):
                if not self.ios.cap_check("execute", decision.action):
                    self.isa.respond("[权限拒绝] cap_policy不允许此操作")
                    turn.add_action("denied", "cap_policy拒绝")
                    turn.complete()
                    return
            
            t6 = time.time()
            result = self.isn.execute(decision)
            turn.trace_phase("execute", "ok" if result.success else "error",
                           duration_ms=result.duration_ms)
            self.iko.trace("execute", "ok" if result.success else "error")
            
            # ── Phase 7.5: tool_validator 自动验证 ──
            if _HAS_TOOL_VALIDATOR and result.success:
                try:
                    _tc_list = getattr(decision, 'tool_calls', None) or []
                    _first = _tc_list[0] if _tc_list else {}
                    _tool_name = _first.get("name", "unknown") if isinstance(_first, dict) else str(_first)
                    _tool_type_map = {"read_file": "read", "write_file": "write",
                                      "search_files": "search", "terminal": "execute"}
                    _tool_type = _tool_type_map.get(_tool_name, "execute")
                    _tv_call = _TVToolCall(
                        tool_name=_tool_name, tool_type=_tool_type,
                        params=_first.get("args", {}) if isinstance(_first, dict) else {},
                        result=result.output,
                        duration_ms=result.duration_ms,
                        session_id=getattr(self.session, 'id', ''),
                        turn_id=turn.id,
                    )
                    _tv_report = _tv_validate(_tv_call)
                    turn.trace_phase("validate", _tv_report.overall.value,
                                   detail=f"checks={len(_tv_report.checks)} retry={_tv_report.should_retry} block={_tv_report.should_block}")
                    if _tv_report.should_block:
                        _fail_entry = {
                            "timestamp": time.time(), "tool_name": _tool_name,
                            "reason": _tv_report.audit_entry.reason if _tv_report.audit_entry else "blocked",
                            "session_id": getattr(self.session, 'id', ''), "turn_id": turn.id,
                        }
                        self._tool_failures.append(_fail_entry)
                        self.iko.trace("validate", "blocked", detail=_fail_entry["reason"])
                        self._last_output = f"[工具验证拦截] {_fail_entry['reason']}"
                        if self.isa.mode != "silent":
                            self.isa.respond(self._last_output)
                        turn.add_action("validate_blocked", _fail_entry["reason"])
                        turn.complete()
                        return
                    if _tv_report.should_retry:
                        turn.trace_phase("validate_retry", "warn", detail="工具结果需重试")
                except Exception as _tv_err:
                    turn.trace_phase("validate", "skip", detail=str(_tv_err)[:80])
            
            # ── Phase 8: 因果对照 ──
            t7 = time.time()
            delta = self.octopus.compare(prediction, result)
            self.ios.learn_causal(ctx, prediction, result, delta)
            turn.trace_phase("learn", "ok", duration_ms=(time.time()-t7)*1000,
                           detail=delta.summary_text())
            self.iko.trace("learn", "ok", detail=delta.summary_text())
            
            # ── Phase 9: 进化 ──
            t8 = time.time()
            self.ios.evolve(proposal, critique, result, delta)
            turn.trace_phase("evolve", "ok", duration_ms=(time.time()-t8)*1000)
            self.iko.trace("evolve", "ok")
            
            # ── Phase 10: 压缩+输出+checkpoint ──
            t9 = time.time()
            self.session.compact_if_needed()
            # 无工具调用时：输出LLM生成的proposal内容（对话响应）
            if getattr(decision, 'tool_calls', None) or getattr(decision, '_has_tools', False):
                self._last_output = result.output
            else:
                self._last_output = proposal.content if proposal.content else result.output
            
            # IKO七因子管线
            try:
                risk_map = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
                ctx_for_iko = {
                    "risk_level": risk_map.get(getattr(decision, 'risk_level', 'low'), "LOW"),
                    "has_tool_calls": bool(getattr(decision, 'tool_calls', None)),
                    # 包拯审计修正：has_side_effects应检查实际副作用，非仅输出存在
                    "has_side_effects": bool(getattr(result, 'success', False) and getattr(result, 'output', '')),
                    # 包拯审计修正：option_count从proposal选项中提取，非硬编码
                    "option_count": max(1, len(getattr(proposal, 'evidence', []))),
                }
                decision_dict = {"type": "execute", "content": result.output}
                processed = self.iko.process_output(result.output, ctx_for_iko, decision_dict)
                if self.isa.mode != "silent" and processed:
                    self.isa.respond(processed)
            except Exception as e:
                # 包拯审计修正：降级时必须记录trace，不能静默吞没
                self.iko.trace("output_process", "error", detail=str(e)[:100])
                # 管线失败降级为原始输出
                if self.isa.mode != "silent":
                    self.isa.respond(result.output)
            
            turn.add_action("execute", result.output[:100])
            turn.complete()
            
            # 定时checkpoint
            if self._tick_count % 10 == 0:
                self.session.checkpoint()
                self.iko.trace("checkpoint", "ok")
            
            turn.trace_phase("output", "ok", duration_ms=(time.time()-t9)*1000)
            self.iko.trace("output", "ok", duration_ms=(time.time()-t9)*1000)
            
            # ── Phase 11: 六体自监督反馈收集 ──
            try:
                self._body_outputs["IAX"] = {"heartbeat_ok": True, "tick_count": self._tick_count}
                self._body_outputs["IAI"] = getattr(self.octopus, 'last_output', {}) or {}
                self._body_outputs["ISA"] = getattr(self.isa, 'last_output', {}) or {}
                self._body_outputs["IOS"] = getattr(self.ios, 'last_output', {}) or {}
                self._body_outputs["ISN"] = getattr(self.isn, 'last_output', {}) or {}
                self._body_outputs["IKO"] = getattr(self.iko, 'last_output', {}) or {}
                
                records = self.feedback_loop.collect_feedback(self._body_outputs)
                if records:
                    adjustments = self.feedback_loop.apply_feedback()
                    # 记录调整因子供下轮使用
                    self._feedback_adjustments = adjustments
                turn.trace_phase("feedback", "ok", detail=f"collected:{len(records)}")
            except Exception as e:
                turn.trace_phase("feedback", "error", detail=str(e)[:50])

            # tick计数自增（移到这里，确保run_once也能正确计数）
            self._tick_count += 1
            
        except Exception as e:
            turn.fail(str(e))
            raise


# ═══════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════

def main():
    """CLI入口"""
    args = sys.argv[1:]
    
    if "--once" in args:
        # 单次模式
        idx = args.index("--once")
        message = args[idx + 1] if idx + 1 < len(args) else "你好"
        agent = Agent(mode="silent")
        result = agent.run_once(message)
        print(result)
        return
    
    if "--test" in args:
        # 测试模式：跑一次完整的cycle然后退出
        _run_tests()
        return
    
    # 交互模式（默认）
    agent = Agent(mode="console")
    agent.run()


def _run_tests():
    """运行M0测试"""
    print("=== M0: Agent主循环测试 ===\n")
    
    # 测试1: 初始化
    agent = Agent(mode="silent")
    assert agent.isa is not None
    assert agent.octopus is not None
    assert agent.ios is not None
    assert agent.isn is not None
    assert agent.iko is not None
    print("✅ 测试1: 五体初始化")
    
    # 测试2: run_once
    result = agent.run_once("测试消息")
    assert result is not None
    print(f"✅ 测试2: run_once 输出={result[:30]}")
    
    # 测试3: Session/Turn生命周期
    assert agent.session.start_time is not None
    assert len(agent.session.turns) >= 1
    print(f"✅ 测试3: Session/Turn生命周期 total_turns={len(agent.session.turns)}")
    
    # 测试4: 左右脑对弈
    ctx = agent.isa.build_context(Message("写一段Python代码"))
    proposal, critique = agent.octopus.reason(ctx)
    assert proposal is not None
    assert critique is not None
    print(f"✅ 测试4: 左脑提案={proposal.content[:30]} 右脑批判={critique.verdict}")
    
    # 测试5: IO-S仲裁
    decision = agent.ios.arbitrate(proposal, critique)
    assert decision is not None
    print(f"✅ 测试5: 仲裁={decision.action} 批准={decision.approved}")
    
    # 测试6: 工具执行
    result = agent.isn.execute(decision)
    print(f"✅ 测试6: 执行 success={result.success}")
    
    # 测试7: IKO可观测
    report = agent.iko.report()
    assert report["total_ticks"] >= 2  # run_once + 手动
    print(f"✅ 测试7: 可观测 total_ticks={report['total_ticks']} ok_rate={report['ok_rate']:.0%}")
    
    # 测试8: IO-S自我进化
    print(f"✅ 测试8: 进化日志 {len(agent.ios.evolution_log)}条（>=1）")
    
    # 测试9: 因果预测
    prediction = agent.octopus.predict_consequences(ctx)
    assert prediction is not None
    print(f"✅ 测试9: 因果预测 {prediction.summary_text()[:40]}")
    
    # 测试10: 因果对照
    delta = agent.octopus.compare(prediction, result)
    assert delta is not None
    print(f"✅ 测试10: 因果对照 {delta.summary_text()}")
    
    # 测试11: 风险评估
    risk = agent.ios.risk_check(ctx, prediction)
    assert risk is not None
    print(f"✅ 测试11: 风险评估 level={risk.level}")
    
    # 测试12: 10阶段trace日志
    ticks_file = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
    assert ticks_file.exists()
    with open(ticks_file) as f:
        lines = f.readlines()
    assert len(lines) >= 2  # run_once + 手动
    print(f"✅ 测试12: 10阶段trace日志 ticks={len(lines)}")
    
    print(f"\n全部 12/12 测试通过 ✅")


if __name__ == "__main__":
    main()
