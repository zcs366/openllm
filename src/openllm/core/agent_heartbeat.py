"""
openLLM Agent心跳执行 — 从main_loop.py提取的_execute_tick

11阶段心跳：listen→context→replay→predict→governance→risk→reason→arbitrate→execute→feedback

数据契约：HeartbeatContext（protocol.py）
- 感知层写：identity, memory, search_results, prediction, left_proposal, right_critique
- 决策层写：risk, decision, rejection_record
- 执行层写：result, output, metrics
"""
import time
from typing import Optional
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult)
from .protocol import HeartbeatContext


def execute_tick(agent, msg: Message):
    """一次完整的11阶段心跳（从Agent._execute_tick提取）
    
    数据流通过HeartbeatContext：
    Phase 1-2:  感知层写 → hc.identity, hc.memory, hc.search_results
    Phase 2.5:  感知层写 → hc.search_results (evidence replay)
    Phase 3:    感知层写 → hc.prediction
    Phase 4:    决策层写 → hc.risk
    Phase 5:    感知层写 → hc.left_proposal, hc.right_critique
    Phase 6:    决策层写 → hc.decision
    Phase 7:    执行层写 → hc.result
    Phase 8-9:  决策层学习（因果+进化）
    Phase 10:   执行层写 → hc.output
    Phase 11:   反馈环路
    """
    t0 = time.time()
    turn = agent.session.new_turn()
    
    # ═══ HeartbeatContext: 6体数据契约 ═══
    hc = HeartbeatContext(
        user_message=msg.text,
        tick_id=f"tick_{agent._tick_count}",
    )
    
    try:
        # ── Phase 1: listen ──
        turn.trace_phase("listen", "ok")
        agent.iko.trace("listen", "ok")
        
        # ── Phase 2: build_context (感知层) ──
        t1 = time.time()
        ctx = agent.isa.build_context(msg, agent.session, agent.octopus, agent.ios)
        hc.memory = getattr(ctx, 'memory', {}) or {}
        hc.identity = getattr(ctx, 'identity', {}) or {}
        turn.trace_phase("context", "ok", duration_ms=(time.time()-t1)*1000)
        agent.iko.trace("context", "ok")
        
        # ── Phase 2.25: search (感知层·章鱼搜索) ──
        try:
            index_brain = agent.octopus.tentacles.get("index")
            if index_brain:
                # 构建索引（首次调用时）
                if not index_brain.index:
                    index_brain.build()
                # 搜索
                search_results = index_brain.search(msg.text, limit=5)
                if search_results:
                    ctx.search_results = getattr(ctx, 'search_results', []) or []
                    for r in search_results:
                        ctx.search_results.append(
                            f"[search:{r.get('filepath','?')}] {r.get('snippet','')}")
                    hc.search_results = ctx.search_results
                    turn.trace_phase("search", "ok",
                                     detail=f"found:{len(search_results)}")
                    agent.iko.trace("search", "ok")
        except Exception as e:
            turn.trace_phase("search", "skip", detail=str(e)[:100])

        # ── Phase 2.5: evidence replay (感知层) ──
        try:
            from ..memory.evidence_replay import create_replay_for_context
            replay_text = create_replay_for_context(msg.text, ctx, top_k=5, max_tokens=512)
            if replay_text:
                ctx.search_results = getattr(ctx, 'search_results', []) or []
                ctx.search_results.append(replay_text)
                hc.search_results = ctx.search_results
                turn.trace_phase("replay", "ok", detail=f"evidence_replay:{len(replay_text)}chars")
                agent.iko.trace("replay", "ok")
        except Exception as e:
            turn.trace_phase("replay", "skip", detail=str(e)[:100])
        
        # ── Phase 3: predict (感知层) ──
        t2 = time.time()
        prediction = agent.octopus.predict_consequences(ctx)
        hc.prediction = prediction
        turn.trace_phase("predict", "ok", duration_ms=(time.time()-t2)*1000,
                       detail=prediction.summary_text())
        agent.iko.trace("predict", "ok", detail=prediction.summary_text())
        agent._record_inference("phase_3_predict")
        
        # ── Phase 3.5: governance check (决策层·预审) ──
        _governance_check(agent, hc, ctx, "phase_3_prediction", prediction)
        
        # ── Phase 4: risk check (决策层) ──
        t3 = time.time()
        risk = agent.ios.risk_check(ctx, prediction)
        hc.risk = risk
        turn.risk_level = risk.level
        turn.trace_phase("risk", "ok" if not risk.is_blocked() else "denied",
                       duration_ms=(time.time()-t3)*1000,
                       detail=f"level={risk.level} blocked={risk.is_blocked()}")
        agent.iko.trace("risk", "ok" if not risk.is_blocked() else "denied", detail=risk.reason)
        
        if risk.is_blocked():
            agent.isa.respond(f"[安全拦截] {risk.reason}")
            hc.rejection_record = {"reason": risk.reason, "phase": "risk"}
            turn.add_action("blocked", risk.reason)
            turn.complete()
            return hc
        
        # ── Phase 5: reason (感知层·左右脑对弈) ──
        d0 = agent.octopus.d0_snapshot()
        ctx.d0_report = d0
        if risk and risk.details:
            ctx.causal_hints = risk.details
        
        t4 = time.time()
        proposal, critique = agent.octopus.reason(ctx, prediction, risk)
        hc.left_proposal = proposal
        hc.right_critique = critique
        turn.trace_phase("reason", "ok", duration_ms=(time.time()-t4)*1000)
        agent.iko.trace("reason", "ok")
        agent._record_inference("phase_5_reason")
        
        # ── Phase 5.5: governance check (决策层·预审) ──
        _governance_check(agent, hc, ctx, "phase_5_reasoning", proposal)
        
        # ── Phase 6: arbitrate (决策层) ──
        t5 = time.time()
        decision = agent.ios.arbitrate(proposal, critique, risk)
        hc.decision = decision
        turn.trace_phase("decide", "ok" if decision.approved else "denied",
                       duration_ms=(time.time()-t5)*1000, detail=decision.reason)
        agent.iko.trace("decide", "ok" if decision.approved else "denied", detail=decision.reason)
        
        if not decision.approved:
            agent.isa.respond(f"[仲裁否决] {decision.reason}")
            hc.rejection_record = {"reason": decision.reason, "phase": "arbitrate"}
            turn.add_action("denied", decision.reason)
            turn.complete()
            return hc
        
        # ── Phase 7: execute (执行层) ──
        if getattr(decision, 'tool_calls', None):
            if not agent.ios.cap_check("execute", decision.action):
                agent.isa.respond("[权限拒绝] cap_policy不允许此操作")
                hc.rejection_record = {"reason": "cap_policy拒绝", "phase": "execute"}
                turn.add_action("denied", "cap_policy拒绝")
                turn.complete()
                return hc
        
        t6 = time.time()
        result = agent.isn.execute(decision)
        hc.result = result
        turn.trace_phase("execute", "ok" if result.success else "error",
                       duration_ms=result.duration_ms)
        agent.iko.trace("execute", "ok" if result.success else "error")
        
        # ── Phase 7.5: tool_validator ──
        _handle_tool_validation(agent, decision, result, turn)
        
        # ── Phase 8: causal compare (决策层·学习) ──
        t7 = time.time()
        delta = agent.octopus.compare(prediction, result)
        agent.ios.learn_causal(ctx, prediction, result, delta)
        turn.trace_phase("learn", "ok", duration_ms=(time.time()-t7)*1000,
                       detail=delta.summary_text())
        agent.iko.trace("learn", "ok", detail=delta.summary_text())
        
        # ── Phase 9: evolve (决策层·进化) ──
        t8 = time.time()
        agent.ios.evolve(proposal, critique, result, delta)
        turn.trace_phase("evolve", "ok", duration_ms=(time.time()-t8)*1000)
        agent.iko.trace("evolve", "ok")
        
        # ── Phase 10: output (执行层) ──
        t9 = time.time()
        agent.session.compact_if_needed()
        if getattr(decision, 'tool_calls', None) or getattr(decision, '_has_tools', False):
            agent._last_output = result.output
        else:
            agent._last_output = proposal.content if proposal.content else result.output
        hc.output = agent._last_output
        
        # IKO pipeline
        try:
            risk_map = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
            ctx_for_iko = {
                "risk_level": risk_map.get(getattr(decision, 'risk_level', 'low'), "LOW"),
                "has_tool_calls": bool(getattr(decision, 'tool_calls', None)),
                "has_side_effects": bool(getattr(result, 'success', False) and getattr(result, 'output', '')),
                "option_count": max(1, len(getattr(proposal, 'evidence', []))),
            }
            decision_dict = {"type": "execute", "content": result.output}
            processed = agent.iko.process_output(result.output, ctx_for_iko, decision_dict)
            if agent.isa.mode != "silent" and processed:
                agent.isa.respond(processed)
        except Exception as e:
            agent.iko.trace("output_process", "error", detail=str(e)[:100])
            if agent.isa.mode != "silent":
                agent.isa.respond(result.output)
        
        turn.add_action("execute", result.output[:100])
        turn.complete()
        
        if agent._tick_count % 10 == 0:
            agent.session.checkpoint()
            agent.iko.trace("checkpoint", "ok")
        
        turn.trace_phase("output", "ok", duration_ms=(time.time()-t9)*1000)
        agent.iko.trace("output", "ok", duration_ms=(time.time()-t9)*1000)
        
        # ── Phase 11: feedback (反馈环路) ──
        try:
            agent._body_outputs["IAX"] = {"heartbeat_ok": True, "tick_count": agent._tick_count}
            agent._body_outputs["IAI"] = getattr(agent.octopus, 'last_output', {}) or {}
            agent._body_outputs["ISA"] = getattr(agent.isa, 'last_output', {}) or {}
            agent._body_outputs["IOS"] = getattr(agent.ios, 'last_output', {}) or {}
            agent._body_outputs["ISN"] = getattr(agent.isn, 'last_output', {}) or {}
            agent._body_outputs["IKO"] = getattr(agent.iko, 'last_output', {}) or {}
            records = agent.feedback_loop.collect_feedback(agent._body_outputs)
            if records:
                agent._feedback_adjustments = agent.feedback_loop.apply_feedback()
            turn.trace_phase("feedback", "ok", detail=f"collected:{len(records)}")
        except Exception as e:
            turn.trace_phase("feedback", "error", detail=str(e)[:50])
        
        agent._tick_count += 1
        return hc
    
    except Exception as e:
        turn.fail(str(e))
        raise


def _governance_check(agent, hc: HeartbeatContext, ctx, phase: str, payload):
    """Phase 3.5 / 5.5: governance routing check (写入hc.phase_log)"""
    try:
        from ..governance.events import RoutingCheckEvent
        event = RoutingCheckEvent(
            actor=f"ios_{phase}",
            session_id=getattr(ctx, 'session_id', 'default'),
            prev_hash="genesis",
            phase=phase,
            route_target=str(type(payload).__name__),
            risk_level="low",
            approved=True,
        )
        if hasattr(agent, '_audit_chain'):
            agent._audit_chain.append(event)
        hc.log_phase(phase, "governance_ok")
    except Exception:
        pass


def _handle_tool_validation(agent, decision, result, turn):
    """Phase 7.5: tool_validator自动验证"""
    from .tool_validator_types import ToolCall, validate_tool_result
    if not result.success:
        return
    
    try:
        _tc_list = getattr(decision, 'tool_calls', None) or []
        _first = _tc_list[0] if _tc_list else {}
        _tool_name = _first.get("name", "unknown") if isinstance(_first, dict) else str(_first)
        _tool_type_map = {"read_file": "read", "write_file": "write",
                          "search_files": "search", "terminal": "execute"}
        _tool_type = _tool_type_map.get(_tool_name, "execute")
        _tv_call = ToolCall(
            tool_name=_tool_name, tool_type=_tool_type,
            params=_first.get("args", {}) if isinstance(_first, dict) else {},
            result=result.output, duration_ms=result.duration_ms,
            session_id=getattr(agent.session, 'id', ''), turn_id=turn.id,
        )
        _tv_report = validate_tool_result(_tv_call)
        turn.trace_phase("validate", _tv_report.overall.value,
                       detail=f"checks={len(_tv_report.checks)} retry={_tv_report.should_retry} block={_tv_report.should_block}")
        if _tv_report.should_block:
            _fail_entry = {
                "timestamp": time.time(), "tool_name": _tool_name,
                "reason": _tv_report.audit_entry.reason if _tv_report.audit_entry else "blocked",
                "session_id": getattr(agent.session, 'id', ''), "turn_id": turn.id,
            }
            agent._tool_failures.append(_fail_entry)
            agent.iko.trace("validate", "blocked", detail=_fail_entry["reason"])
            agent._last_output = f"[工具验证拦截] {_fail_entry['reason']}"
            if agent.isa.mode != "silent":
                agent.isa.respond(agent._last_output)
            turn.add_action("validate_blocked", _fail_entry["reason"])
            turn.complete()
            return
        if _tv_report.should_retry:
            turn.trace_phase("validate_retry", "warn", detail="工具结果需重试")
    except Exception as _tv_err:
        turn.trace_phase("validate", "skip", detail=str(_tv_err)[:80])
