"""
openLLM Agent心跳执行 — 从main_loop.py提取的_execute_tick

10阶段心跳：listen→context→replay→predict→governance→risk→reason→arbitrate→execute→feedback
"""
import time
from typing import Optional
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult)
from .protocol import HeartbeatContext


def execute_tick(agent, msg: Message):
    """一次完整的10阶段心跳（从Agent._execute_tick提取）"""
    t0 = time.time()
    turn = agent.session.new_turn()
    
    try:
        # Phase 1: listen
        turn.trace_phase("listen", "ok")
        agent.iko.trace("listen", "ok")
        
        # Phase 2: build_context
        t1 = time.time()
        ctx = agent.isa.build_context(msg, agent.session, agent.octopus, agent.ios)
        turn.trace_phase("context", "ok", duration_ms=(time.time()-t1)*1000)
        agent.iko.trace("context", "ok")
        
        # Phase 2.5: evidence replay
        try:
            from ..memory.evidence_replay import create_replay_for_context
            replay_text = create_replay_for_context(msg.text, ctx, top_k=5, max_tokens=512)
            if replay_text:
                ctx.search_results = getattr(ctx, 'search_results', []) or []
                ctx.search_results.append(replay_text)
                turn.trace_phase("replay", "ok", detail=f"evidence_replay:{len(replay_text)}chars")
                agent.iko.trace("replay", "ok")
        except Exception as e:
            turn.trace_phase("replay", "skip", detail=str(e)[:100])
        
        # Phase 3: predict
        t2 = time.time()
        prediction = agent.octopus.predict_consequences(ctx)
        turn.trace_phase("predict", "ok", duration_ms=(time.time()-t2)*1000,
                       detail=prediction.summary_text())
        agent.iko.trace("predict", "ok", detail=prediction.summary_text())
        agent._record_inference("phase_3_predict")
        
        # Phase 3.5: governance check
        try:
            from ..governance.events import RoutingCheckEvent
            _g8 = RoutingCheckEvent(
                actor="ios_phase3",
                session_id=getattr(ctx, 'session_id', 'default'),
                prev_hash="genesis", phase="phase_3_prediction",
                route_target=str(type(prediction).__name__),
                risk_level="low", approved=True,
            )
            if hasattr(agent, '_audit_chain'):
                agent._audit_chain.append(_g8)
        except Exception:
            pass
        
        # Phase 4: risk check
        t3 = time.time()
        risk = agent.ios.risk_check(ctx, prediction)
        turn.risk_level = risk.level
        turn.trace_phase("risk", "ok" if not risk.is_blocked() else "denied",
                       duration_ms=(time.time()-t3)*1000,
                       detail=f"level={risk.level} blocked={risk.is_blocked()}")
        agent.iko.trace("risk", "ok" if not risk.is_blocked() else "denied", detail=risk.reason)
        
        if risk.is_blocked():
            agent.isa.respond(f"[安全拦截] {risk.reason}")
            turn.add_action("blocked", risk.reason)
            turn.complete()
            return
        
        # Phase 5: reason
        d0 = agent.octopus.d0_snapshot()
        ctx.d0_report = d0
        if risk and risk.details:
            ctx.causal_hints = risk.details
        
        t4 = time.time()
        proposal, critique = agent.octopus.reason(ctx, prediction, risk)
        turn.trace_phase("reason", "ok", duration_ms=(time.time()-t4)*1000)
        agent.iko.trace("reason", "ok")
        agent._record_inference("phase_5_reason")
        
        # Phase 5.5: governance check
        try:
            from ..governance.events import RoutingCheckEvent
            _g8r = RoutingCheckEvent(
                actor="ios_phase5",
                session_id=getattr(ctx, 'session_id', 'default'),
                prev_hash="genesis", phase="phase_5_reasoning",
                route_target=str(type(proposal).__name__),
                risk_level="low", approved=True,
            )
            if hasattr(agent, '_audit_chain'):
                agent._audit_chain.append(_g8r)
        except Exception:
            pass
        
        # Phase 6: arbitrate
        t5 = time.time()
        decision = agent.ios.arbitrate(proposal, critique, risk)
        turn.trace_phase("decide", "ok" if decision.approved else "denied",
                       duration_ms=(time.time()-t5)*1000, detail=decision.reason)
        agent.iko.trace("decide", "ok" if decision.approved else "denied", detail=decision.reason)
        
        if not decision.approved:
            agent.isa.respond(f"[仲裁否决] {decision.reason}")
            turn.add_action("denied", decision.reason)
            turn.complete()
            return
        
        # Phase 7: execute
        if getattr(decision, 'tool_calls', None):
            if not agent.ios.cap_check("execute", decision.action):
                agent.isa.respond("[权限拒绝] cap_policy不允许此操作")
                turn.add_action("denied", "cap_policy拒绝")
                turn.complete()
                return
        
        t6 = time.time()
        result = agent.isn.execute(decision)
        turn.trace_phase("execute", "ok" if result.success else "error",
                       duration_ms=result.duration_ms)
        agent.iko.trace("execute", "ok" if result.success else "error")
        
        # Phase 7.5: tool_validator
        _handle_tool_validation(agent, decision, result, turn)
        
        # Phase 8: causal compare
        t7 = time.time()
        delta = agent.octopus.compare(prediction, result)
        agent.ios.learn_causal(ctx, prediction, result, delta)
        turn.trace_phase("learn", "ok", duration_ms=(time.time()-t7)*1000,
                       detail=delta.summary_text())
        agent.iko.trace("learn", "ok", detail=delta.summary_text())
        
        # Phase 9: evolve
        t8 = time.time()
        agent.ios.evolve(proposal, critique, result, delta)
        turn.trace_phase("evolve", "ok", duration_ms=(time.time()-t8)*1000)
        agent.iko.trace("evolve", "ok")
        
        # Phase 10: output
        t9 = time.time()
        agent.session.compact_if_needed()
        if getattr(decision, 'tool_calls', None) or getattr(decision, '_has_tools', False):
            agent._last_output = result.output
        else:
            agent._last_output = proposal.content if proposal.content else result.output
        
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
        
        # Phase 11: feedback
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
    
    except Exception as e:
        turn.fail(str(e))
        raise


def _handle_tool_validation(agent, decision, result, turn):
    """Phase 7.5: tool_validator自动验证"""
    from .main_loop import _HAS_TOOL_VALIDATOR, _TVToolCall, _tv_validate
    if not _HAS_TOOL_VALIDATOR or not result.success:
        return
    
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
            result=result.output, duration_ms=result.duration_ms,
            session_id=getattr(agent.session, 'id', ''), turn_id=turn.id,
        )
        _tv_report = _tv_validate(_tv_call)
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
