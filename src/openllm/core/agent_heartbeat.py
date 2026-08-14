"""
openLLM Agent心跳执行 — 5阶段精简心跳

PERCEIVE → DECIDE → EXECUTE → LEARN → FEEDBACK

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
    """5阶段心跳：PERCEIVE→DECIDE→EXECUTE→LEARN→FEEDBACK

    从12阶段精简而来：
    - listen/governance×2 删除（空壳）
    - search+replay+predict 合并为 PERCEIVE
    - risk+reason+arbitrate 合并为 DECIDE
    - execute+tool_validator 合并为 EXECUTE
    - causal_compare+evolve+output 合并为 LEARN
    - feedback+research 合并为 FEEDBACK
    """
    t0 = time.time()
    agent._tick_start_time = t0
    turn = agent.session.new_turn()

    hc = HeartbeatContext(
        user_message=msg.text,
        tick_id=f"tick_{agent._tick_count}",
    )

    try:
        # ═══ Phase 1: PERCEIVE（感知）═══
        _perceive(agent, msg, hc, turn)

        # ═══ Phase 2: DECIDE（决策）═══
        blocked = _decide(agent, msg, hc, turn)
        if blocked:
            turn.complete()
            return hc

        # ═══ Phase 3: EXECUTE（执行）═══
        _execute(agent, hc, turn)

        # ═══ Phase 4: LEARN（学习）═══
        _learn(agent, hc, turn)

        # ═══ Phase 5: FEEDBACK（反馈）═══
        _feedback(agent, hc, turn)

        agent._tick_count += 1
        return hc

    except Exception as e:
        turn.fail(str(e))
        raise


# ── Phase 1: PERCEIVE ──────────────────────────────────────

def _perceive(agent, msg, hc, turn):
    """感知：构建上下文 + 搜索 + 证据回放 + 预测"""
    # 1.1 记忆上下文
    t1 = time.time()
    ctx = agent.isa.build_context(msg, agent.session, agent.octopus, agent.ios)
    hc.memory = getattr(ctx, 'memory', {}) or {}
    hc.identity = getattr(ctx, 'identity', {}) or {}
    turn.trace_phase("context", "ok", duration_ms=(time.time()-t1)*1000)
    agent.iko.trace("context", "ok")

    # 1.2 章鱼索引搜索
    _do_search(agent, msg, ctx, hc, turn)

    # 1.3 证据回放
    _do_replay(agent, msg, ctx, hc, turn)

    # 1.4 因果预测
    t2 = time.time()
    prediction = agent.octopus.predict_consequences(ctx)
    hc.prediction = prediction
    turn.trace_phase("predict", "ok", duration_ms=(time.time()-t2)*1000,
                     detail=prediction.summary_text())
    agent.iko.trace("predict", "ok", detail=prediction.summary_text())
    agent._record_inference("perceive_predict")
    
    # 1.5 上下文漂移检测（接入v1·2026-07-30）
    try:
        if hasattr(agent, '_drift_detector') and hasattr(agent, '_last_message'):
            drift = agent._drift_detector.compute_drift(agent._last_message, msg)
            if drift >= 0.7:
                turn.trace_phase("drift", "warn", detail=f"漂移度={drift:.2f}")
                if agent.mode != "silent":
                    print(f"  ⚠️ 上下文漂移检测: drift={drift:.2f}（阈值0.7）")
        agent._last_message = msg
    except Exception:
        pass  # 漂移检测失败不阻塞主循环


def _do_search(agent, msg, ctx, hc, turn):
    """章鱼索引搜索"""
    try:
        index_brain = agent.octopus.tentacles.get("index")
        if index_brain:
            if not index_brain.index:
                index_brain.build()
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


def _do_replay(agent, msg, ctx, hc, turn):
    """证据回放"""
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


# ── Phase 2: DECIDE ────────────────────────────────────────

def _decide(agent, msg, hc, turn):
    """决策：风险检查 + 方案生成 + 仲裁。返回True表示被拦截。"""
    ctx = agent.isa.build_context(msg, agent.session, agent.octopus, agent.ios)
    prediction = hc.prediction

    # 2.1 风险检查
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
        return True

    # 2.2 左右脑对弈
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
    agent._record_inference("decide_reason")

    # 2.3 仲裁
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
        return True

    return False


# ── Phase 3: EXECUTE ───────────────────────────────────────

def _execute(agent, hc, turn):
    """执行：工具调用 + 验证"""
    decision = hc.decision

    # 3.1 权限检查
    if getattr(decision, 'tool_calls', None):
        if not agent.ios.cap_check("execute", decision.action):
            agent.isa.respond("[权限拒绝] cap_policy不允许此操作")
            hc.rejection_record = {"reason": "cap_policy拒绝", "phase": "execute"}
            turn.add_action("denied", "cap_policy拒绝")
            turn.complete()
            return

    # 3.2 执行
    t6 = time.time()
    result = agent.isn.execute(decision)
    hc.result = result
    turn.trace_phase("execute", "ok" if result.success else "error",
                     duration_ms=result.duration_ms)
    agent.iko.trace("execute", "ok" if result.success else "error")

    # 3.3 工具验证
    _handle_tool_validation(agent, decision, result, turn)


def _handle_tool_validation(agent, decision, result, turn):
    """工具结果验证"""
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


# ── Phase 4: LEARN ─────────────────────────────────────────

def _learn(agent, hc, turn):
    """学习：因果比较 + 进化 + 输出处理"""
    ctx = agent.isa.build_context(
        Message(text=hc.user_message), agent.session, agent.octopus, agent.ios)
    prediction = hc.prediction
    result = hc.result
    decision = hc.decision
    proposal = hc.left_proposal

    # 4.1 因果比较
    t7 = time.time()
    delta = agent.octopus.compare(prediction, result)
    agent.ios.learn_causal(ctx, prediction, result, delta)
    turn.trace_phase("learn", "ok", duration_ms=(time.time()-t7)*1000,
                     detail=delta.summary_text())
    agent.iko.trace("learn", "ok", detail=delta.summary_text())

    # 4.2 进化
    t8 = time.time()
    agent.ios.evolve(proposal, hc.right_critique, result, delta)
    turn.trace_phase("evolve", "ok", duration_ms=(time.time()-t8)*1000)
    agent.iko.trace("evolve", "ok")

    # 4.3 输出处理
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
            "risk_level": risk_map.get(getattr(hc.risk, 'level', 'low'), "LOW") if hc.risk else "LOW",
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


# ── Phase 5: FEEDBACK ──────────────────────────────────────

def _feedback(agent, hc, turn):
    """反馈：研究循环注入 + 六体自监督"""
    # 5.1 研究循环
    try:
        research_ctx = agent.research.to_heartbeat_context()
        if research_ctx.get("active_hypothesis"):
            hc.research = research_ctx
            agent.iko.trace("research", "ok",
                detail=f"phase={research_ctx['current_phase']} hyp={research_ctx['active_hypothesis']}")
    except Exception as _r_err:
        agent.iko.trace("research", "skip", detail=str(_r_err)[:60])

    # 5.2 六体自监督反馈
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
    
    # 5.3 过程透明化摘要
    _emit_summary(agent, hc, turn)



def _emit_summary(agent, hc, turn):
    """过程透明化：每次心跳后输出结构化摘要"""
    t0 = getattr(agent, "_tick_start_time", time.time())
    duration = time.time() - t0
    tick = agent._tick_count
    
    # 收集各阶段信息
    perceive_info = "ok"
    if hc.search_results:
        perceive_info = f"{len(hc.search_results)}条结果"
    
    decide_info = "ok"
    if hc.risk:
        decide_info = f"risk={hc.risk.level}"
    if hc.decision:
        decide_info += f" approved={hc.decision.approved}"
    
    execute_info = "ok"
    if hc.result:
        execute_info = f"{'success' if hc.result.success else 'error'}"
    
    learn_info = "ok"
    if hc.output:
        learn_info = f"output={len(str(hc.output))}chars"
    
    # 输出摘要（一行）
    summary = f"[tick #{tick}] {duration:.1f}s | PERCEIVE:{perceive_info} | DECIDE:{decide_info} | EXECUTE:{execute_info} | LEARN:{learn_info}"
    
    # 只在非静默模式下输出
    if agent.mode != "silent":
        print(summary)
    
    # 记录到IKO
    agent.iko.trace("summary", "ok", detail=summary)
    
    # 引用率检查（接入v1·2026-07-30）
    try:
        if hc.output and hasattr(agent, '_token_economy'):
            from .citation_checker import check_citations
            report = check_citations(str(hc.output))
            if report.total_claims > 0 and report.citation_rate < 0.5:
                agent.iko.trace("citation", "warn",
                    detail=f"引用率低: {report.citation_rate:.0%} ({report.cited_claims}/{report.total_claims})")
                if agent.mode != "silent":
                    print(f"  📎 引用率: {report.citation_rate:.0%} ({report.cited_claims}/{report.total_claims}条有来源)")
    except Exception:
        pass  # 引用检查失败不阻塞主循环
