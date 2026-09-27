"""
openLLM Agent心跳执行 — 5阶段精简心跳

PERCEIVE → DECIDE → EXECUTE → LEARN → FEEDBACK

数据契约：HeartbeatContext（protocol.py）
- 感知层写：identity, memory, search_results, prediction, left_proposal, right_critique
- 决策层写：risk, decision, rejection_record
- 执行层写：result, output, metrics
"""
import time
from pathlib import Path
from typing import Optional
import os
from openllm.core.models import (Message, Context, Prediction, RiskAssessment,
                                  Proposal, Critique, Decision, ActionResult)
from openllm.core.protocol import HeartbeatContext
from openllm.core.session import TurnStatus
from .awakening import AwakeningProtocol

def execute_tick(agent, msg: Message):
    """5阶段心跳：PERCEIVE→DECIDE→EXECUTE→LEARN→FEEDBACK

    从12阶段精简而来：
    - listen 删除（空壳）；governance×2 已恢复为治理trace写入（G-1·2026-09-01）
    - search+replay+predict 合并为 PERCEIVE
    - risk+reason+arbitrate 合并为 DECIDE
    - execute+tool_validator 合并为 EXECUTE
    - causal_compare+evolve+output 合并为 LEARN
    - feedback+research 合并为 FEEDBACK
    """
    t0 = time.time()
    agent._tick_start_time = t0
    turn = agent.session.new_turn()

    # 连续性门控：记录主任务延续状态（十律⑥·蜻蜓门控）
    _current_task = getattr(msg, "text", "") or ""
    _prev_task = getattr(agent, "_gate_current_task", None)
    if _prev_task and _prev_task != _current_task:
        # 主任务切换——发布 blocked=True（保持旧任务轨迹）
        iai = getattr(agent, "iai", None)
        if iai is not None:
            iai.gate("task_switch", _prev_task, blocked=True)
    agent._gate_current_task = _current_task

    hc = HeartbeatContext(
        user_message=msg.text,
        tick_id=f"tick_{agent._tick_count}",
    )

    # 苏醒协议：初始化（首次调用时创建）
    if not hasattr(agent, "_awakening_protocol"):
        agent._awakening_protocol = AwakeningProtocol(agent)

    # E2 时钟：每拍写一行（空拍也写——记录流逝才叫钟）
    try:
        if agent.clock:
            agent.clock.tick("beat")
    except Exception:
        pass

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

        # ═══ Phase 3.5: SYNTHESIZE（综合消化）═══
        _synthesize(agent, msg, hc, turn)

        # ═══ Phase 4: LEARN（学习）═══
        _learn(agent, hc, turn)

        # ═══ Phase 5: FEEDBACK（反馈）═══
        _feedback(agent, hc, turn)

        # 连续性门控：主任务完成→允许切换（blocked=False）
        iai = getattr(agent, "iai", None)
        if iai is not None:
            iai.gate("task_switch", _current_task, blocked=False)

        agent._tick_count += 1
        return hc

    except Exception as e:
        # DR-20260917-01：fail/complete都有状态机守卫，turn已终结时不得二次改状态，
        # 否则此处异常会顶掉真实异常（traceback被吞、CLI直接退出）。
        if getattr(turn, "status", None) == TurnStatus.ACTIVE.value:
            turn.fail(str(e))
        else:
            agent.iko.trace("tick", "error_after_complete", detail=str(e)[:100])
        raise


# ── Phase 1: PERCEIVE ──────────────────────────────────────

def _perceive(agent, msg, hc, turn):
    """感知：构建上下文 + 搜索 + 证据回放 + 预测"""
    # 1.1 记忆上下文
    t1 = time.time()
    ctx = agent.isa.build_context(msg, agent.session, agent.octopus, agent.ios)
    # DR-20260917-04：注入本会话对话历史（修「刚说的话就忘」）。
    # _pending_history 由 run_once/_execute_tick 逐轮置入；空=首轮。
    try:
        ctx.history = list(getattr(agent, "_pending_history", None) or [])
    except Exception:
        pass
    hc.memory = getattr(ctx, 'memory', {}) or {}
    hc.identity = getattr(ctx, 'identity', {}) or {}
    turn.trace_phase("context", "ok", duration_ms=(time.time()-t1)*1000)
    agent.iko.trace("context", "ok")

    # ── 苏醒协议：新session首轮注入苏醒词 ──
    try:
        if agent._awakening_protocol.inject_to_context(ctx, agent.session):
            turn.trace_phase("awakening_inject", "ok")
            agent.iko.trace("awakening_inject", "ok")
    except Exception as _aw_err:
        turn.trace_phase("awakening_inject", "skip", detail=str(_aw_err)[:100])

    # 1.1b 因果疤注入（E2 缺口②·2026-08-23）
    try:
        if agent.isa.causal:
            block = agent.isa.causal.to_context_block(max_entries=5)
            if block and block.strip():
                ctx.causal_block = block
                hc.causal_block = block
                # 同时并入 search_results（兼容旧路径）
                ctx.search_results = getattr(ctx, 'search_results', []) or []
                ctx.search_results.append(block)
                turn.trace_phase("causal_inject", "ok", detail=f"{len(block)}chars")
                agent.iko.trace("causal_inject", "ok")
    except Exception as e:
        turn.trace_phase("causal_inject", "skip", detail=str(e)[:100])

    # 1.1c 苏醒读钟（E2 时钟·2026-08-23）
    try:
        if agent.clock:
            _clk = agent.clock.now_status()
            _clk_text = (
                f"⏰ 时钟: 上次走针{_clk['last_wall_time']:.0f}, "
                f"现在{time.time():.0f}, 间隔{_clk['gap_since_last']:.0f}秒, "
                f"第{_clk['epoch']}拍, 苏醒{_clk['awakening_count']}次"
            )
            ctx.search_results = getattr(ctx, 'search_results', []) or []
            ctx.search_results.append(_clk_text)
            turn.trace_phase("clock_read", "ok", detail=f"epoch={_clk['epoch']}")
            agent.iko.trace("clock_read", "ok")
    except Exception:
        pass

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
        from ..isa.evidence_replay import create_replay_for_context
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

def _emit(agent, event_type: str, payload=None):
    """IAI事件发布（失败不阻塞主流程）。IAI融合·2026-09-02。

    推理链路事件源：context.built / reasoning.proposed / reasoning.critiqued
    / decision.made / action.executed —— 自进化管道（ILM训练三元组）的原料。
    """
    iai = getattr(agent, "iai", None)
    if iai is None:
        return
    iai.emit(event_type, payload)


def _decide(agent, msg, hc, turn):
    """决策：风险检查 + 方案生成 + 仲裁。返回True表示被拦截。"""
    ctx = agent.isa.build_context(msg, agent.session, agent.octopus, agent.ios)
    _emit(agent, "context.built", {
        "user_message": (getattr(msg, "text", "") or "")[:200],
        "tools": len(ctx.tools) if getattr(ctx, "tools", None) else 0,
    })
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
    _emit(agent, "reasoning.proposed", {
        "content": (getattr(proposal, "content", "") or "")[:200],
        "confidence": getattr(proposal, "confidence", 0.0),
    })
    _emit(agent, "reasoning.critiqued", {
        "verdict": getattr(critique, "verdict", ""),
        "concerns": len(getattr(critique, "concerns", []) or []),
    })
    turn.trace_phase("reason", "ok", duration_ms=(time.time()-t4)*1000)
    agent.iko.trace("reason", "ok")
    agent._record_inference("decide_reason")

    # 2.3 仲裁
    t5 = time.time()
    decision = agent.ios.arbitrate(proposal, critique, risk)
    hc.decision = decision
    _emit(agent, "decision.made", {
        "action": getattr(decision, "action", ""),
        "approved": bool(getattr(decision, "approved", False)),
        "reason": (getattr(decision, "reason", "") or "")[:100],
    })
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
    _emit(agent, "action.executed", {
        "success": bool(getattr(result, "success", False)),
        "duration_ms": getattr(result, "duration_ms", 0.0),
        "error": (getattr(result, "error", "") or "")[:100],
    })
    # DR-20260828-01 修复#4：tool_calls写入协议上下文（接通断裂四：IKO场景分类）
    hc.tool_calls = getattr(decision, 'tool_calls', []) or []
    turn.trace_phase("execute", "ok" if result.success else "error",
                     duration_ms=result.duration_ms)
    agent.iko.trace("execute", "ok" if result.success else "error")

    # 3.3 工具验证
    _handle_tool_validation(agent, decision, result, turn)


def _handle_tool_validation(agent, decision, result, turn):
    """工具结果验证"""
    from openllm.core.tool_validator_types import ToolCall, validate_tool_result
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
            # DR-20260917-01：此处提前终结本turn，外层_learn(492)会再次complete
            # → "cannot complete from status complete" 崩溃。幂等防护：仅在ACTIVE时结束。
            if getattr(turn, "status", None) == TurnStatus.ACTIVE.value:
                turn.complete()
            return
        if _tv_report.should_retry:
            turn.trace_phase("validate_retry", "warn", detail="工具结果需重试")
    except Exception as _tv_err:
        turn.trace_phase("validate", "skip", detail=str(_tv_err)[:80])


# ── 刀⑧(2026-09-26): 工具调用摘要（心跳 add_action 入账 + CLI 轨迹前缀用）──

def _tool_call_summary(tc) -> dict:
    """把 decision.tool_calls 的一项压成 {"name":…,"arg":…}。

    兼容 dict({"name","args"}) 与裸对象/字符串；arg 取首个参数值截30字符。
    纯字符串化字段——保证 turn.actions JSON 可序列化（checkpoint 持久化路径）。
    """
    if isinstance(tc, dict):
        name = tc.get("name", "?")
        args = tc.get("args") or {}
        first = next(iter(args.values()), "") if isinstance(args, dict) and args else args
        return {"name": str(name), "arg": str(first)[:30]}
    return {"name": str(tc)[:30], "arg": ""}


# ── Phase 3.5: SYNTHESIZE（综合消化）────────────────────

def _synthesize(agent, msg, hc, turn):
    """综合消化：让模型消化工具结果后生成最终回答，而非裸吐工具输出。

    仅在 decision 带 tool_calls 且执行成功且 provider 可用时触发。
    最多 2 轮：综合→模型输出→(若仍带TOOL_CALLS)→再执行→再综合。
    失败语义：任何异常回退旧行为 _last_output=result.output。
    开关：OPENLLM_HEARTBEAT_SYNTH=0/off 时跳过（默认开）。
    """
    # 开关检查
    if os.environ.get("OPENLLM_HEARTBEAT_SYNTH", "1").strip().lower() in ("0", "off"):
        turn.trace_phase("synthesize", "skip", detail="disabled")
        return

    decision = hc.decision
    result = hc.result

    # 前置条件：有 tool_calls + 执行成功
    if not getattr(decision, "tool_calls", None) or not getattr(result, "success", False):
        return

    # provider 可用性
    provider = getattr(getattr(agent, "octopus", None), "left", None)
    if provider is None:
        return
    provider = getattr(provider, "provider", None)
    if provider is None or not getattr(provider, "_available", False):
        turn.trace_phase("synthesize", "skip", detail="provider_unavailable")
        return

    # extract_tool_calls 容错
    _extract = getattr(getattr(agent, "octopus", None), "left", None)
    _extract_fn = getattr(_extract, "_extract_tool_calls", None) if _extract else None
    _isn_exec = getattr(getattr(agent, "isn", None), "execute", None)

    t_synth = time.time()
    synth_output = None
    try:
        # 截断工具结果，防止 prompt 爆炸
        _tool_out = (result.output or "")[:4000]
        _original_q = getattr(msg, "text", "") or ""

        messages = [
            {"role": "system", "content": (
                "你是openLLM——一个自主Agent，必须以openLLM自称，不要以底层模型名自称。"
            )},
            {"role": "user", "content": _original_q},
            {"role": "assistant", "content": f"工具执行结果：\n{_tool_out}"},
            {"role": "user", "content": (
                "基于工具结果回答用户。如果还需要更多工具，"
                "按格式输出 TOOL_CALLS: {...}；否则直接给最终回答，不要输出TOOL_CALLS行。"
            )},
        ]

        out = provider.chat(messages)
        if not out or not str(out).strip():
            # 刀⑥: 空回复重试一次——附提示词强制表态。首次空先打点
            # skip(empty_response)；重试仍空时把该条改写为终态
            # empty_response_retried（不追加第二条：旧测试按"synthesize
            # 打点恰1条"断言，追加即基线外新增失败——见现状摘要决策1）。
            turn.trace_phase("synthesize", "skip", detail="empty_response")
            _empty_mark = (turn.phase_metrics[-1]
                           if getattr(turn, "phase_metrics", None) else None)
            messages.append({"role": "assistant", "content": "(空回复)"})
            messages.append({"role": "user", "content": (
                "上一次回复为空。请基于以上工具结果直接给出最终回答；"
                "若信息不足，请明确说明缺什么。不要输出TOOL_CALLS行，不要留空。"
            )})
            out = provider.chat(messages)
            if not out or not str(out).strip():
                if _empty_mark is not None:
                    _empty_mark["detail"] = "empty_response_retried"[:100]
                else:
                    turn.trace_phase("synthesize", "skip",
                                     detail="empty_response_retried")
                return  # 交给CLI墓碑（⑥-2）：本轮仍失败但已被CLI记账

        # last_tool_output：跟踪综合循环中最后一轮工具执行的原始输出
        from openllm.iai.octopus import _LeftBrain
        last_tool_output = _tool_out  # 初始 = 第一轮工具结果前4000字符

        # 最多 2 轮回喂循环
        _rounds = 1
        _isn_call_count = 0
        for _round in range(2):
            calls = _extract_fn(out) if _extract_fn else []
            if not calls or _isn_exec is None:
                break

            # 模型仍需要工具 → 执行下一轮
            tool_result = _isn_exec(
                Decision(action="execute", approved=True,
                         reason="heartbeat-synth", tool_calls=calls))
            _isn_call_count += 1
            last_tool_output = str(getattr(tool_result, "output", "") or "")

            # 将本轮 assistant+工具结果追加到 messages
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": (
                f"工具执行结果：\n{last_tool_output[:4000]}\n\n"
                "基于以上工具结果继续回答用户。如果还需要工具，"
                "按同样格式输出TOOL_CALLS；否则直接给出最终回答，"
                "不要输出TOOL_CALLS行。"
            )})
            # 记账
            try:
                agent._record_inference("synthesize")
            except Exception:
                pass
            out = provider.chat(messages)
            _rounds = _round + 2

        # 清理残留 TOOL_CALLS 行，采用清理后文本
        synth_output = _LeftBrain._TOOLCALL_RE.sub("", out).strip() if out else ""

        # ── 强制收口轮：综合正文空但有工具结果时，再逼一次 ──
        if not synth_output and last_tool_output:
            # 确保 last_tool_output 在 messages 中（以防从未进循环）
            if not any("工具执行原始结果" in m.get("content", "") or
                       "工具执行结果" in m.get("content", "")
                       for m in messages if m["role"] == "user"):
                pass  # 已在 messages 中，不重复
            messages.append({"role": "user", "content": (
                "禁止再调用任何工具。基于以上已有的工具执行结果，"
                "直接给出对用户的最终回答（含你对内容的理解与看法）。"
                "不要输出TOOL_CALLS行。"
            )})
            try:
                out2 = provider.chat(messages)
                synth_output = _LeftBrain._TOOLCALL_RE.sub("", out2).strip() if out2 else ""
                _isn_call_count = _isn_call_count  # 收口轮不算 isn_exec
                _rounds += 1  # 收口轮记入 rounds
            except Exception:
                synth_output = ""  # 收口失败，走回退阶梯

        # ── 回退阶梯 + 状态诚实 ──
        if synth_output:
            # a) 综合正文非空 → 用它，ok
            turn.trace_phase("synthesize", "ok",
                             duration_ms=(time.time() - t_synth) * 1000,
                             detail=f"len={len(synth_output)} rounds={_rounds}")
        elif last_tool_output:
            # b) 综合正文空但有工具原始输出 → 降级使用，warn
            synth_output = f"（综合未成文，以下为工具执行原始结果）\n{last_tool_output[:4000]}"
            turn.trace_phase("synthesize", "warn",
                             duration_ms=(time.time() - t_synth) * 1000,
                             detail=f"fallback_tool_output len={len(synth_output)} rounds={_rounds}")
        else:
            # c) 全空 → 不写 hc.synth_output，error
            synth_output = ""
            turn.trace_phase("synthesize", "error",
                             duration_ms=(time.time() - t_synth) * 1000,
                             detail="empty_synth")

    except Exception as e:
        # 综合是增强不是关卡——失败回退现状
        turn.trace_phase("synthesize", "error",
                         duration_ms=(time.time() - t_synth) * 1000,
                         detail=str(e)[:100])
        synth_output = None  # 回退：让 _learn 取 result.output

    # 记账（最后一轮）
    try:
        agent._record_inference("synthesize")
    except Exception:
        pass

    # 传递综合结果给 _learn
    if synth_output:
        hc.synth_output = synth_output


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

    # PAL T-G-2: SelfModificationGuard 激活 — learn_causal 写入受守卫检查
    _causal_target = str(Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl")
    _guard_skipped = False
    try:
        from ..ios.self_modification_guard import SelfModificationGuard
        _guard = SelfModificationGuard()
        # 写入前：频率限制 + 禁区检查
        if _guard.check_rate_limit(_causal_target):
            _guard_skipped = True
            turn.trace_phase("guard_rate_limit", "skip", detail=f"target={_causal_target}")
        elif not _guard.approve_change(_causal_target, change_type="causal_memory"):
            _guard_skipped = True
            turn.trace_phase("guard_forbidden", "skip", detail=f"target={_causal_target}")
        else:
            agent.ios.learn_causal(ctx, prediction, result, delta)
            agent.octopus.learn_causal(ctx, prediction, result, delta)
            # 写入后：记录本次修改
            _guard.record_modification(
                target_file=_causal_target,
                change_type="causal_memory",
                agent_id=getattr(agent, '_agent_id', 'openllm'),
            )
            turn.trace_phase("guard_recorded", "ok", detail=f"target={_causal_target}")
    except Exception as _guard_err:
        # guard失败不阻塞心跳
        turn.trace_phase("guard", "skip", detail=str(_guard_err)[:100])
        if not _guard_skipped:
            # guard检查阶段未跳过，回退执行learn_causal
            try:
                agent.ios.learn_causal(ctx, prediction, result, delta)
                agent.octopus.learn_causal(ctx, prediction, result, delta)
            except Exception:
                pass

    # E2 缺口③：因果度量接线（2026-08-23）
    try:
        if agent.memory_evaluator:
            _had_causal = bool(getattr(hc, 'causal_block', ''))
            _influenced = _had_causal and (not delta.prediction_match)
            agent.memory_evaluator.record_causal(
                action=hc.user_message[:200],
                retrieved=_had_causal,
                influenced=_influenced,
                lesson=delta.delta_summary,
            )
    except Exception:
        pass  # 不阻塞主循环

    # PAL T-F-6：预测律闭环 — 偏差信号回流（2026-09-02）
    # 赫尔墨斯铁律：回流信号必须经过独立验证（match=False才记录）
    try:
        if agent.iai and agent.iai.predictor:
            # 将 delta.prediction_match 注入 prediction dict 供 record_error 使用
            _pred_for_error = dict(prediction) if isinstance(prediction, dict) else {"predicted_type": str(prediction)}
            _pred_for_error["prediction_match"] = delta.prediction_match
            agent.iai.predictor.record_error(
                prediction=_pred_for_error,
                actual=result if isinstance(result, dict) else {"output": result},
                source="compare",
            )
    except Exception:
        pass  # 不阻塞主循环

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
    # P0.5: 优先取综合消化结果（_synthesize 回路），回退裸工具输出
    _synth = getattr(hc, "synth_output", None)
    if _synth:
        agent._last_output = _synth
    elif getattr(decision, 'tool_calls', None) or getattr(decision, '_has_tools', False):
        agent._last_output = result.output
    else:
        agent._last_output = proposal.content if proposal.content else result.output
    hc.output = agent._last_output

    # ── 苏醒协议：检测并记录选择 ──
    try:
        agent._awakening_protocol.detect_choice_and_record(
            agent._last_output, agent.session
        )
    except Exception:
        pass  # 选择检测失败不阻塞主循环

    # IKO pipeline
    try:
        risk_map = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
        ctx_for_iko = {
            "risk_level": risk_map.get(getattr(hc.risk, 'level', 'low'), "LOW") if hc.risk else "LOW",
            # DR-20260828-01 修复#4：优先取协议字段（EXECUTE已填充），回退decision（向后兼容）
            "has_tool_calls": bool(hc.tool_calls) or bool(getattr(decision, 'tool_calls', None)),
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

    turn.add_action("execute", result.output[:100],
                    # 刀⑧(2026-09-26 会话连续性三修): 本轮工具调用摘要随
                    # action 入账，供 CLI 回合轨迹前缀读取。add_action 的
                    # **kwargs 是既有容器（session.py），Turn 结构零改动。
                    # 每项 {"name","arg"}，arg 取首参数截30字符。
                    tool_calls=[_tool_call_summary(tc) for tc in
                                (getattr(decision, "tool_calls", None) or [])])
    # DR-20260917-01：_execute提前终结turn（cap拒绝303/验证拦截359）后仍会走到这里，
    # 二次complete曾致状态机异常+CLI崩溃。幂等：仅ACTIVE才完结。
    if getattr(turn, "status", None) == TurnStatus.ACTIVE.value:
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

    # 5.2b 时钟仪表（E2·2026-08-23）
    try:
        if agent.clock:
            _cs = agent.clock.now_status()
            agent.iko.trace("clock", "ok",
                detail=f"epoch={_cs['epoch']} awakenings={_cs['awakening_count']} gap={_cs['gap_since_last']:.0f}s")
    except Exception:
        pass

    # 5.3 过程透明化摘要
    _emit_summary(agent, hc, turn)

    # 5.4 G-1(2026-09-01): 治理trace写入审计链
    # 零LLM调用、零阻塞。失败静默降级。
    try:
        _risk_level = getattr(hc.risk, 'level', 'low') if hc.risk else 'low'
        _approved = getattr(hc.decision, 'approved', True) if hc.decision else True
        _reason = getattr(hc.decision, 'reason', '') if hc.decision else ''
        _duration_ms = (time.time() - getattr(agent, '_tick_start_time', time.time())) * 1000
        agent.ios.governance_engine.heartbeat_trace(
            tick_id=getattr(hc, 'tick_id', f"tick_{agent._tick_count}"),
            risk_level=_risk_level,
            approved=_approved,
            decision_reason=_reason,
            duration_ms=_duration_ms,
            agent_id=getattr(agent, '_agent_id', 'openllm'),
        )
    except Exception:
        pass  # 心跳治理trace失败不阻断主循环



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
            from openllm.core.citation_checker import check_citations
            report = check_citations(str(hc.output))
            if report.total_claims > 0 and report.citation_rate < 0.5:
                agent.iko.trace("citation", "warn",
                    detail=f"引用率低: {report.citation_rate:.0%} ({report.cited_claims}/{report.total_claims})")
                if agent.mode != "silent":
                    print(f"  📎 引用率: {report.citation_rate:.0%} ({report.cited_claims}/{report.total_claims}条有来源)")
    except Exception:
        pass  # 引用检查失败不阻塞主循环
