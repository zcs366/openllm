"""
tool_executor.py — 工具执行管线

从engine.py提取的execute_tool + execute_tool_with_hindsight。
这两个方法依赖engine的多个属性（tools/security/bus/firewall/failure_tracker/loop），
所以提取为函数，接收engine实例作为参数。
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from ..tools.executor import ToolResult
from ..memory.execution_recorder import record_execution

logger = logging.getLogger("openllm.engine.tool_executor")


def execute_tool(engine, tool_name: str, **kwargs) -> ToolResult:
    """执行工具调用（带安全检查+verify钩子+三层管线）。
    
    从OpenLLMEngine.execute_tool提取。接收engine实例以访问：
    - engine.security: SecurityFoundation
    - engine.tools: ToolRegistry
    - engine.bus: MessageQueue
    - engine.firewall: CredentialFirewall
    - engine._check_tool_risk()
    - engine._verify_tool_params()
    - engine._verify_tool_result()
    - engine._run_pipeline()
    
    执行结果通过execution_recorder记录到RECALL（CCL执行节点）。
    """
    from .engine_integrations import get_isn_metadata
    from ..protocol import MessageType, BodyName

    t0 = time.time()
    _result: Optional[ToolResult] = None  # 单一返回点

    ok, reason = engine.security.check_action(tool_name)
    if not ok:
        _result = ToolResult(tool_name=tool_name, success=False, error=reason)

    # ── 血管 #3: ISN 风险检查 ──
    if _result is None:
        risk_check = engine._check_tool_risk(tool_name)
        if not risk_check["pass"]:
            _result = ToolResult(
                tool_name=tool_name, success=False,
                error=f"ISN风险检查未通过: {risk_check['reason']}"
            )
    else:
        risk_check = {"pass": False, "reason": "skipped"}

    # ── 三层安全管线（老IO-S pipeline） ──
    if _result is None:
        try:
            pipeline_result = engine._run_pipeline(
                input_text=str(kwargs)[:500],
                source=tool_name,
                llm_output="",
                pid="engine",
            )
            if not pipeline_result.passed:
                logger.warning(
                    f"安全管线拦截 {tool_name}: "
                    f"confidence={pipeline_result.confidence}, "
                    f"alerts={pipeline_result.alerts}")
                _result = ToolResult(
                    tool_name=tool_name, success=False,
                    error=f"安全管线拦截: {pipeline_result.alerts}")
        except ImportError:
            logger.info(f"安全管线不可用，跳过 {tool_name} 安全检查")
        except Exception as e:
            logger.error(f"安全管线异常 {tool_name}: {e}")
            _result = ToolResult(
                tool_name=tool_name, success=False,
                error=f"安全管线异常: {e}")

    # ── IOS拒绝检查 ──
    if _result is None:
        try:
            from .governance_engine import GovernanceEngine
            gov = GovernanceEngine()
            if risk_check.get("level") == "critical":
                rejection = gov.reject(
                    instruction=tool_name,
                    reason=f"ISN标记为critical级别工具: {tool_name}",
                    belief_confidence=0.95,
                )
                _result = ToolResult(
                    tool_name=tool_name, success=False,
                    error=f"IOS拒绝: {rejection.reason}")
        except Exception as e:
            logger.debug(f"IOS拒绝检查跳过: {e}")

    # ── verify钩子：调用前参数验证 ──
    if _result is None:
        param_check = engine._verify_tool_params(tool_name, **kwargs)
        if not param_check["pass"]:
            _result = ToolResult(
                tool_name=tool_name, success=False,
                error=f"参数验证未通过: {param_check['reason']}"
            )

    # ── 实际执行 ──
    if _result is None:
        result = engine.tools.execute(tool_name, **kwargs)

        # P2: 凭据防火墙——扫描工具输出
        if result.output:
            result.output = engine.firewall.scan_text(result.output)
        if result.error:
            result.error = engine.firewall.scan_text(result.error)

        # ── verify钩子：调用后结果验证 ──
        result_check = engine._verify_tool_result(tool_name, result)
        if not result_check["pass"]:
            logger.warning(f"结果验证未通过: {tool_name}: {result_check['reason']}")

        # ── 血管 #2: ISA 信念更新 ──
        verdict = "pass" if result.success and result_check["pass"] else "fail"
        engine.bus.publish(
            MessageType.GOVERNANCE_EVENT,
            source=BodyName.IOS, target=BodyName.ISA,
            payload={
                "tool_name": tool_name,
                "params": kwargs,
                "result": {"output": result.output[:500], "error": result.error},
                "verdict": verdict,
            },
        )

        # ── 血管 #4: IKO trace消费 ──
        engine.bus.publish(
            MessageType.OBSERVABILITY_LOG,
            source=BodyName.IOS, target=BodyName.IKO,
            payload={
                "type": "verify_pass" if result.success else "verify_fail",
                "tool_name": tool_name,
                "params": {k: str(v)[:100] for k, v in kwargs.items()},
                "verdict": "pass" if result.success else "fail",
                "timestamp": time.time(),
            },
        )

        # ── 血管 #5: IKO→ISA 反馈闭环 ──
        if result.success:
            engine.bus.publish(
                MessageType.DECISION_RESULT,
                source=BodyName.IKO, target=BodyName.ISA,
                payload={
                    "success": True,
                    "output": result.output[:1000],
                    "error": result.error,
                },
            )

        _result = result

    # ── CCL执行节点：记录工具执行到RECALL ──
    if _result is None:
        _result = ToolResult(tool_name=tool_name, success=False, error="unexpected: no execution path taken")
    duration_ms = (time.time() - t0) * 1000
    try:
        record_execution(
            tool_name=tool_name,
            args_summary={k: str(v)[:100] for k, v in kwargs.items()},
            status="ok" if _result.success else "error",
            duration_ms=duration_ms,
            result_summary=(_result.output or _result.error)[:200],
        )
    except Exception:
        logger.debug(f"record_execution失败(tool={tool_name})", exc_info=True)

    return _result


def execute_tool_with_hindsight(engine, tool_name: str, pid: str = "",
                                goal: str = "", **kwargs) -> ToolResult:
    """执行工具 + verify验证 + hindsight经验自动提取。
    
    三步一体化：
      1. execute_tool()（含verify钩子）
      2. 提取verify结果
      3. 写入hindsight经验（含验证状态、负面案例标记）
    """
    result = execute_tool(engine, tool_name, **kwargs)

    # 提取verify结果
    verify_pass = result.success
    verify_reason = ""
    if result.success:
        v = engine._verify_tool_result(tool_name, result)
        verify_pass = v["pass"]
        verify_reason = v["reason"]
    elif result.error:
        verify_reason = result.error

    # ── Phase 8: Failure Signature 提取 ──
    if not verify_pass and verify_reason:
        ctx = {"user_input": goal[:200] if goal else ""}
        sig = engine.failure_tracker.extract_signature(tool_name, verify_reason, ctx)
        evolve_hint = engine.failure_tracker.learn_causal(sig)
        if evolve_hint:
            logger.info(f"  Phase 8: {evolve_hint}")
            if engine.failure_tracker.should_evolve():
                proposals = engine.failure_tracker.evolve()
                for p in proposals:
                    logger.warning(
                        f"  Phase 9 进化提案: [{p.proposal_type}] "
                        f"{p.description} (置信度={p.confidence:.0%})"
                    )

    # 更新loop的工具成功/失败状态
    engine.loop._last_tool_success = verify_pass

    # 如果有任务上下文，自动提取hindsight经验
    if pid and goal:
        try:
            from .hindsight_loop import extract_hindsight

            hindsight_data = extract_hindsight(
                pid=pid,
                goal=goal[:200],
                result={
                    "success": verify_pass,
                    "execution_time": result.latency_ms / 1000.0,
                    "token_count": 0,
                },
                failure=verify_reason if not verify_pass else ""
            )

            verify_record = {
                "tool": tool_name,
                "verify_pass": verify_pass,
                "verify_reason": verify_reason,
                "hindsight": hindsight_data,
                "timestamp": time.time(),
            }
            hindsight_dir = Path.home() / ".io-s" / "hindsight"
            hindsight_dir.mkdir(parents=True, exist_ok=True)
            verify_path = hindsight_dir / f"verify_{pid}_{int(time.time())}.json"
            verify_path.write_text(json.dumps(verify_record, ensure_ascii=False))

            _trim_verify_files(hindsight_dir, getattr(engine, 'MAX_VERIFY_FILES', 50))

            logger.info(
                f"  hindsight+verify: {pid}.{tool_name} "
                f"{'PASS' if verify_pass else 'FAIL'}"
            )
        except Exception as e:
            logger.debug(f"hindsight提取跳过: {e}")

    return result


def _trim_verify_files(directory: Path, max_files: int):
    """清理旧 verify 文件，只保留最近的 max_files 个。"""
    if not directory.exists():
        return
    files = sorted(directory.glob("verify_*.json"))
    if len(files) > max_files:
        for f in files[:-max_files]:
            f.unlink()
