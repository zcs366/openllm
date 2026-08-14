"""verification_pipeline.py — 验证管线：把验证逻辑从chat()中抽离。

验证管线 = 按顺序执行的一组验证步骤。
每个步骤：接收response → 验证/修正 → 返回response。

设计原则：
  - 独立模块，不修改engine.py的核心逻辑
  - 每个步骤可独立启用/禁用
  - 验证失败不阻塞主流程（graceful degradation）
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .provider import Message

logger = logging.getLogger("openllm.verification_pipeline")


@dataclass
class VerificationResult:
    """单步验证结果。"""
    step_name: str
    passed: bool
    message: str = ""
    modified: bool = False  # 是否修改了response


@dataclass
class PipelineReport:
    """管线完整报告。"""
    results: List[VerificationResult] = field(default_factory=list)
    final_response: str = ""
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0

    def add(self, result: VerificationResult) -> None:
        self.results.append(result)
        self.total_steps += 1
        if result.passed:
            self.passed_steps += 1
        else:
            self.failed_steps += 1


class VerificationPipeline:
    """验证管线——按顺序执行验证步骤。

    Usage:
        pipeline = VerificationPipeline(engine)
        report = pipeline.run(response)
        response = report.final_response
    """

    def __init__(self, engine):
        self.engine = engine
        self._steps: List[Callable[[str], VerificationResult]] = []
        self._register_default_steps()

    def _register_default_steps(self) -> None:
        """注册默认验证步骤。"""
        self._steps = [
            self._step_code_syntax,
            self._step_claim_verification,
            self._step_repetition_detection,
            self._step_checkpoint_save,
        ]

    def add_step(self, step: Callable[[str], VerificationResult]) -> None:
        """添加自定义验证步骤。"""
        self._steps.append(step)

    def run(self, response: str) -> PipelineReport:
        """执行完整验证管线。"""
        report = PipelineReport()

        for step in self._steps:
            try:
                result = step(response)
                report.add(result)
                if result.modified:
                    response = result.message  # step返回修正后的response
            except Exception as e:
                logger.warning(f"验证步骤异常: {e}")
                report.add(VerificationResult(
                    step_name=step.__name__,
                    passed=False,
                    message=f"异常: {e}"
                ))

        report.final_response = response
        return report

    # ── 步骤1：代码语法验证（含自动修正循环） ──

    def _step_code_syntax(self, response: str) -> VerificationResult:
        """验证response中Python代码块的语法正确性，错误时自动修正。

        旧逻辑：检测到语法错误→注入历史让模型修正→重新调用模型→最多2次。
        新逻辑保持一致：检测→注入→重调→重试。
        """
        if not response:
            return VerificationResult(step_name="code_syntax", passed=True)

        MAX_FIX_ROUNDS = 2
        for round_num in range(MAX_FIX_ROUNDS + 1):
            code_blocks = self.engine._extract_python_blocks(response)
            if not code_blocks:
                return VerificationResult(step_name="code_syntax", passed=True)

            first_error = None
            for i, block in enumerate(code_blocks):
                ok, err_msg = self.engine._verify_code_syntax(block)
                if not ok:
                    first_error = (i, err_msg)
                    break

            if first_error is None:
                # 所有代码块语法正确
                if round_num > 0:
                    logger.info(f"代码语法验证: 经过{round_num}次修正后通过")
                return VerificationResult(step_name="code_syntax", passed=True)

            # 有语法错误——注入历史让模型修正
            i, err_msg = first_error
            logger.info(f"代码语法验证失败(块{i+1}): {err_msg}")
            self.engine._history.append(Message(
                role="user",
                content=(
                    f"[代码语法验证] 你输出的第{i+1}个代码块有语法错误:\n"
                    f"{err_msg}\n\n"
                    f"请修正后重新输出完整代码。"
                )
            ))

            # 最后一轮不再重调模型
            if round_num < MAX_FIX_ROUNDS:
                response = self.engine._call_model(stream=True)

        # 修正轮次用完仍有错误
        return VerificationResult(
            step_name="code_syntax",
            passed=False,
            message=f"代码语法错误经{MAX_FIX_ROUNDS}次修正仍未通过"
        )

    # ── 步骤2：事实声明验证 ──

    def _step_claim_verification(self, response: str) -> VerificationResult:
        """用工具回查response中的事实性声明。"""
        if not response or len(response) < 10:
            return VerificationResult(step_name="claim_verification", passed=True)

        verified_response = self.engine._verify_claims(response)
        modified = verified_response != response

        return VerificationResult(
            step_name="claim_verification",
            passed=True,  # 验证本身不阻塞
            message=verified_response,
            modified=modified
        )

    # ── 步骤3：重复检测 ──

    def _step_repetition_detection(self, response: str) -> VerificationResult:
        """检测输出中的重复自我纠正模式。"""
        if not response:
            return VerificationResult(step_name="repetition_detection", passed=True)

        deduplicated = self.engine._check_repetition(response)
        modified = deduplicated != response

        return VerificationResult(
            step_name="repetition_detection",
            passed=True,
            message=deduplicated,
            modified=modified
        )

    # ── 步骤4：Checkpoint保存 ──

    def _step_checkpoint_save(self, response: str) -> VerificationResult:
        """每轮对话后自动保存checkpoint。"""
        try:
            from .checkpoint import TaskCheckpoint
            ckpt = TaskCheckpoint()
            state = {
                "history": [
                    {"role": m.role, "content": m.content[:500]}
                    for m in self.engine._history[-20:]
                ],
                "tool_calls": [],
                "results": [],
            }
            ckpt.save(state, os.path.expanduser("~/.openllm/auto_checkpoint.json"))
            return VerificationResult(
                step_name="checkpoint_save",
                passed=True,
                message="checkpoint已保存"
            )
        except Exception as e:
            return VerificationResult(
                step_name="checkpoint_save",
                passed=False,
                message=f"checkpoint保存失败: {e}"
            )
