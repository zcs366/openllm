"""
治理转换引擎 — P0-b-2

五步管线（论文提取 + 鲁班修正）:
  Step 1: 失败捕获 (Failure Capture)
  Step 2: 失败分类 (Failure Classification)
  Step 3: 治理设计 (Governance Design)
  Step 4: 治理安装 (Governance Installation) — 含冲突检测
  Step 5: 治理验证 (Governance Verification) — 含回滚机制

论文核心循环:
  速度暴露失败 → 分类(局部/结构) → 治理转换 → 后续Agent继承更窄空间 → 治理复合增长

依赖:
  - structural_failure_classifier.StructuralFailureClassifier
  - governance_rule.GovernanceRule, GovernanceRuleStore, RuleType, RuleFamily
  - failure_tracker.FailureSignature, FailureCategory
"""

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from .structural_failure_classifier import (
    StructuralFailureClassifier, ClassificationResult, GovernanceRequest
)
from .governance_rule import (
    GovernanceRule, GovernanceRuleStore, RuleType, RuleFamily, RuleStatus
)
from .failure_tracker import (
    FailureSignature, FailureCategory, HarnessLayer, CATEGORY_TO_LAYER
)


# ── 数据模型 ──────────────────────────────────────────

@dataclass
class RejectionRecord:
    """拒绝记录——IOS对指令说不的唯一凭证。"""
    instruction: str
    reason: str
    timestamp: float
    belief_confidence: float  # ISA信念系统对该决策的置信度
    rejected_by: str = "IOS"  # 拒绝者标识

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "belief_confidence": self.belief_confidence,
            "rejected_by": self.rejected_by,
        }


@dataclass
class VerificationResult:
    """治理验证结果"""
    passed: bool
    regression_passed: bool = True
    side_effects_clean: bool = True
    details: str = ""
    rollback_performed: bool = False


@dataclass
class ConversionResult:
    """治理转换结果"""
    success: bool
    classification: str = ""          # structural | local | ambiguous
    rule: Optional[GovernanceRule] = None
    verification: Optional[VerificationResult] = None
    governance_request: Optional[GovernanceRequest] = None  # ambiguous时
    details: str = ""


# ── 治理转换引擎 ──────────────────────────────────────

class GovernanceEngine:
    """治理转换引擎: 失败→治理规则的自动转换。
    
    论文依据:
    - "governance conversion explains how controls are discovered 
       from failures that become visible only during agentic work"
    - "The scarce human work is not implementation-level review, 
       but recognizing which failures reveal missing governance 
       and converting them into architecture and controls"
    
    五步管线:
    1. capture_failure: 从trace提取FailureSignature
    2. classify_failure: 调用StructuralFailureClassifier
    3. design_governance: 架构响应 or 控制响应
    4. install_governance: 写入治理基质 + 冲突检测
    5. verify_governance: 回归测试 + 副作用检查 + 回滚
    """
    
    # 失败机制→治理规则族映射（论文10族）
    MECHANISM_TO_FAMILY = {
        "tool_loop": RuleFamily.INCORPORATION_GATE,
        "missing_artifact": RuleFamily.VALIDATION,
        "wrong_format": RuleFamily.STATIC_DYNAMIC_ANALYSIS,
        "dependency_missing": RuleFamily.RESOURCE_MEDIATOR,
        "timeout": RuleFamily.RESOURCE_MEDIATOR,
        "logic_error": RuleFamily.STATIC_DYNAMIC_ANALYSIS,
        "permission": RuleFamily.CONTEXT_DISPATCH,
        "state_corruption": RuleFamily.PROVENANCE,
        "exploration_loop": RuleFamily.CONTEXT_DISPATCH,
        "premature_success": RuleFamily.INCORPORATION_GATE,
    }
    
    # mechanism名→FailureCategory枚举映射（C3+M1修复）
    MECHANISM_TO_CATEGORY = {
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
        # FailureCategory枚举名直接映射
        "tool_param": FailureCategory.TOOL_PARAM,
        "tool_timeout": FailureCategory.TOOL_TIMEOUT,
        "tool_permission": FailureCategory.TOOL_PERMISSION,
        "llm_hallucination": FailureCategory.LLM_HALLUCINATION,
        "llm_format": FailureCategory.LLM_FORMAT,
        "context_overflow": FailureCategory.CONTEXT_OVERFLOW,
        "route_wrong": FailureCategory.ROUTE_WRONG,
        "user_correction": FailureCategory.USER_CORRECTION,
    }
    
    # 失败机制→治理设计模板
    GOVERNANCE_TEMPLATES = {
        "tool_loop": {
            "type": RuleType.CONSTRAINT,
            "condition_template": "tool_retry_count >= {threshold}",
            "action_template": "block_and_report(tool_name='{tool}', max_retries={threshold})",
            "description_template": "工具重试上限: {tool} 最多重试{threshold}次",
        },
        "missing_artifact": {
            "type": RuleType.CONTROL,
            "condition_template": "post_execution_check('{artifact}') == missing",
            "action_template": "fail_with_missing_artifact('{artifact}')",
            "description_template": "产出物验证: 执行后必须检查{artifact}存在",
        },
        "wrong_format": {
            "type": RuleType.CONTROL,
            "condition_template": "output_format('{output}') != expected",
            "action_template": "validate_and_repair('{output}', expected_format)",
            "description_template": "格式验证: {output}必须符合预期格式",
        },
        "timeout": {
            "type": RuleType.CONSTRAINT,
            "condition_template": "execution_time > {threshold}s",
            "action_template": "timeout_with_fallback('{tool}', {threshold})",
            "description_template": "超时保护: {tool}执行超过{threshold}秒自动降级",
        },
        "logic_error": {
            "type": RuleType.CONTROL,
            "condition_template": "assertion_violation('{check}')",
            "action_template": "halt_and_report('{check}')",
            "description_template": "断言检查: {check}必须通过",
        },
        "permission": {
            "type": RuleType.ARCHITECTURE,
            "condition_template": "access_level('{resource}') < required",
            "action_template": "deny_with_explanation('{resource}')",
            "description_template": "权限边界: {resource}访问需要明确授权",
        },
        "exploration_loop": {
            "type": RuleType.CONSTRAINT,
            "condition_template": "search_depth > {threshold}",
            "action_template": "terminate_search_with_summary()",
            "description_template": "搜索深度限制: 最多{threshold}轮搜索",
        },
        "premature_success": {
            "type": RuleType.CONTROL,
            "condition_template": "claim_success_without_verification()",
            "action_template": "require_verification_before_success()",
            "description_template": "成功验证: 声明成功前必须通过验证",
        },
    }
    
    def __init__(self):
        self.classifier = StructuralFailureClassifier()
        self.rule_store = GovernanceRuleStore()
        self._storage_dir = Path.home() / ".openllm" / "output" / "ios"
    
    # ── 主管线 ──────────────────────────────────────
    
    def convert(self, trace: dict) -> ConversionResult:
        """完整治理转换管线。
        
        Args:
            trace: 执行失败trace，包含:
                - tool_name: 工具名
                - error: 错误信息
                - context: 上下文dict
                - mechanism: 失败机制（可选，如已知）
                
        Returns:
            ConversionResult
        """
        # Step 1: 失败捕获
        sig = self.capture_failure(trace)
        
        # Step 2: 失败分类
        classification_result = self.classifier.classify(sig)
        
        # Step 3-5: 按分类结果处理
        if classification_result.classification == "local":
            return ConversionResult(
                success=True,
                classification="local",
                details="局部缺陷，记录但不触发治理转换",
            )
        
        if classification_result.classification == "ambiguous":
            return ConversionResult(
                success=True,
                classification="ambiguous",
                governance_request=classification_result.governance_request,
                details=f"待定分类，已发GovernanceRequest: "
                        f"{classification_result.governance_request.request_id if classification_result.governance_request else 'N/A'}",
            )
        
        # structural → 治理转换
        # Step 3: 治理设计
        rule = self.design_governance(sig, classification_result)
        
        # Step 4: 治理安装（含冲突检测）
        installed = self.install_governance(rule)
        
        if not installed:
            return ConversionResult(
                success=False,
                classification="structural",
                rule=rule,
                details=f"治理安装失败（冲突）: {rule.conflicts}",
            )
        
        # Step 5: 治理验证（含回滚）
        verification = self.verify_governance(rule)
        
        return ConversionResult(
            success=verification.passed,
            classification="structural",
            rule=rule,
            verification=verification,
            details=f"治理转换{'成功' if verification.passed else '失败（已回滚）'}: "
                    f"rule_id={rule.rule_id}",
        )
    
    # ── Step 1: 失败捕获 ──────────────────────────────
    
    def capture_failure(self, trace: dict) -> FailureSignature:
        """Step 1: 从执行trace提取FailureSignature。"""
        tool_name = trace.get("tool_name", "unknown")
        error = trace.get("error", "")
        mechanism = trace.get("mechanism", "")
        
        # 用已有分类器的mechanism分类（如果trace中没有）
        if not mechanism:
            from .failure_tracker import FailureSignatureTracker
            tracker = FailureSignatureTracker()
            sig = tracker.extract_signature(tool_name, error, trace.get("context"))
        else:
            # 直接构造（C3修复：用MECHANISM_TO_CATEGORY映射表）
            category = self.MECHANISM_TO_CATEGORY.get(
                mechanism.lower(), FailureCategory.UNKNOWN
            )
            sig = FailureSignature(
                category=category,
                tool_name=tool_name,
                error_pattern=self._generalize_error(error),
                timestamp=time.time(),
                raw_error=error[:500],
            )
        
        # HTIR-Step1: 自动填充harness_layer
        layer = CATEGORY_TO_LAYER.get(sig.category)
        if layer:
            sig.harness_layer = layer.value
        
        return sig
    
    # ── Step 2: 失败分类 ──────────────────────────────
    # (委托给StructuralFailureClassifier)
    
    # ── Step 3: 治理设计 ──────────────────────────────
    
    def design_governance(
        self, sig: FailureSignature, classification: ClassificationResult
    ) -> GovernanceRule:
        """Step 3: 治理设计——两种模式。
        
        论文:
        - 架构响应: "eliminated the failure class by construction"
        - 控制响应: "detecting failures earlier with a control"
        """
        mechanism = sig.category.name.lower()
        template = self.GOVERNANCE_TEMPLATES.get(mechanism)
        family = self.MECHANISM_TO_FAMILY.get(mechanism, RuleFamily.VALIDATION)
        
        if template:
            rule_type = template["type"]
            condition = template["condition_template"].format(
                tool=sig.tool_name, threshold=3, artifact="output", check="invariant"
            )
            action = template["action_template"].format(
                tool=sig.tool_name, threshold=3, artifact="output", output="result"
            )
            description = template["description_template"].format(
                tool=sig.tool_name, threshold=3, artifact="output", check="invariant"
            )
        else:
            # 默认: 控制响应
            rule_type = RuleType.CONTROL
            condition = f"failure_mechanism == '{mechanism}'"
            action = f"log_and_alert(mechanism='{mechanism}')"
            description = f"通用控制: {mechanism}类失败检测"
        
        return GovernanceRule(
            rule_type=rule_type,
            family=family,
            target=sig.tool_name or "general",
            condition=condition,
            action=action,
            source_failure=sig.key(),
            source_mechanism=mechanism,
            source_count=len(
                self.classifier._signature_index.get(sig.key(), [])
            ),
            description=description,
            confidence=min(0.9, 0.5 + classification.composite_score * 0.4),
            status=RuleStatus.PENDING,
        )
    
    # ── Step 4: 治理安装 ──────────────────────────────
    
    def install_governance(self, rule: GovernanceRule) -> bool:
        """Step 4: 治理安装——含冲突检测+阿瑞斯降级。
        
        Returns:
            True: 安装成功（或降级安装）
            False: 存在冲突且不可降级，需要人工解决
        """
        # 冲突检测
        conflicts = self.rule_store.detect_conflicts(rule)
        
        if conflicts:
            rule.conflicts = [c.rule_id for c in conflicts]
            
            # 阿瑞斯降级路径: 拦截类规则可自动降级安装
            action_lower = (rule.action or "").lower()
            if "block" in action_lower or "intercept" in action_lower:
                rule.status = RuleStatus.DEGRADED
                self.rule_store.save(rule)
                self._record_to_causal_memory(rule)
                return True  # 降级安装成功，不阻塞
            
            rule.status = RuleStatus.CONFLICTED
            self.rule_store.save(rule)
            return False
        
        # 无冲突，安装
        rule.status = RuleStatus.ACTIVE
        self.rule_store.save(rule)
        
        # 写入因果记忆（连接回路）
        self._record_to_causal_memory(rule)
        
        return True
    
    # ── Step 5: 治理验证 ──────────────────────────────
    
    def verify_governance(self, rule: GovernanceRule) -> VerificationResult:
        """Step 5: 治理验证——含回滚机制（鲁班要求）。
        
        验证内容:
        1. 回归测试: 已知同类失败不再发生
        2. 副作用检查: 治理不引入新失败
        3. 条件可验证性: rule.condition可被机器评估
        """
        # ① 条件可验证性检查
        if not rule.is_machine_verifiable():
            rule.add_verification(False, "条件不可机器验证")
            self._rollback_governance(rule)
            return VerificationResult(
                passed=False,
                regression_passed=False,
                details="条件不可机器验证",
                rollback_performed=True,
            )
        
        # ② 回归检查: 查看是否有同机制的历史失败被拦截
        regression_passed = self._check_regression(rule)
        
        # ③ 副作用检查: 新规则不与已有active规则冲突
        side_effects_clean = len(rule.conflicts) == 0
        
        passed = regression_passed and side_effects_clean
        
        # 记录验证结果
        rule.add_verification(
            passed,
            f"regression={regression_passed}, side_effects={side_effects_clean}"
        )
        
        if not passed:
            self._rollback_governance(rule)
        
        # 阿佛洛狄忒·成长信号: 治理转换成功后输出"系统学会了"
        if passed:
            logger.info(f"✅ 系统学会了：{rule.description}")
        
        return VerificationResult(
            passed=passed,
            regression_passed=regression_passed,
            side_effects_clean=side_effects_clean,
            details=f"验证{'通过' if passed else '失败'}",
            rollback_performed=not passed,
        )
    
    def _check_regression(self, rule: GovernanceRule) -> bool:
        """回归检查: 新规则是否能拦截已知同类失败"""
        causal_path = self._storage_dir / "causal_memory.jsonl"
        if not causal_path.exists():
            return True  # 无历史数据，默认通过（保守）
        
        found_any = False
        matched_any = False
        
        try:
            with open(causal_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if (entry.get("mechanism") == rule.source_mechanism and
                            entry.get("result_success") is False):
                            found_any = True
                            action = entry.get("action", "")
                            if rule.target in action or rule.target == "general":
                                matched_any = True
                                break
                    except (json.JSONDecodeError, KeyError):
                        continue
        except IOError:
            pass
        
        if not found_any:
            return True  # 无同类失败记录，默认通过
        return matched_any  # 有同类失败但规则无法拦截→失败
    
    def _rollback_governance(self, rule: GovernanceRule):
        """回滚治理规则"""
        rule.status = RuleStatus.ROLLED_BACK
        rule.add_verification(False, "回滚: 验证失败")
        self.rule_store.save(rule)
    
    # ── 辅助函数 ──────────────────────────────────────
    
    def _record_to_causal_memory(self, rule: GovernanceRule):
        """写入因果记忆（连接回路）"""
        entry = {
            "action": f"[治理转换] {rule.description[:80]}",
            "prediction": f"治理规则应拦截{rule.source_mechanism}类失败",
            "actual": f"rule_id={rule.rule_id}, status={rule.status.value}",
            "lesson": f"治理转换: {rule.source_mechanism}→{rule.family.value}",
            "timestamp": time.time(),
            "verifier_cause": "governance_conversion",
            "mechanism": rule.source_mechanism,
            "prediction_match": None,
            "result_success": None,
        }
        
        causal_path = self._storage_dir / "causal_memory.jsonl"
        causal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(causal_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def _generalize_error(self, error: str) -> str:
        """将具体错误泛化为模式（复用failure_tracker逻辑）"""
        import re
        pattern = error
        pattern = re.sub(r'/[\w/.-]+', '<PATH>', pattern)
        pattern = re.sub(r'\d{4,}', '<NUM>', pattern)
        pattern = re.sub(r'[a-f0-9]{8,}', '<HASH>', pattern)
        pattern = re.sub(r'"[^"]{20,}"', '"<LONG_STR>"', pattern)
        return pattern[:200]
    
    # ── 拒绝接口 ──────────────────────────────────────

    def reject(self, instruction: str, reason: str,
               belief_confidence: float = 0.0) -> RejectionRecord:
        """拒绝一个指令。IOS是唯一有权对任何指令说不的体。

        Args:
            instruction: 被拒绝的指令
            reason: 拒绝原因
            belief_confidence: ISA信念系统对该决策的置信度（0~1）

        Returns:
            RejectionRecord 记录
        """
        record = RejectionRecord(
            instruction=instruction,
            reason=reason,
            timestamp=time.time(),
            belief_confidence=belief_confidence,
        )
        logger.warning(f"IOS拒绝: {reason} (instruction={instruction[:80]})")
        return record

    # ── 查询接口 ──────────────────────────────────────
    
    def get_stats(self) -> dict:
        """获取治理引擎统计"""
        rule_stats = self.rule_store.get_stats()
        classifier_stats = self.classifier.get_stats()
        
        return {
            "rules": rule_stats,
            "classifier": classifier_stats,
            "conversion_history": self._count_conversion_history(),
        }
    
    def _count_conversion_history(self) -> dict:
        """统计治理转换历史"""
        causal_path = self._storage_dir / "causal_memory.jsonl"
        if not causal_path.exists():
            return {"total_conversions": 0}
        
        conversions = 0
        try:
            with open(causal_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("verifier_cause") == "governance_conversion":
                            conversions += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except IOError:
            pass
        
        return {"total_conversions": conversions}


# ── G6 委派防护（EdgeCitadel三重防护）──────────────────────

class DelegationGuard:
    """委派防护器——防止递归/循环委派。
    
    三重防护（论文EdgeCitadel arXiv:2606.14710）:
    1. chain_depth ≤ 5: 深度硬限制
    2. content_hash dedup: 循环检测（前200字符hash）
    3. timeout = 90s: 单次委派超时
    
    防御性编程: 当前delegate_task是单层(max_spawn_depth=1)，
    但未来开放嵌套时直接有防护。
    """
    
    MAX_CHAIN_DEPTH = 5
    CONTENT_HASH_LENGTH = 200
    TIMEOUT_SECONDS = 90
    
    def __init__(self):
        import hashlib as _hl
        self._hashlib = _hl
        self._call_stack: list[str] = []  # 当前调用栈的content hashes
        self._depth = 0
    
    def _deterministic_hash(self, content: str) -> str:
        """确定性hash（不依赖PYTHONHASHSEED）"""
        return self._hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def check(self, tool_name: str, kwargs: dict) -> tuple[bool, str]:
        """检查委派是否允许。
        
        Returns:
            (allowed, reason)
        """
        # 1. 深度检查
        if self._depth >= self.MAX_CHAIN_DEPTH:
            return False, f"chain_depth={self._depth} ≥ {self.MAX_CHAIN_DEPTH}: 委派深度超限"
        
        # 2. 循环检测（content hash）
        content = f"{tool_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        content_hash = self._deterministic_hash(content[:self.CONTENT_HASH_LENGTH])
        
        if content_hash in self._call_stack:
            return False, f"循环检测: 相同content hash={content_hash}已在调用栈中"
        
        return True, "通过"
    
    def push(self, tool_name: str, kwargs: dict):
        """入栈: 委派开始"""
        content = f"{tool_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        content_hash = self._deterministic_hash(content[:self.CONTENT_HASH_LENGTH])
        self._call_stack.append(content_hash)
        self._depth += 1
    
    def pop(self):
        """出栈: 委派结束"""
        if self._call_stack:
            self._call_stack.pop()
        self._depth = max(0, self._depth - 1)
    
    def get_depth(self) -> int:
        return self._depth


# ── ④ 审计日志（SQLite链式hash）────────────────────────────

class GovernanceAuditLog:
    """治理审计日志——append-only SQLite + 链式hash验证。
    
    存储: ~/.hermes/hermes.db (governance_audit表)
    链式hash: 每行prev_hash = sha256(前一行内容)
    """
    
    def __init__(self):
        import sqlite3
        self._db_path = os.path.expanduser("~/.hermes/hermes.db")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # 写入并发安全
        self._ensure_table()
    
    def _ensure_table(self):
        self._conn.execute("""CREATE TABLE IF NOT EXISTS governance_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            agent_id TEXT DEFAULT '',
            action TEXT DEFAULT '',
            prev_hash TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            details TEXT DEFAULT ''
        )""")
        self._conn.commit()
    
    def _get_last_hash(self) -> str:
        cur = self._conn.execute(
            "SELECT prev_hash FROM governance_audit ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else "genesis"
    
    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def append(self, event_type: str, action: str = "", 
               agent_id: str = "", session_id: str = "", 
               details: str = "") -> int:
        """追加审计事件，返回行id。"""
        prev_hash = self._get_last_hash()
        # 链式hash: 当前hash = sha256(事件类型 + 动作 + agent + 前hash)
        # 不依赖timestamp，以便verify_chain可以重算
        content = f"{event_type}:{action}:{agent_id}:{prev_hash}"
        current_hash = self._compute_hash(content)
        
        cur = self._conn.execute(
            """INSERT INTO governance_audit 
               (timestamp, event_type, agent_id, action, prev_hash, session_id, details)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), event_type, agent_id, action, current_hash, 
             session_id, details)
        )
        self._conn.commit()
        return cur.lastrowid or 0
    
    def verify_chain(self) -> tuple[bool, int]:
        """验证链式hash完整性。返回 (valid, broken_at_id)。
        
        验证: 重算每行hash并与存储值对比。
        """
        cur = self._conn.execute(
            "SELECT id, event_type, agent_id, action, prev_hash "
            "FROM governance_audit ORDER BY id"
        )
        rows = cur.fetchall()
        
        if not rows:
            return True, -1
        
        prev_hash = "genesis"
        for row in rows:
            id_, etype, agent, action, p_hash = row
            # 重算hash
            content = f"{etype}:{action}:{agent}:{prev_hash}"
            expected = self._compute_hash(content)
            if p_hash != expected:
                return False, id_
            prev_hash = p_hash
        
        return True, -1
    
    def query(self, event_type: str = "", limit: int = 50) -> list[dict]:
        """查询审计事件。"""
        if event_type:
            cur = self._conn.execute(
                "SELECT * FROM governance_audit WHERE event_type=? ORDER BY id DESC LIMIT ?",
                (event_type, limit)
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM governance_audit ORDER BY id DESC LIMIT ?",
                (limit,)
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── ⑤ Heuristics消费闭环 ─────────────────────────────────

class HeuristicsConsumer:
    """经验消费器——从jiak卡片检索relevant heuristics并注入context。
    
    赫尔墨斯启示: 有存无取=图书馆没人去。
    在main_loop Phase 1（任务分解后）调用，检索top_k=3条relevant heuristics。
    
    格式: "⚠️历史经验：{trigger_semantic}→{action}"
    """
    
    def __init__(self):
        self._cards_dir = Path.home() / ".hermes" / "jiak" / "cards"
        self._cache: dict[str, tuple[float, list]] = {}  # path → (mtime, decisions)
    
    def _load_cards(self) -> list[dict]:
        """加载卡片（带文件级缓存）。返回完整卡片而非仅decisions。"""
        all_cards = []
        if not self._cards_dir.exists():
            return all_cards
        
        for card_file in self._cards_dir.glob("*.json"):
            try:
                mtime = card_file.stat().st_mtime
                path_str = str(card_file)
                
                if path_str in self._cache:
                    cached_mtime, cached_cards = self._cache[path_str]
                    if abs(cached_mtime - mtime) < 0.01:
                        all_cards.extend(cached_cards)
                        continue
                
                with open(card_file) as f:
                    card = json.load(f)
                if not isinstance(card, dict):
                    continue
                self._cache[path_str] = (mtime, [card])
                all_cards.append(card)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        
        return all_cards
    
    def _chinese_overlap(self, text: str, query: str) -> float:
        """中文字符级重叠匹配。返回0~1的相似度。"""
        if not text or not query:
            return 0.0
        # 提取中文字符集
        cn_chars = set(c for c in text if '\u4e00' <= c <= '\u9fff')
        q_chars = set(c for c in query if '\u4e00' <= c <= '\u9fff')
        if not cn_chars or not q_chars:
            return 0.0
        overlap = cn_chars & q_chars
        return len(overlap) / len(q_chars)
    
    def retrieve(self, task_description: str, top_k: int = 3) -> list[dict]:
        """检索relevant heuristics。
        
        三路匹配：关键词(权重3) + 中文字符重叠(权重2) + 英文词重叠(权重1)
        """
        task_lower = task_description.lower()
        candidates = []
        
        for card in self._load_cards():
            score = 0.0
            topic = card.get("topic", "")
            keywords = card.get("keywords", [])
            decisions = card.get("decisions", [])
            
            # 路径1: 关键词匹配（权重3）
            for kw in keywords:
                if isinstance(kw, str) and kw.lower() in task_lower:
                    score += 3.0
                elif isinstance(kw, str):
                    # 中文关键词字符级匹配
                    score += self._chinese_overlap(kw, task_description) * 3.0
            
            # 路径2: topic匹配（权重2）
            if topic and isinstance(topic, str):
                score += self._chinese_overlap(topic, task_description) * 2.0
                # 英文topic
                if topic.lower() in task_lower:
                    score += 2.0
            
            # 路径3: decisions匹配（权重1）
            for dec in decisions:
                content = ""
                if isinstance(dec, str):
                    content = dec
                elif isinstance(dec, dict):
                    content = dec.get("content", "")
                if content and isinstance(content, str):
                    # 中文内容字符级匹配
                    score += self._chinese_overlap(content, task_description) * 1.0
                    # 英文内容词匹配
                    words = content.lower().split()
                    score += sum(0.5 for w in words if len(w) > 1 and w in task_lower)
            
            if score > 0:
                candidates.append({
                    "score": score,
                    "content": decisions[0] if decisions else "",
                    "topic": topic,
                    "keywords": keywords[:5],
                })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
    def format_for_context(self, heuristics: list[dict]) -> str:
        """格式化为context注入字符串。"""
        if not heuristics:
            return ""
        
        lines = ["⚠️历史经验:"]
        for h in heuristics:
            trigger = h.get("trigger", {})
            if isinstance(trigger, dict):
                semantic = trigger.get("semantic", h["content"])
            else:
                semantic = str(trigger)
            lines.append(f"  - {semantic}")
        return "\n".join(lines)
