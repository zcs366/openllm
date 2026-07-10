"""
openLLM IOS — 决策·进化·恢复（从main_loop.py提取）

包含：risk_check / arbitrate / learn_causal / evolve / recover / cap_check
集成：CapPolicy / Kernel / Gate / ToolScope / GovernanceEngine / RejectionMechanism
"""

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .models import (Context, Prediction, RiskAssessment, Proposal,
                     Critique, Decision, ActionResult, CausalDelta)
class IOS:
    """决策框架 + 自我进化 + 错误恢复

    集成老IO-S治理模式：
    - CapPolicy: 能力策略（零root）
    - Kernel: 微内核dispatch
    - Gate: 5模式权限门控
    - ToolScope: 工具作用域治理
    - Hindsight: 经验闭环
    - Pipeline: 三层安全管线
    - Checkpoint: Region注册+增量快照
    """
    
    def __init__(self):
        self.evolution_log: list[dict] = []
        self._arbiter_policy = "conservative"

        # ── 老IO-S治理模式集成 ──
        from .cap_policy import CapPolicy
        from .kernel import Kernel
        from .gate import PermissionGate, PermissionMode
        from .tool_scope import ToolScopeManager, ToolScope
        from .checkpoint_manager import CheckpointManager
        # ── 治理转换引擎（P0: arXiv:2607.01087） ──
        from .governance_engine import GovernanceEngine

        self.cap_policy = CapPolicy()
        self.kernel = Kernel(agent_id="openllm-ios")
        self.gate = PermissionGate(mode=PermissionMode.DEV)
        self.tool_scope = ToolScopeManager(scope=ToolScope.DEV)
        self.checkpoint = CheckpointManager(
            interval=300, auto_start=False)
        self.governance_engine = GovernanceEngine()
        
        # ⑤ Heuristics消费闭环（赫尔墨斯启示）
        from .governance_engine import HeuristicsConsumer
        self.heuristics_consumer = HeuristicsConsumer()

        # 兼容旧接口
        self.cap = self.cap_policy

        # [进化] 拒绝权——被否决的proposal可申诉
        try:
            from ..governance.rejection import RejectionMechanism, RejectionReason
            self._rejection_engine = RejectionMechanism()
            self._RejectionReason = RejectionReason
        except Exception:
            self._rejection_engine = None
            self._RejectionReason = None

        # 因果记忆
        self.causal_memory: list[dict] = []

        print(
            f"  IO-S 决策·进化·恢复就绪 "
            f"· 策略={self._arbiter_policy} "
            f"· zero_root={self.cap_policy.is_zero_root()} "
            f"· 治理引擎=active")
    
    def cap_check(self, resource: str, operation: str) -> bool:
        """硬权限检查 — 使用CapPolicy"""
        ok, _ = self.cap_policy.check("agent", operation, resource)
        return ok
    
    def learn_causal(self, ctx: Context, prediction: Prediction,
                     result: ActionResult, delta: CausalDelta):
        """Phase 8: 因果学习 — Self-Harness Weakness Mining + 治理转换
        
        基于 Self-Harness (arXiv 2606.09498):
        - 提取 (verifier_cause, agent_behavior, mechanism) 失败签名
        - 按 mechanism 聚类（非关键词匹配，是语义归因）
        - 记录到 causal_memory + 持久化到磁盘
        
        新增（P0: arXiv:2607.01087 治理转换引擎）:
        - 结构性失败自动触发治理转换
        - ambiguous失败发GovernanceRequest
        """
        entry = {
            "action": ctx.user_message[:100],
            "prediction": prediction.summary[:100],
            "actual": result.output[:100] if result.output else "",
            "lesson": delta.delta_summary,
            "timestamp": time.time(),
            # Self-Harness Weakness Mining 三元组
            "verifier_cause": self._classify_verifier_cause(result, delta),
            "mechanism": self._classify_mechanism(result, delta),
            "prediction_match": delta.prediction_match,
            "result_success": result.success,
        }
        self.causal_memory.append(entry)

        # 持久化到磁盘
        causal_path = Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl"
        causal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(causal_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # 连接4: 回流到jiak world_evolver mismatch_log（五体→记忆层信号传递）
        if not result.success:
            jiak_path = Path.home() / ".hermes" / "jiak" / "mismatch_log.jsonl"
            jiak_path.parent.mkdir(parents=True, exist_ok=True)
            with open(jiak_path, "a") as f:
                f.write(json.dumps({
                    "ts": entry.get("timestamp"),
                    "card_id": "openllm-ios",
                    "prediction_ref": "",
                    "actual": f"[{entry.get('mechanism', 'unknown')}] {result.error[:80] if result.error else result.output[:80]}",
                    "severity": "high" if entry.get("mechanism") in ("tool_loop", "permission", "timeout") else "medium",
                }, ensure_ascii=False) + "\n")
        
        # ── 治理转换（P0: Phase 8.5）──
        if not result.success:
            self._convert_governance(entry, result)

        # ── Hindsight经验闭环（老IO-S模式）──
        from .hindsight_loop import extract_hindsight
        hindsight = extract_hindsight(
            pid="ios",
            goal=ctx.user_message[:200],
            result={
                "success": result.success,
                "execution_time": result.duration_ms / 1000,
                "error": result.error,
            },
            failure=result.error if not result.success else "",
        )
        if hindsight.get("failure_pattern"):
            entry["hindsight_pattern"] = hindsight["failure_pattern"]
            entry["hindsight_alternative"] = hindsight.get(
                "alternative", "")

        # ── P0/P1增强：learn_causal_adapter接入 ──
        if not result.success:
            try:
                import importlib.util as _ilu
                _isa_path = Path.home() / "projects" / "isa" / "learn_causal_adapter.py"
                _lca_spec = _ilu.spec_from_file_location("learn_causal_adapter", _isa_path)
                if _lca_spec and _lca_spec.loader:
                    _lca_mod = _ilu.module_from_spec(_lca_spec)
                    _lca_spec.loader.exec_module(_lca_mod)
                    LearnCausalAdapter = _lca_mod.LearnCausalAdapter
                adapter = LearnCausalAdapter()
                adapter.on_failure(
                    action=entry.get("action", ""),
                    error=result.error or "",
                    mechanism=entry.get("mechanism", "unknown"),
                    tool_name=entry.get("action", "").split("(")[0] if "(" in entry.get("action", "") else "",
                )
                adapter.close()
            except Exception as e:
                print(f"  learn_causal_adapter failed (non-fatal): {e}")
    
    # ── Self-Harness Weakness Mining 工具 ──
    
    _MECHANISM_KEYWORDS = {
        "tool_loop": ["重复", "循环", "loop", "retry", "反复", "无限"],
        "missing_artifact": ["缺失", "没有生成", "未创建", "missing", "not found", "未产出"],
        "wrong_format": ["格式错误", "format error", "解析失败", "parse error", "json解析", "语法错误"],
        "dependency_missing": ["依赖缺失", "import error", "module not found", "package未安装", "ModuleNotFoundError"],
        "timeout": ["超时", "timeout", "hang", "卡住", "无响应"],
        "logic_error": ["逻辑错误", "条件错误", "判断错误", "分支错误", "错误结果", "断言失败"],
        "permission": ["权限", "permission denied", "forbidden", "access denied"],
        "state_corruption": ["脏数据", "不一致", "corrupt", "状态损坏"],
        "exploration_loop": ["探索循环", "搜索失败", "盲目搜索", "找不到目标"],
        "premature_success": ["过早完成", "提前结束", "未验证成功"],
    }
    
    def _classify_mechanism(self, result: ActionResult, delta: CausalDelta) -> str:
        """分类失败机制（Self-Harness failure signature mi）"""
        text = (result.error or "") + " " + (result.output[:200] or "")
        for mech, keywords in self._MECHANISM_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return mech
        return "unknown" if not result.success else "success"
    
    def _classify_verifier_cause(self, result: ActionResult, delta: CausalDelta) -> str:
        """分类verifier层面原因（Self-Harness failure signature ci）"""
        if not result.success:
            return "execution_error" if result.error else "unknown_failure"
        if not delta.prediction_match:
            return "prediction_mismatch"
        return "success"
    
    def _convert_governance(self, entry: dict, result: ActionResult):
        """Phase 8.5: 治理转换 — arXiv:2607.01087 核心循环
        
        从失败中发现治理需求，转换为持久治理规则。
        
        流程:
        1. 构造failure trace
        2. 调用GovernanceEngine.convert()
        3. 记录转换结果
        """
        trace = {
            "tool_name": entry.get("action", "")[:50],
            "error": result.error or "",
            "mechanism": entry.get("mechanism", ""),
            "context": {
                "action": entry.get("action", ""),
                "prediction": entry.get("prediction", ""),
            },
        }
        
        try:
            conversion = self.governance_engine.convert(trace)
            
            if conversion.classification == "structural" and conversion.rule:
                rule = conversion.rule
                print(f"  IO-S 治理转换: {entry.get('mechanism')} → "
                      f"{rule.family.value} (rule_id={rule.rule_id}, "
                      f"status={rule.status.value})")
                
                # 记录到evolution_log
                self.evolution_log.append({
                    "type": "governance_conversion",
                    "mechanism": entry.get("mechanism"),
                    "rule_id": rule.rule_id,
                    "rule_family": rule.family.value,
                    "rule_status": rule.status.value,
                    "classification": conversion.classification,
                    "timestamp": time.time(),
                })
                
                # 审计日志：治理转换事件
                try:
                    from .governance_engine import GovernanceAuditLog
                    audit = GovernanceAuditLog()
                    audit.append(
                        event_type="governance_conversion",
                        action=f"{rule.family.value}:{rule.status.value}",
                        agent_id="ios",
                        details=f"mechanism={entry.get('mechanism')}, rule_id={rule.rule_id}"
                    )
                except Exception:
                    pass  # 审计失败不应影响主循环
            
            elif conversion.classification == "ambiguous" and conversion.governance_request:
                req = conversion.governance_request
                print(f"  IO-S 治理请求: {entry.get('mechanism')} → "
                      f"ambiguous (request_id={req.request_id})")
                
                self.evolution_log.append({
                    "type": "governance_request",
                    "mechanism": entry.get("mechanism"),
                    "request_id": req.request_id,
                    "classification": conversion.classification,
                    "timestamp": time.time(),
                })
        
        except Exception as e:
            # 治理转换失败不应影响主循环
            print(f"  IO-S 治理转换异常（不影响主循环）: {e}")
    
    def _check_governance_rules(
        self, ctx: Context, prediction: Prediction
    ) -> Optional[RiskAssessment]:
        """查询GovernanceRuleStore中匹配当前上下文的active规则。
        
        如果有active规则匹配当前操作，返回对应的风险评估。
        这是治理转换引擎的"消费端"——治理规则真正影响Agent行为。
        """
        try:
            store = self.governance_engine.rule_store
            active_rules = store.get_active()
            
            if not active_rules:
                return None
            
            # 匹配规则：target作为子串出现在消息中（非单词交集）
            msg_lower = ctx.user_message.lower()
            matched_rules = []
            
            for rule in active_rules:
                target = rule.target.lower().strip()
                if not target or target == "general":
                    continue
                # 子串匹配：target完整出现在消息中
                if target in msg_lower:
                    matched_rules.append(rule)
            
            if not matched_rules:
                return None
            
            # 根据匹配规则评估风险
            high_confidence = [r for r in matched_rules if r.confidence >= 0.7]
            if high_confidence:
                rule_descs = [r.description[:60] for r in high_confidence[:3]]
                return RiskAssessment(
                    level="high",
                    blocked=True,
                    reason=f"治理规则拦截: {len(high_confidence)}条active规则匹配",
                    details=rule_descs,
                )
            
            medium_confidence = [r for r in matched_rules if r.confidence >= 0.5]
            if medium_confidence:
                rule_descs = [r.description[:60] for r in medium_confidence[:3]]
                return RiskAssessment(
                    level="medium",
                    blocked=False,
                    reason=f"治理规则提示: {len(medium_confidence)}条规则匹配",
                    details=rule_descs,
                )
        
        except Exception:
            pass  # 治理规则检查失败不影响主循环
        
        return None
    
    def risk_check(self, ctx: Context, prediction: Prediction) -> RiskAssessment:
        """Phase 4: 风险评估 — 血管接线engine.py验证链 + 因果历史检索 + 治理规则"""
        # 血管接线：engine.py的安全检查
        from .engine_bridge import check_action, check_tool_risk
        ok, reason = check_action("risk_check")
        if not ok:
            return RiskAssessment(level="high", blocked=True, reason=reason)
        
        # 血管接线：engine.py的ISN风险检查（如有工具调用意图）
        if prediction.risk_signals:
            risk = check_tool_risk("unknown")
            if not risk["pass"]:
                return RiskAssessment(level="high", blocked=True, 
                    reason=f"ISN风险: {risk.get('reason','')}")
        
        # ── 治理规则检查（P0: arXiv:2607.01087）──
        # 查询GovernanceRuleStore中匹配当前上下文的active规则
        governance_risk = self._check_governance_rules(ctx, prediction)
        if governance_risk and governance_risk.is_blocked():
            return governance_risk
        
        # 因果历史检索（Self-Harness升级：mechanism感知 + 关键词匹配）
        related = []
        user_words = set(ctx.user_message.lower().split())
        
        # 先按mechanism匹配（更精准）
        for m in self.causal_memory[-50:]:
            mech = m.get("mechanism", "unknown")
            if mech != "success" and mech != "unknown":
                # 检查当前消息是否涉及同类机制的场景
                action_words = set(m.get("action", "").lower().split())
                word_overlap = len(user_words & action_words)
                if word_overlap >= 2 or (word_overlap >= 1 and m.get("result_success") == False):
                    related.append(m)
        
        # 补充：关键词匹配（兜底）
        if not related:
            for m in self.causal_memory[-50:]:
                action_words = set(m.get("action", "").lower().split())
                if len(user_words & action_words) >= 2:
                    related.append(m)
        
        if related:
            lessons = [r.get("lesson", "") for r in related[-3:]]
            # Self-Harness升级：风险等级与mechanism类型关联
            mechanisms = [r.get("mechanism", "unknown") for r in related[-3:]]
            high_risk_mechs = {"tool_loop", "permission", "timeout", "exploration_loop"}
            if any(m in high_risk_mechs for m in mechanisms):
                return RiskAssessment(level="high", blocked=False,
                    reason=f"因果历史(高风险机制): {len(related)}条, 机制={mechanisms}", details=lessons)
            return RiskAssessment(level="medium", blocked=False,
                reason=f"因果历史: {len(related)}条, 机制={mechanisms}", details=lessons)
        
        # 连接5: 注入learned_skills到context（五体学习→风险评估感知）
        skills_details = []
        try:
            skill_path = Path.home() / ".openllm" / "output" / "isn" / "learned_skills.jsonl"
            if skill_path.exists():
                import json as _json
                with open(skill_path) as f:
                    for line in f:
                        try:
                            sk = _json.loads(line)
                            if sk.get("mechanism") or sk.get("proposal"):
                                skills_details.append(f"[{sk.get('mechanism', '?')}] {sk.get('proposal', sk.get('pattern', ''))[:60]}")
                        except:
                            pass
                skills_details = skills_details[-5:]  # 最近5条
        except Exception:
            pass
        
        if skills_details:
            return RiskAssessment(level="low", blocked=False,
                reason=f"已学习技能: {len(skills_details)}条", details=skills_details)
        
        return RiskAssessment(level="low", blocked=False)
    
    def arbitrate(self, proposal: Proposal, critique: Critique, 
                  risk: Optional[RiskAssessment] = None) -> Decision:
        """
        Phase 6: 仲裁 — 5级策略栈 + 拒绝权记录
        
        Level 1: single — 只用左脑提案，跳过右脑
        Level 2: self_consistency — 左脑多次采样→投票
        Level 3: multi_persona — 左脑不同人格→聚合
        Level 4: debate — 左脑↔️右脑对弈（当前实现）
        Level 5: debate_then_verify — 对弈→第三方验证
        """
        # 风险拦截
        if risk and risk.is_blocked():
            decision = Decision(
                action="deny",
                approved=False,
                reason=f"安全拦截: {risk.reason}",
                risk_ref=risk,
            )
            self._record_rejection(proposal, decision, risk.reason)
            return decision
        
        # 风险驱动策略选择
        level = self._select_strategy(risk)
        
        # Level 1: single — 快速·直接用左脑
        if level == 1:
            return Decision(
                action="execute",
                approved=True,
                reason=f"Level 1: 单左脑提案通过（风险={risk.level if risk else 'low'}）",
                risk_ref=risk,
            )
        
        # Level 2: self_consistency — TODO: M1实现真正的self_consistency
        if level == 2:
            return Decision(
                action="execute",
                approved=True,
                reason=f"Level 2: self_consistency TODO·M1实现（降级为Level 1）",
                risk_ref=risk,
            )
        
        # Level 3: multi_persona — TODO: M1实现真正的multi_persona
        if level == 3:
            return Decision(
                action="execute",
                approved=True,
                reason=f"Level 3: multi_persona TODO·M1实现（降级为Level 1）",
                risk_ref=risk,
            )
        
        # Level 4: debate — 左脑↔️右脑对弈
        if level == 4:
            if critique.verdict == "approve":
                return Decision(action="execute", approved=True, reason="Level 4: 右脑通过", risk_ref=risk)
            if critique.verdict == "reject":
                decision = Decision(action="deny", approved=False, 
                              reason=f"Level 4: 右脑否决: {critique.concerns[0] if critique.concerns else '无理由'}",
                              risk_ref=risk)
                self._record_rejection(proposal, decision, decision.reason)
                return decision
            if critique.verdict == "revise":
                if self._arbiter_policy == "conservative":
                    return Decision(action="revise", approved=False, reason="Level 4: 右脑建议修改，暂缓", risk_ref=risk)
                return Decision(action="execute", approved=True, reason="Level 4: 策略允许存疑执行", risk_ref=risk)
        
        # Level 5: debate_then_verify — 对弈+验证
        if level == 5:
            if critique.verdict == "approve":
                return Decision(action="execute", approved=True, reason="Level 5: 对弈+验证通过", risk_ref=risk)
            else:
                decision = Decision(action="deny", approved=False, 
                              reason=f"Level 5: critical风险需要明确批准，当前={critique.verdict}", risk_ref=risk)
                self._record_rejection(proposal, decision, decision.reason)
                return decision
        
        # 默认放行
        return Decision(action="execute", approved=True, reason="默认放行", risk_ref=risk)
    
    def _record_rejection(self, proposal: Proposal, decision: Decision, reason: str):
        """记录拒绝——通过RejectionEngine创建不可变记录"""
        if not self._rejection_engine:
            return
        try:
            # 映射到RejectionReason
            if "安全" in reason or "拦截" in reason:
                rej_reason = self._RejectionReason.CONSTITUTIONAL
            elif "否决" in reason:
                rej_reason = self._RejectionReason.OUT_OF_SCOPE
            else:
                rej_reason = self._RejectionReason.UNCERTAIN_SAFETY
            
            self._rejection_engine.reject(
                instruction=proposal.content[:200],
                reason=rej_reason,
                reasoning=reason,
                context={"proposal_confidence": proposal.confidence, "action": decision.action},
            )
        except Exception:
            pass  # 拒绝权记录失败不阻塞主流程
    
    def _select_strategy(self, risk: Optional[RiskAssessment]) -> int:
        """风险驱动策略选择"""
        if risk is None:
            return 1
        if risk.level == "low":
            return 1  # 快速
        if risk.level == "medium":
            return 3  # 多样
        if risk.level == "high":
            return 4  # 对弈
        return 5  # critical→对弈+验证
    
    def evolve(self, proposal: Proposal, critique: Critique,
               result: Optional[ActionResult] = None, delta: Optional[CausalDelta] = None):
        """
        Phase 9: 因果进化引擎 — Self-Harness Harness Proposal + Validation
        
        基于 Self-Harness (arXiv 2606.09498):
        1. 记录（保留原有逻辑）
        2. 机制聚类：按mechanism字段聚合失败（非关键词交集）
        3. Harness修改提案：当高频机制≥阈值时，生成五体配置修改建议
        4. 策略自适应：根据失败率调整策略
        """
        # ① 记录
        self.evolution_log.append({
            "timestamp": time.time(),
            "proposal": proposal.content[:100],
            "verdict": critique.verdict,
            "result_success": result.success if result else None,
            "delta_match": delta.prediction_match if delta else None,
        })
        
        # ② 机制聚类（Self-Harness: 按failure signature聚合）
        if result and not result.success and len(self.causal_memory) >= 3:
            recent = self.causal_memory[-20:]
            mechanism_counts: dict[str, int] = {}
            for m in recent:
                mech = m.get("mechanism", "unknown")
                if mech != "success":
                    mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1
            
            # 高频机制（≥3次或占比≥30%）
            total_failures = sum(mechanism_counts.values())
            high_freq = {
                mech: count for mech, count in mechanism_counts.items()
                if count >= 3 or (total_failures > 0 and count / total_failures >= 0.3)
            }
            
            if high_freq:
                # ③ Harness修改提案（Self-Harness Harness Proposal）
                # P0-3回归验证：检查同一机制是否已有pending提案
                prop_path = Path.home() / ".openllm" / "output" / "ios" / "harness_proposals.jsonl"
                existing_mechs = set()
                if prop_path.exists():
                    try:
                        with open(prop_path) as f:
                            for line in f:
                                try:
                                    p = json.loads(line)
                                    if p.get("status") == "pending":
                                        existing_mechs.add(p.get("mechanism", ""))
                                except:
                                    pass
                    except:
                        pass
                
                top_mech = max(high_freq, key=lambda k: high_freq[k])
                if top_mech in existing_mechs:
                    # 已有pending提案，不重复生成
                    pass
                else:
                    proposal_text = self._propose_harness_change(top_mech, high_freq[top_mech])
                    if proposal_text:
                        self.evolution_log.append({
                            "type": "harness_proposal",
                            "mechanism": top_mech,
                            "support": high_freq[top_mech],
                            "proposal": proposal_text,
                            "timestamp": time.time(),
                        })
                        print(f"  IO-S Self-Harness提案: [{top_mech}] {proposal_text[:80]}")
                        
                        # 持久化提案（等待人工审批）
                        prop_path = Path.home() / ".openllm" / "output" / "ios" / "harness_proposals.jsonl"
                        prop_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(prop_path, "a") as f:
                            f.write(json.dumps({
                                "mechanism": top_mech,
                                "support": high_freq[top_mech],
                                "proposal": proposal_text,
                                "timestamp": time.time(),
                                "status": "pending",  # pending/approved/rejected
                            }, ensure_ascii=False) + "\n")
                        
                        # 连接2: 提案回流到causal_memory（闭环：evolve→learn_causal）
                        # 注意：提案不是失败，用 None 标记（非失败非成功），避免污染失败率统计
                        self.causal_memory.append({
                            "action": f"[Self-Harness提案] {top_mech}",
                            "prediction": "高频机制触发",
                            "actual": proposal_text[:100],
                            "lesson": f"机制={top_mech}, 支持={high_freq[top_mech]}次",
                            "timestamp": time.time(),
                            "verifier_cause": "self_harness_proposal",
                            "mechanism": top_mech,
                            "prediction_match": None,  # 非失败非成功，不计入统计
                            "result_success": None,
                        })
                    
                    # ④ 技能固化（血管#3: ios→isn）
                skill_entry = {
                    "name": f"learned_{top_mech}_{len(self.evolution_log)}",
                    "mechanism": top_mech,
                    "support": high_freq[top_mech],
                    "proposal": proposal_text,
                    "created_at": time.time(),
                    "source": "self_harness_evolution",
                }
                skill_path = Path.home() / ".openllm" / "output" / "isn" / "learned_skills.jsonl"
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                with open(skill_path, "a") as f:
                    f.write(json.dumps(skill_entry, ensure_ascii=False) + "\n")
                print(f"  IO-S→ISN 血管#3: 技能已固化 [{top_mech}]")
        
        # ⑤ 策略自适应（保留原有逻辑，排除None：提案回流条目不计入）
        recent = self.evolution_log[-10:]
        failed = sum(1 for e in recent if e.get("result_success") is False)
        total = max(len(recent), 1)
        deny_rate = failed / total
        
        if deny_rate > 0.5 and self._arbiter_policy != "conservative":
            self._arbiter_policy = "conservative"
            print(f"  IO-S 自主调降策略→conservative (失败率{deny_rate:.0%})")
        elif deny_rate < 0.1 and len(self.evolution_log) > 20 and self._arbiter_policy == "conservative":
            self._arbiter_policy = "balanced"
            print(f"  IO-S 自主调松策略→balanced (失败率{deny_rate:.0%})")
    
    def _propose_harness_change(self, mechanism: str, support: int) -> str:
        """Self-Harness Harness Proposal: 为高频失败机制生成五体配置修改建议
        
        不同mechanism映射到不同五体层的修改：
        - tool_loop → ISN工具策略（限制重试次数）
        - missing_artifact → ISN产出验证（完成后检查）
        - exploration_loop → IO-S搜索约束（限制搜索轮次）
        - premature_success → IO-S验证门控（声明成功前必须验证）
        """
        proposals = {
            "tool_loop": f"ISN工具策略: 最大重试3次后切换方案（支持{support}次）",
            "missing_artifact": f"ISN产出验证: 工具执行后检查产出物是否存在（支持{support}次）",
            "wrong_format": f"ISN格式规范: 工具输出前做格式校验（支持{support}次）",
            "dependency_missing": f"ISN依赖检查: 执行前检查依赖是否可用（支持{support}次）",
            "timeout": f"IO-S超时策略: 设置工具执行超时上限（支持{support}次）",
            "logic_error": f"ISA因果提示: 注入类似失败的教训到context（支持{support}次）",
            "exploration_loop": f"IO-S搜索约束: 搜索不超过5轮后必须行动（支持{support}次）",
            "premature_success": f"IO-S验证门控: 声明成功前必须执行验证步骤（支持{support}次）",
        }
        return proposals.get(mechanism, "")
    
    def _extract_common(self, texts: list[str]) -> list[str]:
        """提取多段文本的共同词"""
        if not texts:
            return []
        words_sets = [set(t.lower().split()) for t in texts if t]
        if not words_sets:
            return []
        return list(set.intersection(*words_sets))[:5]
    
    def recover(self, error: Exception) -> bool:
        """错误恢复"""
        print(f"  IO-S 错误恢复: {error}")
        return True  # M0简化：总是恢复成功
