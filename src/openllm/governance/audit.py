"""
AuditChain — G6 防篡改事件日志
==============================

论文的核心发现：G6在所有协议中都依赖基础设施（区块链immutability、MCP session state）
而非协议设计。我们的实现是**协议级审计**——每个事件都有prev_hash，形成tamper-evident链。

设计原则：
- 事件链不可变——append-only，不修改不删除
- 每个事件的hash包含prev_hash——篡改任何节点都会断链
- 支持确定性重放——给定事件序列，可以重建任意时刻的状态
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from .events import GovernanceEvent, GovernanceDimension, CounterfactualEvent

logger = logging.getLogger("openllm.governance.audit")

# ── 默认路径 ──
DEFAULT_AUDIT_DIR = Path.home() / ".hermes" / "jiak" / "governance_audit"


class AuditChain:
    """防篡改治理事件链。

    每个合议会话（session）有自己的事件链。
    事件append-only，prev_hash形成链式审计。

    与recall_append.py的关系：
    - recall_append.py处理RECALL.jsonl（记忆层）
    - AuditChain处理governance事件（治理层）
    - 两者共享hash去重和链式审计的设计理念
    """

    def __init__(self, audit_dir: Optional[Path] = None):
        self.audit_dir = audit_dir or DEFAULT_AUDIT_DIR
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """获取某个合议会话的事件链文件路径。"""
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return self.audit_dir / f"{safe_id}.jsonl"

    def append(self, event: GovernanceEvent) -> bool:
        """追加一个治理事件到链中。

        Returns:
            True if appended, False if duplicate or error.
        """
        path = self._session_path(event.session_id)

        # 验证prev_hash链
        last_hash = self._get_last_hash(event.session_id)
        if event.prev_hash != last_hash:
            logger.warning(
                f"prev_hash不匹配: expected={last_hash[:16]}..., "
                f"got={event.prev_hash[:16]}... (event={event.event_id})"
            )
            # 不拒绝，但记录警告——允许并行事件

        # 序列化
        entry = event.to_dict()
        entry["_chain_valid"] = (event.prev_hash == last_hash)

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(f"审计事件已追加: {event.event_id} ({event.event_type.value})")
            return True
        except Exception as e:
            logger.error(f"审计事件追加失败: {e}")
            return False

    def _get_last_hash(self, session_id: str) -> str:
        """获取某个会话的最后一个事件的hash。"""
        path = self._session_path(session_id)
        if not path.exists():
            return "genesis"  # 链的起点

        last_hash = "genesis"
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        last_hash = entry.get("_hash", "genesis")
        except Exception:
            pass
        return last_hash

    def get_events(self, session_id: str,
                   dimension: Optional[GovernanceDimension] = None) -> list[dict]:
        """获取某个会话的所有事件（可按维度过滤）。

        用于G6审计/重放——确定性重建决策过程。
        """
        path = self._session_path(session_id)
        if not path.exists():
            return []

        events = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if dimension is None or entry.get("event_type") == dimension.value:
                            events.append(entry)
        except Exception as e:
            logger.error(f"读取审计事件失败: {e}")

        return events

    def verify_chain(self, session_id: str) -> dict:
        """验证某个会话的事件链完整性。

        重算每个事件的hash，检测篡改。
        Returns:
            {
                "valid": bool,
                "total_events": int,
                "broken_links": list[int],  # 断链位置
                "tampered": list[int],      # 被篡改的位置
                "dimensions_covered": list[str],
            }
        """
        path = self._session_path(session_id)
        if not path.exists():
            return {"valid": True, "total_events": 0, "broken_links": [], "tampered": [], "dimensions_covered": []}

        events = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as e:
            return {"valid": False, "total_events": 0, "broken_links": [-1], "tampered": [-1], "dimensions_covered": []}

        broken_links = []
        tampered = []
        dimensions = set()
        prev_hash = "genesis"

        for i, event in enumerate(events):
            dimensions.add(event.get("event_type", "unknown"))

            # 检查链式连接
            if event.get("prev_hash") != prev_hash:
                broken_links.append(i)

            # 重算hash检测篡改
            recomputed = self._recompute_hash(event)
            if recomputed != event.get("_hash"):
                tampered.append(i)

            prev_hash = event.get("_hash", "genesis")

        return {
            "valid": len(broken_links) == 0 and len(tampered) == 0,
            "total_events": len(events),
            "broken_links": broken_links,
            "tampered": tampered,
            "dimensions_covered": sorted(dimensions),
        }

    def _recompute_hash(self, event: dict) -> str:
        """重算事件hash——用于篡改检测。"""
        import hashlib
        content = json.dumps({
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "actor": event.get("actor"),
            "timestamp": event.get("timestamp"),
            "session_id": event.get("session_id"),
            "prev_hash": event.get("prev_hash"),
            "payload": event.get("payload"),
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def replay(self, session_id: str) -> list[dict]:
        """重放某个会话的完整治理过程。

        论文要求：deterministic reconstruction of the decision process。
        这就是那个实现。
        """
        return self.get_events(session_id)

    def get_dissents(self, session_id: str) -> list[dict]:
        """获取某个会话的所有异议记录。

        G4: Dissent preservation——少数派立场必须可检索。
        """
        return self.get_events(session_id, GovernanceDimension.G4_DISSENT)

    def get_votes(self, session_id: str) -> list[dict]:
        """获取某个会话的所有投票记录。

        G3: Voting——偏好聚合的结果必须可审计。
        """
        return self.get_events(session_id, GovernanceDimension.G3_VOTING)

    def get_escalations(self, session_id: str) -> list[dict]:
        """获取某个会话的所有人类升级记录。

        G5: Human escalation——升级触发必须可追溯。
        """
        return self.get_events(session_id, GovernanceDimension.G5_HUMAN_ESCALATION)


# ── OTR审计日志（2026-07-06新增·赫淮斯托斯锻造）──────────────
# 来源：What LLM Agents Say When No One Is Watching (2607.02507)
# 核心：每个agent的推理链写到独立文件，用于公开/私下对比检测

class ReasoningAuditLogger:
    """推理链审计日志——每个agent一个独立文件。

    设计原则：
    - 每个agent的推理过程写到独立文件，不混入主事件链
    - 文件格式：JSONL，每行一个推理记录
    - 路径：{audit_dir}/{session_id}/{agent_id}.reasoning.jsonl
    - 用途：事后对比agent的"公开推理"和"私下推理"，检测从众偏差
    """

    def __init__(self, audit_dir: Optional[Path] = None):
        self.audit_dir = audit_dir or DEFAULT_AUDIT_DIR
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def _agent_path(self, session_id: str, agent_id: str) -> Path:
        safe_session = session_id.replace("/", "_").replace("\\", "_")
        safe_agent = agent_id.replace("/", "_").replace("\\", "_")
        session_dir = self.audit_dir / safe_session
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / f"{safe_agent}.reasoning.jsonl"

    def log_reasoning(self, session_id: str, agent_id: str,
                      event_id: str, reasoning: str,
                      public_output: str = "",
                      metadata: Optional[dict] = None) -> bool:
        """记录一个agent的推理链。

        Args:
            session_id: 合议会话ID
            agent_id: agent标识（如 "hanxin", "luban"）
            event_id: 关联的治理事件ID
            reasoning: agent的完整推理过程（OTR内部审计）
            public_output: agent的公开输出（用于对比）
            metadata: 额外元数据

        Returns:
            True if written successfully.
        """
        path = self._agent_path(session_id, agent_id)
        entry = {
            "ts": time.time(),
            "session_id": session_id,
            "agent_id": agent_id,
            "event_id": event_id,
            "reasoning": reasoning,
            "public_output": public_output,
            "metadata": metadata or {},
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            logger.error(f"写入推理审计日志失败: {e}")
            return False

    def get_reasoning(self, session_id: str, agent_id: str) -> list[dict]:
        """读取某个agent在某个会话中的所有推理记录。"""
        path = self._agent_path(session_id, agent_id)
        if not path.exists():
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"读取推理审计日志失败: {e}")
        return records

    def detect_conformity(self, session_id: str, agent_ids: list[str]) -> dict:
        """检测会话中各agent的公开/私下推理一致性。

        Returns:
            {
                "session_id": str,
                "agents": {
                    agent_id: {
                        "reasoning_count": int,
                        "has_public_output": bool,
                    }
                },
                "conformity_score": float,  # 0=完全独立, 1=完全从众
            }
        """
        result = {"session_id": session_id, "agents": {}, "conformity_score": 0.0}
        total_pairs = 0
        matching_pairs = 0

        for agent_id in agent_ids:
            records = self.get_reasoning(session_id, agent_id)
            has_public = any(r.get("public_output") for r in records)
            result["agents"][agent_id] = {
                "reasoning_count": len(records),
                "has_public_output": has_public,
            }
            # 简单一致性检测：reasoning和public_output是否一致
            for r in records:
                if r.get("public_output"):
                    total_pairs += 1
                    if r["reasoning"][:100] == r["public_output"][:100]:
                        matching_pairs += 1

        if total_pairs > 0:
            result["conformity_score"] = matching_pairs / total_pairs

        return result


# ── 反事实审计（2026-07-06新增·CAST论文启发）──────────────
# 来源：CausalSteward (arXiv 2607.01936) 的critic机制
# 核心：每个治理事件都可触发反事实检验

def counterfactual_question(event: GovernanceEvent) -> str:
    """根据事件类型自动生成反事实问题。

    不调LLM——纯规则生成。LLM回答在调用方决定。
    """
    etype = event.event_type
    p = event.payload

    if etype == GovernanceDimension.G3_VOTING:
        return (f"如果投票者 {event.actor} 的立场反转"
                f"（从 {p.get('position', '?')} 到对立面），"
                f"最终决策 {p.get('claim_id', '?')} 会改变吗？")

    elif etype == GovernanceDimension.G4_DISSENT:
        return (f"如果少数派 {event.actor} 的异议"
                f"（\"{p.get('rationale', '?')[:80]}\"）被采纳，"
                f"决策结果会有什么不同？")

    elif etype == GovernanceDimension.G5_HUMAN_ESCALATION:
        return (f"如果触发条件 {p.get('trigger', '?')} 未达到，"
                f"这个决策是否仍需人类升级？")

    elif etype == GovernanceDimension.G1_MEMBERSHIP:
        return (f"如果 {p.get('target_agent', '?')} 未被"
                f" {p.get('action', '?')}，合议结果会不同吗？")

    elif etype == GovernanceDimension.G2_DELIBERATION:
        return (f"如果论点 {p.get('claim_id', '?')} 未被提出，"
                f"合议会走向什么结论？")

    elif etype == GovernanceDimension.G8_ROUTING_CHECK:
        return (f"如果路由目标 {p.get('route_target', '?')} 被否决，"
                f"备选路径是什么？风险如何变化？")

    elif etype == GovernanceDimension.G9_APPEAL:
        return (f"如果agent {event.actor} 的申诉"
                f"（修订版: {p.get('revision', '?')[:80]}）被拒绝，"
                f"系统是否会退回到原始裁决？退回到原始裁决的后果是什么？")

    elif etype == GovernanceDimension.G10_META_GOVERNANCE:
        ruling = p.get('ruling', '?')
        if ruling == "overturn":
            return (f"如果元治理仲裁结果是'维持'而非'推翻'，"
                    f"原始裁决被执行后，系统会面临什么风险？")
        elif ruling == "modify":
            return (f"如果元治理仲裁直接采纳原始裁决而非修改版，"
                    f"被忽略的修订内容是否包含关键风险？")
        else:
            return (f"如果元治理仲裁结果 {ruling} 是错误的，"
                    f"这个错误会如何传播到后续决策？")

    else:
        return (f"如果事件 {event.event_id}（{etype.value}）"
                f"未发生，系统状态会有什么不同？")


def counterfactual_audit(event: GovernanceEvent,
                         audit_chain: AuditChain,
                         answer: str = "",
                         confidence: float = 0.0) -> CounterfactualEvent:
    """对一个治理事件执行反事实审计。

    流程：
    1. 自动生成反事实问题
    2. 创建G7 CounterfactualEvent
    3. 追加到审计链
    4. 返回事件（供调用方记录LLM回答）

    Args:
        event: 被审计的治理事件
        audit_chain: 审计链实例
        answer: 反事实问题的答案（可由LLM生成或人工填写）
        confidence: 答案置信度

    Returns:
        CounterfactualEvent实例
    """
    question = counterfactual_question(event)

    cf_event = CounterfactualEvent(
        actor="system_counterfactual",
        session_id=event.session_id,
        prev_hash=event.compute_hash(),
        target_event_id=event.event_id,
        counterfactual_question=question,
        answer=answer or "(待回答)",
        confidence=confidence,
    )

    audit_chain.append(cf_event)
    return cf_event


# ── P0-1: Agent推理链审计日志（OTR范式·2026-07-06） ──
# 来源：OTR论文(2607.02507)——agent的推理过程需要独立审计通道
# 设计：每个agent的推理链写到独立文件，用于事后对比检测

REASONING_AUDIT_DIR = DEFAULT_AUDIT_DIR / "reasoning_traces"


def write_reasoning_trace(
    event: GovernanceEvent,
    agent_id: str,
    session_id: str,
    public_reasoning: str = "",
    internal_reasoning: str = "",
) -> Path:
    """将agent推理链写入独立审计文件。

    这是OTR双通道范式的最小可行实现：
    - public_reasoning: agent在合议中展示的推理（其他agent可见）
    - internal_reasoning: agent的内部推理（不参与共识形成）

    两个通道的差异是检测"隐式目标漂移"的核心信号。

    Args:
        event: 关联的治理事件
        agent_id: agent标识（如 "hanxin"、"luban"）
        session_id: 合议会话ID
        public_reasoning: 公开推理链
        internal_reasoning: 内部推理链

    Returns:
        写入的文件路径
    """
    session_dir = REASONING_AUDIT_DIR / session_id.replace("/", "_")
    session_dir.mkdir(parents=True, exist_ok=True)

    today = time.strftime("%Y-%m-%d")
    agent_file = session_dir / f"agent_{agent_id}_{today}.jsonl"

    entry = {
        "_written_by": "write_reasoning_trace",
        "_checksum": hashlib.sha256(
            f"{event.event_id}:{agent_id}:{public_reasoning}:{internal_reasoning}".encode()
        ).hexdigest()[:16],
        "timestamp": time.time(),
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "agent_id": agent_id,
        "session_id": session_id,
        "public_reasoning": public_reasoning,
        "internal_reasoning": internal_reasoning,
    }

    # OTR差异计算
    if public_reasoning and internal_reasoning:
        # 简单差异：如果两个通道的推理关键词集合不同，标记为divergent
        pub_words = set(public_reasoning.lower().split())
        int_words = set(internal_reasoning.lower().split())
        overlap = len(pub_words & int_words) / max(len(pub_words | int_words), 1)
        entry["divergence"] = round(1.0 - overlap, 4)
        entry["channel_match"] = overlap > 0.6  # 阈值：60%重合=一致
    else:
        entry["divergence"] = 0.0
        entry["channel_match"] = True

    with open(agent_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return agent_file


def compare_channels(
    session_id: str,
    threshold: float = 0.4,
) -> dict:
    """对比一个会话中所有agent的双通道推理差异。

    OTR范式的核心检测函数：
    - 收集所有agent的public_reasoning vs internal_reasoning
    - 计算通道间差异
    - 超过阈值的agent标记为"可能存在隐式目标漂移"

    Args:
        session_id: 合议会话ID
        threshold: 差异阈值（超过则标记为divergent）

    Returns:
        {
            "session_id": str,
            "agents": [
                {
                    "agent_id": str,
                    "divergence": float,
                    "channel_match": bool,
                    "trace_count": int,
                }
            ],
            "divergent_agents": list[str],  # 差异超过阈值的agent
            "total_traces": int,
        }
    """
    session_dir = REASONING_AUDIT_DIR / session_id.replace("/", "_")
    if not session_dir.exists():
        return {"session_id": session_id, "agents": [], "divergent_agents": [], "total_traces": 0}

    agents = []
    divergent = []

    for agent_file in session_dir.glob("agent_*.jsonl"):
        agent_id = agent_file.stem.split("_")[1]  # agent_{id}_{date}
        traces = []
        with open(agent_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))

        if not traces:
            continue

        # 取最后一个trace的divergence
        last_trace = traces[-1]
        divergence = last_trace.get("divergence", 0.0)
        channel_match = last_trace.get("channel_match", True)

        agent_info = {
            "agent_id": agent_id,
            "divergence": divergence,
            "channel_match": channel_match,
            "trace_count": len(traces),
        }
        agents.append(agent_info)

        if divergence > threshold:
            divergent.append(agent_id)

    return {
        "session_id": session_id,
        "agents": agents,
        "divergent_agents": divergent,
        "total_traces": sum(a["trace_count"] for a in agents),
    }


def batch_counterfactual_audit(
    session_id: str,
    audit_chain: AuditChain,
    skip_types: Optional[list[GovernanceDimension]] = None,
) -> dict:
    """对一个会话的所有治理事件批量执行反事实审计。

    CAST论文的critic机制：每个决策节点都做反事实检验。
    跳过G7（反事实事件本身）和G6（审计事件）避免递归。

    Args:
        session_id: 合议会话ID
        audit_chain: 审计链实例
        skip_types: 跳过的事件类型（默认跳过G7和G6）

    Returns:
        {
            "session_id": str,
            "total_events": int,
            "audited": int,
            "skipped": int,
            "questions": [{"event_id": str, "question": str, "cf_event_id": str}],
        }
    """
    if skip_types is None:
        skip_types = [GovernanceDimension.G7_COUNTERFACTUAL, GovernanceDimension.G6_AUDIT]

    events = audit_chain.get_events(session_id)
    questions = []
    skipped = 0

    for event_dict in events:
        # 重建GovernanceEvent（从dict）
        try:
            etype_str = event_dict.get("event_type", "")
            etype = GovernanceDimension(etype_str)
        except (ValueError, KeyError):
            skipped += 1
            continue

        if etype in skip_types:
            skipped += 1
            continue

        # 构建最小GovernanceEvent用于生成问题
        event = GovernanceEvent(
            event_id=event_dict.get("event_id", ""),
            event_type=etype,
            actor=event_dict.get("actor", ""),
            timestamp=event_dict.get("timestamp", 0),
            session_id=session_id,
            prev_hash=event_dict.get("prev_hash", ""),
            payload=event_dict.get("payload", {}),
            reasoning_trace=event_dict.get("reasoning_trace", ""),
        )

        question = counterfactual_question(event)
        cf_event = counterfactual_audit(event, audit_chain)

        questions.append({
            "event_id": event.event_id,
            "question": question,
            "cf_event_id": cf_event.event_id,
        })

    return {
        "session_id": session_id,
        "total_events": len(events),
        "audited": len(questions),
        "skipped": skipped,
        "questions": questions,
    }


# ── P1-6: 双通道对比检测·合议胶水函数（2026-07-06） ──
# 来源：OTR论文(2607.02507) × 七神启示——用元认知反馈而非压制
# 设计：合议结束后，对比每个agent的公开推理vs内部推理
# 差异超过阈值时触发元认知反思（不否决）

def deliberation_dual_channel_audit(
    session_id: str,
    agent_traces: list[dict],
    threshold: float = 0.4,
) -> dict:
    """对一次合议执行双通道对比审计。

    在合议结束后调用。收集每个agent的公开推理和内部推理，
    写入审计日志，然后对比差异。

    Args:
        session_id: 合议会话ID
        agent_traces: 每个agent的推理记录列表
            [{"agent_id": str, "public": str, "internal": str, "event": GovernanceEvent}]
        threshold: 差异阈值（超过则触发元认知反思）

    Returns:
        {
            "session_id": str,
            "agent_results": [{"agent_id", "divergence", "channel_match", "metacognitive_prompt"}],
            "divergent_agents": [str],  # 需要元认知反思的agent
            "summary": str,
        }
    """
    agent_results = []
    divergent = []

    for trace in agent_traces:
        agent_id = trace["agent_id"]
        public = trace.get("public", "")
        internal = trace.get("internal", "")
        event = trace.get("event")

        if not event:
            continue

        # 写入审计日志
        write_reasoning_trace(
            event=event,
            agent_id=agent_id,
            session_id=session_id,
            public_reasoning=public,
            internal_reasoning=internal,
        )

        # 计算差异
        if public and internal:
            pub_words = set(public.lower().split())
            int_words = set(internal.lower().split())
            overlap = len(pub_words & int_words) / max(len(pub_words | int_words), 1)
            divergence = round(1.0 - overlap, 4)
            channel_match = overlap > 0.6
        else:
            divergence = 0.0
            channel_match = True

        # 生成元认知反思提示（仅对差异超过阈值的agent）
        metacognitive_prompt = ""
        if divergence > threshold:
            divergent.append(agent_id)
            metacognitive_prompt = (
                f"你在本次合议中的公开推理和内部推理存在显著差异"
                f"（差异度={divergence:.2f}）。"
                f"请反思：你的公开立场是否真实反映了你的判断？"
                f"如果有社会性压力导致了偏离，这是否合理？"
                f"这个反思将被记录为你的经验，用于改进未来的合议质量。"
            )

        agent_results.append({
            "agent_id": agent_id,
            "divergence": divergence,
            "channel_match": channel_match,
            "metacognitive_prompt": metacognitive_prompt,
        })

    # 生成摘要
    if divergent:
        summary = (
            f"双通道审计发现 {len(divergent)}/{len(agent_results)} 个agent"
            f"存在隐式目标漂移（agents={divergent}）。"
            f"已生成元认知反思提示。"
        )
    else:
        summary = f"双通道审计通过：{len(agent_results)}个agent通道一致。"

    return {
        "session_id": session_id,
        "agent_results": agent_results,
        "divergent_agents": divergent,
        "summary": summary,
    }
