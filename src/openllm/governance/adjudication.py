"""adjudication.py — 传输裁决模块（P0-1 · "该不该传/醒"）

七神启示终裁（2026-09-04）钉死的架构判断：多 agent 通信缺的不是管道，
是裁决"该不该传"的那一层——这是架构层缺失，不是模型能力问题。本模块
即该层的 v0 工程实现，落 governance（IO-S 治理层），可被五体/ISA/壳
适配器调用，不寄生任何壳。

分层（层层否决制）：
  L0 硬规则层 —— 一票否决（红线类型/凭据模式/黑名单/频控硬顶）
  L1 信号层   —— 软评分加权（importance 自评 / semantic 相关度 / 冷却期）
  L2/L3       —— 治理策略 + 人类升级（v1+；接口预留 action 枚举）

失败安全分层（内部放宽、对外收紧）：
  裁决器本身异常 → channel=internal 放行（感知不阻断）、否则 BLOCK（防失控）。
  信号依赖异常（semantic 挂）→ 该信号中性化（0.5），不瘫痪裁决。

判据一句话：传，当且仅当接收方的世界模型会因此变得更准，且代价可担。
（对齐成市 AI 哲学：智能 = 对世界越来越准确的预测）

DR-20260904-01 · v0.1 · 2026-09-04
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.governance.adjudication")
DEFAULT_LOG_DIR = Path.home() / ".openllm" / "adjudication"

# L0 · 红线类型（对齐 event_bus Event._BLOCKED —— 任何体不得终止对话）
REDLINE_TYPES = frozenset({"terminate", "shutdown", "kill", "abort"})

# L0 · 凭据/密钥模式轻量检测（对齐凭据防火墙精神——受限内容不外传）
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),        # OpenAI/Anthropic 风格密钥
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),           # AWS Access Key
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),       # GitHub PAT
    re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{10,})\b"),  # Slack token
    re.compile(r"(api[_-]?key|secret|token|password)\s*[:=]\s*\S{8,}", re.I),
]


# ═══════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TransmissionRequest:
    """一次待裁决的传输/唤醒请求。"""
    from_agent: str           # 发起方（IAI / IO-S / 军师 / ...）
    to_agent: Optional[str]   # 接收方（None=广播）
    msg_type: str             # message / wake / wave / insight / heartbeat ...
    body: str = ""            # 内容（原文或摘要）
    importance: float = 0.5   # 发起方自评 [0,1]——自评不可信，只作低权重参考
    urgency: float = 0.0      # 时效紧迫度 [0,1]（v0 未参与评分，留接口）
    channel: str = "internal" # internal / im / shell / ...（对外传输收紧）
    brain_id: str = "default"
    context: dict = field(default_factory=dict)   # 扩展（引用/前序等）


@dataclass
class Verdict:
    """裁决结果。action: PASS / BLOCK / DEFER / THROTTLE（v0 实际产出 PASS/BLOCK）。"""
    action: str
    reason: str
    policy_hits: list[str] = field(default_factory=list)
    confidence: float = 1.0
    wait_seconds: int = 0
    adjudicator_version: str = "0.1.0"
    score: float = 0.0        # L1 软评分（BLOCK by L0 时为 0）


@dataclass
class PolicyRule:
    """一条自定义硬规则。check 命中返回 reason 字符串，未命中返回 None。"""
    rule_id: str
    check: Callable[[TransmissionRequest], Optional[str]]

    def __call__(self, req: TransmissionRequest) -> Optional[str]:
        return self.check(req)


# ═══════════════════════════════════════════════════════════════════
# 频控追踪（L0 硬顶数据源）
# ═══════════════════════════════════════════════════════════════════

class _RateTracker:
    """滑动窗口计数：同 (from,to) 在窗口内的事件次数。"""
    def __init__(self, window_sec: float = 60.0):
        self._window = window_sec
        self._buckets: dict[tuple, deque] = defaultdict(deque)

    def record(self, key: tuple, now: float) -> int:
        dq = self._buckets[key]
        cutoff = now - self._window
        while dq and dq[0] < cutoff:
            dq.popleft()
        dq.append(now)
        return len(dq)

    def count(self, key: tuple, now: float) -> int:
        dq = self._buckets[key]
        cutoff = now - self._window
        return sum(1 for ts in dq if ts >= cutoff)

    def reset(self):
        self._buckets.clear()


# ═══════════════════════════════════════════════════════════════════
# 裁决器
# ═══════════════════════════════════════════════════════════════════

class TransmissionAdjudicator:
    """传输裁决器 v0：L0 硬规则一票否决 + L1 信号加权评分。

    L1 评分 = Σ(w_i · sig_i) / Σ(w_i)。中性基线（importance=0.5/semantic=0.5/
    冷却已过 rec=1.0）时 score=0.65 —— 默认信任放行（安全由 L0 把关）；
    semantic 高相关抬分至 0.85；冷却未过 rec=0 → 0.35，显著压到阈值下。
    依赖全部可注入：semantic（SemanticMatcher 兼容 similarity）、
    interests（接收方兴趣文本）、now（时钟，测试用）、log_dir（审计路径）。
    """

    DEFAULT_WEIGHTS = {"importance": 0.2, "semantic": 0.5, "recency": 0.3}

    def __init__(
        self,
        *,
        log_dir: Optional[Path] = None,
        semantic: Any = None,                # 可选：语义匹配器（similarity 接口）
        interests: Optional[dict[str, str]] = None,  # 接收方兴趣文本 {to_agent: interest}
        pass_threshold: float = 0.5,
        weights: Optional[dict[str, float]] = None,
        rate_limit: int = 10,                # L0 频控硬顶（窗口内次数）
        rate_window_sec: float = 60.0,
        min_interval_sec: Optional[float] = None,   # L1 冷却期（None=关闭）
        now: Optional[Callable[[], float]] = None,  # 时钟注入（测试）
    ) -> None:
        self._log_dir = log_dir or DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "adjudication.jsonl"
        self._semantic = semantic
        self._interests = dict(interests or {})
        self._threshold = pass_threshold
        self._weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self._rate_limit = rate_limit
        self._rate_window = rate_window_sec
        self._min_interval = min_interval_sec
        self._now = now or time.time
        self._rate = _RateTracker(rate_window_sec)
        self._last_pass: dict[tuple, float] = {}
        self._blocked_pairs: set[tuple[str, str]] = set()
        self._extra_rules: list[PolicyRule] = []

    # -- 规则注册 ---------------------------------------------------------

    def register_rule(self, rule: PolicyRule) -> None:
        """动态注册自定义硬规则（一票否决语义）。"""
        self._extra_rules.append(rule)

    def block_pair(self, from_agent: str, to_agent: str) -> None:
        """黑名单：禁止某对 (from,to) 传输。to_agent 传 "*" 表示禁止该发起方所有外传。"""
        self._blocked_pairs.add((from_agent, to_agent))

    # -- 主入口 -----------------------------------------------------------

    def decide(self, req: TransmissionRequest) -> Verdict:
        try:
            key = (req.from_agent, req.to_agent or "*")
            now = self._now()
            self._rate.record(key, now)  # 所有 attempt 计入频控

            # ── L0 硬规则层：一票否决 ──
            l0_reason = self._check_hard_rules(req, key, now)
            if l0_reason:
                v = Verdict(action="BLOCK", reason=l0_reason,
                            policy_hits=[l0_reason], score=0.0)
                self._audit(req, v)
                return v

            # ── L1 信号层：软评分 ──
            score = self._score(req, key, now)
            if score >= self._threshold:
                self._last_pass[key] = now
                v = Verdict(action="PASS",
                            reason=f"L1 评分 {score:.2f} ≥ 阈值 {self._threshold:.2f}",
                            score=score)
            else:
                v = Verdict(action="BLOCK",
                            reason=f"L1 信号不足 score={score:.2f} < 阈值 {self._threshold:.2f}",
                            score=score)
            self._audit(req, v)
            return v
        except Exception as exc:  # 失败安全：内部放行 / 对外收紧
            logger.exception("[adjudication] 裁决器异常: %s", exc)
            action = "PASS" if req.channel == "internal" else "BLOCK"
            v = Verdict(action=action,
                        reason=f"裁决器异常降级（channel={req.channel}）: {exc}",
                        policy_hits=["adjudicator_error"], confidence=0.0)
            self._audit(req, v)
            return v

    # -- L0 硬规则 --------------------------------------------------------

    def _check_hard_rules(self, req: TransmissionRequest, key: tuple, now: float) -> Optional[str]:
        # 红线类型
        if req.msg_type.lower() in REDLINE_TYPES:
            return f"红线类型 '{req.msg_type}' 被禁止——任何体不得终止对话"
        # 凭据/密钥模式
        if req.body:
            for pat in SECRET_PATTERNS:
                if pat.search(req.body):
                    return "内容含凭据/密钥模式，禁止外传"
        # 黑名单
        if key in self._blocked_pairs or (req.from_agent, "*") in self._blocked_pairs:
            return f"黑名单命中: {req.from_agent} → {req.to_agent or '*'}"
        # 频控硬顶
        if self._rate.count(key, now) > self._rate_limit:
            return (f"频控硬顶: {req.from_agent}→{req.to_agent or '*'} "
                    f"{self._rate_window:.0f}s 内超 {self._rate_limit} 次")
        # 自定义规则
        for rule in self._extra_rules:
            reason = rule(req)
            if reason:
                return f"[{rule.rule_id}] {reason}"
        return None

    # -- L1 软评分 --------------------------------------------------------

    def _score(self, req: TransmissionRequest, key: tuple, now: float) -> float:
        w = self._weights
        imp_w, sem_w, rec_w = w.get("importance", 0.2), w.get("semantic", 0.5), w.get("recency", 0.3)

        # importance：发起方自评，压低权重（自评不可信）
        imp = max(0.0, min(1.0, req.importance))

        # semantic：接收方兴趣 × 内容 相似度；无 semantic/无兴趣/异常 → 中性 0.5
        sem = 0.5
        interest = self._interests.get(req.to_agent or "") if req.to_agent else None
        if self._semantic is not None and interest and req.body:
            try:
                sim = self._semantic.similarity(interest, req.body)
                if sim is not None:
                    sem = max(0.0, min(1.0, sim))
            except Exception as exc:
                logger.warning("[adjudication] semantic 信号异常，中性化: %s", exc)
                sem = 0.5

        # recency：冷却期内 → 0（惩罚），已过/未启用 → 1.0
        rec = 1.0
        if self._min_interval is not None:
            last = self._last_pass.get(key)
            if last is not None and now - last < self._min_interval:
                rec = 0.0

        total_w = imp_w + sem_w + rec_w
        if total_w <= 0:
            return 0.5
        return (imp_w * imp + sem_w * sem + rec_w * rec) / total_w

    # -- 审计 -------------------------------------------------------------

    def _audit(self, req: TransmissionRequest, v: Verdict) -> None:
        try:
            rec = {
                "ts": round(self._now(), 3),
                "from": req.from_agent, "to": req.to_agent,
                "type": req.msg_type, "channel": req.channel,
                "action": v.action, "reason": v.reason,
                "hits": v.policy_hits, "score": round(v.score, 3),
            }
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except IOError as exc:
            logger.warning("[adjudication] 审计落盘失败: %s", exc)


# ═══════════════════════════════════════════════════════════════════
# 便捷入口
# ═══════════════════════════════════════════════════════════════════

_default_adjudicator: Optional[TransmissionAdjudicator] = None


def should_transmit(req: TransmissionRequest) -> Verdict:
    """默认单例裁决器。调用方一行接入。"""
    global _default_adjudicator
    if _default_adjudicator is None:
        _default_adjudicator = TransmissionAdjudicator()
    return _default_adjudicator.decide(req)


def make_request(from_agent: str, to_agent: Optional[str] = None,
                 msg_type: str = "message", body: str = "",
                 importance: float = 0.5, channel: str = "internal",
                 **kw: Any) -> TransmissionRequest:
    """便捷构造请求。"""
    return TransmissionRequest(from_agent=from_agent, to_agent=to_agent,
                               msg_type=msg_type, body=body,
                               importance=importance, channel=channel, **kw)
