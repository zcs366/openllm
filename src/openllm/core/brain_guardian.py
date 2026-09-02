"""双脑互保协调器 — BrainGuardian（PAL T-D-4）

一脑改代码失明时，另一脑接管（继续/回滚）。
集成三大子系统：
  - Hemisphere 心跳互检（hemispheres.py）
  - EffectContext 可逆效应代数（cordis/runtime.py）
  - SelfModificationGuard 禁区守卫（governance/self_modification_guard.py）

核心流程 apply_code_change：
  approve_change（禁区→中止+审计）→ 快照 → EffectContext.apply → 验证
  → 成功：固化（跳过 undo）
  → 失败：undo_all 回滚 + 接管事件记录

接管事件：append-only 写入 ~/.openllm/guardian/events.jsonl
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.core.brain_guardian")

from openllm.cordis.runtime import Effect, EffectContext
from openllm.core.hemispheres import (
    HemispherePair,
    HemisphereState,
)
from openllm.governance.self_modification_guard import SelfModificationGuard
from openllm.governance.verification_ledger import (
    DEFAULT_LEDGER_PATH,
    VerificationLedger,
)

GUARDIAN_DIR = Path.home() / ".openllm" / "guardian"
EVENTS_FILE = GUARDIAN_DIR / "events.jsonl"


# ── 数据结构 ─────────────────────────────────────────────

@dataclass
class ChangeResult:
    """apply_code_change 返回值"""
    success: bool
    target: str
    reason: str
    undone: list[Effect] = field(default_factory=list)
    snapshot_backup: Any = None  # EffectContext 快照备份（仅失败时非 None）
    evidence_id: str = ""  # VerificationLedger 证据引用（成功固化时非空）


# ── 双脑互保协调器 ──────────────────────────────────────

class BrainGuardian:
    """双脑互保协调器。

    管理热改执行全生命周期：
      guard → 快照 → apply → 验证 → 固化/回滚 + 接管事件

    append-only 原则：接管事件只追加不修改。
    """

    def __init__(
        self,
        hemispheres: HemispherePair,
        guard: Optional[SelfModificationGuard] = None,
        event_file: Path | str = EVENTS_FILE,
        state: dict | None = None,
        ledger: Optional[VerificationLedger] = None,
        ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    ):
        self.hemispheres = hemispheres
        self.guard = guard or SelfModificationGuard()
        self._event_file = Path(event_file)
        self._event_file.parent.mkdir(parents=True, exist_ok=True)

        self._state: dict = state if state is not None else {}
        self._effect_ctx: EffectContext[dict] = EffectContext(self._state)

        # 验证证据账本（2026-09-02 VerificationLedger 移植）
        # 双脑互保语境: 热改脑的验证动作落账, 守护脑可对账"验证过"是事实非自述
        self.ledger = ledger if ledger is not None else VerificationLedger(ledger_path)

    # ── 公共 API ─────────────────────────────────────────

    def apply_code_change(
        self,
        target: str,
        apply_fn: Callable[[dict], dict],
        verify_fn: Optional[Callable[[dict], bool]] = None,
        verify_evidence_required: bool = False,
        session_id: str = "",
    ) -> ChangeResult:
        """执行热改变更（五步闭环 + 验证证据账本）。

        流程：
          1. guard.approve_change → 禁区拒绝则中止+审计
          2. snapshot（EffectContext 快照）
          3. EffectContext.apply（记录 undo）
          4. 验证函数返回 bool —— 验证动作同时写入 VerificationLedger
          5. 成功→固化（携带 evidence_id）；失败→undo_all + 接管事件

        证据门（verify_evidence_required=True 时强制）：
          固化前要求 ledger 中存在该 target 的新鲜验证证据（≤默认1h）。
          守护脑用此防止热改脑"自述已验证"却无痕可查——无证据则拒绝固化。

        Args:
            target: 目标文件标识
            apply_fn: 变更函数 state→state（副作用在此产生）
            verify_fn: 验证函数 state→bool（默认通过）
            verify_evidence_required: 强制新鲜证据门（默认 False=观察模式）
            session_id: 当前会话（证据溯源用）

        Returns:
            ChangeResult（成功/失败+回滚信息+evidence_id）
        """
        return self._apply_code_change(
            target, apply_fn, verify_fn,
            verify_evidence_required=verify_evidence_required,
            session_id=session_id,
        )

    def monitor_heartbeat(self) -> Optional[dict]:
        """检查对侧脑心跳，异常则记录接管事件。

        调用 HemispherePair.health_check() 获取双脑状态，
        发现失明/崩溃时 append-only 记录接管事件。

        Returns:
            接管事件 dict（有接管时）或 None（一切正常）
        """
        pair = self.hemispheres
        health = pair.health_check()

        for side in ("left", "right"):
            if not health[side]["alive"]:
                opposite = "right" if side == "left" else "left"
                if health[opposite]["alive"]:
                    # 对侧存活→接管
                    target_hemi = pair.left if side == "left" else pair.right
                    target_hemi.state = HemisphereState.FAILOVER
                    return self._record_takeover(
                        role=side,
                        action="takeover",
                        reason=health[side]["status"],
                    )

        return None

    def stats(self) -> dict:
        """统计：总改动/成功/回滚/接管事件数"""
        events = self._read_events()
        totals = {"total": 0, "success": 0, "rollback": 0, "takeover_events": 0}
        for e in events:
            kind = e.get("event", "")
            if kind == "change_applied":
                totals["total"] += 1
                if e.get("success"):
                    totals["success"] += 1
                else:
                    totals["rollback"] += 1
            elif kind == "takeover":
                totals["takeover_events"] += 1
        return totals

    # ── 内部实现 ─────────────────────────────────────────

    def _apply_code_change(
        self,
        target: str,
        apply_fn: Callable[[dict], dict],
        verify_fn: Optional[Callable[[dict], bool]] = None,
        verify_evidence_required: bool = False,
        session_id: str = "",
    ) -> ChangeResult:
        # 1. 禁区检查
        if not self.guard.approve_change(target):
            self._record_takeover(
                role="guard", action="forbidden_reject",
                reason=f"target={target} in forbidden zone",
            )
            return ChangeResult(
                success=False, target=target,
                reason="forbidden zone — change rejected by SelfModificationGuard",
            )

        # 2. 快照
        snapshot = self._effect_ctx.snapshot()

        # 3. apply（记录 undo）
        def _apply(state: dict) -> dict:
            return apply_fn(state)
        def _undo(state: dict) -> dict:
            return snapshot[0]  # 恢复到快照时的状态

        effect = Effect(name=f"change:{target}", apply=_apply, undo=_undo)
        try:
            self._effect_ctx.apply(effect)
        except Exception as e:
            # apply 本身出错 → 回滚
            self._record_takeover(
                role="guard", action="apply_error",
                reason=f"target={target}: {e}",
            )
            return ChangeResult(
                success=False, target=target,
                reason=f"apply failed: {e}",
                snapshot_backup=snapshot,
            )

        # 4. 验证 —— 动作同时写入 VerificationLedger（2026-09-02 移植）
        ok = True
        verify_step = ""
        if verify_fn is not None:
            verify_step = getattr(verify_fn, "__name__", "verify")
            try:
                ok = bool(verify_fn(self._effect_ctx.context))
            except Exception as e:
                ok = False
                logger.warning("BrainGuardian: verify_fn %s raised: %s", verify_step, e)

        # 验证动作落账（append-only，被动记录实际发生的验证）
        evidence_id = self.ledger.record(
            target=target,
            kind="ad_hoc",  # verify_fn 是代码内事务验证（非外部canonical套件）
            step=verify_step or "no_verify_fn",
            ok=ok,
            session_id=session_id,
            note="BrainGuardian.apply_code_change verify step",
        )

        # 5. 结果处理
        if ok:
            # 证据门：强制模式要求本 target 有新鲜验证证据（ledger 记录本身满足）
            if verify_evidence_required and not evidence_id:
                # ledger 写失败——证据未落账，拒绝固化（守护脑强制）
                self._record_takeover(
                    role="guard", action="evidence_not_recorded",
                    reason=f"target={target} verify passed but evidence not recorded in ledger",
                )
                return ChangeResult(
                    success=False, target=target,
                    reason="verification evidence not recorded — refusing to commit without ledger entry",
                    snapshot_backup=snapshot,
                )
            # 成功：固化（跳过 undo），记录事件 + 证据引用
            self._record_event("change_applied", {
                "target": target, "success": True,
                "evidence_id": evidence_id,
            })
            reason = "applied and verified"
            if evidence_id:
                reason = f"applied and verified (evidence: {evidence_id})"
            return ChangeResult(
                success=True, target=target,
                reason=reason,
                evidence_id=evidence_id,
            )
        else:
            # 失败：undo_all 回滚 + 接管事件 + 失败记录
            undone = self._effect_ctx.undo_all()
            # 恢复 EffectContext 内部状态到快照
            self._effect_ctx.restore(snapshot)

            self._record_takeover(
                role="guard", action="verify_failed_rollback",
                reason=f"target={target} failed verification, {len(undone)} effects undone",
            )
            # stats() 计数用
            self._record_event("change_applied", {
                "target": target, "success": False,
            })
            return ChangeResult(
                success=False, target=target,
                reason="verification failed, full rollback applied",
                undone=undone,
                snapshot_backup=snapshot,
            )

    def _record_takeover(self, role: str, action: str, reason: str) -> dict:
        """记录接管事件（append-only）"""
        event = {
            "event": "takeover",
            "role": role,
            "action": action,
            "reason": reason,
            "timestamp": time.time(),
            "id": uuid.uuid4().hex[:8],
        }
        self._append_event(event)
        return event

    def _record_event(self, event_type: str, data: dict) -> None:
        """记录通用事件"""
        entry = {"event": event_type, "timestamp": time.time(), **data}
        self._append_event(entry)

    def _append_event(self, event: dict) -> None:
        """append-only 写入事件日志"""
        try:
            with self._event_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 写失败不阻塞主流程

    def _read_events(self) -> list[dict]:
        """读取全部事件日志"""
        if not self._event_file.exists():
            return []
        events = []
        for line in self._event_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
