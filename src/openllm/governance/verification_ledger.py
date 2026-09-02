"""VerificationLedger — 验证证据账本（openLLM创造性移植自 Hermes verification_evidence.py）。

Hermes 原版哲学（本模块照搬）：
- deliberately passive —— 从不自己决定跑什么验证，只记录实际发生的验证动作
- 从不把 ad-hoc 验证升级为"repo 绿"（kind 字段区分，账本不撒谎）
- append-only，不修改不删除（对齐 openLLM AuditChain / guardian events 惯例）

openLLM 语境差异：这里的"验证者"就是 LLM 自己（双脑互保：一脑热改一脑接管）。
账本让守护脑能对账热改脑的验证声明——"applied and verified"从自述变成可审计记录。

用途：
    ledger = VerificationLedger(path)
    eid = ledger.record(target="/src/utils/a.py", kind="ad_hoc", step="verify_syntax", ok=True)
    ledger.has_fresh_evidence(target="/src/utils/a.py", max_age_s=3600)  # True
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openllm.governance.verification_ledger")

DEFAULT_LEDGER_PATH = Path.home() / ".openllm" / "governance" / "verification_ledger.jsonl"
DEFAULT_MAX_AGE_S = 3600  # 证据新鲜度默认1小时


@dataclass(frozen=True)
class VerificationEvidence:
    """一条验证证据。"""

    ts: str  # ISO8601 UTC
    evidence_id: str
    target: str
    kind: str  # "canonical" | "ad_hoc"
    step: str  # verify_fn.__name__ 或命令名
    ok: bool
    session_id: str = ""
    exit_code: Optional[int] = None
    note: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VerificationLedger:
    """被动验证证据账本（append-only JSONL）。"""

    def __init__(self, path: Path | str = DEFAULT_LEDGER_PATH):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── 写 ────────────────────────────────────────────────

    def record(
        self,
        target: str,
        kind: str = "ad_hoc",
        step: str = "",
        ok: bool = True,
        session_id: str = "",
        exit_code: Optional[int] = None,
        note: str = "",
    ) -> str:
        """记录一次验证动作，返回 evidence_id。append-only。"""
        evidence = VerificationEvidence(
            ts=_utc_now(),
            evidence_id=uuid.uuid4().hex[:12],
            target=target,
            kind=kind if kind in ("canonical", "ad_hoc") else "ad_hoc",
            step=step or "verify",
            ok=bool(ok),
            session_id=session_id,
            exit_code=exit_code,
            note=note,
        )
        line = json.dumps(asdict(evidence), ensure_ascii=False)
        try:
            with self._lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as e:
            logger.warning("VerificationLedger: write failed: %s", e)
            # 写失败返回空 id——调用方应视为"证据未落账"
            return ""
        return evidence.evidence_id

    # ── 读 ────────────────────────────────────────────────

    def _read_all(self) -> list[VerificationEvidence]:
        if not self._path.exists():
            return []
        out: list[VerificationEvidence] = []
        with self._lock:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    out.append(VerificationEvidence(**d))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("VerificationLedger: skip bad line: %s", e)
                    continue
        return out

    def fresh_evidence_for(
        self, target: str, max_age_s: int = DEFAULT_MAX_AGE_S
    ) -> Optional[VerificationEvidence]:
        """该 target 最近一次验证证据；超过 max_age_s 视为不新鲜返回 None。

        append-only 文件顺序 = 时间顺序；同秒多条时取文件里最后一条匹配
        （_read_all 按行序返回，天然有序——直接反向扫描取第一条匹配即可）。
        """
        all_ev = self._read_all()
        if not all_ev:
            return None
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_s
        # 反向扫描：文件尾部 = 最新（append-only 保证行序即时间序）
        for ev in reversed(all_ev):
            if ev.target != target:
                continue
            try:
                ts = datetime.fromisoformat(ev.ts).timestamp()
            except ValueError:
                continue
            if ts < cutoff:
                return None  # 已过期——更早的也不会更新（行序即时间序）
            return ev
        return None

    def has_fresh_evidence(self, target: str, max_age_s: int = DEFAULT_MAX_AGE_S) -> bool:
        return self.fresh_evidence_for(target, max_age_s) is not None

    def verify_claims_vs_evidence(
        self, claims: list[str], max_age_s: int = DEFAULT_MAX_AGE_S
    ) -> dict:
        """声称-证据对账：claims 里说的验证，账本里有对应记录吗？

        claims 例: ["pytest tests/test_brain_guardian.py", "verify_syntax"]
        匹配规则（宽松）: claim 字符串是某条证据 step 的子串，或反之。
        返回: {"matched": [...], "unmatched": [...], "total_records": N}
        """
        all_ev = self._read_all()
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_s
        fresh = []
        for ev in all_ev:
            try:
                ts = datetime.fromisoformat(ev.ts).timestamp()
            except ValueError:
                continue
            if ts >= cutoff:
                fresh.append(ev)

        matched: list[str] = []
        unmatched: list[str] = []
        fresh_steps = {ev.step for ev in fresh}
        for claim in claims:
            c = claim.strip()
            if not c:
                continue
            if any(c in s or s in c for s in fresh_steps):
                matched.append(c)
            else:
                unmatched.append(c)
        return {
            "matched": matched,
            "unmatched": unmatched,
            "total_records": len(all_ev),
            "fresh_records": len(fresh),
        }

    def stats(self) -> dict:
        all_ev = self._read_all()
        ok_count = sum(1 for ev in all_ev if ev.ok)
        return {
            "total": len(all_ev),
            "ok": ok_count,
            "failed": len(all_ev) - ok_count,
            "last_10": [asdict(ev) for ev in all_ev[-10:]],
        }
