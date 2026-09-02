"""VerificationLedger 验证证据账本测试。

覆盖：
  - record → fresh_evidence_for 新鲜证据
  - 过期证据判定
  - 无证据返回 None
  - claims 对账（声称 vs 账本）
  - append-only + stats
  - 写入失败返回空 id
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from openllm.governance.verification_ledger import (
    VerificationEvidence,
    VerificationLedger,
)


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "verification_ledger.jsonl"


@pytest.fixture
def ledger(ledger_path):
    return VerificationLedger(ledger_path)


# ── record + fresh ─────────────────────────────────────────

class TestRecordAndFresh:
    def test_record_returns_id_and_appends(self, ledger, ledger_path):
        eid = ledger.record(target="/src/utils/a.py", step="verify_syntax", ok=True)
        assert eid  # 非空
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["evidence_id"] == eid
        assert d["target"] == "/src/utils/a.py"
        assert d["step"] == "verify_syntax"
        assert d["ok"] is True

    def test_fresh_evidence_found(self, ledger):
        eid = ledger.record(target="/src/utils/a.py", step="pytest", ok=True)
        ev = ledger.fresh_evidence_for("/src/utils/a.py", max_age_s=3600)
        assert ev is not None
        assert ev.evidence_id == eid
        assert ev.step == "pytest"

    def test_has_fresh_evidence(self, ledger):
        ledger.record(target="/src/utils/b.py", step="verify", ok=True)
        assert ledger.has_fresh_evidence("/src/utils/b.py", max_age_s=3600) is True
        assert ledger.has_fresh_evidence("/src/utils/never.py", max_age_s=3600) is False

    def test_stale_evidence_returns_none(self, ledger):
        """写入一条伪造的旧证据（ts 在 max_age 之前）"""
        ledger.record(target="/src/utils/old.py", step="verify", ok=True)
        # 直接篡改文件时间戳为 2 小时前
        ledger_path = ledger._path
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        d = json.loads(lines[0])
        d["ts"] = "2020-01-01T00:00:00+00:00"
        ledger_path.write_text(json.dumps(d, ensure_ascii=False) + "\n", encoding="utf-8")
        # 重建实例（避免缓存）
        fresh_ledger = VerificationLedger(ledger_path)
        assert fresh_ledger.fresh_evidence_for("/src/utils/old.py", max_age_s=3600) is None
        assert fresh_ledger.has_fresh_evidence("/src/utils/old.py", max_age_s=3600) is False

    def test_latest_wins(self, ledger):
        eid1 = ledger.record(target="/src/utils/c.py", step="v1", ok=True)
        time.sleep(0.01)
        eid2 = ledger.record(target="/src/utils/c.py", step="v2", ok=True)
        ev = ledger.fresh_evidence_for("/src/utils/c.py")
        assert ev.evidence_id == eid2  # 最新一条


# ── 声称-证据对账 ─────────────────────────────────────────

class TestClaimsVsEvidence:
    def test_claims_match_steps(self, ledger):
        ledger.record(target="/src/utils/d.py", step="pytest tests/test_brain_guardian.py", ok=True)
        ledger.record(target="/src/utils/d.py", step="verify_syntax", ok=True)
        result = ledger.verify_claims_vs_evidence(
            ["pytest tests/test_brain_guardian.py", "verify_syntax", "no_such_step"],
        )
        assert "pytest tests/test_brain_guardian.py" in result["matched"]
        assert "verify_syntax" in result["matched"]
        assert "no_such_step" in result["unmatched"]
        assert result["total_records"] == 2

    def test_claims_empty(self, ledger):
        result = ledger.verify_claims_vs_evidence([])
        assert result["matched"] == []
        assert result["unmatched"] == []


# ── stats / 健壮性 ────────────────────────────────────────

class TestStatsAndRobustness:
    def test_stats_counts(self, ledger):
        ledger.record(target="/src/a.py", step="s1", ok=True)
        ledger.record(target="/src/a.py", step="s2", ok=False)
        ledger.record(target="/src/b.py", step="s3", ok=True)
        stats = ledger.stats()
        assert stats["total"] == 3
        assert stats["ok"] == 2
        assert stats["failed"] == 1
        assert len(stats["last_10"]) == 3

    def test_empty_ledger(self, ledger):
        assert ledger.stats()["total"] == 0
        assert ledger.fresh_evidence_for("/src/x.py") is None
        assert ledger.verify_claims_vs_evidence(["pytest"])["unmatched"] == ["pytest"]

    def test_append_only_preserves_history(self, ledger, ledger_path):
        ledger.record(target="/src/a.py", step="first", ok=True)
        ledger.record(target="/src/a.py", step="second", ok=True)
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2  # 追加不覆盖

    def test_kind_normalized(self, ledger):
        eid = ledger.record(target="/src/a.py", step="s", ok=True, kind="bogus")
        ev = ledger.fresh_evidence_for("/src/a.py")
        assert ev.kind == "ad_hoc"  # 非法 kind 归一化
        eid2 = ledger.record(target="/src/b.py", step="s", ok=True, kind="canonical")
        ev2 = ledger.fresh_evidence_for("/src/b.py")
        assert ev2.kind == "canonical"

    def test_write_failure_returns_empty(self, ledger_path):
        """写入失败（目录只读）返回空 id"""
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text("", encoding="utf-8")
        ledger_path.chmod(0o400)  # 只读
        try:
            ledger = VerificationLedger(ledger_path)
            eid = ledger.record(target="/src/a.py", step="s", ok=True)
            assert eid == ""
        finally:
            ledger_path.chmod(0o644)
