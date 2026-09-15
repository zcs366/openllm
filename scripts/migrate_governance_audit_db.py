#!/usr/bin/env python3
"""把 IOS 治理审计库从 Hermes 家目录平移到 openLLM 自己的家目录。

背景（2026-09-15 医师接骨）：``GovernanceAuditLog`` 原先硬编码
``~/.hermes/hermes.db``，IOS 的治理审计寄生在 Hermes 家目录里——换 HOME 或
沙箱启动就 ``sqlite3.OperationalError: unable to open database file``。
默认路径已改为 ``~/.openllm/governance/audit.db``。

本脚本**只读源库、只写目标库，绝不删除或修改源库**，用于把已有审计链平移过去，
保持 id/timestamp/prev_hash 原样，因而链式 hash 连续、``verify_chain()`` 仍通过。

用法：
    OPENLLM_SECURITY_LEVEL=3 .venv/bin/python scripts/migrate_governance_audit_db.py
    # 预演（只看不写）
    ... --dry-run
    # 自定义源/目标
    ... --from ~/.hermes/hermes.db --to ~/.openllm/governance/audit.db
    # 目标已有数据时强制重来（会清空目标表，源库不动）
    ... --force

幂等：目标库已有行且未加 --force 时，直接跳过并报告。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_FROM = os.path.expanduser("~/.hermes/hermes.db")
DEFAULT_TO = os.path.expanduser("~/.openllm/governance/audit.db")

COLUMNS = ("id", "timestamp", "event_type", "agent_id", "action",
           "prev_hash", "session_id", "details")


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]


def _row_count(conn: sqlite3.Connection) -> int:
    if "governance_audit" not in _tables(conn):
        return 0
    return conn.execute("SELECT COUNT(*) FROM governance_audit").fetchone()[0]


def _tail_hash(conn: sqlite3.Connection) -> str:
    if "governance_audit" not in _tables(conn):
        return "genesis"
    row = conn.execute(
        "SELECT prev_hash FROM governance_audit ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else "genesis"


def _headers(conn: sqlite3.Connection) -> dict:
    """采集源库审计元信息：行数 + 末条 hash + 全表内容 sha256（作为对账指纹）。"""
    if "governance_audit" not in _tables(conn):
        return {"rows": 0, "tail_hash": "genesis", "fingerprint": "-"}
    rows = conn.execute(
        f"SELECT {', '.join(COLUMNS)} FROM governance_audit ORDER BY id").fetchall()
    blob = "\n".join("|".join("" if v is None else str(v) for v in r) for r in rows)
    return {
        "rows": len(rows),
        "tail_hash": _tail_hash(conn),
        "fingerprint": hashlib.sha256(blob.encode()).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="平移 IOS 治理审计库")
    ap.add_argument("--from", dest="src", default=DEFAULT_FROM)
    ap.add_argument("--to", dest="dst", default=DEFAULT_TO)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="目标已有数据时清空目标表重来（源库不动）")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    print(f"源库  : {src}")
    print(f"目标库: {dst}")

    if not src.exists():
        print("⚠️  源库不存在——无需迁移（全新环境按新默认路径即可）")
        return 0
    if src.resolve() == dst.resolve():
        print("❌ 源与目标相同，拒绝执行")
        return 2

    # 源库只读打开，物理上不可能改到它
    sconn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        before = _headers(sconn)
        print(f"源库状态: {before['rows']} 行, 末条 hash={before['tail_hash'][:16]}, "
              f"指纹={before['fingerprint'][:16]}")
        if before["rows"] == 0:
            print("⚠️  源库 governance_audit 为空——无需迁移")
            return 0

        rows = sconn.execute(
            f"SELECT {', '.join(COLUMNS)} FROM governance_audit ORDER BY id").fetchall()

        if args.dry_run:
            print(f"[dry-run] 将写入 {len(rows)} 行到目标库，未做任何改动")
            return 0

        dst.parent.mkdir(parents=True, exist_ok=True)
        dconn = sqlite3.connect(dst)
        try:
            dconn.execute("""CREATE TABLE IF NOT EXISTS governance_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                agent_id TEXT DEFAULT '',
                action TEXT DEFAULT '',
                prev_hash TEXT DEFAULT '',
                session_id TEXT DEFAULT '',
                details TEXT DEFAULT ''
            )""")
            existing = _row_count(dconn)
            if existing and not args.force:
                print(f"⚠️  目标库已有 {existing} 行——幂等跳过（要重来加 --force）")
                return 0
            if existing:
                dconn.execute("DELETE FROM governance_audit")
                dconn.commit()

            dconn.executemany(
                f"INSERT INTO governance_audit ({', '.join(COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(COLUMNS))})", rows)
            dconn.commit()

            after = _headers(dconn)
            ok = (after["rows"] == before["rows"]
                  and after["tail_hash"] == before["tail_hash"]
                  and after["fingerprint"] == before["fingerprint"])
            print(f"目标库状态: {after['rows']} 行, 末条 hash={after['tail_hash'][:16]}, "
                  f"指纹={after['fingerprint'][:16]}")
            if not ok:
                print("❌ 对账不一致——目标库数据可疑，请人工检查（源库未动）")
                return 1
            print("✅ 行数/末条hash/全表指纹 三重对账一致，链式hash连续")
        finally:
            dconn.close()
    finally:
        sconn.close()

    # 独立复核：用代码自身的 verify_chain()
    from openllm.core.governance_engine import GovernanceAuditLog
    log = GovernanceAuditLog(db_path=str(dst))
    try:
        valid, broken_at = log.verify_chain()
        print(f"{'✅' if valid else '❌'} verify_chain() → valid={valid} broken_at={broken_at}")
        return 0 if valid else 1
    finally:
        log._conn.close()


if __name__ == "__main__":
    raise SystemExit(main())