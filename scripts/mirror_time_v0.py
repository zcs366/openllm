#!/usr/bin/env python3
"""镜子时间 v0 —— 每晚翻账本的审计日志生成器。

三个数据源（全部只读）：
  1. ~/.openllm/iai/preference_pairs.jsonl   偏好对
  2. ~/.openllm/isl_chain.jsonl              ISL 身份链（epoch 哈希链）
  3. ~/.openllm/memory/causal/*.json         因果伤疤

输出（append-only）：~/.openllm/mirror_time_log.jsonl
首行为免责声明；之后每次运行追加一条 mirror_run 记录。

⚠️ 本日志的产出不构成审计结论，判读权在造物主。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

DISCLAIMER = "本日志的产出不构成审计结论，判读权在造物主"

PREFERENCE_PAIRS: Path | None = None  # None = 运行时解析 Path.home()，便于测试 monkeypatch
ISL_CHAIN: Path | None = None
CAUSAL_DIR: Path | None = None
LOG_PATH: Path | None = None


def _default_log_path() -> Path:
    return Path.home() / ".openllm" / "mirror_time_log.jsonl"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def collect_preferences(path: Path) -> dict:
    """统计偏好对文件：条目数（跳过 # 注释行）+ 文件字节数。"""
    try:
        raw = path.read_text(encoding="utf-8")
        entries = 0
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            json.loads(line)  # 损坏行抛异常，进入 errors
            entries += 1
        return {"total": entries, "file_bytes": path.stat().st_size}
    except FileNotFoundError:
        return {"total": None, "file_bytes": None, "errors": ["file not found: %s" % path]}
    except json.JSONDecodeError as e:
        return {"total": None, "file_bytes": None, "errors": ["json parse: %s" % e]}


def collect_isl_chain(path: Path) -> dict:
    """ISL 链统计：epoch 数 + 链接完整性（prev_hash 链 + epoch 递增）。v0 不重算哈希。"""
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except FileNotFoundError:
        return {"epochs": None, "linkage_ok": None, "last_epoch": None, "last_hash": None,
                "errors": ["file not found: %s" % path]}
    epochs = []
    hashes = []
    parse_errors = []
    for idx, ln in enumerate(lines, 1):
        try:
            j = json.loads(ln)
            epochs.append(int(j.get("epoch", -1)))
            hashes.append(str(j.get("hash", "")))
        except (json.JSONDecodeError, ValueError) as e:
            parse_errors.append("line %d: %s" % (idx, e))
    linkage_ok = True
    if not epochs:
        linkage_ok = False
    else:
        if epochs[0] != 1:
            linkage_ok = False
        for i in range(1, len(epochs)):
            if epochs[i] != epochs[i - 1] + 1:
                linkage_ok = False
                break
        try:
            for i in range(1, len(lines)):
                prev = json.loads(lines[i - 1])
                cur = json.loads(lines[i])
                if cur.get("prev_hash") != prev.get("hash"):
                    linkage_ok = False
                    break
        except json.JSONDecodeError:
            linkage_ok = False
    out = {"epochs": len(epochs), "linkage_ok": linkage_ok,
           "last_epoch": epochs[-1] if epochs else None,
           "last_hash": hashes[-1] if hashes else None}
    if parse_errors:
        out["errors"] = parse_errors
    return out


def collect_causal_scars(directory: Path) -> dict:
    """伤疤目录统计：数量 / actual_success 均值 / importance 均值 / 最新 created_at。"""
    try:
        files = sorted(glob.glob(str(directory / "*.json")))
    except Exception as e:  # 目录不存在等
        return {"count": None, "success_rate": None, "importance_mean": None,
                "latest_created_at": None, "errors": [str(e)]}
    if not files:
        return {"count": 0, "success_rate": None, "importance_mean": None,
                "latest_created_at": None}
    successes = []
    importances = []
    latest_ts = None
    errors = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as fh:
                j = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            errors.append("%s: %s" % (os.path.basename(fp), e))
            continue
        if isinstance(j.get("actual_success"), (int, float)):
            successes.append(float(j["actual_success"]))
        if isinstance(j.get("importance"), (int, float)):
            importances.append(float(j["importance"]))
        ts = j.get("created_at")
        if isinstance(ts, (int, float)) and (latest_ts is None or ts > latest_ts):
            latest_ts = float(ts)
    out = {
        "count": len(files),
        "success_rate": round(sum(successes) / len(successes), 3) if successes else None,
        "importance_mean": round(sum(importances) / len(importances), 3) if importances else None,
        "latest_created_at": _iso(latest_ts),
    }
    if errors:
        out["errors"] = errors[:10]  # 最多记10条，防日志膨胀
    return out


def append_log(log_path: Path, metrics: dict) -> dict:
    """追加一条 mirror_run 记录；首次运行先写免责声明行。返回本条记录。"""
    record = {
        "event_id": uuid.uuid4().hex,
        "type": "mirror_run",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "source": "mirror_time_v0",
        "metrics": metrics,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        disclaimer = {
            "type": "disclaimer",
            "text": DISCLAIMER,
            "created_at": record["timestamp"],
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(disclaimer, ensure_ascii=False) + "\n")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def build_metrics(pref_path: Path | None = None, isl_path: Path | None = None,
                  causal_path: Path | None = None) -> dict:
    """路径默认运行时解析 Path.home()（不可在定义时绑定，否则测试 monkeypatch 失效）。"""
    pref_path = pref_path or (Path.home() / ".openllm" / "iai" / "preference_pairs.jsonl")
    isl_path = isl_path or (Path.home() / ".openllm" / "isl_chain.jsonl")
    causal_path = causal_path or (Path.home() / ".openllm" / "memory" / "causal")
    return {
        "preference_pairs": collect_preferences(pref_path),
        "isl_chain": collect_isl_chain(isl_path),
        "causal_scars": collect_causal_scars(causal_path),
    }


def main() -> int:
    metrics = build_metrics()
    record = append_log(_default_log_path(), metrics)
    # stdout 只报数字，不做任何"一切正常"式叙述
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
