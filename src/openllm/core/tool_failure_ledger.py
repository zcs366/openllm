"""
openLLM 工具失败账 — 可计数落点（加固⑤·第二步 2a，2026-10-02）。

一行 JSON 一次失败信号，append-only 追加到
``~/.openllm/guard/tool_failure_ledger.jsonl``。
写它的人有三路（source 字段区分）：
  · "marker"       — 执行后文本里检出失败标记（`_guard_tool_failure_signal`）
  · "input_shape"  — 执行后检出 shell 输入"显然不像命令"（同上，形态并记）
  · "input_reject" — 执行前输入侧拦截，本 turn 未执行（2-pre 闸）

**仪表铁律：绝不抛异常。** 落盘失败（权限/磁盘/序列化）静默吞掉——
账记不上，最多瞎了眼；把异常顶进心跳，是把手里的病治成新的病。

按天×来源计数一行命令（直接读分布）::

    python3 -c "import json,collections,pathlib,time; p=pathlib.Path.home()/'.openllm/guard/tool_failure_ledger.jsonl'; rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]; print(collections.Counter((time.strftime('%Y-%m-%d', time.localtime(r['ts'])), r.get('source'), r.get('kind')) for r in rows))"

字段：ts(epoch float)、tool(list或str)、kind、exit_code(如有)、
input_shape(如有)、source、session_id、turn_id。
"""
import json
import time
from pathlib import Path

LEDGER_PATH = Path.home() / ".openllm" / "guard" / "tool_failure_ledger.jsonl"


def record(sig: dict, *, session_id: str = "", turn_id: str = "") -> None:
    """追加一行失败信号到账。**任何失败静默**（仪表不得拖垮主循环）。"""
    try:
        sig = sig if isinstance(sig, dict) else {}
        tool = sig.get("tool")
        if tool in (None, ""):
            tool = sig.get("tools", "")
        entry = {
            "ts": time.time(),
            "tool": tool,
            "kind": sig.get("kind", ""),
            "source": sig.get("source", ""),
            "session_id": session_id or "",
            "turn_id": turn_id or "",
        }
        if sig.get("exit_code") is not None:
            entry["exit_code"] = sig["exit_code"]
        if sig.get("input_shape"):
            entry["input_shape"] = sig["input_shape"]
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
