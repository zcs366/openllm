#!/usr/bin/env python3
"""IKO ↔ Hermes Bridge — 纯CLI，无Hermes依赖
===============================================

暴露 IKO（输出体）核心功能为命令行接口，供 Hermes 调用：

  status            — 查看 IKO 模块状态
  classify          — 意图分类（context + decision → intent）
  audit             — 审计链管理（添加/验证/查看条目）

用法：
    python3 scripts/iko_hermes_bridge.py status
    python3 scripts/iko_hermes_bridge.py classify \\
        --context '{"risk_level":"LOW","has_tool_calls":false,
                    "has_side_effects":false,"option_count":0}' \\
        --decision '{"content":"Hello world"}'
    python3 scripts/iko_hermes_bridge.py audit --chain-file /tmp/audit.json \\
        --add '{"output_id":"out-001","intent":"INFORM","content":"hi",
                "decision_source":"IKO","risk_level":0.1,"confidence":0.95,
                "reasoning_chain_hash":"abc"}'
    python3 scripts/iko_hermes_bridge.py audit --chain-file /tmp/audit.json --verify
    python3 scripts/iko_hermes_bridge.py audit --chain-file /tmp/audit.json --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── 确保 src/ 在 sys.path 中，使 `from openllm.iko import ...` 可用 ──
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openllm.iko import (
    ClassificationResult,
    IntentClassifier,
    OutputAuditChain,
    OutputAuditEntry,
    OutputIntent,
    SilenceAuditor,
    __version__,
)

# ═══════════════════════════════════════════════════════════════
# 1. status — 查看 IKO 模块状态
# ═══════════════════════════════════════════════════════════════

def cmd_status(_args: argparse.Namespace) -> None:
    """打印 IKO 模块状态摘要。"""
    intents = [e.value for e in OutputIntent]
    print(json.dumps({
        "module": "openllm.iko",
        "version": __version__,
        "status": "ok",
        "intents": intents,
        "path": str(_SRC / "openllm" / "iko"),
    }, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════
# 2. classify — 意图分类
# ═══════════════════════════════════════════════════════════════

def cmd_classify(args: argparse.Namespace) -> None:
    """根据 context + decision 返回分类结果。"""
    try:
        context = json.loads(args.context)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid --context JSON: {exc}"}))
        sys.exit(1)

    try:
        decision = json.loads(args.decision)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid --decision JSON: {exc}"}))
        sys.exit(1)

    classifier = IntentClassifier()
    try:
        result: ClassificationResult = classifier.classify(context, decision)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    # 经过 SilenceAuditor 校验
    auditor = SilenceAuditor()
    final_intent = auditor.audit(
        intent=result.intent,
        context=context,
        output_id=f"classify-{id(result)}",
    )

    print(json.dumps({
        "intent": final_intent.value,
        "reason": result.reason,
        "audited": final_intent != result.intent,
    }, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════
# 3. audit — 审计链管理
# ═══════════════════════════════════════════════════════════════

def _load_chain(chain_path: Path) -> OutputAuditChain:
    """从 JSON 文件加载审计链（反序列化）。"""
    chain = OutputAuditChain()
    if not chain_path.exists():
        return chain
    try:
        data = json.loads(chain_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return chain
    for entry_data in data.get("entries", []):
        chain.append(
            output_id=entry_data["output_id"],
            intent=entry_data["intent"],
            content=entry_data.get("content", "").encode("utf-8"),
            decision_source=entry_data["decision_source"],
            risk_level=entry_data["risk_level"],
            confidence=entry_data["confidence"],
            reasoning_chain_hash=entry_data["reasoning_chain_hash"],
        )
    return chain


def _save_chain(chain: OutputAuditChain, chain_path: Path) -> None:
    """将审计链序列化为 JSON 文件。"""
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for e in chain.entries:
        entries.append({
            "output_id": e.output_id,
            "intent": e.intent,
            "content_hash": e.content_hash,
            "decision_source": e.decision_source,
            "risk_level": e.risk_level,
            "confidence": e.confidence,
            "prev_hash": e.prev_hash,
            "timestamp": e.timestamp,
            "reasoning_chain_hash": e.reasoning_chain_hash,
        })
    chain_path.write_text(
        json.dumps({"version": __version__, "entries": entries},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cmd_audit(args: argparse.Namespace) -> None:
    """审计链子命令：--add / --verify / --list。"""
    chain_path = Path(args.chain_file)
    chain = _load_chain(chain_path)

    if args.add:
        _audit_add(chain, chain_path, args.add)
    elif args.verify:
        _audit_verify(chain)
    elif args.list_entries:
        _audit_list(chain)
    else:
        print(json.dumps({"error": "Specify --add, --verify, or --list"}))
        sys.exit(1)


def _audit_add(chain: OutputAuditChain, chain_path: Path, add_json: str) -> None:
    """向审计链追加一条条目。"""
    try:
        data = json.loads(add_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid --add JSON: {exc}"}))
        sys.exit(1)

    content_str = data.pop("content", "")
    try:
        entry = chain.append(
            output_id=data["output_id"],
            intent=data["intent"],
            content=content_str.encode("utf-8") if isinstance(content_str, str) else content_str,
            decision_source=data["decision_source"],
            risk_level=float(data["risk_level"]),
            confidence=float(data["confidence"]),
            reasoning_chain_hash=data["reasoning_chain_hash"],
        )
    except (KeyError, ValueError, TypeError) as exc:
        print(json.dumps({"error": f"Failed to append entry: {exc}"}))
        sys.exit(1)

    _save_chain(chain, chain_path)
    print(json.dumps({
        "status": "appended",
        "output_id": entry.output_id,
        "chain_length": len(chain),
        "verified": chain.verify(),
    }, ensure_ascii=False))


def _audit_verify(chain: OutputAuditChain) -> None:
    """验证审计链完整性。"""
    valid = chain.verify()
    print(json.dumps({
        "verified": valid,
        "chain_length": len(chain),
        "entries_valid": all(
            len(e.content_hash) == 16 for e in chain.entries
        ),
    }))


def _audit_list(chain: OutputAuditChain) -> None:
    """列出审计链中的所有条目摘要。"""
    entries = []
    for e in chain.entries:
        entries.append({
            "output_id": e.output_id,
            "intent": e.intent,
            "content_hash": e.content_hash,
            "decision_source": e.decision_source,
            "risk_level": e.risk_level,
            "confidence": e.confidence,
            "reasoning_chain_hash": e.reasoning_chain_hash,
        })
    print(json.dumps({
        "chain_length": len(chain),
        "verified": chain.verify(),
        "entries": entries,
    }, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iko_hermes_bridge",
        description="IKO ↔ Hermes Bridge — 纯CLI，无Hermes依赖",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="查看 IKO 模块状态")

    # classify
    p_cls = sub.add_parser("classify", help="意图分类")
    p_cls.add_argument("--context", required=True, help="上下文 JSON 字符串")
    p_cls.add_argument("--decision", required=True, help="决策 JSON 字符串")

    # audit
    p_aud = sub.add_parser("audit", help="审计链管理")
    p_aud.add_argument("--chain-file", required=True, help="审计链 JSON 文件路径")
    p_aud.add_argument("--add", help="追加审计条目 (JSON 字符串)")
    p_aud.add_argument("--verify", action="store_true", help="验证链完整性")
    p_aud.add_argument("--list", dest="list_entries", action="store_true",
                       help="列出所有审计条目")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "status": cmd_status,
        "classify": cmd_classify,
        "audit": cmd_audit,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        return 1
    fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
