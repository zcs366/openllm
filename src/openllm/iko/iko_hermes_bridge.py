"""
IKO ↔ Hermes 桥接层
====================

将 IKO（输出体）的七个因子管线封装为 Hermes 可调用的简洁接口：

- status()     → JSON 状态摘要
- classify()   → 意图分类结果（dict）
- audit()      → 写入审计链文件

设计约束：
- 零 LLM 调用：所有逻辑复用 IKO 确定性规则
- 持久化审计：审计链以 JSONL 文件形式落盘（德墨忒尔约束）
- 可独立验证：审计文件可被外部工具 parse+verify
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from openllm.iko.intent_classifier import (
    IntentClassifier,
    ClassificationResult,
    OutputIntent,
)
from openllm.iko.output_audit import OutputAuditChain, compute_entry_hash
from openllm.iko.lambda_calibrator import LambdaCalibrator

# ── 全局单例 ──
_classifier = IntentClassifier()
_calibrator = LambdaCalibrator()

__all__ = ["status", "classify", "audit"]


def status() -> dict[str, Any]:
    """返回 IKO 桥接层的 JSON 状态摘要。

    Returns:
        dict 包含 version, modules, lambda_value, timestamp 等字段。
    """
    return {
        "version": "0.1.0",
        "modules": [
            "IntentClassifier",
            "SilenceAuditor",
            "OutputRouter",
            "OutputAuditChain",
            "SymmetricCodec",
            "FeedbackCollector",
            "LambdaCalibrator",
            "ProbingTrainer",
        ],
        "lambda_value": _calibrator.get_current_lambda(),
        "timestamp": time.time(),
    }


def classify(
    context: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, str]:
    """将决策内容分类为输出意图。

    Args:
        context: 上下文信息（risk_level, has_tool_calls 等）。
        decision: 决策内容（必须含 content 字段）。

    Returns:
        {"intent": "<intent_value>", "reason": "<reason>"}

    Raises:
        ValueError: context 或 decision 缺少必需字段。
    """
    result: ClassificationResult = _classifier.classify(context, decision)
    return {
        "intent": result.intent.value,
        "reason": result.reason,
    }


def audit(
    chain_dir: str | Path,
    output_id: str,
    intent: str,
    content: bytes,
    decision_source: str = "IKO",
    risk_level: float = 0.0,
    confidence: float = 1.0,
    reasoning_chain_hash: str = "",
) -> dict[str, Any]:
    """向审计链追加一条记录并持久化到文件。

    每次调用在 chain_dir 下创建/追加 JSONL 文件：
        chain.jsonl — 每行一条审计条目（含 content_hex 用于重建）

    同时生成对应条目的 hash 快照文件：
        <output_id>.hash.json — 该条目的完整哈希（阿波罗约束）

    Args:
        chain_dir: 审计链文件存放目录。
        output_id: 输出唯一标识。
        intent: 输出意图描述。
        content: 输出原始内容（bytes）。
        decision_source: 决策来源。
        risk_level: 风险等级 0.0-1.0。
        confidence: 置信度 0.0-1.0。
        reasoning_chain_hash: 推理链哈希。

    Returns:
        {"output_id", "content_hash", "entry_hash", "chain_length", "path"}

    Raises:
        ValueError: output_id 重复或 chain_dir 不可写。
    """
    chain_dir = Path(chain_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)

    chain_file = chain_dir / "chain.jsonl"

    # 加载已有链（如果存在）
    chain = OutputAuditChain()
    if chain_file.exists():
        _load_chain_from_file(chain, chain_file)

    # 追加新条目
    entry = chain.append(
        output_id=output_id,
        intent=intent,
        content=content,
        decision_source=decision_source,
        risk_level=risk_level,
        confidence=confidence,
        reasoning_chain_hash=reasoning_chain_hash,
    )

    # 计算条目哈希
    entry_hash = compute_entry_hash(entry)

    # 追加写入 JSONL（content_hex 用于重建）
    record = {
        "output_id": entry.output_id,
        "intent": entry.intent,
        "content_hash": entry.content_hash,
        "content_hex": content.hex(),
        "decision_source": entry.decision_source,
        "risk_level": entry.risk_level,
        "confidence": entry.confidence,
        "prev_hash": entry.prev_hash,
        "timestamp": entry.timestamp,
        "reasoning_chain_hash": entry.reasoning_chain_hash,
        "_entry_hash": entry_hash,
    }
    with open(chain_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 写入独立 hash 文件（阿波罗约束：外部可独立验证）
    hash_file = chain_dir / f"{output_id}.hash.json"
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump({"output_id": output_id, "entry_hash": entry_hash}, f)

    return {
        "output_id": output_id,
        "content_hash": entry.content_hash,
        "entry_hash": entry_hash,
        "chain_length": len(chain),
        "path": str(chain_file),
    }


def _load_chain_from_file(chain: OutputAuditChain, chain_file: Path) -> None:
    """从 JSONL 文件恢复审计链。

    每行 JSONL 包含 content_hex，可重建原始 bytes 供 chain.append() 使用。
    """
    with open(chain_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            content = bytes.fromhex(rec["content_hex"])
            chain.append(
                output_id=rec["output_id"],
                intent=rec["intent"],
                content=content,
                decision_source=rec["decision_source"],
                risk_level=rec["risk_level"],
                confidence=rec["confidence"],
                reasoning_chain_hash=rec.get("reasoning_chain_hash", ""),
            )
