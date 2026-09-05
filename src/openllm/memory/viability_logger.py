"""
viability_logger.py — P1-2 观测仪表：V值日志 + 技能生命周期日志

signal_bus → V → selection_gate → 技能生死 链路的一周观测仪表。
只造记录器（append-only jsonl）+ CLI，不挂钩心跳、不动调度。
采样节奏由人工拍板。

用法:
    # 记录一次V值
    python -m openllm.memory.viability_logger

    # 读最后N条V值日志
    python -m openllm.memory.viability_logger --read 20

    # 编程调用
    from openllm.memory.viability_logger import log_viability, log_skill_event, read_log
    log_viability()
    log_skill_event("adopted", "search-pipeline")
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openllm.viability_logger")

# ── 路径常量 ──
VIABILITY_LOG = Path.home() / ".openllm" / "viability_log.jsonl"
SKILL_LIFECYCLE_LOG = Path.home() / ".openllm" / "skill_lifecycle_log.jsonl"


def _load_compute_viability():
    """懒加载 io-s 的 compute_viability()，遵循 engine_integrations.py 的加载模式。

    Returns:
        compute_viability 函数，或 None（io-s 不可用时）。
    """
    import importlib.util

    io_s_path = Path.home() / "io-s"
    if not io_s_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "io_s_viability", io_s_path / "viability.py"
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "compute_viability", None)
    except Exception:
        return None


def _load_read_current_weights():
    """懒加载 io-s 的 weight_adaptation.read_current_weights()。

    Returns:
        weights dict，或 None（io-s/weight_adaptation 不可用时）。
    """
    import importlib.util

    io_s_path = Path.home() / "io-s"
    if not io_s_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "io_s_weight_adaptation", io_s_path / "weight_adaptation.py"
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "read_current_weights", lambda: None)()
    except Exception:
        return None


def _append_jsonl(path: Path, record: dict) -> None:
    """追加一条 JSON 记录到 jsonl 文件（append-only）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    """返回 ISO8601 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


# ═══ 公开 API ═══


def log_viability(log_path: Optional[Path] = None, bus: Any = "auto") -> Optional[dict]:
    """调用 io-s 的 compute_viability()，将结果追加写入 V 值日志。

    Args:
        log_path: 日志文件路径覆盖（测试用），为 None 时用默认 VIABILITY_LOG。
        bus: 进化总线接线（T3 P0-C）。"auto"=生产（自建真实总线）；
             None=不接总线（测试隔离，防污染 ~/.openllm/evolution）；
             传入实例=用注入的总线（测试可传mock/临时总线）。

    Returns:
        写入的记录 dict，含 timestamp、source、V值及分量；失败返回 None。
    """
    compute_viability = _load_compute_viability()
    if compute_viability is None:
        logger.warning("io-s compute_viability 不可用，跳过本次V值采样")
        return None
    # 懒加载自适应权重（失败零阻塞）
    adapted = None
    try:
        adapted = _load_read_current_weights()
    except Exception:
        pass
    try:
        if adapted is not None:
            v_result = compute_viability(weights=adapted)
        else:
            v_result = compute_viability()
    except Exception as exc:
        logger.warning(f"compute_viability 调用失败: {exc}")
        return None

    record = {
        "timestamp": _now_iso(),
        "source": "openllm.viability_logger",
        "v_result": v_result,
    }
    if adapted is not None:
        record["weights_used"] = adapted
    target = log_path or VIABILITY_LOG
    try:
        _append_jsonl(target, record)
    except Exception as exc:
        logger.warning(f"写入V值日志失败: {exc}")
        return None
    # ── T3（BurnInGate P0-C, 2026-09-06）：V值并入进化总线，拆断头管 ──
    # bus.py 声明 viability.update 类型已久但无生产者=断头管重演中。
    # BurnInGate 注册为消费者（"谁读它读完触发什么"闭环：V值→门禁判定）。
    # 降级铁律：总线故障静默，不阻塞V值日志主流程。
    if bus is not None:
        try:
            if bus == "auto":
                from openllm.evolution.bus import EvolutionBus
                bus_obj = EvolutionBus()
            else:
                bus_obj = bus
            from openllm.evolution.bus import DigestEvent
            bus_obj.registry.register("viability.update", "burnin_gate")
            v_result = record.get("v_result", {})
            bus_obj.append(DigestEvent(
                type="viability.update",
                producer="viability_logger",
                payload={
                    "V": v_result.get("V"),
                    "components": v_result.get("components"),
                    "missing": v_result.get("missing"),
                    "timestamp": record.get("timestamp"),
                },
            ), warn_unconsumed=False)
        except Exception as exc:
            logger.debug(f"V值总线事件降级静默: {exc}")
    return record


def log_skill_event(
    event: str,
    skill: str,
    details: Optional[dict] = None,
    log_path: Optional[Path] = None,
) -> Optional[dict]:
    """记录一条技能生命周期事件。

    Args:
        event: 事件类型（建议: 'adopted'/'retired'/'gated_pass'/'gated_reject'）。
        skill: 技能名称。
        details: 可选附加信息。
        log_path: 日志文件路径覆盖（测试用）。

    Returns:
        写入的记录 dict；失败返回 None。
    """
    record: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "event": event,
        "skill": skill,
    }
    if details is not None:
        record["details"] = details

    target = log_path or SKILL_LIFECYCLE_LOG
    try:
        _append_jsonl(target, record)
    except Exception as exc:
        logger.warning(f"写入技能生命周期日志失败: {exc}")
        return None
    return record


def read_log(log_path: Path, last_n: int = 10) -> List[dict]:
    """读取日志文件的最后 N 条记录。

    Args:
        log_path: 日志文件路径。
        last_n: 返回的记录数上限。

    Returns:
        最后 N 条有效记录的列表（文件不存在返回空列表，损坏行跳过）。
    """
    if not log_path.exists():
        return []
    records: List[dict] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return records[-last_n:]


# ═══ CLI ═══


def main() -> None:
    """CLI 入口：执行一次 log_viability() 并输出，或 --read N 读日志。"""
    parser = argparse.ArgumentParser(
        description="openLLM P1-2 观测仪表：V值日志采样与查看"
    )
    parser.add_argument(
        "--read",
        type=int,
        default=None,
        metavar="N",
        help="读取最后 N 条 V 值日志并输出",
    )
    args = parser.parse_args()

    if args.read is not None:
        records = read_log(VIABILITY_LOG, last_n=args.read)
        for rec in records:
            print(json.dumps(rec, ensure_ascii=False))
        return

    record = log_viability()
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
