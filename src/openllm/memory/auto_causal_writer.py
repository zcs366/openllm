"""
AutoCausalWriter — 自动因果记忆写入器
=====================================

让因果记忆的数据流自动灌入，不再"管线通了但没水"。

用法:
    from openllm.memory.auto_causal_writer import AutoCausalWriter
    
    writer = AutoCausalWriter()
    writer.record(
        action="pip install torch",
        prediction="会成功",
        actual="ModuleNotFoundError",
        success=False,
        context="dependency check"
    )

写入:
    - ~/.openllm/memory/causal/<memory_id>.json (CausalMemoryStore格式)
    - ~/.openllm/memory/causal/write_log.jsonl (审计日志)
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openllm.auto_causal_writer")


class AutoCausalWriter:
    """自动因果记忆写入器。"""

    def __init__(self, store_dir: Optional[Path] = None):
        """
        Args:
            store_dir: 因果记忆存储目录，默认取共享常量
                       causal_memory.DEFAULT_STORE_DIR（~/.openllm/memory/causal）。
                       延迟导入该常量，使测试的 conftest 重定向在调用时生效。
        """
        if store_dir is None:
            from .causal_memory import DEFAULT_STORE_DIR
            store_dir = DEFAULT_STORE_DIR
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.store_dir / "write_log.jsonl"

    def record(
        self,
        action: str,
        prediction: str,
        actual: str,
        success: bool,
        context: str = "",
    ) -> dict:
        """
        记录一条因果记忆。

        自动计算 delta_magnitude:
            - success=True → delta_magnitude=0 (预测正确)
            - success=False → delta_magnitude=min(1.0, len(delta_text)/100)

        Args:
            action: 做了什么
            prediction: 预测会发生什么
            actual: 实际发生了什么
            success: 操作是否成功
            context: 上下文信息

        Returns:
            写入的因果记忆字典
        """
        # 计算 delta_magnitude
        delta_text = f"{prediction} -> {actual}" if prediction and actual else actual
        if success:
            delta_magnitude = 0.0
        else:
            delta_magnitude = min(1.0, len(delta_text) / 100.0)

        # 生成 lesson
        status = "成功" if success else "失败"
        lesson = f"{status}: {action[:50]} -> {actual[:50]}"

        # 构建因果记忆记录
        memory_id = f"auto-{int(time.time() * 1000) % 100000000:08d}"
        record_data = {
            "memory_id": memory_id,
            "created_at": time.time(),
            "action_signature": action,
            "context_features": context.split()[:5] if context else [],
            "prediction": prediction,
            "prediction_confidence": 0.5,
            "actual_result": actual,
            "actual_success": success,
            "delta": delta_text,
            "delta_magnitude": delta_magnitude,
            "lesson": lesson,
            "source": "auto_causal_writer",
            "trust_level": "internal",
            "importance": 0.5,
            "last_accessed": time.time(),
            "access_count": 0,
            "tags": [],
            "session_id": "",
        }

        # 写入因果记忆文件
        try:
            path = self.store_dir / f"{memory_id}.json"
            path.write_text(
                json.dumps(record_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to write causal memory {memory_id}: {e}")
            return record_data

        # 写入审计日志
        try:
            log_entry = {
                "timestamp": time.time(),
                "memory_id": memory_id,
                "action": action[:100],
                "success": success,
                "delta_magnitude": delta_magnitude,
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write audit log for {memory_id}: {e}")

        logger.info(
            f"AutoCausalWriter: {memory_id} | "
            f"delta={delta_magnitude:.2f} | {lesson[:60]}"
        )
        return record_data
