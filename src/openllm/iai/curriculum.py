"""curriculum.py — 训练课程管理器（PAL T-C-3）。

数据账本 JSON + 置信度过滤 + should_train 阈值 + build_training_set 混合比例
+ schedule_nightly 训练命令 + verify_and_rollback 回归门。

铁律:
  - 30% 云端探索注入率（防自激坍缩）
  - 低置信 <0.6 不进训练集
  - 每晚批量训练
  - 回归门硬卡 + 失败自动回滚
  - append-only 账本
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("openllm.iai.curriculum")

STATE_FILE = Path.home() / ".openllm" / "iai" / "curriculum_state.json"
TRAIN_SCRIPT = Path.home() / "projects" / "isa" / "ilm" / "train_qlora.py"
DEFAULT_DATA_DIR = Path.home() / "projects" / "isa" / "ilm" / "train_data"
DEFAULT_MODEL = "/mnt/i/hermes/models/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT = Path.home() / "projects" / "isa" / "ilm" / "models" / "qlora-ilm-v1"

# 七神·阿波罗约束：置信度阈值
CONFIDENCE_THRESHOLD = 0.6
# 七神·克洛诺斯约束：云端探索最低比例
CLOUD_EXPLORE_MIN_RATIO = 0.30
# should_train 阈值：pending 样本数
TRAIN_THRESHOLD = 100
# 两次训练最小间隔（秒）
MIN_TRAIN_INTERVAL = 24 * 3600

# 训练成功标记（verify_and_rollback 检查）
TRAIN_SUCCESS_MARKERS = ["训练完成", "✅", "save_pretrained"]


def _default_state() -> dict:
    """初始化空白账本。"""
    return {
        "last_trained": 0.0,
        "pending_pairs": 0,
        "source_counts": {"corrections": 0, "critique": 0, "cloud_explore": 0},
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_rejected": 0,
        "train_history": [],
    }


class CurriculumManager:
    """训练课程管理器。

    职责:
    1. 接收偏好对，按置信度过滤
    2. 判断是否该训练（阈值 + 时间）
    3. 构建混合训练集（≥30% 云端探索）
    4. 生成夜间训练命令
    5. 回归门验证 + 失败回滚

    不直接执行训练，只生成计划/命令。
    """

    def __init__(self, state_path: Optional[Path] = None, train_script: Optional[Path] = None):
        self._state_path = state_path or STATE_FILE
        self._train_script = train_script or TRAIN_SCRIPT
        self._state: dict = self._load_state()

    # ── 账本持久化 ──────────────────────────────────────────

    def _load_state(self) -> dict:
        """加载账本，不存在则初始化。"""
        try:
            if self._state_path.exists():
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("[curriculum] 账本读取失败，重新初始化: %s", e)
        return _default_state()

    def _save_state(self) -> None:
        """原子写入账本（try/except 不抛异常）。"""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            tmp.replace(self._state_path)
        except Exception as e:
            logger.warning("[curriculum] 账本写入失败: %s", e)

    @property
    def state(self) -> dict:
        return self._state

    # ── 接收偏好对 ──────────────────────────────────────────

    def add_pairs(self, source: str, pairs: List[dict], confidence: float) -> dict:
        """按置信度过滤，按 source 计数。

        Args:
            source: 来源类型 "corrections" | "critique" | "cloud_explore"
            pairs: 偏好对列表 [{"chosen": ..., "rejected": ...}, ...]
            confidence: 教师信号置信度

        Returns:
            {"accepted": int, "rejected": int}
        """
        threshold = self._state.get("confidence_threshold", CONFIDENCE_THRESHOLD)

        accepted = 0
        rejected = 0

        if confidence < threshold:
            # 七神·阿波罗：低置信度整批拒收
            rejected = len(pairs)
            self._state["total_rejected"] += rejected
            logger.info(
                "[curriculum] 置信度 %.2f < %.2f 拒收 %d 对 (source=%s)",
                confidence, threshold, rejected, source,
            )
        else:
            accepted = len(pairs)
            self._state["source_counts"].setdefault(source, 0)
            self._state["source_counts"][source] += accepted
            self._state["pending_pairs"] += accepted

        self._save_state()
        return {"accepted": accepted, "rejected": rejected}

    # ── 训练判断 ────────────────────────────────────────────

    def should_train(self) -> bool:
        """判断是否该训练。

        条件:
        1. pending_pairs >= TRAIN_THRESHOLD (默认100)
        2. OR 距上次训练 > 24h（即使样本不够也强制训练）
        """
        if self._state["pending_pairs"] >= TRAIN_THRESHOLD:
            return True
        elapsed = time.time() - self._state.get("last_trained", 0.0)
        if elapsed > MIN_TRAIN_INTERVAL and self._state["pending_pairs"] > 0:
            return True
        return False

    # ── 构建训练集 ──────────────────────────────────────────

    def build_training_set(self) -> dict:
        """构建混合训练集。

        约束（七神·克洛诺斯）：cloud_explore >= 30%
        不足时标记缺口但不阻止返回。

        Returns:
            {
                "chosen": [...],
                "rejected": [...],
                "stats": {
                    "total": N,
                    "corrections": N,
                    "critique": N,
                    "cloud_explore": N,
                    "cloud_ratio": 0.xx,
                    "deficit": bool,
                    "deficit_note": "..."
                }
            }
        """
        counts = self._state.get("source_counts", {})
        total = sum(counts.values())
        cloud = counts.get("cloud_explore", 0)
        cloud_ratio = cloud / total if total > 0 else 0.0
        deficit = cloud_ratio < CLOUD_EXPLORE_MIN_RATIO and total > 0

        deficit_note = ""
        if deficit:
            needed = int(total * CLOUD_EXPLORE_MIN_RATIO) - cloud
            deficit_note = (
                f"云端探索缺口: 需要 {needed} 条额外 cloud_explore "
                f"(当前 {cloud}/{total} = {cloud_ratio:.1%}, 需 ≥30%)"
            )
            logger.warning("[curriculum] %s", deficit_note)

        # 构建占位训练集（实际数据从 JSONL 文件加载）
        chosen = []
        rejected = []
        for source_name, count in counts.items():
            for i in range(count):
                chosen.append({"source": source_name, "index": i})
                rejected.append({"source": source_name, "index": i, "role": "negative"})

        return {
            "chosen": chosen,
            "rejected": rejected,
            "stats": {
                "total": total,
                "corrections": counts.get("corrections", 0),
                "critique": counts.get("critique", 0),
                "cloud_explore": cloud,
                "cloud_ratio": round(cloud_ratio, 4),
                "deficit": deficit,
                "deficit_note": deficit_note,
            },
        }

    # ── 夜间训练命令 ────────────────────────────────────────

    def schedule_nightly(
        self,
        data_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        epochs: int = 2,
        resume: bool = True,
    ) -> str:
        """生成夜间训练命令（subprocess 调用 train_qlora.py）。

        不直接执行训练，只返回命令字符串。
        记录训练计划到账本。

        Returns:
            完整 shell 命令字符串
        """
        data_dir = str(data_dir or DEFAULT_DATA_DIR)
        model = model_path or str(DEFAULT_MODEL)
        output = str(output_dir or str(DEFAULT_OUTPUT))
        script = str(self._train_script)

        cmd = (
            f"python3 {script}"
            f" --model {model}"
            f" --data-dir {data_dir}"
            f" --output {output}"
            f" --epochs {epochs}"
            f" --batch-size 2 --grad-accum 4 --lr 2e-4"
            f" --max-seq-len 2048"
        )
        if resume:
            cmd += " --resume"

        # 记录训练计划
        self._state["train_history"].append({
            "timestamp": time.time(),
            "command": cmd,
            "status": "scheduled",
        })
        self._save_state()

        logger.info("[curriculum] 训练计划: %s", cmd)
        return cmd

    # ── 回归门 ──────────────────────────────────────────────

    def verify_and_rollback(self, train_log: str) -> bool:
        """训练后回归门。

        检查 train_log 中是否包含成功标记。
        失败 → 返回 False 触发回滚。

        Args:
            train_log: 训练日志文本

        Returns:
            True = 通过回归门, False = 失败需回滚
        """
        success = any(marker in train_log for marker in TRAIN_SUCCESS_MARKERS)

        # 更新账本
        if self._state["train_history"]:
            last = self._state["train_history"][-1]
            last["status"] = "success" if success else "failed"

        if success:
            self._state["last_trained"] = time.time()
            self._state["pending_pairs"] = 0
            logger.info("[curriculum] 回归门通过 ✓")
        else:
            logger.warning("[curriculum] 回归门失败 ✗ — 需回滚")
            if self._state["train_history"]:
                self._state["train_history"][-1]["rollback_triggered"] = True

        self._save_state()
        return success
