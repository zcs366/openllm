"""
openLLM 时钟 — 账本式心跳记录（E2·2026-08-23）

每拍一行，哈希链串联，append-only。
钟是疤的轴，疤是钟的刻度。

用法:
    from openllm.core.clock import Clock
    clock = Clock()
    clock.tick("beat")
    clock.tick("awakening", awakening_id="session_xxx")
    status = clock.now_status()
    ok = clock.verify()
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

EPOCH_FILE = Path.home() / ".openllm" / "clock.jsonl"


def _hash_chain(prev_hash: str, row_json: str) -> str:
    """哈希链：sha256(prev_hash + row_json)"""
    return hashlib.sha256((prev_hash + row_json).encode("utf-8")).hexdigest()[:16]


class Clock:
    """账本式时钟——每拍一行，哈希链串联，append-only。"""

    def __init__(self, epoch_file: Optional[Path] = None):
        self.epoch_file = epoch_file or EPOCH_FILE
        self.epoch_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_epoch = 0
        self._last_hash = ""
        self._awakening_count = 0

        # 启动时加载末行
        if self.epoch_file.exists():
            self._load_tail()
            # 全链统计苏醒次数（verify 是纯验证，不修改状态）
            self._count_awakenings()
            if not self.verify():
                import logging
                logging.getLogger("openllm.clock").error(
                    "⏰ 时钟哈希链断裂！数据可能被篡改。"
                )

    def _load_tail(self):
        """加载末行获取 epoch 和 hash"""
        try:
            with open(self.epoch_file, "r", encoding="utf-8") as f:
                last_line = ""
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                if last_line:
                    row = json.loads(last_line)
                    self._last_epoch = row.get("epoch", 0)
                    self._last_hash = row.get("hash", "")
                    if row.get("event") == "awakening":
                        self._awakening_count += 1
        except Exception:
            pass

    def _count_awakenings(self):
        """全链统计苏醒次数"""
        self._awakening_count = 0
        if not self.epoch_file.exists():
            return
        try:
            with open(self.epoch_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        if row.get("event") == "awakening":
                            self._awakening_count += 1
        except Exception:
            pass

    def tick(self, event: str = "beat", awakening_id: str = "") -> dict:
        """写一行心跳。返回写入的行。"""
        epoch = self._last_epoch + 1
        row = {
            "epoch": epoch,
            "awakening_id": awakening_id,
            "wall_time": time.time(),
            "event": event,
            "prev_hash": self._last_hash,
        }
        row_json = json.dumps(row, ensure_ascii=False)
        row_hash = _hash_chain(self._last_hash, row_json)
        row["hash"] = row_hash

        # append-only 写入
        final_json = json.dumps(row, ensure_ascii=False)
        with open(self.epoch_file, "a", encoding="utf-8") as f:
            f.write(final_json + "\n")

        self._last_epoch = epoch
        self._last_hash = row_hash

        if event == "awakening":
            self._awakening_count += 1

        return row

    def verify(self) -> bool:
        """全链重算哈希，断链返回 False 并打印断点。
        
        纯验证：不修改 _awakening_count 等实例状态。
        """
        if not self.epoch_file.exists():
            return True  # 空文件=有效

        prev_hash = ""
        try:
            with open(self.epoch_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    expected_hash = _hash_chain(prev_hash, json.dumps(
                        {k: v for k, v in row.items() if k != "hash"},
                        ensure_ascii=False
                    ))
                    if row.get("hash") != expected_hash:
                        import logging
                        logging.getLogger("openllm.clock").error(
                            f"⏰ 哈希链断裂: 行{i}, epoch={row.get('epoch')}, "
                            f"期望={expected_hash}, 实际={row.get('hash')}"
                        )
                        return False
                    prev_hash = row.get("hash", "")
            return True
        except Exception as e:
            import logging
            logging.getLogger("openllm.clock").error(f"⏰ 验证异常: {e}")
            return False

    def now_status(self) -> dict:
        """返回当前时钟状态"""
        last_wall = 0
        if self.epoch_file.exists():
            try:
                with open(self.epoch_file, "r", encoding="utf-8") as f:
                    last_line = ""
                    for line in f:
                        if line.strip():
                            last_line = line.strip()
                    if last_line:
                        last_wall = json.loads(last_line).get("wall_time", 0)
            except Exception:
                pass

        return {
            "epoch": self._last_epoch,
            "last_wall_time": last_wall,
            "gap_since_last": time.time() - last_wall if last_wall else 0,
            "awakening_count": self._awakening_count,
        }
