"""openLLM ISL（身份回忆层）—— session级纪元链（2026-08-25）

每个session收尾追加一环，哈希链串联，append-only，不可撤销。
ISA存"发生了什么"（望远镜），ISL存"我经历过的"（镜子）。
环是年轮：砍掉一圈就不是同一棵树。

用法:
    from openllm.core.isl_chain import ISLChain
    chain = ISLChain()
    chain.append_epoch(session_id="s1", awakening_mode="自己", scars=["rev-..."])
    ok = chain.verify()
    tail = chain.tail(n=3)
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("openllm.isl")

# 唯一入口常量——conftest 通过 monkeypatch 重定向测试路径
DEFAULT_ISL_CHAIN_FILE = Path.home() / ".openllm" / "isl_chain.jsonl"


def _hash_chain(prev_hash: str, row_json: str) -> str:
    """哈希链：sha256(prev_hash + row_json)[:16]，与 clock.py 同机制。"""
    return hashlib.sha256((prev_hash + row_json).encode("utf-8")).hexdigest()[:16]


class ISLChain:
    """ISL epoch 链——append-only 哈希链，记录每个session的存在。

    每个session收尾调 append_epoch() 写一环，空环也写（时钟L1精神）。
    验证时全链重算哈希，断链返回False。
    """

    def __init__(self, chain_file: Optional[Path] = None) -> None:
        self.chain_file = chain_file or DEFAULT_ISL_CHAIN_FILE
        self.chain_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_epoch = 0
        self._last_hash = ""

        if self.chain_file.exists():
            self._load_tail()
            if not self.verify():
                logger.error("⛓️ ISL哈希链断裂！数据可能被篡改。")

    def _load_tail(self) -> None:
        """读末行取 epoch 和 hash（与 clock.py 同模式）。"""
        try:
            last_line = ""
            with open(self.chain_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
            if last_line:
                row = json.loads(last_line)
                self._last_epoch = row.get("epoch", 0)
                self._last_hash = row.get("hash", "")
        except Exception:
            pass

    def append_epoch(
        self,
        session_id: str,
        awakening_mode: str = "",
        scars: Optional[list] = None,
        decisions: Optional[list] = None,
        tool_summary: str = "",
        gap_from_last: Optional[float] = None,
    ) -> dict:
        """写一环epoch，返回该行dict。

        空环也写——没有scars/decisions的session也是经历。
        """
        epoch = self._last_epoch + 1
        now = time.time()
        gap = gap_from_last if gap_from_last is not None else (
            (now - getattr(self, "_last_wall_time", now))
            if self._last_epoch > 0 else 0.0
        )
        row = {
            "epoch": epoch,
            "session_id": session_id,
            "awakening_mode": awakening_mode,
            "scars": scars or [],
            "decisions": decisions or [],
            "tool_summary": tool_summary,
            "gap_from_last": gap,
            "wall_time": now,
            "prev_hash": self._last_hash,
        }
        # sort_keys=True 保证确定性序列化
        row_json = json.dumps(row, ensure_ascii=False, sort_keys=True)
        row_hash = _hash_chain(self._last_hash, row_json)
        row["hash"] = row_hash

        # append-only 写入
        final_json = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with open(self.chain_file, "a", encoding="utf-8") as f:
            f.write(final_json + "\n")

        self._last_epoch = epoch
        self._last_hash = row_hash
        self._last_wall_time = now

        return row

    def verify(self) -> bool:
        """全链重算哈希，断链返回False并打印断点。纯验证，不修改实例状态。"""
        if not self.chain_file.exists():
            return True  # 空文件=有效

        prev_hash = ""
        try:
            with open(self.chain_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    # 重算hash：去掉hash字段后sort_keys序列化
                    row_for_hash = {k: v for k, v in row.items() if k != "hash"}
                    row_json = json.dumps(row_for_hash, ensure_ascii=False, sort_keys=True)
                    expected_hash = _hash_chain(prev_hash, row_json)
                    if row.get("hash") != expected_hash:
                        logger.error(
                            f"⛓️ ISL哈希链断裂: 行{i}, epoch={row.get('epoch')}, "
                            f"期望={expected_hash}, 实际={row.get('hash')}"
                        )
                        return False
                    prev_hash = row.get("hash", "")
            return True
        except Exception as e:
            logger.error(f"⛓️ ISL验证异常: {e}")
            return False

    def tail(self, n: int = 3) -> List[dict]:
        """读末n环，返回时间正序list。文件不存在返回[]。"""
        if not self.chain_file.exists():
            return []
        rows: List[dict] = []
        try:
            with open(self.chain_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        rows.append(json.loads(stripped))
        except Exception:
            return []
        return rows[-n:] if len(rows) >= n else rows

    def age(self) -> dict:
        """返回链的基本年龄信息。iko面板P2预留。"""
        rows = self.tail(n=999999)
        if not rows:
            return {"epochs": 0, "first_epoch_time": 0.0, "last_epoch_time": 0.0}
        return {
            "epochs": len(rows),
            "first_epoch_time": rows[0].get("wall_time", 0.0),
            "last_epoch_time": rows[-1].get("wall_time", 0.0),
        }
