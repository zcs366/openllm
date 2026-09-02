"""厌倦律 IoR（Inhibition of Return）过滤器。

蜻蜓十律⑤：切换目标后旧目标轨迹被抑制——处理过的主题不再重复关注。
IoR = Inhibition of Return

设计：
  - append-only JSONL 存储：~/.openllm/iai/ior_topics.jsonl
  - mark_handled(topic)：记录已处理主题（按关键词去重）
  - suppress(candidates)：过滤掉已处理主题（返回未处理的新主题）
  - get_recent(limit)：最近处理的（供注入上下文，抑制提示）
  - 默认不影响：无已处理主题时行为不变（军规十一/十三：增强不替代）

红线：不改变搜索执行本身，只改上下文注入。
"""
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional


def _topic_hash(topic: str) -> str:
    """主题去重键：归一化后取sha256前8位。"""
    normalized = " ".join(topic.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


class IoRFilter:
    """厌倦律过滤器——抑制已处理主题的重复关注。

    append-only JSONL 存储，永不删除，永不修改。
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or (Path.home() / ".openllm" / "iai" / "ior_topics.jsonl")
        self._seen_hashes: set = set()
        self._load_existing()

    def _load_existing(self):
        """加载已有记录到内存索引（不修改文件）。"""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            self._seen_hashes.add(record["hash"])
                        except (json.JSONDecodeError, KeyError):
                            continue
            except OSError:
                pass

    def mark_handled(self, topic: str):
        """记录已处理主题（append-only，自动去重）。"""
        h = _topic_hash(topic)
        if h in self._seen_hashes:
            return  # 已记录，跳过
        self._seen_hashes.add(h)
        record = {
            "topic": topic,
            "hash": h,
            "timestamp": time.time(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def suppress(self, candidates: List[str]) -> List[str]:
        """过滤已处理主题，返回未处理的新主题。"""
        return [c for c in candidates if _topic_hash(c) not in self._seen_hashes]

    def get_recent(self, limit: int = 10) -> List[str]:
        """最近处理的主题列表（最新在前）。"""
        records = []
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except (json.JSONDecodeError, KeyError):
                            continue
            except OSError:
                pass
        records.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        return [r["topic"] for r in records[:limit]]
