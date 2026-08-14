"""
IO-S 迷雾区 #1: Ecosystem Integrity — AI内容污染监控
=====================================================

问题：Agent输出的内容进入信息生态，可能被爬取作为训练数据，
造成模型坍塌（model collapse）反馈循环。
同时，Agent自身也可能在消费被AI污染的输入。

MVP：ContentFingerprinter
- 追踪所有输出的内容指纹（SHA-256）
- 检测输入是否来自AI生态（自引用检测）
- 监控内容多样性趋势（坍塌信号）

验收标准：
1. 每次Agent输出自动记录fingerprint
2. 输入中出现历史输出的片段时触发告警
3. 内容多样性指数持续下降时触发告警
"""

import hashlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional


class ContentFingerprinter:
    """AI内容生态完整性监控器。"""

    DB_PATH = Path.home() / ".openllm" / "output" / "ecosystem" / "fingerprints.jsonl"
    # 最近N个输出的n-gram集合，用于检测自引用
    _RECENT_WINDOW = 200
    # 多样性滑动窗口（最近K个周期的指纹数）
    _DIVERSITY_WINDOW = 50
    # 多样性告警阈值（unique_ratio < threshold → 坍塌风险）
    _DIVERSITY_THRESHOLD = 0.3

    def __init__(self):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._recent_outputs: deque[str] = deque(maxlen=self._RECENT_WINDOW)
        self._diversity_window: deque[float] = deque(maxlen=self._DIVERSITY_WINDOW)
        self._seen_hashes: set[str] = set()
        self._load_existing()

    def _load_existing(self):
        """启动时加载历史指纹，重建状态。"""
        if not self.DB_PATH.exists():
            return
        try:
            with open(self.DB_PATH) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        h = entry.get("hash", "")
                        if h:
                            self._seen_hashes.add(h)
                            self._recent_outputs.append(entry.get("content_snippet", ""))
        except Exception:
            pass

    def _hash_content(self, content: str) -> str:
        """SHA-256内容指纹。"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _extract_ngrams(self, text: str, n: int = 6) -> set[str]:
        """提取字符级n-gram，用于片段匹配。"""
        text = text.lower().strip()
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    # ── 核心API ──────────────────────────────────────

    def record_output(self, content: str, model_id: str = "",
                      session_id: str = "") -> dict:
        """记录一次Agent输出。
        
        Returns: {hash, is_duplicate, diversity_score, alerts}
        """
        content_hash = self._hash_content(content)
        is_duplicate = content_hash in self._seen_hashes
        self._seen_hashes.add(content_hash)
        self._recent_outputs.append(content[:200])

        # 写入磁盘
        entry = {
            "ts": time.time(),
            "hash": content_hash[:16],
            "model": model_id,
            "session": session_id,
            "content_snippet": content[:120],
            "length": len(content),
        }
        with open(self.DB_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 检测告警
        alerts = []
        if is_duplicate:
            alerts.append("DUPLICATE_HASH")

        diversity = self._check_diversity()
        # 数据量太少时不做diversity告警（样本不足）
        if len(self._seen_hashes) >= 5 and diversity < self._DIVERSITY_THRESHOLD:
            alerts.append(f"LOW_DIVERSITY:{diversity:.3f}")

        return {
            "hash": content_hash[:16],
            "is_duplicate": is_duplicate,
            "diversity_score": round(diversity, 4),
            "alerts": alerts,
        }

    def check_input(self, user_input: str) -> dict:
        """检测用户输入是否包含AI生态内容（自引用检测）。
        
        Returns: {is_self_referential, match_count, alerts}
        """
        input_ngrams = self._extract_ngrams(user_input, n=8)
        if not input_ngrams:
            return {"is_self_referential": False, "match_count": 0, "alerts": []}

        match_count = 0
        matched_outputs = []

        for past_output in self._recent_outputs:
            if not past_output:
                continue
            out_ngrams = self._extract_ngrams(past_output, n=8)
            overlap = len(input_ngrams & out_ngrams)
            total = len(input_ngrams | out_ngrams)
            if total > 0 and overlap / total > 0.3:
                match_count += 1
                matched_outputs.append(past_output[:60])

        alerts = []
        if match_count >= 2:
            alerts.append(f"SELF_REFERENTIAL:{match_count}")

        return {
            "is_self_referential": match_count >= 2,
            "match_count": match_count,
            "matched_snippets": matched_outputs[:3],
            "alerts": alerts,
        }

    def _check_diversity(self) -> float:
        """计算内容多样性指数：唯一hash占比。"""
        if not self._seen_hashes:
            return 1.0
        unique_ratio = len(self._seen_hashes) / max(len(self._seen_hashes) + 10, 1)
        self._diversity_window.append(unique_ratio)
        if len(self._diversity_window) < 3:
            return unique_ratio
        # 趋势：最近5个周期的均值
        recent = list(self._diversity_window)[-5:]
        return sum(recent) / len(recent)

    def report(self) -> dict:
        """生成生态完整性报告。"""
        return {
            "total_tracked": len(self._seen_hashes),
            "recent_outputs": len(self._recent_outputs),
            "diversity_score": round(self._check_diversity(), 4),
            "diversity_alert": self._check_diversity() < self._DIVERSITY_THRESHOLD,
            "db_path": str(self.DB_PATH),
        }


# ── 集成点 ──────────────────────────────────────────

_fingerprinter: Optional[ContentFingerprinter] = None

def get_fingerprinter() -> ContentFingerprinter:
    global _fingerprinter
    if _fingerprinter is None:
        _fingerprinter = ContentFingerprinter()
    return _fingerprinter

def on_output_emitted(content: str, model_id: str = "", session_id: str = "") -> dict:
    """IO-S Phase 11 钩子：Agent输出后自动调用。"""
    return get_fingerprinter().record_output(content, model_id, session_id)

def on_input_received(user_input: str) -> dict:
    """IO-S Phase 1 钩子：用户输入时自动调用。"""
    return get_fingerprinter().check_input(user_input)
