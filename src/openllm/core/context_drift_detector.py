"""context_drift_detector.py — IAX+IOS 上下文漂移检测（Layer 11 SSVP）

纯规则驱动，零LLM调用。检测对话中上下文主题的漂移程度。
  - 关键词Jaccard相似度
  - TF-IDF向量余弦相似度（简化版）
  - 结构漂移（长度/格式突变）
  - 综合漂移 = 1 - 加权平均相似度
状态持久化到 ~/.hermes/core/drift_state.json
"""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openllm.context_drift")

# 持久化路径
_STATE_DIR = Path.home() / ".hermes" / "core"
_STATE_PATH = _STATE_DIR / "drift_state.json"

# 中英文停用词（精简集）
_STOPWORDS = frozenset(
    ("的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
     "自己 这 他 她 它 们 那 里 为 么 什么 怎么 呢 吧 吗 嗯 啊 哈 呀 "
     "a an the is are was were be been being "
     "i me my we our you your he him his she her it its they them their "
     "this that these those and or but if then so for of in on at to with "
     "from by as not no can will do does did have has had may might shall "
     "just also very really much more most some any all each every "
    ).split()
)


def _tokenize(text: str) -> list[str]:
    """中英文分词：英文按空格+去标点，中文按字符bigram + 单字。"""
    tokens = []
    # 英文单词
    for m in re.finditer(r"[a-zA-Z]{2,}", text.lower()):
        w = m.group()
        if w not in _STOPWORDS:
            tokens.append(w)
    # 中文：2-gram
    cn = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in cn:
        # 去单字符停用词
        chars = [c for c in seg if c not in _STOPWORDS]
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])
        tokens.extend(chars)
    return tokens


def _jaccard(a: set, b: set) -> float:
    """Jaccard相似度。"""
    if not a and not b:
        return 1.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _cosine_sim(vec_a: Counter, vec_b: Counter) -> float:
    """简化TF-IDF余弦相似度（用TF近似，无IDF）。"""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _structural_drift(len_a: int, len_b: int) -> float:
    """结构漂移：对数尺度长度比，减少长句偏差。返回0-1。"""
    import math as _math
    if max(len_a, len_b) == 0:
        return 0.0
    # 对数尺度：10字vs100字的漂移 < 10字vs1000字
    log_a = _math.log1p(len_a)
    log_b = _math.log1p(len_b)
    if max(log_a, log_b) == 0:
        return 0.0
    ratio = min(log_a, log_b) / max(log_a, log_b)
    return 1.0 - ratio


class ContextDriftDetector:
    """上下文漂移检测器——Layer 11 SSVP（IAX+IOS）。

    纯统计方法，不调用LLM。综合三个维度评估上下文漂移：
    1. 关键词重叠率（Jaccard）
    2. 主题一致性（TF-IDF余弦）
    3. 结构漂移（长度突变）

    漂移值 = 1 - 加权平均相似度，范围0-1，越高漂移越严重。
    """

    def __init__(
        self,
        w_keyword: float = 0.4,
        w_topic: float = 0.4,
        w_struct: float = 0.2,
    ):
        """初始化权重。默认关键词40%，主题40%，结构20%。"""
        self.w_keyword = w_keyword
        self.w_topic = w_topic
        self.w_struct = w_struct
        self._history: list[dict] = []
        self._load_state()

    def compute_drift(self, message_a: str, message_b: str) -> float:
        """计算两条消息之间的漂移度（0-1）。

        Args:
            message_a: 前一条消息
            message_b: 后一条消息

        Returns:
            漂移度，0表示完全一致，1表示完全不同
        """
        tokens_a = _tokenize(message_a)
        tokens_b = _tokenize(message_b)
        set_a, set_b = set(tokens_a), set(tokens_b)
        counter_a, counter_b = Counter(tokens_a), Counter(tokens_b)

        # 1. 关键词Jaccard相似度 → 漂移
        kw_sim = _jaccard(set_a, set_b)
        kw_drift = 1.0 - kw_sim

        # 2. TF-IDF余弦相似度 → 漂移
        topic_sim = _cosine_sim(counter_a, counter_b)
        topic_drift = 1.0 - topic_sim

        # 3. 结构漂移
        struct_drift = _structural_drift(len(message_a), len(message_b))

        # 综合漂移
        drift = (
            self.w_keyword * kw_drift
            + self.w_topic * topic_drift
            + self.w_struct * struct_drift
        )
        return round(min(max(drift, 0.0), 1.0), 4)

    def detect_drift_in_conversation(
        self, messages: list[str], threshold: float = 0.7
    ) -> list[dict]:
        """检测对话中的漂移点。

        Args:
            messages: 消息列表
            threshold: 漂移阈值，超过则视为漂移点

        Returns:
            漂移点列表，每个元素包含索引、漂移度、消息摘要
        """
        drift_points = []
        for i in range(1, len(messages)):
            drift = self.compute_drift(messages[i - 1], messages[i])
            if drift >= threshold:
                entry = {
                    "index": i,
                    "drift": drift,
                    "msg_a": messages[i - 1][:80],
                    "msg_b": messages[i][:80],
                }
                drift_points.append(entry)
                self._record(entry)
        return drift_points

    def get_drift_report(self) -> dict:
        """获取漂移统计报告。

        Returns:
            包含总检测次数、平均漂移、最大漂移、漂移分布等统计
        """
        if not self._history:
            return {"total_checks": 0, "message": "尚无漂移检测记录"}
        drifts = [e["drift"] for e in self._history]
        high = sum(1 for d in drifts if d >= 0.7)
        mid = sum(1 for d in drifts if 0.4 <= d < 0.7)
        low = sum(1 for d in drifts if d < 0.4)
        return {
            "total_checks": len(drifts),
            "avg_drift": round(sum(drifts) / len(drifts), 4),
            "max_drift": round(max(drifts), 4),
            "min_drift": round(min(drifts), 4),
            "distribution": {"high(>=0.7)": high, "mid(0.4-0.7)": mid, "low(<0.4)": low},
        }

    def _record(self, entry: dict) -> None:
        """记录漂移事件并持久化。"""
        self._history.append(entry)
        # 保留最近200条
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self._save_state()

    def _save_state(self) -> None:
        """持久化到JSON。"""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            _STATE_PATH.write_text(
                json.dumps(self._history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"[drift] 状态持久化失败: {e}")

    def _load_state(self) -> None:
        """从JSON恢复状态。"""
        try:
            if _STATE_PATH.exists():
                self._history = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[drift] 状态加载失败，重置: {e}")
            self._history = []
