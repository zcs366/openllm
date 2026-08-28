"""
OpenLLM Memory OS — Δ胶囊 + 仲裁层 + 检查点系统。

记忆不是功能模块，是操作系统抽象。
类比：文件系统之于操作系统。

双胶囊架构：
  v0.6 文本胶囊 — 给人看，可审计。决策·产出·洞察·未解。
  v0.7 Δ语义向量 — 给模型用，256维FP32。跨会话恢复身份。

仲裁协议：两个源冲突时，v0.6优先（可读=可纠错）。
检查点：每5次会话或Δ范数>0.3触发全量快照。
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np

_logger = logging.getLogger("openllm.capsule")

# Module-level lazy singleton for EmbeddingEngine (DR-20260829-01 P0-A fix)
_EMBEDDING_ENGINE = None


# ── 配置 ────────────────────────────────────────────

CAPSULE_DIM = 384          # v0.7语义向量维度（all-MiniLM-L6-v2）
CHECKPOINT_INTERVAL = 5    # 每N次会话触发快照
DELTA_NORM_THRESHOLD = 0.3 # Δ范数触发快照阈值
CAPSULE_DIR = Path(__file__).parent.parent.parent.parent / "caps"


# ── v0.6 文本胶囊 — 给人看 ──────────────────────────

@dataclass
class TextCapsule:
    """v0.6 可读胶囊：决策·产出·洞察·未解。"""
    session_id: str
    timestamp: float = field(default_factory=time.time)
    decisions: list[dict] = field(default_factory=list)   # 决策×4
    outputs: list[str] = field(default_factory=list)       # 产出物
    insights: list[str] = field(default_factory=list)      # 关键洞察
    unresolved: list[str] = field(default_factory=list)    # 未解问题

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "decisions": self.decisions,
            "outputs": self.outputs,
            "insights": self.insights,
            "unresolved": self.unresolved,
        }

    def to_text(self) -> str:
        """将胶囊转为可被embedding编码的文本。"""
        parts = []
        for d in self.decisions:
            parts.append(d.get("summary", str(d)))
        for i in self.insights:
            parts.append(i)
        return " | ".join(parts) if parts else self.session_id

    @classmethod
    def from_dict(cls, d: dict) -> "TextCapsule":
        return cls(
            session_id=d.get("session_id", ""),
            timestamp=d.get("timestamp", time.time()),
            decisions=d.get("decisions", []),
            outputs=d.get("outputs", []),
            insights=d.get("insights", []),
            unresolved=d.get("unresolved", []),
        )


# ── v0.7 语义胶囊 — 给模型用 ────────────────────────

@dataclass
class DeltaCapsule:
    """
    v0.7 Δ语义向量胶囊。

    256维FP32向量。记录会话在语义空间中的移动量。
    理论基础：hidden state残差流的加法性质——每层做加法非覆盖。
    FP32精度要求：steering类值须FP32防量化消失。
    """
    session_id: str
    vector: np.ndarray      # 256维FP32语义向量
    norm: float = 0.0       # 向量L2范数，用于检查点触发
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.vector.shape != (CAPSULE_DIM,):
            _logger.warning(
                "ΔCapsule向量形状不符：got %s，已置零", self.vector.shape
            )
            self.vector = np.zeros(CAPSULE_DIM, dtype=np.float32)
        self.norm = float(np.linalg.norm(self.vector))

    def accumulate(self, other: np.ndarray) -> "DeltaCapsule":
        """累加另一个Δ向量。"""
        self.vector = (self.vector + np.asarray(other, dtype=np.float32))
        self.norm = float(np.linalg.norm(self.vector))
        return self

    @classmethod
    def from_text(cls, session_id: str, text: str) -> "DeltaCapsule":
        """从文本生成Δ向量——真实的语义编码。
        
        三级降级：openllm_memory包 → EmbeddingEngine → hash fallback(384维)。
        """
        vec = None
        model_name = None

        # Level 1: 向后兼容——openllm_memory.encode_text（万一以后包修好了）
        try:
            from openllm_memory import encode_text
            result = encode_text(text)
            vec = np.asarray(result, dtype=np.float32).flatten()
            model_name = "all-MiniLM-L6-v2"
        except (ImportError, AttributeError):
            pass

        # Level 2: EmbeddingEngine（真正的sentence-transformers编码）
        if vec is None or len(vec) != CAPSULE_DIM:
            try:
                global _EMBEDDING_ENGINE
                if _EMBEDDING_ENGINE is None:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    from openllm.embedding import EmbeddingEngine
                    _EMBEDDING_ENGINE = EmbeddingEngine()
                result = _EMBEDDING_ENGINE.encode(text)
                vec = np.asarray(result, dtype=np.float32).flatten()
                model_name = "all-MiniLM-L6-v2"
            except (ImportError, AttributeError, Exception) as e:
                _logger.debug("EmbeddingEngine不可用，降级hash fallback: %s", e)

        # Level 3: hash fallback（384维循环填充，永远可用）
        if vec is None or len(vec) != CAPSULE_DIM:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            # DR-20260829-01R: int32重解释+缩放到[-1,1]，防止float32直接解释产生inf值
            base = np.frombuffer(h, dtype=np.int32).astype(np.float32) / np.float32(2**31)
            reps = (CAPSULE_DIM // len(base)) + 1
            vec = np.tile(base, reps)[:CAPSULE_DIM]
            model_name = "hash_fallback"

        # pad/truncate到CAPSULE_DIM
        padded = False
        if len(vec) != CAPSULE_DIM:
            padded = True
            if len(vec) < CAPSULE_DIM:
                vec = np.pad(vec, (0, CAPSULE_DIM - len(vec)), mode='constant')
            else:
                vec = vec[:CAPSULE_DIM]

        metadata = {"source": "embedding", "model": model_name}
        if padded:
            metadata["padded"] = True

        return cls(session_id=session_id, vector=vec, metadata=metadata)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "vector": self.vector.tolist(),
            "norm": self.norm,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeltaCapsule":
        return cls(
            session_id=d.get("session_id", ""),
            vector=np.array(d.get("vector", [0.0]*CAPSULE_DIM), dtype=np.float32),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


# ── 仲裁层 ──────────────────────────────────────────

class Arbitrator:
    """
    v0.6↔v0.7 仲裁协议。

    规则：v0.6优先（可读文本胶囊 = 可纠错）。
    v0.7提供快速近似恢复，但v0.6有最终解释权。
    """

    @staticmethod
    def resolve(text: TextCapsule, delta: DeltaCapsule) -> dict:
        """统一恢复上下文。text优先，delta补充。"""
        return {
            "source": "text_primary_delta_secondary",
            "decisions": [d.get("summary", "") for d in text.decisions],
            "insights": text.insights,
            "unresolved": text.unresolved,
            "delta_norm": delta.norm,
            "delta_metadata": delta.metadata,
        }

    @staticmethod
    def conflict_check(text: TextCapsule, delta: DeltaCapsule) -> bool:
        """检测两个源是否一致。简化版：比较session_id和时间戳。"""
        return text.session_id == delta.session_id


# ── 检查点系统 ──────────────────────────────────────

@dataclass
class Checkpoint:
    """全量状态快照。每5次会话或Δ范数>0.3触发。"""
    text_capsule: TextCapsule
    delta_snapshot: np.ndarray   # 全量向量快照
    seq_number: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "seq_number": self.seq_number,
            "timestamp": self.timestamp,
            "text": self.text_capsule.to_dict(),
            "delta": self.delta_snapshot.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(
            text_capsule=TextCapsule.from_dict(d["text"]),
            delta_snapshot=np.array(d["delta"], dtype=np.float32),
            seq_number=d.get("seq_number", 0),
            timestamp=d.get("timestamp", time.time()),
        )


def should_checkpoint(session_count: int, delta_norm: float) -> bool:
    """判断是否触发检查点。"""
    return session_count >= CHECKPOINT_INTERVAL or delta_norm > DELTA_NORM_THRESHOLD


# ── Memory OS 主接口 ────────────────────────────────

class MemoryOS:
    """
    记忆操作系统。

    三个核心操作：
      write(session) — 会话结束时写入Δ胶囊
      read()          — 新会话开始时恢复记忆
      checkpoint()    — 定期全量快照
    """

    def __init__(self, capsule_dir: Path = CAPSULE_DIR):
        self.capsule_dir = Path(capsule_dir)
        self.capsule_dir.mkdir(parents=True, exist_ok=True)
        self.session_count = 0
        self.delta: Optional[DeltaCapsule] = None
        self.text: Optional[TextCapsule] = None
        self.checkpoints: list[Checkpoint] = []

    def write(self, text: TextCapsule, delta: Optional[DeltaCapsule] = None) -> str:
        """写入胶囊。返回文件路径。"""
        self.text = text
        self.session_count += 1

        # 写v0.6文本胶囊
        v06_path = self.capsule_dir / f"v06_{text.session_id}.json"
        with open(v06_path, "w", encoding="utf-8") as f:
            json.dump(text.to_dict(), f, ensure_ascii=False, indent=2)

        # 累加或新建v0.7 Δ向量
        if delta is not None:
            if self.delta is not None:
                self.delta.accumulate(delta.vector)
            else:
                self.delta = delta
            v07_path = self.capsule_dir / f"v07_{delta.session_id}.json"
            with open(v07_path, "w", encoding="utf-8") as f:
                json.dump(self.delta.to_dict(), f, ensure_ascii=False, indent=2)
        else:
            v07_path = ""

        # 检查是否需要快照
        if self.delta and should_checkpoint(self.session_count, self.delta.norm):
            self.checkpoint()

        return str(v06_path)

    def read(self, session_id: Optional[str] = None) -> dict:
        """读取最新胶囊，恢复记忆。"""
        # 找最新v0.6
        v06_files = sorted(self.capsule_dir.glob("v06_*.json"), reverse=True)
        if not v06_files:
            return {"status": "empty", "context": "无记忆。第一次对话？"}

        latest_v06 = v06_files[0]
        with open(latest_v06, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.text = TextCapsule.from_dict(data)

        # 找最新v0.7
        v07_files = sorted(self.capsule_dir.glob("v07_*.json"), reverse=True)
        if v07_files:
            with open(v07_files[0], "r", encoding="utf-8") as f:
                self.delta = DeltaCapsule.from_dict(json.load(f))

        # 如果有检查点，从最近检查点恢复
        cp_files = sorted(self.capsule_dir.glob("checkpoint_*.json"), reverse=True)
        if cp_files:
            with open(cp_files[0], "r", encoding="utf-8") as f:
                cp = Checkpoint.from_dict(json.load(f))
                self.checkpoints.append(cp)
                # 从检查点的delta基准+累加后续Δ
                if self.delta and cp.delta_snapshot is not None:
                    self.delta.vector = cp.delta_snapshot + self.delta.vector
                    self.delta.norm = float(np.linalg.norm(self.delta.vector))

        # 仲裁
        if self.text and self.delta:
            context = Arbitrator.resolve(self.text, self.delta)
        elif self.text:
            context = {"source": "text_only", "decisions": [d.get("summary", "") for d in self.text.decisions]}
        else:
            context = {"source": "empty"}

        context["status"] = "restored"
        return context

    def checkpoint(self) -> str:
        """创建全量快照。"""
        if not self.text or self.delta is None:
            return ""
        cp = Checkpoint(
            text_capsule=self.text,
            delta_snapshot=self.delta.vector.copy(),
            seq_number=len(self.checkpoints) + 1,
        )
        self.checkpoints.append(cp)
        cp_path = self.capsule_dir / f"checkpoint_{cp.seq_number:03d}.json"
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(cp.to_dict(), f, ensure_ascii=False, indent=2)
        return str(cp_path)


# ── 四层记忆分层 ────────────────────────────────────

class MemoryLayers:
    """
    四层记忆：session → project → global → user
    对齐Claude Code的4层模型。
    """
    def __init__(self):
        self.session: dict = {}   # L1: 当前会话
        self.project: dict = {}   # L2: 项目上下文
        self.global_: dict = {}   # L3: 跨项目模式
        self.user: dict = {}      # L4: 用户偏好

    def inject(self, layer: str, key: str, value: Any) -> None:
        getattr(self, layer)[key] = value

    def recall(self, layer: str, key: str) -> Any:
        return getattr(self, layer).get(key)

    def all_context(self) -> dict:
        return {
            "session": self.session,
            "project": self.project,
            "global": self.global_,
            "user": self.user,
        }
