"""iai/brain.py — SA/ZA 头脑系统（token级头脑设计·2026-09-02）。

头脑 = 有限原型词表（BRAIN_VOCAB）× 两种生命周期（SA/ZA）
- ZA = 一次性头脑（有始有终，如 delegate_task 子agent）
- SA = 持久头脑（跨会话记忆累积，如军师）
- 意识单焦点：一次一个活跃头脑（BrainActivator）

设计依据：SA/ZA时空论 + 蜻蜓十律（单头脑×多眼睛）
"""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal

# ── BRAIN_VOCAB 词表 ────────────────────────────────────
# 5 个预置头脑原型，key = persona_name
BRAIN_VOCAB: dict[str, dict] = {
    "军师": {
        "persona": "战略分析与全局指挥，从混沌中提炼行动路径",
        "mode": "SA",
        "memory_domain": "strategy",
    },
    "子贡": {
        "persona": "执行调度与任务分发，将军令变为可交付物",
        "mode": "SA",
        "memory_domain": "execution",
    },
    "包拯": {
        "persona": "审计合规与质量把关，每一步都留痕迹可追溯",
        "mode": "SA",
        "memory_domain": "audit",
    },
    "韩信": {
        "persona": "远景推演与终局思维，从目标倒推每一步",
        "mode": "ZA",
        "memory_domain": "vision",
    },
    "探照灯": {
        "persona": "深度思考与单点穿透，照亮被忽略的角落",
        "mode": "ZA",
        "memory_domain": "deep_think",
    },
}

# ── BrainRegistry 注册表 ──────────────────────────────────

DEFAULT_REGISTRY_PATH = Path.home() / ".openllm" / "brain_registry.json"


@dataclass
class BrainRecord:
    """注册表中一条头脑记录。"""
    brain_id: str
    persona_name: str
    kind: Literal["SA", "ZA"]
    persona: str
    memory_domain: str
    created_at: str
    archived_at: Optional[str] = None
    adapter_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BrainRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BrainRegistry:
    """头脑注册表 — 持久化到 JSON 文件。

    每个头脑由 brain_id 唯一标识，支持创建/查询/归档。
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or DEFAULT_REGISTRY_PATH
        self._data: dict[str, dict] = {}
        self._load()

    # ── 内部 ──

    def _load(self) -> None:
        """从磁盘加载，文件不存在则初始化空注册表。"""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """持久化到磁盘（目录不存在自动创建）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── 公开 API ──

    def create(self, persona_name: str, kind: Literal["SA", "ZA"] = "SA") -> str:
        """从词表创建一个头脑实例，返回 brain_id。

        如果 persona_name 不在 BRAIN_VOCAB 中，使用默认 persona。
        """
        vocab_entry = BRAIN_VOCAB.get(persona_name, {})
        brain_id = f"brain-{uuid.uuid4().hex[:12]}"
        record = BrainRecord(
            brain_id=brain_id,
            persona_name=persona_name,
            kind=kind,
            persona=vocab_entry.get("persona", f"{persona_name} 头脑"),
            memory_domain=vocab_entry.get("memory_domain", "general"),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._data[brain_id] = record.to_dict()
        self._save()
        return brain_id

    def list(self, include_archived: bool = False) -> list[dict]:
        """列出所有头脑记录。默认排除已归档的。"""
        if include_archived:
            return list(self._data.values())
        return [r for r in self._data.values() if r.get("archived_at") is None]

    def get(self, brain_id: str) -> Optional[dict]:
        """按 brain_id 查询，不存在返回 None。"""
        return self._data.get(brain_id)

    def archive(self, brain_id: str) -> bool:
        """归档头脑（标记 archived_at，不删除记录）。返回是否成功。"""
        if brain_id not in self._data:
            return False
        self._data[brain_id]["archived_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save()
        return True


# ── BrainActivator 单意识序列化激活器 ───────────────────────

DEFAULT_ACTIVE_BRAIN_PATH = Path.home() / ".openllm" / "active_brain.json"


class BrainActivator:
    """单意识激活器 — 全局同一时刻只有一个活跃头脑。

    激活状态持久化到 active_brain.json，重启不丢失。
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or DEFAULT_ACTIVE_BRAIN_PATH
        self._state: dict = {}
        self._load()

    # ── 内部 ──

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._state = {}
        else:
            self._state = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    # ── 公开 API ──

    def activate(self, brain_id: str) -> None:
        """激活一个头脑（自动替换当前活跃头脑）。

        不检查 brain_id 是否在注册表中——激活是瞬时行为，
        注册表验证由调用方负责。
        """
        self._state = {
            "active_brain_id": brain_id,
            "activated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()

    def deactivate(self) -> None:
        """取消当前活跃头脑。"""
        self._state = {}
        self._save()

    def current(self) -> Optional[str]:
        """返回当前活跃头脑的 brain_id，无则返回 None。"""
        return self._state.get("active_brain_id")
