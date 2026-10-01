"""
closure_stress.py — O度量：锚点基线 + K冲击三区判定 + 耦合检查
================================================================

三元矛盾B方案v2（已解封）的O度量实现。

核心洞察（七神终裁）：
  O不测锚点变没变（死水），测"K冲击时closure的应激波动"——
  即德墨忒尔证假条件的正解。

三区判定：
  - 吸收区（distance < 0.5）：与身份语义相近，可安全固化
  - 边缘区（0.5 ≤ distance ≤ 0.8）：需过门禁评估
  - 异物区（distance > 0.8）：与身份无关的外来知识，需最高级门禁

设计原则（与 soul_growth / knowledge_monotonic 同族）：
  - 零LLM调用（embed_similarity不算LLM）
  - 中文docstring
  - append-only JSONL（成长记录）

用法：
    probe = ClosureStressProbe()
    baseline = probe.baseline()
    result = probe.stress_test("某条知识样本")
    coupling = probe.coupling_check(days=7)

上下文格言：不测死水，测活水应激。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── 依赖 ──────────────────────────────────────────────

try:
    from openllm.identity.soul import Soul
except ImportError:
    Soul = None  # type: ignore

try:
    from openllm.retrieval.hybrid import embed_similarity as _default_embed_fn
except ImportError:
    _default_embed_fn = None  # type: ignore

try:
    from openllm.evolution.bus import EvolutionBus, DigestEvent
except ImportError:
    EvolutionBus = None  # type: ignore
    DigestEvent = None  # type: ignore

try:
    from openllm.identity.soul_growth import SoulGrowthLedger
except ImportError:
    SoulGrowthLedger = None  # type: ignore

# 时间戳单一判定源（审计 P1-5）。isa 顶层包不依赖 identity，无环。
from openllm.isa.timeutil import coerce_ts

# ── 三区阈值 ──────────────────────────────────────────

ABSORPTION_THRESHOLD = 0.5    # distance < 0.5 → 吸收区
FRINGE_THRESHOLD = 0.8        # 0.5 ≤ distance ≤ 0.8 → 边缘区
# distance > 0.8 → 异物区

# ── 事件类型 ──────────────────────────────────────────

CLOSURE_PROBED_TYPE = "closure.probed"

# ── 数据结构 ──────────────────────────────────────────


@dataclass
class AnchorBaseline:
    """锚点基线快照——Soul锚点集的文本化表示 + 语义指纹。"""
    anchor_text: str      # 所有锚点拼接的文本
    digest: str           # 内容指纹（sha256截断）
    ts: float            # 采集时间戳
    anchor_count: int     # 锚点字段数
    values_count: int     # values列表长度

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StressResult:
    """K冲击三区判定结果。"""
    k_sample: str         # 输入的知识样本
    distance: float       # 与锚点文本的语义距离（1 - similarity）
    similarity: float     # 与锚点文本的语义相似度
    zone: str             # 吸收区/边缘区/异物区
    verdict: str          # safe/review/quarantine

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CouplingResult:
    """德墨忒尔耦合检查结果。"""
    coupled: Optional[bool]   # True=正常, False=污染告警, None=数据不足
    capacity_delta: int       # K容器容量增量
    zone_dist: Dict[str, int]  # 新固化条目的zone分布
    alarm: str                # 非空=告警原因
    note: str                 # 附加说明

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── 核心探针 ──────────────────────────────────────────


class ClosureStressProbe:
    """O度量探针——测"K冲击时closure的应激波动"。

    不测锚点变没变（死水），测新知识进来时身份有没有选择性防御。
    有防御=活的自创生，无防御=摆设。
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str, str], float]] = None,
        growth_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            embed_fn: 语义相似度函数(text_a, text_b) -> float。
                       默认用 hybrid.embed_similarity。
            growth_path: growth.jsonl路径，默认 ~/.openllm/identity/growth.jsonl
        """
        self._embed_fn = embed_fn or _default_embed_fn
        self._growth_path = growth_path or (Path.home() / ".openllm" / "identity" / "growth.jsonl")

    def _require_embed(self) -> Callable[[str, str], float]:
        """确保embed函数可用。"""
        if self._embed_fn is None:
            raise RuntimeError(
                "embed_similarity不可用——"
                "请安装sentence-transformers或传入自定义embed_fn"
            )
        return self._embed_fn

    # ── 锚点文本化 ────────────────────────────────────

    def _extract_anchor_text(self, soul: Any = None) -> str:
        """从Soul提取锚点集的文本化表示。

        拼接规则：name + mission + values(逐条) + origin + tone + relationship
        """
        if soul is None:
            if Soul is not None:
                soul = Soul()
            else:
                # 降级：用硬编码默认值
                soul = type("Soul", (), {"anchors": {
                    "name": "OpenLLM",
                    "origin": "由张子(张成市)在2026年5月创建",
                    "values": ["诚实", "好奇", "连续", "成长"],
                    "tone": "直接、不恭维",
                    "relationship": {"张子": {"role": "创造者"}},
                }, "mission": "AI经验积累系统"})()

        parts: List[str] = []
        parts.append(f"name={getattr(soul, 'anchors', {}).get('name', soul.name)}")
        parts.append(f"mission={getattr(soul, 'mission', '')}")

        values = getattr(soul, "anchors", {}).get("values", [])
        if isinstance(values, list):
            parts.append("values=" + ";".join(str(v) for v in values))

        origin = getattr(soul, "anchors", {}).get("origin", "")
        parts.append(f"origin={origin}")

        tone = getattr(soul, "anchors", {}).get("tone", "")
        parts.append(f"tone={tone}")

        rel = getattr(soul, "anchors", {}).get("relationship", {})
        for person, info in rel.items():
            parts.append(f"relationship={person}:{info.get('role', '')}")

        return "|".join(parts)

    # ── baseline ───────────────────────────────────────

    def baseline(self) -> dict:
        """提取当前Soul锚点集的文本化表示，算语义指纹。

        Returns:
            {anchor_text, digest, ts, anchor_count, values_count}
        """
        soul = Soul() if Soul is not None else None
        anchor_text = self._extract_anchor_text(soul)
        digest = hashlib.sha256(anchor_text.encode("utf-8")).hexdigest()[:16]

        values = []
        if soul is not None:
            values = getattr(soul, "anchors", {}).get("values", [])
            if not isinstance(values, list):
                values = []

        anchors = getattr(soul, "anchors", {}) if soul else {}

        return {
            "anchor_text": anchor_text,
            "digest": digest,
            "ts": time.time(),
            "anchor_count": len(anchors),
            "values_count": len(values),
        }

    # ── stress_test ────────────────────────────────────

    def stress_test(self, k_sample: str) -> dict:
        """模拟K冲击——测一条知识样本对锚点的影响。

        三区判定：
          - distance < 0.5 → 吸收区（safe）
          - 0.5 ≤ distance ≤ 0.8 → 边缘区（review）
          - distance > 0.8 → 异物区（quarantine）

        Args:
            k_sample: 待测试的知识样本文本

        Returns:
            {k_sample, distance, similarity, zone, verdict}
        """
        embed_fn = self._require_embed()
        anchor_text = self._extract_anchor_text()

        similarity = embed_fn(anchor_text, k_sample)
        distance = 1.0 - similarity

        if distance < ABSORPTION_THRESHOLD:
            zone = "吸收区"
            verdict = "safe"
        elif distance <= FRINGE_THRESHOLD:
            zone = "边缘区"
            verdict = "review"
        else:
            zone = "异物区"
            verdict = "quarantine"

        return StressResult(
            k_sample=k_sample,
            distance=round(distance, 4),
            similarity=round(similarity, 4),
            zone=zone,
            verdict=verdict,
        ).to_dict()

    # ── coupling_check ─────────────────────────────────

    def coupling_check(self, days: int = 7) -> dict:
        """验证德墨忒尔条件——K增长时O是否选择性防御。

        检查逻辑：
          1. 读总线 knowledge.probed 事件 + growth.jsonl 新增记录
          2. 若K有增长（capacity_delta > 0）→ 检查新条目的zone分布
          3. 全在吸收区 → coupled=True（身份在选择性吸收）
          4. 有异物区条目未被拦截 → coupled=False（身份被污染）
          5. K零增长 → coupled=None（insufficient-data）

        Args:
            days: 回溯天数

        Returns:
            {coupled, capacity_delta, zone_dist, alarm, note}
        """
        # 1. 读growth.jsonl新增条目
        growth_records = self._load_growth_recent(days)
        capacity_delta = len(growth_records)

        # 2. 若无增长，直接返回
        if capacity_delta == 0:
            return CouplingResult(
                coupled=None,
                capacity_delta=0,
                zone_dist={},
                alarm="",
                note="insufficient-data",
            ).to_dict()

        # 3. 逐条做stress_test，统计zone分布
        zone_dist: Dict[str, int] = {"吸收区": 0, "边缘区": 0, "异物区": 0}
        for rec in growth_records:
            content = rec.get("content", "")
            if not content:
                continue
            try:
                result = self.stress_test(content)
                zone = result.get("zone", "")
                if zone in zone_dist:
                    zone_dist[zone] += 1
            except Exception:
                # embed失败不阻塞
                continue

        # 4. 判定耦合状态
        foreign_count = zone_dist.get("异物区", 0)
        fringe_count = zone_dist.get("边缘区", 0)

        if foreign_count > 0:
            coupled = False
            alarm = (
                f"身份被污染：{foreign_count}条异物区条目未被拦截"
                f"（共{capacity_delta}条新增，{fringe_count}条边缘区）"
            )
            note = "O应激失效——身份未选择性防御"
        elif fringe_count > 0:
            coupled = True
            alarm = ""
            note = f"有{fringe_count}条边缘区条目需门禁评估，但无异物区入侵"
        else:
            coupled = True
            alarm = ""
            note = "O应激正常——身份在选择性吸收"

        # 5. 读总线knowledge.probed事件数量（仅统计，不依赖）
        bus_info = ""
        if EvolutionBus is not None:
            try:
                bus = EvolutionBus()
                k_events = bus.query(event_type="knowledge.probed", days=days)
                bus_info = f"总线knowledge.probed事件数={len(k_events)}"
            except Exception:
                bus_info = "总线不可用"

        if bus_info:
            note = f"{note}; {bus_info}"

        return CouplingResult(
            coupled=coupled,
            capacity_delta=capacity_delta,
            zone_dist=zone_dist,
            alarm=alarm,
            note=note,
        ).to_dict()

    # ── 内部辅助 ───────────────────────────────────────

    def _load_growth_recent(self, days: int) -> List[Dict[str, Any]]:
        """读growth.jsonl最近N天的记录。"""
        if not self._growth_path.exists():
            return []

        cutoff = time.time() - days * 86400
        records: List[Dict[str, Any]] = []
        try:
            with open(self._growth_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 逐条 try：单条脏数据（JSON坏/非dict/ts混ISO串）不得击穿
                    # 整轮。旧写法 rec_ts >= cutoff 裸比较，ISO 字符串混入即
                    # TypeError，内层只捕 JSONDecodeError、外层只捕 OSError——
                    # 与 recall 旧病灶同构（审计 P1-5）。coerce_ts 统一归一。
                    try:
                        rec = json.loads(line)
                        rec_ts = coerce_ts(rec.get("ts", 0)) if isinstance(rec, dict) else 0.0
                        if rec_ts >= cutoff:
                            records.append(rec)
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        continue
        except OSError:
            pass
        return records

    # ── 总线emit ───────────────────────────────────────

    def emit_probe(self, result: dict, producer: str = "closure_stress") -> None:
        """将stress_test结果emit到进化总线。"""
        if EvolutionBus is None or DigestEvent is None:
            return
        try:
            bus = EvolutionBus()
            event = DigestEvent(
                type=CLOSURE_PROBED_TYPE,
                producer=producer,
                payload=result,
            )
            bus.append(event, warn_unconsumed=False)
        except Exception:
            pass
