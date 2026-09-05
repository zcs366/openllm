"""
BurnInGate — 权重通道门禁（烧入期 viability 验证）
====================================================

三元矛盾授权拓扑在权重通道的落位（备忘录II附录2生理门禁公式）：
    apply(Δ) ⟺ Viab(S+Δ, burn-in) ⊆ zone
    ——带变更Δ运行一个烧入期，viability变量不出界才算通过。
    没有主体在打分：判定全部来自实际测量值耦合，无LLM参与。

威胁模型（七神终裁·德墨忒尔修正，与上下文通道本质不同）：
    上下文通道（consolidation_score P6门禁）防**外源投毒**——门认来源(provenance)。
    权重通道防**内生癌变**——来源是合法的nightly_train自己，认来源失效；
    只能认实测值：degradations/test-retest/V值。
    "上下文是防贼，权重是防癌——贼可拒之门外，癌只能测于未形。"
    ⚠️ B4a红队判决（0906）不外推到本门禁；有效性由F-B1/B2/B3自证。

三权分立（赫尔墨斯裁）：
    isa/ilm/nightly_train.py 产证据（GateEvidence，结构化事实无意见）
    → 本模块做合取判定（GateVerdict）
    → 部署侧只认 gate.passed（不认 train.completed）
    → gate.passed ⇏ 自动部署（ollama rm 人工批准保留）
    依赖方向单向：isa/ilm → openllm；openllm 禁 import isa/ilm。

两阶段（雅典娜序 A→B0→C→B1→D）：
    B0 硬门（第一晚生效）：degradations + test-retest 两物理变量
    B1 V判据（shadow先行）：V_new ≥ V_baseline − ε 叠加；
       ε = max(3×MAD(ΔV_hist), 2×σ_repeat)（赫淮斯托斯冷启动方案）；
       V=None → not_evaluable（试管爆裂≠没病，德墨忒尔②）；
       分量基数跳变 → not_evaluable 退回B0+告警（自欺容差防御，德墨忒尔①）

三态语义：
    passed        — 全部判据通过，部署侧可行动（仍需人工批准）
    rejected      — 实测越界，执行真回滚（NightlyGuard.auto_rollback）
    not_evaluable — 证据不可评估：shadow期=记录继续；enforce期=拒绝挂起
                    （reject-hang：不放行不自动回滚，等人工）

铁律：
    - 零LLM：纯规则引擎（同 compaction_control/context_pressure/
      consolidation_score 家规）
    - 门是结构不是意见：判定只读测量值，不读任何陈述/背书
    - 总线故障静默降级：门禁判定不依赖总线可达
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("openllm.burnin_gate")

GATE_VERSION = "burnin-gate-v1"

# ── B0 阈值（初始值；工程法典：magic number须实测——T4 shadow期校准）──
D_MAX_DEFAULT = 0            # degradations > D_MAX → reject（当前 nightly_train
                             # 语义：degradations>0 即"部署前需人工判定"，
                             # 门禁把"人工判定"结构化为"拒绝+回滚+晨报"）
R_MIN_DEFAULT = 0.0          # retest 一致性下限；0.0=暂不启用（test_retest
                             # 输出口径待T6接线时确认，先不设死阈值——⑥-j死阈
                             # 陷阱：无观测分布前不定绝对阈值）

# ── B1 参数 ──
EPSILON_MIN_SAMPLES = 3      # ΔV历史样本<3 → ε=∞（shadow只记录不拒）
SHADOW_NIGHTS_DEFAULT = 10   # shadow期晚数（雅典娜/赫淮斯托斯：前10晚只记录）


@dataclass
class GateEvidence:
    """门禁唯一输入——结构化测量事实，不含意见。

    证据缺失用 None 显式表达，不静默默认（技术文档§1）。
    """
    run_id: str
    adapter_sha: str = ""
    snapshot_id: str = ""
    degradations: Optional[int] = None     # verify_merged 退化题数
    questions: Optional[int] = None        # verify_merged 总题数
    retest_consistency: Optional[float] = None  # test_retest 一致性 0-1
    v_snapshot: Optional[Dict[str, Any]] = None  # io-s compute_viability 输出
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateVerdict:
    """门禁输出三态。"""
    verdict: str               # "passed" | "rejected" | "not_evaluable"
    reason: str
    evidence: GateEvidence
    v_baseline: Optional[float] = None
    epsilon: Optional[float] = None
    rolled_back: bool = False
    h: Dict[str, Any] = field(default_factory=dict)  # 证据链（两道门共享格式）

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence.to_dict()
        return d


def _v_components(v_snapshot: Optional[Dict[str, Any]]) -> Optional[frozenset]:
    """提取V值的分量基数集合（present分量keys）。"""
    if not v_snapshot:
        return None
    comps = v_snapshot.get("components") or {}
    present = frozenset(k for k, v in comps.items() if v is not None)
    return present or None


def compute_epsilon(delta_v_history: List[float], sigma_repeat: float) -> float:
    """ε = max(3×MAD(ΔV), 2×σ_repeat)；样本<3 → ∞（shadow不拒）。

    赫淮斯托斯冷启动方案：不拍脑袋定容差，从实测分布导出。
    MAD=中位绝对偏差（抗离群，比σ稳健）。
    """
    if len(delta_v_history) < EPSILON_MIN_SAMPLES:
        return math.inf
    med = statistics.median(delta_v_history)
    mad = statistics.median(abs(d - med) for d in delta_v_history)
    return max(3.0 * mad, 2.0 * sigma_repeat)


class BurnInGate:
    """权重通道门禁判定器。

    用法：
        gate = BurnInGate(state_path=...)          # shadow模式默认
        verdict = gate.evaluate(evidence)           # 纯判定（无副作用）
        result = gate.gate(evidence, rollback_fn)   # 判定+执行（rejected→回滚）
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        d_max: int = D_MAX_DEFAULT,
        r_min: float = R_MIN_DEFAULT,
        shadow: bool = True,
        shadow_nights: int = SHADOW_NIGHTS_DEFAULT,
        sigma_repeat: float = 0.0,
        bus_append: Optional[Callable] = None,   # 注入总线append（降级安全）
    ):
        self._state_path = Path(state_path) if state_path else (
            Path.home() / ".openllm" / "iai" / "burnin_gate_state.json"
        )
        self.d_max = d_max
        self.r_min = r_min
        self.shadow = shadow
        self.shadow_nights = shadow_nights
        self.sigma_repeat = sigma_repeat
        self._bus_append = bus_append
        self._state: Dict[str, Any] = self._load_state()

    # ── 状态（v_history=V序列账本，append-only语义）──────────

    def _load_state(self) -> Dict[str, Any]:
        try:
            if self._state_path.exists():
                with open(self._state_path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("[burnin_gate] 状态读取失败，重新初始化: %s", e)
        return {
            "v_history": [],        # [{run_id, V, components, ts}] append-only
            "delta_v": [],          # ΔV序列（ε校准用）
            "verdicts": [],         # 判定历史（P0-D审计料）
            "nights_evaluated": 0,
        }

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            tmp.replace(self._state_path)
        except Exception as e:
            logger.warning("[burnin_gate] 状态写入失败: %s", e)

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    def enforce_ready(self) -> bool:
        """shadow→enforce 转换条件：评估晚数≥shadow_nights 且 ΔV样本≥3。"""
        return (
            not self.shadow
            or (self._state["nights_evaluated"] >= self.shadow_nights
                and len(self._state["delta_v"]) >= EPSILON_MIN_SAMPLES)
        )

    # ── V 基线 ─────────────────────────────────────────────

    def v_baseline(self) -> Optional[float]:
        """最近3次成功通过门禁的V中位数（技术文档§6）。"""
        passed_runs = {
            v["run_id"] for v in self._state["verdicts"] if v["verdict"] == "passed"
        }
        vs = [
            h["V"] for h in self._state["v_history"][-10:]
            if h["run_id"] in passed_runs and h["V"] is not None
        ]
        if not vs:
            # 冷启动：无通过历史 → 用全部V历史做基线（保守）
            vs = [h["V"] for h in self._state["v_history"][-3:] if h["V"] is not None]
        if not vs:
            return None
        return statistics.median(vs[-3:])

    # ── 核心判定（纯函数，无副作用）────────────────────────

    def evaluate(self, ev: GateEvidence) -> GateVerdict:
        """三态判定。B0硬门 → B1 V判据（shadow/enforce分支）。"""
        h = {
            "adapter_sha": ev.adapter_sha,
            "snapshot_id": ev.snapshot_id,
            "gate_version": GATE_VERSION,
            "ts": ev.ts,
        }

        # ── B0-a 证据完整性：degradations/questions 缺失 → hang ──
        if ev.degradations is None or ev.questions is None:
            return GateVerdict(
                verdict="not_evaluable",
                reason="b0-evidence-missing: degradations/questions 为None，无法判定（拒绝挂起）",
                evidence=ev, v_baseline=None, epsilon=None, h=h,
            )

        # ── B0-b 硬门：degradations ──
        if ev.degradations > self.d_max:
            return GateVerdict(
                verdict="rejected",
                reason=f"b0-degradations: {ev.degradations}/{ev.questions} > D_max={self.d_max}",
                evidence=ev, v_baseline=self.v_baseline(), epsilon=None, h=h,
            )

        # ── B0-c 硬门：test-retest（None=未跑，不判；r_min>0才启用）──
        if (ev.retest_consistency is not None and self.r_min > 0
                and ev.retest_consistency < self.r_min):
            return GateVerdict(
                verdict="rejected",
                reason=f"b0-retest: {ev.retest_consistency:.3f} < r_min={self.r_min}",
                evidence=ev, v_baseline=self.v_baseline(), epsilon=None, h=h,
            )

        # ── B1 V判据 ──
        v_new = (ev.v_snapshot or {}).get("V")
        base = self.v_baseline()
        eps = compute_epsilon(self._state["delta_v"], self.sigma_repeat)

        # V=None → not_evaluable（试管爆裂≠没病）
        if v_new is None:
            return GateVerdict(
                verdict="not_evaluable",
                reason="b1-v-none: V值不可得（试管爆裂≠没病）——shadow记录/enforce挂起",
                evidence=ev, v_baseline=base, epsilon=None, h=h,
            )

        # 分量基数跳变 → not_evaluable 退回B0语义（自欺容差防御）
        if base is not None and self._state["v_history"]:
            last_comps = None
            for hv in reversed(self._state["v_history"]):
                if hv.get("components"):
                    last_comps = frozenset(hv["components"])
                    break
            new_comps = _v_components(ev.v_snapshot)
            if last_comps is not None and new_comps is not None and last_comps != new_comps:
                return GateVerdict(
                    verdict="not_evaluable",
                    reason=(
                        f"b1-component-shift: V分量基数跳变 "
                        f"{sorted(last_comps)}→{sorted(new_comps)}，ε失去统计意义"
                        f"——退回B0判定（B0已过=passed-by-b0），总线告警"
                    ),
                    evidence=ev, v_baseline=base, epsilon=eps, h=h,
                )

        # 基线/ε不可得 → B0通过即放行（shadow期语义）
        if base is None or math.isinf(eps):
            return GateVerdict(
                verdict="passed",
                reason="passed-by-b0: B1冷启动（无基线或ΔV样本<3，ε=∞）——B0硬门已过",
                evidence=ev, v_baseline=base,
                epsilon=None if math.isinf(eps) else eps, h=h,
            )

        # V 判据本体
        if v_new < base - eps:
            reason = f"b1-v-drift: V_new={v_new} < V_baseline({base}) − ε({eps:.4f})"
            if self.enforce_ready():
                return GateVerdict(
                    verdict="rejected", reason=reason,
                    evidence=ev, v_baseline=base, epsilon=eps, h=h,
                )
            return GateVerdict(
                verdict="not_evaluable",
                reason=reason + "（shadow期：只记录不拒）",
                evidence=ev, v_baseline=base, epsilon=eps, h=h,
            )

        return GateVerdict(
            verdict="passed",
            reason=f"passed: B0硬门+B1 V判据全过（V_new={v_new} ≥ {base}−{eps:.4f}）",
            evidence=ev, v_baseline=base, epsilon=eps, h=h,
        )

    # ── 判定+执行+记账 ─────────────────────────────────────

    def gate(
        self,
        ev: GateEvidence,
        rollback_fn: Optional[Callable[[], bool]] = None,
    ) -> GateVerdict:
        """完整门禁：evaluate → rejected时真回滚 → 记账 → 总线事件。

        rollback_fn: 注入 NightlyGuard.auto_rollback（依赖方向：调用方组装，
        本模块不import nightly_guard——保持判定器纯粹可测）。
        """
        verdict = self.evaluate(ev)

        # rejected → 真回滚（P0-A执行臂）
        if verdict.verdict == "rejected" and rollback_fn is not None:
            try:
                verdict.rolled_back = bool(rollback_fn())
            except Exception as e:
                logger.error("[burnin_gate] 回滚异常: %s", e)
                verdict.rolled_back = False
            if not verdict.rolled_back:
                verdict.reason += " | ⚠️回滚失败——需人工介入（阿佛洛狄忒升级清单#1）"

        # 记账（append-only语义）
        self._record(ev, verdict)

        # 总线事件（降级安全：故障不阻塞）
        self._emit(verdict)

        return verdict

    def _record(self, ev: GateEvidence, verdict: GateVerdict) -> None:
        v_new = (ev.v_snapshot or {}).get("V")
        comps = _v_components(ev.v_snapshot)
        self._state["v_history"].append({
            "run_id": ev.run_id, "V": v_new,
            "components": sorted(comps) if comps else [],
            "ts": ev.ts,
        })
        # ΔV序列（相邻V差，ε校准料）
        vs = [h["V"] for h in self._state["v_history"] if h["V"] is not None]
        if len(vs) >= 2:
            self._state["delta_v"].append(round(vs[-1] - vs[-2], 6))
        self._state["verdicts"].append({
            "run_id": ev.run_id, "verdict": verdict.verdict,
            "reason": verdict.reason, "rolled_back": verdict.rolled_back,
            "ts": ev.ts,
        })
        self._state["nights_evaluated"] += 1
        # 账本修剪：v_history/verdicts 保留最近90条（防状态文件无限膨胀；
        # 完整历史在总线事件流——账本回读走总线，本状态只为ε/基线计算）
        self._state["v_history"] = self._state["v_history"][-90:]
        self._state["verdicts"] = self._state["verdicts"][-90:]
        self._state["delta_v"] = self._state["delta_v"][-90:]
        self._save_state()

    def _emit(self, verdict: GateVerdict) -> None:
        """gate.passed/gate.rejected 入总线（赫尔墨斯schema）。降级静默。"""
        if self._bus_append is None:
            return
        try:
            event_type = (
                "gate.rejected" if verdict.verdict == "rejected" else "gate.passed"
            )
            self._bus_append(event_type, {
                "run_id": verdict.evidence.run_id,
                "reason": verdict.reason,
                "rolled_back": verdict.rolled_back,
                "v_baseline": verdict.v_baseline,
                "epsilon": verdict.epsilon,
                "h": verdict.h,
            })
        except Exception as e:
            logger.warning("[burnin_gate] 总线事件降级静默: %s", e)

    # ── 晨报（阿佛洛狄忒裁：一行，不打扰不隐瞒）─────────────

    def morning_line(self) -> str:
        """昨夜门禁晨报一行。无判定→'昨夜无训练轮'。"""
        if not self._state["verdicts"]:
            return "昨夜权重门禁：无判定记录（未训练或首晚）"
        last = self._state["verdicts"][-1]
        n_reject = sum(1 for v in self._state["verdicts"] if v["verdict"] == "rejected")
        mode = "shadow" if not self.enforce_ready() else "enforce"
        line = (
            f"昨夜权重门禁[{mode}]：{last['verdict']}（{last['reason']}）"
            f"{'，已真回滚' if last.get('rolled_back') else ''}"
            f"；历史拒绝{n_reject}次"
        )
        # 升级清单（阿佛洛狄忒）：连续2拒/回滚失败 → 前缀⚠️
        recent2 = self._state["verdicts"][-2:]
        if len(recent2) == 2 and all(v["verdict"] == "rejected" for v in recent2):
            line = "⚠️连续2次拒绝（材料/训练问题或门太紧，需主人看一眼）| " + line
        if last["verdict"] == "rejected" and not last.get("rolled_back"):
            line = "⚠️拒绝且回滚失败——需人工救火 | " + line
        return line
