"""
MemoryEvaluator — ISA记忆自评估引擎
====================================

RQGM启示：评估器必须和Agent共同进化。
本模块让ISA能评估自己的记忆质量并自动调整。

epoch机制（来自RQGM）：
  epoch内：冻结评估，积累数据
  epoch边界：评估 → 调整策略 → 选择性淘汰 → 新epoch

评估维度（来自AML 7维 + RQGM对抗性目标）：
  1. 检索命中率：MemoryBus.query()返回的结果中，多少被Agent实际使用？
  2. 因果利用率：CausalMemoryStore中的教训，多少被检索并影响了决策？
  3. 遗忘效率：scan_for_eviction()发现的evictable，有多少真的被淘汰？
  4. 行为改善度：跨epoch对比，Agent的决策质量是否提升？

数据流：
  每次tick → 记录(检索query, 返回结果, 使用结果)
  epoch边界 → evaluate() → 计算指标 → 调整策略 → 写入调整日志
"""
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

logger = logging.getLogger("openllm.memory_evaluator")

# ── 配置 ──
EVAL_DIR = Path.home() / ".openllm" / "memory_eval"
EPOCH_SIZE = 10  # 每10次tick为一个epoch
STRATEGY_FILE = EVAL_DIR / "current_strategy.json"
HISTORY_FILE = EVAL_DIR / "eval_history.jsonl"
ADJUST_LOG = EVAL_DIR / "adjust_log.jsonl"


@dataclass
class RetrievalEvent:
    """一次检索事件"""
    timestamp: float
    query: str
    results_count: int
    sources: List[str]
    top_score: float
    used: bool = False  # Agent是否实际使用了检索结果
    used_result_idx: int = -1  # 使用了第几条


@dataclass
class CausalEvent:
    """一次因果记忆事件"""
    timestamp: float
    action: str
    was_retrieved: bool  # 因果记忆是否被检索到
    influenced_decision: bool  # 是否影响了决策
    lesson: str = ""


@dataclass
class EpochStats:
    """一个epoch的统计"""
    epoch_id: int
    start_time: float
    end_time: float
    tick_count: int
    # 检索质量
    retrieval_hit_rate: float  # 被使用的检索结果比例
    avg_top_score: float
    source_diversity: float  # 来源多样性（entropy）
    # 因果利用
    causal_retrieval_rate: float  # 因果记忆被检索到的比例
    causal_influence_rate: float  # 因果记忆影响决策的比例
    # 遗忘
    evictable_count: int  # epoch结束时有多少evictable
    # 行为改善
    success_rate: float  # 成功决策比例
    improvement: float = 0.0  # 与上一epoch的改善


@dataclass
class Strategy:
    """记忆检索策略（可调整）"""
    # MemoryBus参数
    top_k: int = 10
    token_budget: int = 2000
    # 来源权重
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "capsule": 1.0,
        "causal": 1.0,
        "jiak": 1.0,
        "recall": 1.0,
        "unified": 1.0,
        "source_index": 0.5,
    })
    # 遗忘阈值
    eviction_threshold: float = 0.3
    # 自上次调整以来的epoch数
    epochs_since_adjust: int = 0


class MemoryEvaluator:
    """ISA记忆自评估引擎"""

    def __init__(self, eval_dir: Optional[Path] = None):
        self.eval_dir = eval_dir or EVAL_DIR
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self._current_events: List[RetrievalEvent] = []
        self._causal_events: List[CausalEvent] = []
        self._tick_count = 0
        self._epoch_id = self._load_epoch_id()
        self._strategy = self._load_strategy()

    def _load_epoch_id(self) -> int:
        """加载当前epoch编号"""
        state_file = self.eval_dir / "state.json"
        if state_file.exists():
            try:
                return json.loads(state_file.read_text()).get("epoch_id", 0)
            except Exception:
                pass
        return 0

    def _save_state(self):
        """持久化状态"""
        state_file = self.eval_dir / "state.json"
        state_file.write_text(json.dumps({
            "epoch_id": self._epoch_id,
            "tick_count": self._tick_count,
            "last_update": time.time(),
        }))

    def _load_strategy(self) -> Strategy:
        """加载当前策略"""
        if STRATEGY_FILE.exists():
            try:
                data = json.loads(STRATEGY_FILE.read_text())
                return Strategy(**data)
            except Exception:
                pass
        return Strategy()

    def _save_strategy(self):
        """持久化策略"""
        STRATEGY_FILE.write_text(json.dumps(asdict(self._strategy), indent=2))

    # ── 数据记录 ──

    def record_retrieval(self, query: str, results: list, used_idx: int = -1):
        """记录一次检索事件"""
        if not results:
            return
        sources = list(set(getattr(r, "source", "unknown") for r in results))
        event = RetrievalEvent(
            timestamp=time.time(),
            query=query[:200],
            results_count=len(results),
            sources=sources,
            top_score=getattr(results[0], "score", 0.0),
            used=used_idx >= 0,
            used_result_idx=used_idx,
        )
        self._current_events.append(event)
        self._tick_count += 1

        # epoch边界检查
        if self._tick_count % EPOCH_SIZE == 0:
            self.evaluate()

    def record_causal(self, action: str, retrieved: bool, influenced: bool, lesson: str = ""):
        """记录一次因果记忆事件"""
        self._causal_events.append(CausalEvent(
            timestamp=time.time(),
            action=action[:200],
            was_retrieved=retrieved,
            influenced_decision=influenced,
            lesson=lesson[:200],
        ))

    # ── 评估 ──

    def evaluate(self) -> EpochStats:
        """评估当前epoch"""
        # 计算检索质量
        if self._current_events:
            hit_rate = sum(1 for e in self._current_events if e.used) / len(self._current_events)
            avg_top_score = sum(e.top_score for e in self._current_events) / len(self._current_events)
            # 来源多样性（简单entropy）
            all_sources = []
            for e in self._current_events:
                all_sources.extend(e.sources)
            source_counts = {}
            for s in all_sources:
                source_counts[s] = source_counts.get(s, 0) + 1
            total = len(all_sources) or 1
            diversity = -sum((c/total) * __import__("math").log2(c/total + 1e-10)
                           for c in source_counts.values())
        else:
            hit_rate = 0.0
            avg_top_score = 0.0
            diversity = 0.0

        # 因果利用
        causal_retrieval = (sum(1 for c in self._causal_events if c.was_retrieved)
                          / max(len(self._causal_events), 1))
        causal_influence = (sum(1 for c in self._causal_events if c.influenced_decision)
                          / max(len(self._causal_events), 1))

        # 遗忘
        try:
            from .temperature_engine import scan_for_eviction
            evictable = len(scan_for_eviction())
        except Exception:
            evictable = 0

        stats = EpochStats(
            epoch_id=self._epoch_id,
            start_time=time.time() - EPOCH_SIZE * 60,  # 估算
            end_time=time.time(),
            tick_count=self._tick_count,
            retrieval_hit_rate=round(hit_rate, 4),
            avg_top_score=round(avg_top_score, 4),
            source_diversity=round(diversity, 4),
            causal_retrieval_rate=round(causal_retrieval, 4),
            causal_influence_rate=round(causal_influence, 4),
            evictable_count=evictable,
            success_rate=1.0,  # 默认值，后续由Agent反馈填充
        )

        # 持久化
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(asdict(stats), ensure_ascii=False) + "\n")

        logger.info(f"Epoch {self._epoch_id}: hit={hit_rate:.2%} causal={causal_retrieval:.2%} evict={evictable}")

        # 策略调整
        self._adjust_strategy(stats)

        # 进入新epoch
        self._epoch_id += 1
        self._current_events.clear()
        self._causal_events.clear()
        self._save_state()

        return stats

    # ── 策略调整（RQGM启示：评估器即Agent自身）──

    def _adjust_strategy(self, stats: EpochStats):
        """LLM驱动的自我评估——Agent用自己的推理引擎决定改什么"""
        adjustments = []

        # 构造评估数据
        history = self.get_history(5)
        history_summary = "\n".join(
            f"  E{h['epoch_id']}: hit={h['retrieval_hit_rate']:.2%} "
            f"causal={h['causal_retrieval_rate']:.2%} diversity={h['source_diversity']:.2f} "
            f"evict={h['evictable_count']}"
            for h in history
        ) if history else "  (无历史数据)"

        strategy_summary = (
            f"top_k={self._strategy.top_k} token_budget={self._strategy.token_budget} "
            f"eviction_threshold={self._strategy.eviction_threshold} "
            f"weights={json.dumps(self._strategy.source_weights)}"
        )

        prompt = f"""你是一个Agent记忆系统的自评估引擎。分析以下epoch数据，决定如何调整记忆检索策略。

## 当前策略
{strategy_summary}

## 最近epoch数据
{history_summary}

## 当前epoch
hit_rate={stats.retrieval_hit_rate:.2%} causal_rate={stats.causal_retrieval_rate:.2%} diversity={stats.source_diversity:.2f} evictable={stats.evictable_count}

## 可调整参数
- top_k: 检索返回条数 (当前{self._strategy.top_k}, 范围5-30)
- token_budget: token预算 (当前{self._strategy.token_budget}, 范围1000-8000)
- eviction_threshold: 淘汰温度阈值 (当前{self._strategy.eviction_threshold}, 范围0.1-0.8)
- source_weights: 各来源权重 (capsule/causal/jiak/recall/unified, 范围0.1-3.0)

## 要求
分析数据趋势，给出1-3个最有效的调整。每个调整必须说明：参数名、当前值、建议值、理由。
直接输出JSON格式的调整列表，不要其他内容。
示例：[{{"param": "top_k", "from": 10, "to": 15, "reason": "命中率持续低于30%"}}]"""

        # 调用LLM自我评估
        try:
            from ..core.provider import create_provider
            # 从config.json读取mimo的api_key（与engine.py同路径）
            api_key = ""
            cfg_path = Path.home() / ".openllm" / "config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    api_key = cfg.get("providers", {}).get("mimo", {}).get("api_key", "")
                except Exception:
                    pass
            provider = create_provider(provider_type="mimo", model="mimo-v2.5", api_key=api_key)
            from ..core.provider import ChatMessage
            response = provider.chat([ChatMessage(role="user", content=prompt)])
            content = response.content.strip()

            # 解析JSON
            if content.startswith("["):
                proposed = json.loads(content)
            else:
                # 尝试提取JSON部分
                import re
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    proposed = json.loads(match.group())
                else:
                    proposed = []

            # 应用调整（带安全边界）
            for adj in proposed:
                param = adj.get("param", "")
                to_val = adj.get("to")
                reason = adj.get("reason", "")

                if param == "top_k" and isinstance(to_val, int):
                    old = self._strategy.top_k
                    self._strategy.top_k = max(5, min(30, to_val))
                    adjustments.append(f"LLM: top_k {old}→{self._strategy.top_k} ({reason})")
                elif param == "token_budget" and isinstance(to_val, int):
                    old = self._strategy.token_budget
                    self._strategy.token_budget = max(1000, min(8000, to_val))
                    adjustments.append(f"LLM: budget {old}→{self._strategy.token_budget} ({reason})")
                elif param == "eviction_threshold" and isinstance(to_val, (int, float)):
                    old = self._strategy.eviction_threshold
                    self._strategy.eviction_threshold = max(0.1, min(0.8, float(to_val)))
                    adjustments.append(f"LLM: eviction {old}→{self._strategy.eviction_threshold} ({reason})")
                elif param == "source_weights" and isinstance(to_val, dict):
                    for k, v in to_val.items():
                        if k in self._strategy.source_weights and isinstance(v, (int, float)):
                            old = self._strategy.source_weights[k]
                            self._strategy.source_weights[k] = max(0.1, min(3.0, float(v)))
                            adjustments.append(f"LLM: {k} {old}→{self._strategy.source_weights[k]} ({reason})")

        except Exception as e:
            logger.warning(f"LLM自我评估失败，降级为规则调整: {e}")
            # 降级：保留最简单的规则
            if stats.retrieval_hit_rate < 0.3 and self._strategy.top_k < 20:
                self._strategy.top_k += 2
                adjustments.append(f"fallback: top_k+2={self._strategy.top_k}")

        if adjustments:
            self._strategy.epochs_since_adjust = 0
            self._save_strategy()
            with open(ADJUST_LOG, "a") as f:
                f.write(json.dumps({
                    "epoch": self._epoch_id,
                    "time": time.time(),
                    "adjustments": adjustments,
                    "mode": "llm_self_evaluate",
                    "stats": asdict(stats),
                }, ensure_ascii=False) + "\n")
        else:
            self._strategy.epochs_since_adjust += 1

    # ── 查询接口 ──

    def get_strategy(self) -> Strategy:
        """获取当前策略（供MemoryBus使用）"""
        return self._strategy

    def get_history(self, last_n: int = 10) -> list:
        """获取最近N个epoch的历史"""
        if not HISTORY_FILE.exists():
            return []
        lines = HISTORY_FILE.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-last_n:] if l.strip()]

    def get_adjust_log(self, last_n: int = 5) -> list:
        """获取最近N次调整"""
        if not ADJUST_LOG.exists():
            return []
        lines = ADJUST_LOG.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-last_n:] if l.strip()]


# ── CLI ──

def main():
    """命令行接口"""
    import sys
    evaluator = MemoryEvaluator()

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        strategy = evaluator.get_strategy()
        history = evaluator.get_history(5)
        print(f"Epoch: {evaluator._epoch_id}")
        print(f"Strategy: top_k={strategy.top_k} budget={strategy.token_budget} eviction={strategy.eviction_threshold}")
        print(f"Weights: {strategy.source_weights}")
        print(f"History ({len(history)} epochs):")
        for h in history:
            print(f"  E{h['epoch_id']}: hit={h['retrieval_hit_rate']:.2%} causal={h['causal_retrieval_rate']:.2%} diversity={h['source_diversity']:.2f}")

    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        stats = evaluator.evaluate()
        print(f"Epoch {stats.epoch_id} evaluated:")
        print(f"  Retrieval hit: {stats.retrieval_hit_rate:.2%}")
        print(f"  Causal rate: {stats.causal_retrieval_rate:.2%}")
        print(f"  Diversity: {stats.source_diversity:.2f}")
        print(f"  Evictable: {stats.evictable_count}")

    else:
        print("用法: python3 memory_evaluator.py [status|evaluate]")


if __name__ == "__main__":
    main()
