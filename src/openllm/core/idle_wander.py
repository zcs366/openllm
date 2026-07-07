"""
空闲期协议（间隙体最小实现）

阿波罗启示：好奇心的最小单位是注意力的主动偏移——
系统在没有外部指令时，把注意力移到未被探索的领域。

设计：
- 追踪连续空闲心跳次数
- 达到阈值后进入"散步模式"
- 用IAI搜索能力随机探索一个从未访问过的领域
- 发现写入ISA记忆池（标记source=idle_wander）
- 约束：散步发现不触发行动，仅积累记忆

克洛诺斯修正：散步不是任务——是间隙。
间隙不是浪费——是好奇心的土壤。
"""
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ── 散步配置 ──
IDLE_THRESHOLD = 3          # 连续N次无输入后进入散步模式
MAX_WANDER_DISCOVERIES = 5  # 单次散步最多探索几个领域
WANDER_DOMAINS = [          # 散步的候选领域
    "认知科学", "信息论", "因果推理", "记忆巩固",
    "注意力机制", "自监督学习", "多Agent协调", "治理理论",
    "涌现行为", "复杂系统", "博弈论", "意识哲学",
    "知识图谱", "语义搜索", "压缩感知", "元学习",
]


@dataclass
class WanderDiscovery:
    """一次散步发现"""
    domain: str             # 探索的领域
    query: str              # 搜索查询
    finding: str            # 发现内容（摘要）
    relevance: float        # 与当前任务的相关性 0.0-1.0
    timestamp: float = field(default_factory=time.time)
    source: str = "idle_wander"

    def to_memory_entry(self) -> dict:
        """转换为ISA记忆条目格式"""
        return {
            "content": f"[散步·{self.domain}] {self.finding}",
            "source": self.source,
            "domain": self.domain,
            "relevance": self.relevance,
            "timestamp": self.timestamp,
        }


class IdleWanderer:
    """间隙体——空闲期的注意力自由偏移"""

    def __init__(self, threshold: int = IDLE_THRESHOLD):
        self.threshold = threshold
        self._idle_count = 0
        self._visited_domains: set[str] = set()
        self._total_wanders: int = 0
        self._discoveries: list[WanderDiscovery] = []

    def tick_idle(self) -> bool:
        """报告一次空闲心跳。返回True表示应该进入散步模式。"""
        self._idle_count += 1
        return self._idle_count >= self.threshold

    def tick_active(self) -> None:
        """报告一次活跃心跳（有用户输入）。重置空闲计数。"""
        self._idle_count = 0

    def wander(self, search_fn=None) -> list[WanderDiscovery]:
        """
        执行一次散步——随机探索未访问的领域。

        search_fn: 可选的搜索函数 (query: str) -> str
                   如果为None，生成模拟发现
        """
        # 选择未访问过的领域
        unvisited = [d for d in WANDER_DOMAINS if d not in self._visited_domains]
        if not unvisited:
            # 全部访问过，清空重来
            self._visited_domains.clear()
            unvisited = WANDER_DOMAINS[:]

        # 随机选几个领域探索
        n = min(MAX_WANDER_DISCOVERIES, len(unvisited))
        selected = random.sample(unvisited, n)

        discoveries = []
        for domain in selected:
            self._visited_domains.add(domain)
            query = f"{domain}的最新进展"

            if search_fn:
                try:
                    finding = search_fn(query)
                except Exception:
                    finding = f"[散步] {domain}搜索失败，跳过"
            else:
                finding = self._simulate_finding(domain)

            discovery = WanderDiscovery(
                domain=domain,
                query=query,
                finding=finding,
                relevance=random.uniform(0.1, 0.5),  # 散步发现通常相关性较低
            )
            discoveries.append(discovery)

        self._discoveries.extend(discoveries)
        self._total_wanders += 1
        self._idle_count = 0  # 散步后重置

        return discoveries

    def _simulate_finding(self, domain: str) -> str:
        """模拟发现（无真实搜索时的占位）"""
        templates = [
            f"{domain}领域近期有关于自监督方法的新讨论",
            f"{domain}中的某些概念可能与Agent治理相关",
            f"{domain}提供了理解复杂系统的新视角",
            f"{domain}的研究者正在探索可解释性方法",
        ]
        return random.choice(templates)

    def get_status(self) -> dict:
        """获取散步状态"""
        return {
            "idle_count": self._idle_count,
            "threshold": self.threshold,
            "total_wanders": self._total_wanders,
            "total_discoveries": len(self._discoveries),
            "visited_domains": len(self._visited_domains),
            "mode": "walking" if self._idle_count >= self.threshold else "idle",
        }

    def get_recent_discoveries(self, n: int = 5) -> list[dict]:
        """获取最近的散步发现"""
        return [d.to_memory_entry() for d in self._discoveries[-n:]]

    def should_act_on_discovery(self, discovery: WanderDiscovery) -> bool:
        """
        判断是否应该对某个发现采取行动。

        铁律：散步发现不触发行动，仅积累记忆。
        只有relevance > 0.8的发现才记录为"值得跟进"。
        """
        return discovery.relevance > 0.8
