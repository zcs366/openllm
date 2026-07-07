# T-PAL-5 MemoryBus 接口规范 v1.0

## 1. 问题

ISA当前有5套独立存储，各自有独立API：
1. **UnifiedMemory** (hot/warm/cold) — 温度衰减记忆
2. **CausalMemoryStore** — 因果教训链
3. **jiak cards** (~/.hermes/jiak/) — 结构化知识卡片
4. **RECALL** (recall_append.py) — 时间线日志
5. **Δ capsule** (SALifecycle) — 跨session共振

**痛点：**
- ISA.__init__直接实例化6个子系统，紧耦合
- ICE的MultiSourceMemorySelector硬编码4个selector
- 写入路径分散（ISA写UnifiedMemory，jiak写cards/，RECALL写jsonl）
- 检索路径分散（ISA检索causal，ICE检索jiak+recall）
- 新增记忆源需要改ISA/ICE两处代码

## 2. 设计原则

1. **单一写入口** — 所有记忆写入经过MemoryBus → 免疫检查 → 路由到正确provider
2. **统一检索接口** — ICE不需要知道记忆在哪个provider，只向MemoryBus查询
3. **Provider注册制** — 新增记忆源只需实现MemoryProvider接口并注册
4. **向后兼容** — ISA现有API不删，MemoryBus是新增层，不是替换
5. **零LLM调用** — MemoryBus本身是纯Python路由层，不调用LLM

## 3. 数据结构

```python
@dataclass
class MemoryRecord:
    """统一记忆记录——所有provider返回此格式"""
    record_id: str          # 唯一ID
    content: str            # 原文内容（verbatim-first）
    source: str             # 来源标识："isa/causal" | "jiak" | "recall" | "delta" | "capsule"
    record_type: str        # 类型："causal" | "opinion" | "lesson" | "event" | "insight"
    importance: float       # 重要性 0.0-1.0
    temperature: float      # 温度 0.0-1.0（衰减后）
    trust_level: str        # 信任分级："trusted" | "internal" | "untrusted" | "unknown"
    tags: List[str]         # 标签
    timestamp: float        # 创建时间
    context: Dict[str, Any] # provider特定元数据（因果链、card结构等）
    score: float = 0.0      # 检索相关性分数（检索时填充）
    provider: str = ""      # 实际provider名称（检索时填充）

@dataclass
class WriteRequest:
    """统一写入请求"""
    content: str
    source: str             # 写入者标识
    trust_level: str = "internal"
    record_type: str = "insight"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WriteResult:
    """写入结果"""
    success: bool
    record_id: str = ""
    provider: str = ""
    blocked: bool = False    # 被免疫系统拦截
    reason: str = ""

@dataclass
class Query:
    """统一检索查询"""
    text: str               # 查询文本
    top_k: int = 5
    token_budget: int = 1000
    record_types: Optional[List[str]] = None  # 过滤类型
    sources: Optional[List[str]] = None       # 过滤来源
    tags: Optional[List[str]] = None          # 过滤标签
    min_importance: float = 0.0
    min_temperature: float = 0.0
```

## 4. Provider接口

```python
class MemoryProvider(Protocol):
    """记忆Provider协议——任何实现此接口的类都可注册到MemoryBus"""

    @property
    def name(self) -> str:
        """Provider唯一名称"""
        ...

    @property
    def priority(self) -> int:
        """检索优先级（数值越小越优先）。0=jiak, 10=recall, 20=causal, 30=delta"""
        ...

    def search(self, query: Query) -> List[MemoryRecord]:
        """检索——返回与query相关的记忆记录"""
        ...

    def store(self, request: WriteRequest) -> WriteResult:
        """写入——将记忆存储到此provider
        返回WriteResult(success=False)表示不接受此类型写入"""
        ...

    def count(self) -> int:
        """当前存储的记忆数量"""
        ...

    def health(self) -> Dict[str, Any]:
        """健康状态"""
        ...
```

## 5. MemoryBus核心API

```python
class MemoryBus:
    """统一记忆总线——ISA和ICE的唯一记忆接口"""

    def __init__(self):
        self._providers: Dict[str, MemoryProvider] = {}
        self._write_log: List[WriteResult] = []  # 写入审计日志

    # ── Provider管理 ──

    def register(self, provider: MemoryProvider) -> None:
        """注册记忆provider"""

    def unregister(self, name: str) -> None:
        """注销provider"""

    def get_providers(self) -> List[str]:
        """列出所有已注册provider"""

    # ── 写入路径 ──

    def write(self, request: WriteRequest) -> WriteResult:
        """
        统一写入——路由到合适的provider。

        路由规则：
        1. 遍历所有provider，询问store()
        2. 第一个返回success=True的provider接受写入
        3. 都不接受 → 返回WriteResult(success=False)
        4. 写入结果记入审计日志
        """

    # ── 检索路径 ──

    def query(self, query: Query) -> List[MemoryRecord]:
        """
        统一检索——多源并行查询 + 融合排序。

        流程：
        1. 按priority排序，依次查询各provider
        2. 合并所有结果
        3. 按score降序排序
        4. 去重（同一record_id只保留最高分）
        5. 贪心填充token_budget
        6. 返回top_k条
        """

    # ── 管理 ──

    def stats(self) -> Dict[str, Any]:
        """全局统计"""

    def health_check(self) -> Dict[str, Any]:
        """所有provider健康检查"""

    def get_write_log(self, last_n: int = 20) -> List[WriteResult]:
        """最近N条写入记录（审计用）"""
```

## 6. 路由规则

### 写入路由
```
WriteRequest → MemoryBus.write()
  → 遍历provider按priority
  → provider.store(request)
    → 返回success=True → 写入成功，记录审计
    → 返回success=False → 继续下一个
  → 全部不接受 → 返回失败
```

### 检索路由
```
Query → MemoryBus.query()
  → 并行/顺序查询所有provider
  → 合并结果 (all_records)
  → 按score排序
  → 去重
  → 贪心token_budget填充
  → 返回top_k
```

### 路由优先级（默认）
| Provider | Priority | 说明 |
|----------|----------|------|
| jiak | 0 | 结构化知识，最精确 |
| recall | 10 | 时间线日志，最全面 |
| causal | 20 | 因果教训，最深刻 |
| delta | 30 | 跨session共振，最广 |
| unified | 40 | 温度记忆，兜底 |

## 7. 集成路径

### ISA集成
```python
# ISA.__init__中新增
self.bus = MemoryBus()
self.bus.register(JiakProvider())
self.bus.register(RecallProvider())
self.bus.register(CausalProvider(self.causal))
self.bus.register(UnifiedProvider(self.memory))
# DeltaProvider待Δ胶囊成熟后注册
```

### ICE集成
```python
# MultiSourceMemorySelector改为
class MultiSourceMemorySelector:
    def __init__(self, bus: MemoryBus):
        self.bus = bus  # 不再硬编码4个selector

    def select(self, topics, top_k=5, token_budget=1000):
        query = self._topics_to_query(topics, top_k, token_budget)
        return self.bus.query(query)
```

### MistakeLedger集成
```python
# MistakeLedger写入后自动进入CausalProvider
mistake → MistakeLedger.append() → mistake_bridge → CausalMemory.store()
                                                       ↓
                                              MemoryBus可检索
```

## 8. 文件位置

```
src/openllm/memory/
  memory_bus.py          # MemoryBus + MemoryProvider协议 + 数据结构
  providers/
    __init__.py
    jiak_provider.py     # JiakProvider（读jiak cards + index）
    recall_provider.py   # RecallProvider（读RECALL jsonl）
    causal_provider.py   # CausalProvider（包装CausalMemoryStore）
    unified_provider.py  # UnifiedProvider（包装UnifiedMemory）
```

## 9. 测试计划

| 测试 | 说明 |
|------|------|
| test_memory_bus.py | MemoryBus核心：注册/写入/检索/路由/去重 |
| test_providers.py | 各Provider：search/store/count/health |
| test_integration.py | ISA→MemoryBus→ICE端到端 |
| test_performance.py | 多源检索延迟（目标<50ms） |

## 10. 验收标准

1. ✅ MemoryBus可注册/注销provider
2. ✅ 写入自动路由到正确provider
3. ✅ 检索多源融合+去重+token_budget填充
4. ✅ ICE的MultiSourceMemorySelector改为使用MemoryBus
5. ✅ ISA通过MemoryBus写入/检索
6. ✅ MistakeLedger→Causal链路通过MemoryBus可检索
7. ✅ 所有测试通过
8. ✅ 多源检索延迟<50ms
