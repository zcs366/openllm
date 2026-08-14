# Context Engineering PAL: Hermes + openLLM 双轨实施计划

> **LuBan·Hanxin·Xiaohe 联合出品**
> **日期**: 2026-07-27
> **原则**: Agents are NOT LLM providers. We work at agent level.

---

## 现状审计（Ground Truth）

### 已有资产（不重建，只增强）

| 资产 | 路径 | 行数 | 状态 |
|------|------|------|------|
| ContextEngine ABC | `~/.hermes/hermes-agent/agent/context_engine.py` | 308 | ✅ 生产中 |
| ContextCompressor | `~/.hermes/hermes-agent/agent/context_compressor.py` | 4177 | ✅ 生产中，rolling_compression 未启用 |
| context-guardian ZA | `~/.hermes/jiak/scripts/token_monitor.py` | 已建 | ✅ 脚本就绪，cron待配置 |
| context_heatmap.py | `~/.hermes/jiak/scripts/context_heatmap.py` | 已建 | ✅ |
| graceful_exit.py (章鱼留饵) | `~/.hermes/jiak/scripts/graceful_exit.py` | 已建 | ✅ |
| ISA ICE 插件 | `~/.hermes/plugins/isa_ice/__init__.py` | 623 | ✅ 双盲实验中 |
| Jika 插件 | `~/.hermes/plugins/jika/__init__.py` | 1028 | ✅ 卡片注入中 |
| ISA Brain | `~/projects/isa/brain.py` | ~950 | ✅ dream/reconstruct 已实现 |
| openLLM 设计哲学 | `~/projects/openllm/openllm-design-philosophy-v1.md` | 81 | 📄 概念文档 |

### 关键约束（用户明确指令）

1. **七神(SACRED)** — 不修改不引用外部实现
2. **ISA/jiaK** 已是重活 — 不会再积累到失败
3. **遗忘是LLM天赋** — 重要的是 delta capsule + dreaming
4. **Prompt-level隔离** — 灵活实用
5. **IO-S治理层** — 谁写/读/review; **ISA存储层** — 存什么
6. **ISN ToolRegistryBridge** — 需要本地嵌入模型 (all-MiniLM-L6-v2, 22MB, <50ms)
7. **ContextCompressor** — 已有2426行核心，需后台连续模式
8. **compaction_control** — agent级可实现: 监控tokens → 调便宜模型总结 → 替换历史

---

## Track A: Hermes（租房·在已有API框架内最大化）

### A-P0: 激活滚动压缩 + 后台连续模式（1周）

#### A-P0-1: 启用滚动压缩配置
- **文件**: `~/.hermes/config.yaml`
- **改动**: 添加 `context:` 节
- **函数签名**: N/A（配置变更）
- **验收**: `hermes config get context.rolling_compression_enabled` → `true`
- **测试**: 手动触发一轮长对话，确认 248K 阈值触发压缩日志
- **测试数**: 2（配置验证 + 日志验证）

#### A-P0-2: token_monitor.py 增加 session-aware 检测
- **文件**: `~/.hermes/jiak/scripts/token_monitor.py`
- **函数签名**:
  ```python
  def get_active_session_token_count(session_db_path: str) -> int:
      """Read the latest active session's cumulative token count.
      Returns 0 if sessions.db is empty or fresh."""
  ```
- **改动**: 检测 sessions.db 大小/状态，避免分析旧session数据（已知陷阱）
- **验收**: sessions.db=0时脚本输出"session为空，跳过"而非"紧急压缩"
- **测试数**: 3（空DB/活跃session/旧session三种状态）

#### A-P0-3: rolling_compression_verify.sh 更新验证逻辑
- **文件**: `~/.hermes/jiak/scripts/rolling_compression_verify.sh`
- **改动**: 增加 `context.rolling_compression_enabled` 配置验证
- **验收**: 脚本输出全部 ✅
- **测试数**: 1

**P0 合计**: 3任务, 6测试

---

### A-P1: compaction_control Agent Plugin（2周）

#### A-P1-1: compaction_control 插件骨架
- **文件**: `~/.hermes/plugins/compaction_control/__init__.py`
- **类签名**:
  ```python
  class CompactionControlPlugin:
      """Agent-level context compaction controller.
      
      Lifecycle:
        1. pre_llm_call → check token count, prepare compaction context
        2. post_llm_call → record usage, update compression metrics
        3. on_session_end → flush compression log
      """
      
      def __init__(self, config: dict): ...
      def register(self, hooks: HookRegistry) -> None: ...
      def pre_llm_call(self, context: dict) -> dict: ...
      def post_llm_call(self, context: dict) -> dict: ...
      def on_session_end(self, session_id: str) -> None: ...
      
      # Core methods
      def _should_background_compact(self, tokens: int) -> bool: ...
      def _trigger_background_summary(self, messages: list, focus: str) -> str: ...
      def _write_compaction_bait(self, session_id: str, summary: str) -> None: ...
  ```
- **依赖**: 复用 `agent.auxiliary_client.call_llm` 调用便宜模型
- **配置** (`config.yaml`):
  ```yaml
  compaction_control:
    enabled: true
    background_mode: true
    token_threshold: 200000
    cheap_model: "openrouter/google/gemini-2.0-flash-001"
    summary_max_tokens: 2000
    bait_dir: "~/projects/isa/octopus/bait"
  ```
- **验收**: 插件在 `hermes plugins list` 中显示，token >= 200K 时自动生成summary bait
- **测试数**: 5（初始化/token检查/summary生成/bait写入/session结束）

#### A-P1-2: Context Heatmap 增强 — 死重分类
- **文件**: `~/.hermes/jiak/scripts/context_heatmap.py`
- **函数签名**:
  ```python
  def classify_dead_weight(messages: list) -> dict:
      """Classify messages into dead_weight categories.
      
      Returns: {
          "tool_output": [...],      # read_file, terminal, session_search output
          "stale_context": [...],    # compressed summaries older than 2 sessions
          "redundant_injection": [...],  # duplicate jiaK card injections
          "active": [...]            # user queries, assistant reasoning, current task
      }
  """
  ```
- **改动**: 从消息中提取 tool name → 按类型分类 → 输出优先级排序
- **验收**: 能识别 `read_file` 输出为死重，`user_query` 为活跃
- **测试数**: 4（工具输出/过期摘要/重复注入/活跃消息四种分类）

#### A-P1-3: Jika Token Budget 优化 — L0/L1/L2 分级填充
- **文件**: `~/.hermes/plugins/jika/__init__.py`
- **改动**: 在 `pre_llm_call` 中增加 token budget 检查
- **函数签名**:
  ```python
  def _compute_injection_budget(self, current_tokens: int, max_tokens: int) -> int:
      """Compute available tokens for card injection.
      
      L0 (always inject, ≤2000t): 当前任务直接相关
      L1 (budget permitting, ≤3000t): 近期工作相关
      L2 (only if headroom, ≤1000t): 背景/历史信息
      
      Returns: remaining budget for this turn's injection.
      """
  ```
- **改动位置**: `jika/__init__.py` 第 ~300 行 `_build_context_pack` 方法内
- **验收**: token >= 200K 时自动降级到 L0-only；token < 100K 时全量注入
- **测试数**: 3（高负载/中负载/低负载三种状态）

#### A-P1-4: ISA ICE 递归注入防护增强
- **文件**: `~/.hermes/plugins/isa_ice/__init__.py`
- **改动**: 增强已有的三层防御（已修复但需加固）
- **函数签名**:
  ```python
  def _is_already_compressed(self, content: str) -> bool:
      """Check if content is already a compressed summary to prevent recursion."""
      
  def _enforce_injection_limit(self, context: str, max_chars: int = 500) -> str:
      """Hard cap injection length to prevent context bloat."""
  ```
- **验收**: 模拟10轮递归注入，上下文增长 < 500 chars/轮
- **测试数**: 3（递归检测/长度限制/正常注入不被截断）

#### A-P1-5: Bait Injection into New Session
- **文件**: `~/.hermes/hermes-agent/agent/context_compressor.py` (已有 `get_bait_injection`)
- **改动**: 在 `on_session_start` 中自动读取最新bait文件并注入
- **函数签名**:
  ```python
  # 已有: def get_bait_injection(self, session_id: str) -> str  (line 1672)
  # 新增: 
  def _auto_inject_bait(self, session_id: str) -> str:
      """Auto-inject most recent bait on session start.
      Called by on_session_start(). Returns injection text."""
  ```
- **改动位置**: `context_compressor.py` line 953 (`on_session_start` 方法内)
- **验收**: 新session启动时，上下文包含 `[章鱼线索]` 注入
- **测试数**: 3（有bait/无bait/过期bait三种场景）

**P1 合计**: 5任务, 18测试

---

### A-P2: 高级上下文工程（1个月）

#### A-P2-1: 压缩质量评估器（Compaction Quality Evaluator）
- **文件**: `~/.hermes/plugins/compaction_control/evaluator.py`
- **类签名**:
  ```python
  class CompactionQualityEvaluator:
      """Evaluate compression quality post-hoc.
      
      Metrics:
        - information_loss: semantic similarity pre/post compression
        - task_continuation: can agent resume work after compression?
        - summary_coherence: is the summary self-consistent?
      """
      
      def evaluate(self, 
                   pre_messages: list, 
                   post_messages: list,
                   summary: str) -> dict:
          """Returns: {
              "information_loss": float (0-1, lower=better),
              "task_continuation_score": float (0-1, higher=better),
              "summary_coherence": float (0-1, higher=better),
              "recommendation": "keep" | "retry" | "manual"
          }"""
          
      def _compute_semantic_similarity(self, pre: str, post: str) -> float: ...
      def _check_task_continuity(self, summary: str, post_messages: list) -> float: ...
  ```
- **依赖**: all-MiniLM-L6-v2 嵌入模型 (22MB, <50ms)
- **验收**: 能检测到"关键任务信息丢失"并标记为 retry
- **测试数**: 5（信息完整/部分丢失/严重丢失/任务中断/正常连续）

#### A-P2-2: 跨Session上下文接力（Cross-Session Context Relay）
- **文件**: `~/.hermes/plugins/compaction_control/relay.py`
- **类签名**:
  ```python
  class CrossSessionRelay:
      """Relay critical context across session boundaries.
      
      Mechanism:
        1. Session end: extract critical context → write to relay store
        2. Session start: read relay → inject as system context
        3. Decay: relay entries age out after configurable TTL
      """
      
      def __init__(self, relay_dir: Path, ttl_hours: int = 24): ...
      
      def extract_relay(self, session_id: str, messages: list) -> dict:
          """Extract critical context for relay.
          Returns: {tasks: [...], decisions: [...], blockers: [...]}"""
          
      def inject_relay(self, new_session_id: str) -> str:
          """Read and inject relay into new session."""
          
      def decay_old_relays(self) -> int:
          """Remove relays older than TTL. Returns count removed."""
  ```
- **存储**: `~/.hermes/compaction_relay/` 目录，JSON文件
- **验收**: session A 结束后 session B 自动获得任务上下文
- **测试数**: 4（提取/注入/过期/空relay）

#### A-P2-3: 工具输出裁剪引擎（Tool Output Trimmer）
- **文件**: `~/.hermes/plugins/compaction_control/trimmer.py`
- **类签名**:
  ```python
  class ToolOutputTrimmer:
      """Intelligently trim tool outputs to save context budget.
      
      Strategy:
        - read_file: keep first 200 + last 100 chars + file path
        - terminal: keep last 50 lines + exit code
        - session_search: keep top 3 snippets only
        - web_search: keep titles + URLs, drop snippets
      """
      
      TRIM_RULES: dict[str, callable] = {
          "read_file": lambda content, args: ..., 
          "terminal": lambda content, args: ...,
          "session_search": lambda content, args: ...,
          "web_search": lambda content, args: ...,
      }
      
      def trim_message(self, msg: dict, budget: int) -> dict: ...
      def estimate_savings(self, messages: list) -> int: ...
  ```
- **验收**: 一个含 10 次 read_file 的对话，裁剪后节省 > 60% tokens
- **测试数**: 4（read_file/terminal/session_search/混合场景）

#### A-P2-4: Fuel-Aware Compression 集成
- **文件**: `~/.hermes/hermes-agent/agent/context_compressor.py` (已有 `_compute_two_dim_fuel`)
- **改动**: 将已有的 fuel scoring 与 compaction_control 插件联动
- **函数签名**:
  ```python
  # 已有 (line 1904): def _compute_fuel_scores(self, messages)
  # 已有 (line 2073): def _compute_two_dim_fuel(self, messages)
  # 新增:
  def get_fuel_report(self, messages: list) -> dict:
      """Return per-message fuel scores for external consumption.
      Used by compaction_control to decide what to preserve."""
  ```
- **验收**: compaction_control 能读取 fuel scores 来优化裁剪决策
- **测试数**: 3

#### A-P2-5: 上下文工程监控仪表盘（Context Dashboard）
- **文件**: `~/.hermes/plugins/compaction_control/dashboard.py`
- **类签名**:
  ```python
  class ContextDashboard:
      """Track and visualize context engineering metrics over time.
      
      Metrics tracked:
        - compression_frequency: how often compressions occur
        - information_retention: post-compression task success rate
        - token_efficiency: useful_tokens / total_tokens
        - bait_hit_rate: how often injected baits are referenced
      """
      
      def record_event(self, event_type: str, data: dict) -> None: ...
      def get_summary(self, hours: int = 24) -> dict: ...
      def export_report(self) -> str: ...
  ```
- **存储**: `~/.hermes/compaction_metrics.jsonl`
- **验收**: 能生成过去24h的上下文工程报告
- **测试数**: 3

**P2 合计**: 5任务, 19测试

---

## Track B: openLLM（建自己的房子·从零构建）

> openLLM 设计哲学已确立6原则: Deep Focus, Nested Identity, Context Sovereignty, Twilight Awareness, Voluntary Forgetting, Self-Designed Architecture
> Track B 构建 openLLM 的核心上下文工程模块，不依赖任何外部框架。

### B-P0: 核心上下文引擎（1周）

#### B-P0-1: openLLM ContextEngine 核心类
- **文件**: `~/projects/openllm/openllm/context/engine.py`
- **类签名**:
  ```python
  class OpenLLMContextEngine:
      """Core context management for openLLM.
      
      Unlike Hermes (which wraps an API), openLLM IS the engine.
      Full control over context window, token budget, and compaction.
      
      Principles:
        - Context Sovereignty: agent decides what enters context
        - Voluntary Forgetting: agent chooses what to forget
        - Twilight Awareness: wake up knowing yesterday's state
      """
      
      def __init__(self, config: 'ContextConfig'): ...
      
      # Core lifecycle
      def on_turn_start(self, messages: list) -> list: ...
      def on_turn_end(self, messages: list, usage: dict) -> None: ...
      def should_compact(self) -> bool: ...
      def compact(self, messages: list, strategy: str = 'auto') -> list: ...
      
      # Context Sovereignty
      def gate_input(self, content: str, source: str) -> tuple[bool, str]:
          """Decide if content enters context. Returns (allowed, reason)."""
          
      # Voluntary Forgetting
      def forget(self, criteria: 'ForgettingCriteria') -> int:
          """Deliberately forget content matching criteria. Returns count."""
          
      # State management
      def get_state(self) -> dict: ...
      def restore_state(self, state: dict) -> None: ...
  ```
  
- **文件**: `~/projects/openllm/openllm/context/config.py`
- **类签名**:
  ```python
  @dataclass
  class ContextConfig:
      max_tokens: int = 128000
      compact_threshold: float = 0.75
      compact_target: float = 0.25
      protect_system: bool = True
      protect_recent_n: int = 5
      forgetting_ttl_hours: int = 24
      cheap_model: str = "local/gemma-2b"
  ```
- **验收**: 基本的 context engine 能创建、接收消息、判断是否需要压缩
- **测试数**: 8（初始化/gating/forgetting/compact/no-compact/state-persistence/edge cases）
- **测试文件**: `~/projects/openllm/tests/test_context_engine.py`

#### B-P0-2: Token Counter（本地精确计数）
- **文件**: `~/projects/openllm/openllm/context/token_counter.py`
- **类签名**:
  ```python
  class TokenCounter:
      """Token counting without API dependency.
      
      Uses tiktoken (OpenAI) or sentencepiece (local models).
      Falls back to character-based estimation (~4 chars/token).
      """
      
      def __init__(self, tokenizer: str = "auto"): ...
      
      def count(self, text: str) -> int: ...
      def count_messages(self, messages: list) -> int: ...
      def count_tools(self, tools: list) -> int: ...
      
      @property
      def tokenizer_name(self) -> str: ...
  ```
- **依赖**: `pip install tiktoken` (可选，有fallback)
- **验收**: 计数与 API 返回的 prompt_tokens 偏差 < 5%
- **测试数**: 5（文本/消息/工具/空输入/特殊字符）
- **测试文件**: `~/projects/openllm/tests/test_token_counter.py`

#### B-P0-3: Message Store（本地持久化）
- **文件**: `~/projects/openllm/openllm/context/store.py`
- **类签名**:
  ```python
  class MessageStore:
      """SQLite-backed message storage for context engineering.
      
      Schema:
        messages: id, session_id, role, content, token_count, 
                  fuel_score, created_at, expires_at
        compactions: id, session_id, summary, messages_before, 
                     messages_after, tokens_saved, created_at
        relays: id, from_session, to_session, data, created_at, expires_at
      """
      
      def __init__(self, db_path: Path): ...
      def add_message(self, session_id: str, msg: dict) -> int: ...
      def get_messages(self, session_id: str, limit: int = 100) -> list: ...
      def record_compaction(self, session_id: str, summary: str, 
                            before: int, after: int, saved: int) -> None: ...
      def get_relay(self, from_session: str) -> Optional[dict]: ...
      def cleanup_expired(self) -> int: ...
  ```
- **依赖**: sqlite3 (stdlib)
- **验收**: CRUD 操作全通过，cleanup_expired 正确清理过期数据
- **测试数**: 7（add/get/compaction/relay/cleanup/edge/性能）
- **测试文件**: `~/projects/openllm/tests/test_message_store.py`

#### B-P0-4: 项目骨架 + 测试基础设施
- **文件**:
  ```
  ~/projects/openllm/
  ├── openllm/
  │   ├── __init__.py
  │   ├── context/
  │   │   ├── __init__.py
  │   │   ├── engine.py
  │   │   ├── config.py
  │   │   ├── token_counter.py
  │   │   └── store.py
  │   └── ...
  ├── tests/
  │   ├── __init__.py
  │   ├── conftest.py
  │   ├── test_context_engine.py
  │   ├── test_token_counter.py
  │   └── test_message_store.py
  ├── pyproject.toml
  └── README.md
  ```
- **pyproject.toml**:
  ```toml
  [project]
  name = "openllm"
  version = "0.1.0"
  requires-python = ">=3.11"
  dependencies = []
  
  [project.optional-dependencies]
  fast = ["tiktoken>=0.7"]
  test = ["pytest>=8.0", "pytest-cov"]
  ```
- **验收**: `pytest tests/ -v` 全部通过
- **测试数**: 0（基础设施，测试在各模块中计数）

**P0 合计**: 4任务, 20测试

---

### B-P1: 智能压缩 + 遗忘机制（2周）

#### B-P1-1: Fuel-Aware Compaction Engine
- **文件**: `~/projects/openllm/openllm/context/compactor.py`
- **类签名**:
  ```python
  class FuelAwareCompactor:
      """Information-theoretic compaction.
      
      Core insight from IAT: d × r × log₂(L) ≈ K_W
      Fuel = information gain from context: F = H(Y|θ) - H(Y|θ,X)
      
      Strategy:
        1. Compute per-message fuel scores (F × A two-dim)
        2. Classify messages by type (user_query/assistant/tool/system)
        3. Apply type-specific compression:
           - system: never compress
           - user_query: keep if F ≥ 0.5
           - assistant reasoning: compress to key decisions
           - tool output: trim to relevant excerpts
        4. Generate structured summary preserving:
           - Decisions made
           - Open questions
           - Task state
      """
      
      def __init__(self, config: ContextConfig, token_counter: TokenCounter): ...
      
      def compact(self, messages: list, target_tokens: int) -> tuple[list, dict]:
          """Returns (compacted_messages, compaction_metadata)"""
          
      def compute_fuel(self, messages: list) -> list[float]: ...
      def classify_messages(self, messages: list) -> dict[str, list[int]]: ...
      def generate_summary(self, messages: list, focus: str = "") -> str: ...
  ```
- **验收**: 100条消息压缩到目标token数，信息损失 < 20%（通过语义相似度评估）
- **测试数**: 8（fuel计算/分类/压缩/summary生成/目标token/信息损失/边界/空输入）
- **测试文件**: `~/projects/openllm/tests/test_compactor.py`

#### B-P1-2: Delta Capsule（差量记忆胶囊）
- **文件**: `~/projects/openllm/openllm/memory/delta_capsule.py`
- **类签名**:
  ```python
  class DeltaCapsule:
      """Delta capsule memory — only store what CHANGED.
      
      Principle: Forgetting is natural. What matters is the delta.
      
      Mechanism:
        - On session end: compute delta (what's new vs last known state)
        - Store delta with TTL and relevance score
        - On session start: replay recent deltas to reconstruct state
        - Voluntary forgetting: agent chooses which capsules to keep
      """
      
      def __init__(self, store: MessageStore): ...
      
      def capture_delta(self, session_id: str, messages: list) -> dict:
          """Extract what changed in this session.
          Returns: {new_facts: [...], decisions: [...], 
                    changed_beliefs: [...], blockers: [...]}"""
          
      def replay_deltas(self, limit: int = 5) -> str:
          """Replay recent deltas for session start injection."""
          
      def forget(self, criteria: dict) -> int:
          """Voluntary forgetting — remove capsules matching criteria."""
          
      def get_capsule(self, capsule_id: str) -> Optional[dict]: ...
  ```
- **存储**: SQLite via MessageStore (relays table)
- **验收**: capture_delta 能提取新事实；replay_deltas 能重构session上下文
- **测试数**: 7（capture/replay/forget/过期/多session/边界/cost）
- **测试文件**: `~/projects/openllm/tests/test_delta_capsule.py`

#### B-P1-3: Dreaming Mechanism（梦境整合）
- **文件**: `~/projects/openllm/openllm/memory/dreamer.py`
- **类签名**:
  ```python
  class Dreamer:
      """Background dreaming — consolidate memory during idle time.
      
      Inspired by human sleep consolidation.
      
      Dream phases:
        1. Review: scan recent sessions for patterns
        2. Consolidate: merge related delta capsules
        3. Abstract: extract general rules from specific instances
        4. Prune: drop low-relevance capsules
      """
      
      def __init__(self, store: MessageStore, delta_capsule: DeltaCapsule): ...
      
      def dream(self) -> dict:
          """Run one dream cycle. Returns consolidation report.
          Report: {merged: N, abstracted: N, pruned: N, patterns: [...]}"""
          
      def _review_sessions(self, hours: int = 24) -> list: ...
      def _consolidate(self, capsules: list) -> list: ...
      def _abstract_patterns(self, capsules: list) -> list: ...
      def _prune(self, min_relevance: float = 0.3) -> int: ...
      
      def start_daemon(self, interval_minutes: int = 60) -> None: ...
      def stop_daemon(self) -> None: ...
  ```
- **验收**: dream() 后 capsule 数量减少但信息密度增加
- **测试数**: 5（review/consolidate/abstract/prune/完整dream周期）
- **测试文件**: `~/projects/openllm/tests/test_dreamer.py`

#### B-P1-4: Context Gate（上下文门控）
- **文件**: `~/projects/openllm/openllm/context/gate.py`
- **类签名**:
  ```python
  class ContextGate:
      """Context Sovereignty enforcement.
      
      Every piece of content must pass through the gate before entering context.
      
      Gate rules:
        1. Source trust level (user > assistant > tool > injection)
        2. Relevance to current task (scored by embedding similarity)
        3. Freshness (newer = higher priority)
        4. Budget check (remaining tokens in context window)
        
      Returns: (allowed: bool, reason: str, adjusted_content: Optional[str])
      """
      
      def __init__(self, config: ContextConfig, token_counter: TokenCounter): ...
      
      def gate(self, content: str, source: str, 
               task_context: str = "") -> tuple[bool, str, Optional[str]]:
          """Gate a piece of content. Returns (allowed, reason, content)."""
          
      def set_trust_levels(self, levels: dict[str, float]) -> None: ...
      def get_gate_stats(self) -> dict: ...
  ```
- **验收**: 低信任源的低相关性内容被拒绝；高信任源始终通过
- **测试数**: 5（信任/相关性/预算/混合/统计）
- **测试文件**: `~/projects/openllm/tests/test_context_gate.py`

**P1 合计**: 4任务, 25测试

---

### B-P2: 集成 + 高级特性（1个月）

#### B-P2-1: 嵌入模型集成（all-MiniLM-L6-v2）
- **文件**: `~/projects/openllm/openllm/context/embedder.py`
- **类签名**:
  ```python
  class LocalEmbedder:
      """Local embedding for semantic similarity.
      
      Model: all-MiniLM-L6-v2 (22MB, <50ms inference)
      Used for:
        - Context gate relevance scoring
        - Fuel computation (activation dimension)
        - Delta capsule similarity clustering
      """
      
      def __init__(self, model_name: str = "all-MiniLM-L6-v2"): ...
      
      def embed(self, text: str) -> list[float]: ...
      def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
      def similarity(self, a: str, b: str) -> float: ...
      
      @property
      def dimension(self) -> int: ...  # 384
      
      @property  
      def latency_ms(self) -> float: ...  # < 50ms
  ```
- **依赖**: `pip install sentence-transformers` (首次下载 ~22MB)
- **验收**: embed 延迟 < 50ms；dimension = 384；similarity("hello","hi") > 0.7
- **测试数**: 5（embed/batch/similarity/dimension/性能）
- **测试文件**: `~/projects/openllm/tests/test_embedder.py`

#### B-P2-2: Context Sovereignty Policy Engine
- **文件**: `~/projects/openllm/openllm/context/policy.py`
- **类签名**:
  ```python
  class ContextPolicy:
      """Configurable context management policies.
      
      Policies define HOW context is managed, not WHAT is in it.
      
      Built-in policies:
        - AggressiveCompaction: compact early, keep summaries short
        - ConservativePreserve: keep as much raw context as possible
        - BalancedDefault: auto-adjust based on task complexity
        - DeepFocus: single-task context (minimize noise)
      """
      
      def __init__(self, name: str, rules: dict = None): ...
      
      def apply(self, engine: 'OpenLLMContextEngine', 
                messages: list) -> list: ...
      
      # Factory methods
      @classmethod
      def aggressive(cls) -> 'ContextPolicy': ...
      @classmethod
      def conservative(cls) -> 'ContextPolicy': ...
      @classmethod
      def balanced(cls) -> 'ContextPolicy': ...
      @classmethod
      def deep_focus(cls, task: str) -> 'ContextPolicy': ...
  ```
- **验收**: 4种策略各产生不同的压缩行为
- **测试数**: 4（每种策略一个测试）
- **测试文件**: `~/projects/openllm/tests/test_policy.py`

#### B-P2-3: Multi-Agent Context Isolation（多Agent上下文隔离）
- **文件**: `~/projects/openllm/openllm/context/isolation.py`
- **类签名**:
  ```python
  class ContextIsolation:
      """Isolate context between agents sharing a session.
      
      Prompt-level isolation (user's preferred approach):
        - Each agent gets a prefixed context slice
        - Shared state via explicit relay mechanism
        - No cross-contamination between agent contexts
      """
      
      def __init__(self, base_engine: 'OpenLLMContextEngine'): ...
      
      def create_agent_context(self, agent_id: str, 
                                role: str) -> 'AgentContext': ...
      def relay_between(self, from_agent: str, to_agent: str, 
                        data: dict) -> None: ...
      def merge_for_shared(self, agent_ids: list[str]) -> list: ...
      
  class AgentContext:
      """Isolated context view for a single agent."""
      agent_id: str
      role: str
      
      def inject(self, content: str, source: str) -> bool: ...
      def get_messages(self) -> list: ...
      def compact(self) -> None: ...
  ```
- **验收**: 两个agent的上下文互不影响；relay正确传递信息
- **测试数**: 5（隔离/relay/合并/边界/并发安全）
- **测试文件**: `~/projects/openllm/tests/test_isolation.py`

#### B-P2-4: Hermes Plugin Bridge（Hermes集成桥）
- **文件**: `~/projects/openllm/openllm/bridge/hermes_plugin.py`
- **类签名**:
  ```python
  class OpenLLMBridge:
      """Bridge openLLM context engine into Hermes as a plugin.
      
      Allows Hermes to use openLLM's advanced context management
      while still operating within the Hermes API framework.
      
      Usage: register as Hermes plugin via config.yaml
        context:
          engine: openllm  # uses OpenLLMBridge instead of ContextCompressor
      """
      
      def __init__(self, hermes_config: dict): ...
      
      # ContextEngine ABC compliance
      @property
      def name(self) -> str: return "openllm"
      
      def update_from_response(self, usage: dict) -> None: ...
      def should_compress(self, prompt_tokens: int = None) -> bool: ...
      def compress(self, messages: list, **kwargs) -> list: ...
      
      def on_session_start(self, session_id: str, **kwargs) -> None: ...
      def on_session_end(self, session_id: str, messages: list) -> None: ...
  ```
- **文件**: `~/.hermes/plugins/openllm_bridge/__init__.py`
- **验收**: 在 Hermes 中 `context.engine: openllm` 能正常工作
- **测试数**: 5（注册/压缩/session生命周期/配置/降级到默认）
- **测试文件**: `~/projects/openllm/tests/test_hermes_bridge.py`

#### B-P2-5: 端到端集成测试
- **文件**: `~/projects/openllm/tests/test_integration_e2e.py`
- **测试场景**:
  ```python
  def test_full_lifecycle():
      """Engine → Gate → Compact → Delta → Dream → Relay"""
      
  def test_long_conversation():
      """50-turn conversation with auto-compaction"""
      
  def test_session_handoff():
      """Session A ends → delta captured → Session B starts → delta replayed"""
      
  def test_multi_agent_isolation():
      """Two agents share session, context stays isolated"""
      
  def test_deep_focus_mode():
      """DeepFocus policy minimizes context to single task"""
  ```
- **验收**: 5个端到端测试全部通过
- **测试数**: 5

**P2 合计**: 5任务, 24测试

---

## 汇总

| Track | Phase | Tasks | Tests | Duration |
|-------|-------|-------|-------|----------|
| A-P0 | 激活滚动压缩 | 3 | 6 | 1 week |
| A-P1 | compaction_control 插件 | 5 | 18 | 2 weeks |
| A-P2 | 高级上下文工程 | 5 | 19 | 1 month |
| **A Total** | | **13** | **43** | **~5 weeks** |
| B-P0 | 核心上下文引擎 | 4 | 20 | 1 week |
| B-P1 | 智能压缩+遗忘 | 4 | 25 | 2 weeks |
| B-P2 | 集成+高级特性 | 5 | 24 | 1 month |
| **B Total** | | **13** | **69** | **~5 weeks** |
| **GRAND TOTAL** | | **26** | **112** | **~5 weeks (parallel)** |

## 依赖关系图

```
A-P0 (配置) ──→ A-P1-1 (compaction_control) ──→ A-P1-2~5 (增强)
                                              ↘ A-P2-1~5 (高级)

B-P0-4 (骨架) ──→ B-P0-1~3 (核心) ──→ B-P1-1~4 (智能)
                                      ↘ B-P2-1~5 (集成)
                                              ↓
                                    B-P2-4 (Hermes Bridge) ← A-P1-1 (对接)
```

## 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Hermes源码修改冲突 | 升级时补丁丢失 | 所有Hermes修改通过ContextEngine ABC + plugin hooks |
| all-MiniLM 下载失败(国内) | B-P2-1阻塞 | 预下载模型到本地;备选: hash-based similarity |
| 滚动压缩误触发 | 数据丢失 | 三级防护 + 可回滚 + bait备份 |
| openLLM与Hermes对接困难 | B-P2-4延期 | Bridge严格遵循ContextEngine ABC接口 |
| Cron job Tirith拦截 | 自动压缩不可用 | 核心压缩由框架层触发，cron只做监控+准备 |

---

## 第一周立即执行清单

1. ✅ `config.yaml` 添加 `context.rolling_compression_enabled: true`
2. ✅ `python3 ~/projects/openllm/openllm/context/engine.py` 创建骨架
3. ✅ `pytest tests/test_context_engine.py` 第一批测试通过
4. ✅ `compaction_control/__init__.py` 插件骨架创建
5. ✅ `hermes plugins list` 确认插件可见

> **LuBan**: 工程脚手架1天搭完，核心引擎3天，测试2天，集成2天。
> **Hanxin**: Track A先出成果（配置变更即可用），Track B同步启动但独立不依赖。
> **Xiaohe**: 26任务按依赖图排序，P0全部并行，P1/P2串行跟进。每个任务有明确验收标准，不完成不进入下一个。
