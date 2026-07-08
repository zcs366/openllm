# 接口契约 v1.0 — 17层执行清单 Phase 0

> **产出日期**：2026-07-07
> **来源**：军师审核v2 + 赫淮斯托斯界碑（"先定契约再锻造"）
> **范围**：Sprint 1（Layer 3/10/1）+ Sprint 2（Layer 4）的4个新模块

---

## 设计原则

1. **dataclass优先**：所有输入/输出用`@dataclass(frozen=True)`，不可变
2. **六体边界明确**：每个模块标注归属六体，跨体调用走函数参数（不走全局状态）
3. **错误分层**：`ValidationError`（可恢复）/ `SecurityError`（需审计）/ `FatalError`（需回滚）
4. **日志审计**：所有关键操作写`logging`，安全相关写`AuditEntry`
5. **零LLM依赖**：Sprint 1的4个模块全部纯规则/纯代码，不调LLM

---

## 模块1：ToolResultValidator（Layer 3 · ISN）

**职责**：工具调用后验证结果正确性和完整性

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

class ValidationResult(str, Enum):
    PASS = "pass"           # 验证通过
    WARN = "warn"           # 有警告但可用
    FAIL = "fail"           # 验证失败，结果不可用
    BLOCKED = "blocked"     # 安全拦截，结果应丢弃

class ValidationCheck(str, Enum):
    NON_EMPTY = "non_empty"             # 返回非空
    TYPE_MATCH = "type_match"           # 类型匹配
    ERROR_CODE = "error_code"           # 错误码检查
    LOGICAL_CONSISTENCY = "logical_consistency"  # 结果与调用参数逻辑一致
    SIZE_BUDGET = "size_budget"         # 结果大小在预算内

@dataclass(frozen=True)
class ToolCall:
    """一次工具调用的完整记录"""
    tool_name: str                      # 工具名
    tool_type: str                      # search/read/write/execute/network
    params: dict                        # 调用参数
    result: Any                         # 工具返回结果
    duration_ms: float = 0.0            # 耗时
    session_id: str = ""
    turn_id: str = ""

@dataclass(frozen=True)
class ValidationResult_Detail:
    """单条验证结果"""
    check: ValidationCheck
    result: ValidationResult
    message: str = ""
    details: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ToolValidationReport:
    """完整验证报告"""
    tool_call: ToolCall
    checks: list                        # list[ValidationDetail]
    overall: ValidationResult           # 最终判定
    should_retry: bool = False          # 是否建议重试
    should_block: bool = False          # 是否建议阻断
    audit_entry: Optional[Any] = None   # AuditEntry（如有安全问题）

# ── 核心接口 ──

def validate_tool_result(
    tool_call: ToolCall,
    *,
    size_budget: int = 100_000,         # 结果最大字符数
    allowed_error_codes: set = None,    # 允许的错误码集合
    consistency_rules: dict = None,     # 自定义一致性规则
) -> ToolValidationReport:
    """验证工具调用结果。

    由 main_loop.py 在工具执行后调用。
    返回验证报告，调用方根据 overall 判断：
      PASS/WARN → 继续
      FAIL → 重试或降级
      BLOCKED → 丢弃+审计+告警
    """
    ...

# ── 跨体调用约定 ──
# ISN(本模块) → IOS: 验证失败时写入 failure_tracker（IOS管辖）
# IOS → ISN: IOS 可调用 validate_tool_result 做二次验证
# ISA → ISN: ISA 的 supersession 检测可引用验证结果
```

---

## 模块2：SelfHarness（Layer 10 · IOS+ISA）

**职责**：自动发现性能瓶颈→生成改进提案→评估效果

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class BottleneckType(str, Enum):
    FAILURE_CLUSTER = "failure_cluster"     # 失败模式聚类
    SLOW_TOOL = "slow_tool"                 # 工具调用耗时异常
    LOW_CONFIDENCE = "low_confidence"       # 置信度持续偏低
    MEMORY_BLOAT = "memory_bloat"           # 记忆膨胀
    REPETITION = "repetition"               # 重复犯错
    CONTEXT_PRESSURE = "context_pressure"   # 上下文压力过大

class ProposalStatus(str, Enum):
    PENDING = "pending"         # 待评估
    APPROVED = "approved"       # 已采纳
    REJECTED = "rejected"       # 已拒绝
    DEPLOYED = "deployed"       # 已部署
    ROLLED_BACK = "rolled_back" # 已回滚

@dataclass(frozen=True)
class Bottleneck:
    """识别到的瓶颈"""
    bottleneck_type: BottleneckType
    description: str
    evidence: dict                 # 支撑数据
    severity: float                # 0-1 严重程度
    affected_module: str           # 受影响的模块名
    first_seen: float              # 首次发现时间戳
    occurrence_count: int = 1      # 出现次数

@dataclass(frozen=True)
class ImprovementProposal:
    """改进提案"""
    proposal_id: str
    bottleneck: Bottleneck          # 针对的瓶颈
    action_type: str                # "modify_skill" / "adjust_config" / "add_rule" / "remove_code"
    target_file: str                # 目标文件路径
    target_change: str              # 变更描述（自然语言）
    expected_effect: str            # 预期效果
    confidence: float               # 0-1 提案置信度
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: float = 0.0
    evaluated_at: float = 0.0
    rollback_target: Optional[str] = None  # 回滚目标（修改前的快照）

# ── 核心接口 ──

def scan_bottlenecks(
    *,
    failure_tracker: Any = None,     # FailureTracker实例
    reliability_log: list = None,     # 可靠性日志
    metacognition: Any = None,        # 元认知数据
    window_hours: int = 24,           # 扫描时间窗口
) -> list[Bottleneck]:
    """扫描系统瓶颈。由SelfHarness定期调用（如每6小时）。
    返回按severity降序排列的瓶颈列表。
    """
    ...

def generate_proposal(
    bottleneck: Bottleneck,
    *,
    precedent_log: Any = None,        # 判例日志（避免重复提案）
    max_proposals: int = 3,
) -> list[ImprovementProposal]:
    """对单个瓶颈生成改进提案。
    检查precedent_log避免重复，返回最多max_proposals个提案。
    """
    ...

def evaluate_proposal(
    proposal: ImprovementProposal,
    *,
    validation_set: list = None,      # held-out验证集（RSEA思想）
    metric_fn: callable = None,       # 效果评估函数
) -> ImprovementProposal:
    """评估提案效果。返回更新状态后的提案。
    如果 validation_set 不为空，做 held-out 验证。
    """
    ...

def deploy_proposal(proposal: ImprovementProposal) -> bool:
    """部署已批准的提案。返回是否成功。
    部署后自动记录到precedent_log。
    """
    ...

def rollback_proposal(proposal: ImprovementProposal) -> bool:
    """回滚已部署的提案。返回是否成功。
    使用proposal.rollback_target恢复修改前状态。
    """
    ...

# ── 跨体调用约定 ──
# IOS(本模块) 调用 ISA: 读取记忆数据判断 MEMORY_BLOAT 瓶颈
# IOS(本模块) 调用 ISN: 评估工具调用瓶颈（SLOW_TOOL）
# IOS(本模块) 写入 IOS: 部署/回滚结果写入 precedent_log
# IOS(本模块) → 外部: deploy_proposal 可修改 skills/ 目录（ISN管辖）
```

---

## 模块3：SecurityLayeredFilter（Layer 1 · ISA+IAI）

**职责**：检索结果按安全等级分层注入上下文

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class SecurityLayer(str, Enum):
    SAFE = "safe"               # 直接注入
    CAUTION = "caution"         # 注入但加标记
    BLOCKED = "blocked"         # 丢弃+审计

@dataclass(frozen=True)
class RetrievalItem:
    """单条检索结果"""
    content: str                # 内容
    source: str                 # 来源标识
    trust_level: str            # 来自 immune.py 的 TrustLevel
    relevance_score: float      # 相关性分数
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class FilteredResult:
    """过滤后的结果"""
    item: RetrievalItem
    layer: SecurityLayer
    reason: str                 # 分层理由
    prefix: str = ""            # 注入时的前缀（如"[UNTRUSTED]"）

@dataclass(frozen=True)
class SecurityFilterReport:
    """过滤报告"""
    total_items: int
    safe_count: int
    caution_count: int
    blocked_count: int
    blocked_items: list          # list[RetrievalItem] — 被阻断的条目（用于审计）
    filtered_results: list       # list[FilteredResult] — 可注入的结果

# ── 核心接口 ──

def filter_by_security_layer(
    items: list[RetrievalItem],
    *,
    trust_thresholds: dict = None,     # 自定义信任阈值
    audit: bool = True,                # 是否记录审计
) -> SecurityFilterReport:
    """对检索结果做安全分层过滤。

    由 context_router.py 在 build_context_pack() 中调用。
    分层逻辑：
      trust_level in (trusted, internal) → SAFE
      trust_level == unknown → CAUTION（加"[UNTRUSTED]"前缀）
      trust_level == untrusted → BLOCKED（丢弃+审计）
    """
    ...

def format_for_injection(
    filtered: list[FilteredResult],
    *,
    max_tokens: int = 4000,
) -> str:
    """将过滤后的结果格式化为可注入上下文的文本。
    SAFE层直接拼接，CAUTION层加前缀标记。
    按token预算裁剪。
    """
    ...

# ── 跨体调用约定 ──
# ISA(本模块) 读取 immune.py: TrustLevel 来源
# IAI(感知层) 调用本模块: 检索结果经过安全过滤后注入上下文
# 本模块 → IOS: blocked_items 写入审计日志
```

---

## 模块4：ExecutionBroker（Layer 4 · IOS）

**职责**：syscall与实际执行之间的策略层——执行前检查、执行后确认、失败回滚

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

class BrokerDecision(str, Enum):
    ALLOW = "allow"             # 允许执行
    DENY = "deny"               # 拒绝执行
    ESCALATE = "escalate"       # 升级给人类
    MODIFIED = "modified"       # 修改参数后执行

class ExecutionPhase(str, Enum):
    PRE_CHECK = "pre_check"     # 执行前检查
    EXECUTING = "executing"     # 执行中
    POST_CONFIRM = "post_confirm"  # 执行后确认
    ROLLBACK = "rollback"       # 回滚中

@dataclass(frozen=True)
class SyscallRequest:
    """syscall请求"""
    syscall_name: str           # syscall名称
    params: dict                # 参数
    caller: str                 # 调用者标识
    session_id: str = ""
    turn_id: str = ""

@dataclass(frozen=True)
class BrokerVerdict:
    """Broker裁决"""
    decision: BrokerDecision
    reason: str
    modified_params: Optional[dict] = None   # 如果MODIFIED，返回修改后的参数
    risk_level: str = "low"                  # 本次操作风险等级

@dataclass(frozen=True)
class ExecutionResult:
    """执行结果"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    phase: ExecutionPhase = ExecutionPhase.EXECUTING
    rollback_available: bool = False         # 是否可回滚
    audit_entry: Optional[Any] = None

# ── 核心接口 ──

def pre_check(
    request: SyscallRequest,
    *,
    policy_rules: dict = None,          # 策略规则集
    tool_validator: Any = None,         # ToolResultValidator（Layer 3）
) -> BrokerVerdict:
    """执行前策略检查。
    检查：参数合规→权限匹配→风险评估。
    返回 ALLOW/DENY/ESCALATE/MODIFIED。
    """
    ...

def execute_with_broker(
    request: SyscallRequest,
    handler: Callable,                  # 实际执行函数
    *,
    pre_check_fn: Callable = pre_check,
    post_confirm_fn: Callable = None,
) -> ExecutionResult:
    """带Broker的执行流程。
    1. pre_check → 裁决
    2. 如果ALLOW/MODIFIED → 执行
    3. post_confirm → 确认结果
    4. 如果失败且rollback_available → 自动回滚
    """
    ...

def post_confirm(
    request: SyscallRequest,
    result: ExecutionResult,
) -> ExecutionResult:
    """执行后确认。
    检查：结果类型匹配→无错误码→一致性。
    返回更新后的ExecutionResult。
    """
    ...

# ── 跨体调用约定 ──
# IOS(本模块) 调用 ISN: pre_check 中调用 validate_tool_result
# IOS(本模块) 拦截 Kernel.dispatch: 在 kernel.py 的 dispatch() 中插入 broker
# IOS(本模块) → IOS: 审计写入 governance audit log
# 失败回滚: 调用 handler 的 undo 方法（如果存在）
```

---

## 跨模块数据流

```
用户输入
  ↓
IAX main_loop.py Phase 2 (CONTEXT BUILD)
  ↓
ISA SecurityLayeredFilter.filter_by_security_layer()  ← Layer 1
  ↓  (SAFE/CAUTION结果)
IAX main_loop.py Phase 3 (LLM CALL)
  ↓
ISN 工具执行
  ↓
ISN ToolResultValidator.validate_tool_result()  ← Layer 3
  ↓  (PASS/WARN继续, FAIL重试, BLOCKED阻断)
IOS ExecutionBroker.execute_with_broker()  ← Layer 4
  ↓  (ALLOW执行, DENY拒绝, ESCALATE升级)
IOS SelfHarness.scan_bottlenecks()  ← Layer 10 (定期, 非每轮)
  ↓  (瓶颈→提案→评估→部署/回滚)
```

---

## 错误处理约定

| 错误类型 | 定义位置 | 处理方式 | 审计 |
|---------|---------|---------|------|
| `ValidationError` | 各模块内部 | 重试或降级 | 不写审计 |
| `SecurityError` | immune/filter/broker | 丢弃+告警 | **必须写AuditEntry** |
| `FatalError` | broker/self_harness | 回滚+告警 | **必须写AuditEntry+precedent_log** |

---

## 版本演进

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-07 | 初始接口契约：4个Sprint 1/2模块 |
| v1.1 | (待定) | Sprint 2完成后补充Layer 10.b/11/13的接口 |
