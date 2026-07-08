# IKO 架构文档 — 输出体

> openLLM 六体架构（IAX·IAI·ISA·IOS·ISN·**IKO**）中唯一面向用户的器官。

---

## 1. IKO 定位

六体架构是一个仿生的 AI Agent 认知-行动闭环：

| 器官 | 职责 | 面向 |
|------|------|------|
| IAX (心跳) | 健康探测、存活信号 | 系统内部 |
| IAI (感知) | 信息检索、多源搜索 | 系统内部 |
| ISA (记忆) | 会话记忆、知识图谱 | 系统内部 |
| IOS (决策) | 治理裁决、风险评估 | 系统内部 |
| ISN (执行) | 工具调用、Skill 路由 | 系统内部 |
| **IKO (输出)** | **意图分类、渲染路由、反馈校准** | **面向用户** |

IKO 是六体与人类之间的最后一道桥梁。所有五体的内部运算结果，
**必须**经过 IKO 的七因子管线才能呈现给用户。

核心设计约束：
- **零 LLM 调用**：所有分类、路由、校准逻辑均为确定性规则
- **可解释性**：每次输出附带 `reason` 字段（雅典娜约束）
- **不可变审计**：每次输出生成链式哈希审计记录（赫拉约束）

---

## 2. 七因子公式

$$
OutputQuality = Intent \times Adaptive \times Trust \times SharedState \times SelfCalibrate \times Muscle \times Refuse
$$

| 因子 | 模块 | 作用 |
|------|------|------|
| **Intent** | IntentClassifier | 将决策内容分类为 6 种输出意图 |
| **Adaptive** | OutputRouter + RendererRegistry | 根据意图选择渲染器和输出格式 |
| **Trust** | OutputAuditChain | 为每次输出建立不可变审计记录 |
| **SharedState** | SilenceAuditor | 审计静默意图的合理性，防止信息吞没 |
| **SelfCalibrate** | LambdaCalibrator | 根据反馈信号动态校准输出质量置信度 λ |
| **Muscle** | SymmetricCodec | 压缩/解压推理链，控制输出密度 |
| **Refuse** | FeedbackSignal + ProbingTrainer | 从用户行为中学习拒绝模式和追问偏好 |

每个因子独立可测，串联形成完整的输出质量管线。

---

## 3. 八个模块

### 3.1 IntentClassifier（意图分类器）

> 将决策内容分类为 6 种输出意图，纯规则、零 LLM 调用。

```
classify(context, decision) → ClassificationResult
```

**分类优先级链**（从高到低）：

| 优先级 | 意图 | 触发条件 |
|--------|------|----------|
| 1 | ERROR | risk_level == "HIGH" |
| 2 | CONFIRM | risk_level == "MEDIUM" |
| 3 | ACT | has_tool_calls 或 has_side_effects |
| 4 | DECIDE | option_count >= 2 |
| 5 | INFORM | decision.content 非空 |
| 6 | SILENT | 默认兜底 |

**公开类**：`OutputIntent`, `IntentClassifier`, `ClassificationResult`

### 3.2 SilenceAuditor（沉默审计器）

> 审计 SILENT 意图是否合理，防止不当沉默。

```
audit(intent, context, reversible) → OutputIntent
explain_silence(output_id) → str
```

**审计规则**（按优先级）：
1. 非 SILENT → 直接放行
2. SILENT + reversible=False → 回退为 INFORM（赫尔墨斯约束）
3. SILENT + risk_level > LOW → 回退为 CONFIRM（阿瑞斯约束）
4. SILENT + 用户首次在该领域交互 → 回退为 INFORM
5. 否则 SILENT 通过

**公开类**：`SilenceAuditor`

### 3.3 OutputRouter（输出路由引擎）

> 根据 OutputIntent 选择渲染器，生成 RenderPlan。

```
route(intent, content, user_prefs) → RenderPlan
render(intent, content, user_prefs, confidence) → str
```

**内置渲染器**：

| 渲染器 | 意图 | 输出格式 |
|--------|------|----------|
| TextRenderer | INFORM | 纯文本 |
| StructuredRenderer | DECIDE | 表格/列表 |
| DiffRenderer | ACT | Diff 变更预览 |
| ConfirmationRenderer | CONFIRM | 操作确认卡片 |
| ErrorCardRenderer | ERROR | 错误报告卡片 |
| SilentRenderer | SILENT | 无输出 |

**公开类**：`BaseRenderer`, `OutputRouter`, `RendererRegistry`, `RenderPlan`

### 3.4 OutputAuditChain（输出审计链）

> 为每次输出建立不可变的链式哈希审计记录。

```
append(output_id, intent, content, ...) → OutputAuditEntry
verify() → bool
get_provenance(output_id) → list[OutputAuditEntry]
```

审计链遵循 `protocol.py` 的 `prev_hash` 模式：
`genesis → hash₁ → hash₂ → ...`

**公开类**：`OutputAuditEntry`, `OutputAuditChain`

### 3.5 SymmetricCodec（对称编解码器）

> 压缩推理链为轻量摘要，支持可逆恢复。

```
compress(chain) → CompressedReasoning
decompress(compressed) → FullReasoningChain
verify_reversibility(compressed) → bool
```

**压缩策略**：
- summary: phases[0].description
- key_decisions: risk_assessments 的 decision 字段
- confidence: phases 的平均置信度
- full_chain_ref: 链 JSON 的 sha256[:16]

**公开类**：`CompressedReasoning`, `FullReasoningChain`, `SymmetricCodec`

### 3.6 OutputFeedbackCollector（反馈收集器）

> 收集用户隐式/显式反馈，推断偏好，动态调整输出密度。

```
detect_signal(user_action, output_id) → FeedbackSignal
get_user_preferences(user_id) → dict
should_adjust_density(user_id) → float
```

**信号检测规则**：
- 立即追问 → CLARIFIED
- 采纳执行 → ACCEPTED
- 修改后使用 → MODIFIED
- 沉默30s换话题 → IGNORED
- 明确拒绝 → REJECTED

**公开类**：`FeedbackSignal`, `OutputFeedbackCollector`

### 3.7 LambdaCalibrator（λ 自校准器）

> 根据反馈信号动态校准输出质量置信度 λ ∈ [0.0, 1.0]。

```
update(signal, output_confidence) → None
should_rollback() → bool
trigger_transparency_rollback(domain) → None
get_current_lambda() → float
```

**校准规则**：
- ACCEPTED + 高置信度 → λ += 0.02
- ACCEPTED + 低置信度 → λ += 0.01
- REJECTED + 高置信度 → λ -= 0.05（阿瑞斯惩罚）
- REJECTED + 低置信度 → λ -= 0.02
- CLARIFIED → λ -= 0.03 + 触发透明回滚

**回滚条件**（任一满足）：
- 连续 3 次 CLARIFIED
- 单次高置信度 REJECTED
- λ 跌至 0.3 以下

**公开类**：`LambdaState`, `LambdaCalibrator`

### 3.8 ProbingTrainer（追问训练器）

> 根据用户会话成熟度（M1/M2/M3）生成追问建议。

```
should_prompt_probing(user_id, session_count) → bool
get_probing_suggestion(last_output) → str
```

**成熟度模型**：
- M1 (<30 sessions)：每 5 次提示 1 次
- M2 (30-90 sessions)：每 10 次提示 1 次
- M3 (>90 sessions)：不再提示

**公开类**：`ProbingTrainer`

---

## 4. 数据流拓扑

```
                          ┌─────────────────────────────────┐
                          │        用户交互层               │
                          └───────────┬─────────────────────┘
                                      │
                          ┌───────────▼─────────────────────┐
                          │  1. IntentClassifier            │
                          │     classify(ctx, dec)          │
                          │     → OutputIntent              │
                          └───────────┬─────────────────────┘
                                      │ OutputIntent
                          ┌───────────▼─────────────────────┐
                          │  2. SilenceAuditor              │
                          │     audit(intent, ctx)          │
                          │     → OutputIntent (可修正)     │
                          └───────────┬─────────────────────┘
                                      │ OutputIntent
                          ┌───────────▼─────────────────────┐
                          │  3. OutputRouter                │
                          │     route(intent, content)      │
                          │     → RenderPlan                │
                          └───────────┬─────────────────────┘
                                      │ RenderPlan
                          ┌───────────▼─────────────────────┐
                          │  4. SymmetricCodec              │
                          │     compress(chain)             │
                          │     → CompressedReasoning       │
                          └───────────┬─────────────────────┘
                                      │
                          ┌───────────▼─────────────────────┐
                          │  5. OutputAuditChain            │
                          │     append(...)                 │
                          │     → OutputAuditEntry          │
                          └───────────┬─────────────────────┘
                                      │
                          ┌───────────▼─────────────────────┐
                          │  6. 渲染器 (BaseRenderer)       │
                          │     render(intent, content)     │
                          │     → str (最终输出)            │
                          └───────────┬─────────────────────┘
                                      │
                          ┌───────────▼─────────────────────┐
                          │        用户接收输出              │
                          └───────────┬─────────────────────┘
                                      │ (用户行为反馈)
                          ┌───────────▼─────────────────────┐
                          │  7. FeedbackCollector           │
                          │     detect_signal(action)       │
                          │     → FeedbackSignal            │
                          └───────────┬─────────────────────┘
                                      │
                          ┌───────────▼─────────────────────┐
                          │  8. LambdaCalibrator            │
                          │     update(signal, conf)        │
                          │     → λ ∈ [0.0, 1.0]           │
                          └─────────────────────────────────┘
                                      │
                          ┌───────────▼─────────────────────┐
                          │  ProbingTrainer (周期性)         │
                          │     should_prompt_probing()     │
                          │     → 追问建议                  │
                          └─────────────────────────────────┘
```

---

## 5. 与其他五体的数据接口

### 5.1 IKO ← IOS（决策体）

IOS 通过 `DECISION_RESULT` 消息向 IKO 传递决策内容：

```python
# IOS 发送
from openllm.protocol import Protocol, BodyName, MessageType
protocol = Protocol()
envelope = protocol.send(
    source=BodyName.IOS,
    target=BodyName.IKO,
    message_type=MessageType.DECISION_RESULT,
    payload={
        "decision_id": "dec-001",
        "content": {"type": "answer", "content": "..."},
        "context": {"risk_level": "LOW", ...},
    },
)

# IKO 接收
result = classifier.classify(
    context=envelope.payload["context"],
    decision=envelope.payload["content"],
)
```

### 5.2 IKO ← ISN（执行体）

ISN 通过 `TOOL_RESULT` 消息向 IKO 传递执行结果：

```python
# ISN 执行完毕后，IKO 根据 has_tool_calls 判断 ACT 意图
context = {"risk_level": "LOW", "has_tool_calls": True, ...}
result = classifier.classify(context, decision)
# → OutputIntent.ACT
```

### 5.3 IKO → ISA（记忆体）

IKO 将输出审计记录写入 ISA：

```python
# 审计链条目可作为溯源记录写入 ISA
entry = audit_chain.append(...)
# entry 可序列化后存入 ISA 的 PROVENANCE_RECORD
```

### 5.4 IKO ← IAI（感知体）

IAI 的搜索结果可能影响输出密度和渲染策略：

```python
# IAI 的搜索结果作为 context 的一部分传入
context["search_depth"] = len(search_results)  # 新增字段
```

### 5.5 IKO → IAX（心跳体）

IKO 通过 `OBSERVABILITY_LOG` 消息向 IAX 报告输出健康状态：

```python
# IKO 的输出审计统计可作为可观测信号
protocol.send(
    source=BodyName.IKO,
    target=BodyName.IAX,
    message_type=MessageType.OBSERVABILITY_LOG,
    payload={
        "audit_chain_length": len(audit_chain),
        "lambda_value": calibrator.get_current_lambda(),
    },
)
```

---

## 6. 三阶段里程碑

### 阶段一：透明（Transparent）

**目标**：用户能看到系统的每一步推理。

- IntentClassifier 为每次分类提供 `reason` 字段
- OutputAuditChain 记录所有输出的完整溯源
- SymmetricCodec 保留完整推理链（reversible=True）
- LambdaCalibrator 记录每次校准事件
- 所有输出附带置信度标签

**验收标准**：
- [ ] 100% 输出附带 reason 字段
- [ ] 审计链 verify() 100% 通过
- [ ] 推理链可逆率 > 95%

### 阶段二：可信（Trustworthy）

**目标**：系统输出是可信赖的，用户可以放心采纳。

- LambdaCalibrator 稳定在 [0.6, 0.8] 区间
- FeedbackCollector 的 ACCEPTED 比率 > 70%
- OutputAuditChain 无篡改记录
- SilenceAuditor 拦截 100% 的不当沉默
- ProbingTrainer 帮助用户提升追问质量

**验收标准**：
- [ ] λ 值稳定在 [0.6, 0.8]
- [ ] ACCEPTED 反馈比率 > 70%
- [ ] 审计链完整性 100%

### 阶段三：默契（Tacit）

**目标**：系统理解用户偏好，输出几乎不需要用户追问。

- FeedbackCollector 推断准确率 > 85%
- 密度调整因子自动收敛到用户偏好范围
- ProbingTrainer 频率降低（M3 用户不再提示）
- 输出格式自适应（结构化/纯文本/Diff 自动切换）
- 用户满意度评分 > 4.5/5.0

**验收标准**：
- [ ] 偏好推断准确率 > 85%
- [ ] 密度因子自动收敛
- [ ] 用户满意度 > 4.5/5.0

---

## 7. 七神约束清单

| 神 | 约束 | 模块 | 实现 |
|----|------|------|------|
| 赫淮斯托斯 | 对称性 | SymmetricCodec | compress ↔ decompress 互为逆操作 |
| 赫尔墨斯 | 开口性 | SilenceAuditor | reversible=False 时回退为 INFORM |
| 雅典娜 | 可解释性 | IntentClassifier | 每次分类附带 reason 字段 |
| 阿瑞斯 | 防御性 | LambdaCalibrator | 高置信度 REJECTED → λ -= 0.05 |
| 德墨忒尔 | 持久化 | FeedbackCollector | 反馈历史持久化到 JSON |
| 赫拉 | 链式审计 | OutputAuditChain | prev_hash 链式链接，genesis → hash₁ → ... |
| 阿波罗 | 透明性 | OutputAuditChain | verify() 可被外部工具独立验证 |

**约束优先级**（冲突时从高到低）：
1. 赫尔墨斯（开口性）— 不能静默返回残缺数据
2. 阿瑞斯（防御性）— 高风险必须确认
3. 赫淮斯托斯（对称性）— compress/decompress 必须互逆
4. 雅典娜（可解释性）— 必须附带 reason
5. 德墨忒尔（持久化）— 关键状态必须可持久化
6. 赫拉（链式审计）— 输出必须有审计记录
7. 阿波罗（透明性）— 审计链必须可独立验证

---

## 8. 退役接口设计

当某个 IKO 模块需要退役时，遵循以下流程：

### 8.1 退役条件

- 模块已不再被任何其他模块引用
- 新的实现已完全覆盖其功能
- 所有依赖方已迁移到新接口

### 8.2 退役流程

```python
# 第一步：标记为 deprecated
class OldModule:
    """.. deprecated:: 0.2.0
    Use NewModule instead. Will be removed in 0.3.0.
    """
    def __init__(self):
        import warnings
        warnings.warn(
            "OldModule is deprecated, use NewModule",
            DeprecationWarning,
            stacklevel=2,
        )

# 第二步：保留 __all__ 条目（一个版本周期）
__all__ = [
    "OldModule",      # deprecated
    "NewModule",      # replacement
]

# 第三步：从 __all__ 中移除，删除模块文件
```

### 8.3 版本兼容矩阵

| 版本 | 旧接口 | 新接口 | 迁移方式 |
|------|--------|--------|----------|
| 0.1.x | 可用 | — | — |
| 0.2.x | deprecated 警告 | 可用 | 参考迁移指南 |
| 0.3.x | 移除 | 可用 | 强制迁移 |

### 8.4 退役接口审计

退役前必须通过以下检查：

- [ ] 运行完整测试套件，确认无 breaking change
- [ ] 更新 `__all__` 列表
- [ ] 更新 `__init__.py` 的模块 docstring
- [ ] 在 `docs/iko-architecture.md` 中标记退役
- [ ] 保留一个版本周期的 deprecation warning
- [ ] 写入 changelog 和迁移指南

---

## 附录：快速参考

```python
# 一键导入所有公开 API
from openllm.iko import (
    OutputIntent, IntentClassifier, ClassificationResult,
    FeedbackSignal, OutputFeedbackCollector,
    OutputAuditEntry, OutputAuditChain,
    ProbingTrainer,
    BaseRenderer, OutputRouter, RendererRegistry, RenderPlan,
    SilenceAuditor,
    CompressedReasoning, FullReasoningChain, SymmetricCodec,
    LambdaState, LambdaCalibrator,
)
```
