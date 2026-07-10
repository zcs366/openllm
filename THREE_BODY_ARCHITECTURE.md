# openLLM三体架构文档

## 一、架构概览

```
main_loop.py (533行)
├── perception.py ──→ ISA + 章鱼I = "我知道什么"
├── decision.py ────→ IOS = "我决定什么"
├── execution.py ───→ ISN + IKO = "我做什么"
├── protocol.py ────→ HeartbeatContext 通信合同
├── three_body_heartbeat.py → 三体并行流水线
├── degradation_trace.py → 降级追踪(8个trace点)
└── 7个impl文件 → 各体独立实现
```

## 二、三层定义

| 层 | 文件 | 行数 | 职责 |
|:--:|------|:----:|------|
| **感知层** | perception.py | 11 | ISA记忆+章鱼I搜索 = "我知道什么" |
| **决策层** | decision.py | 9 | IOS治理+仲裁+拒绝权 = "我决定什么" |
| **执行层** | execution.py | 11 | ISN工具+IKO输出 = "我做什么" |

## 三、通信协议

HeartbeatContext数据流：
```
感知层写：identity, memory, search_results, prediction, left_proposal
决策层写：risk, decision
执行层写：result, output, metrics
```

## 四、进化点

| 体 | 进化内容 | 验证 |
|:--:|---------|:----:|
| ISA | unified_memory + CausalMemory + evidence_replay | ✅ |
| IOS | RejectionMechanism拒绝权 | ✅ |
| ISN | ToolRegistry verify hooks | ✅ |
| 章鱼I | PredictionEngine双路径 | ✅ |
| IKO | 七因子管线(已是完整实现) | ✅ |

## 五、降级追踪

8个trace点覆盖所有进化try/except：
- `~/.openllm/output/degradation_log.jsonl`

## 六、测试结果

| 测试 | 结果 |
|------|:----:|
| 全量pytest | 693通过 |
| e2e集成测试 | 29/29 |
| 复杂查询召回 | 10/10 |
| 三体心跳 | 15/15 |

## 七、Hermes集成

插件：`~/.hermes/plugins/openllm_three_body/`
- pre_llm_call: 感知层记忆召回注入system_prompt
- post_llm_call: 记录三体状态

## 八、unified_memory

36条记忆，覆盖：
- openLLM三体架构
- ISA/IOS/ISN/章鱼I
- 三体通信协议
- 降级追踪机制
- Harness四组件
- 更多...

搜索算法：关键词拆分+字符级匹配(50%阈值)

---

*版本: v1.0 | 日期: 2026-07-10 | 状态: 已验证*
