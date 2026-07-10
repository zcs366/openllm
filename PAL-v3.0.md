# PAL v3.0 · main_loop.py拆分 · 从上帝文件到模块化

> 军师祭酒 · 2026-07-10 12:10
> 七神启示已消化。工程法典已注入。核战队工程专家执行。

## 一句话

将main_loop.py从2077行上帝文件拆分为独立模块，每体一个文件，Agent组装层<300行。

## Non-Goal

1. **不改_execute_tick()的Phase顺序** — 11阶段心跳不动
2. **不改任何体的公开接口** — 所有import方零改动
3. **不造新框架** — 不写orchestrator.py，不写新dataclass

## 工程约束

| 约束 | 来源 |
|------|------|
| 拆分后main_loop.py<300行 | 阿波罗 |
| 每体独立文件+故障隔离 | 阿瑞斯 |
| 拆分后立即跑全量pytest | 工程法典·循证 |
| 每个拆分有ad-hoc验证 | 工程法典·调查 |

---

## Phase 1: IOS提取（最大块·624+43行）

### T-1-1: 提取IOS到ios_impl.py

**从main_loop.py提取：**
- `class IOS` (line 558-1055, ~500行)
- `_select_strategy()` 方法
- `_record_rejection()` 方法
- `_check_governance_rules()` 方法

**目标文件：** `src/openllm/core/ios_impl.py`

**main_loop.py改动：**
```python
# 替换内嵌class IOS:
from .ios_impl import IOS
```

**验收：** pytest全量通过 + `from openllm.core.ios_impl import IOS`成功

### T-1-2: 提取数据类到models.py

**从main_loop.py提取：**
- `Message` (line 57-61)
- `Context` (line 63-80)
- `Prediction` (line 83-93)
- `RiskAssessment` (line 95-105)
- `Proposal` (line 107-115)
- `Critique` (line 117-123)
- `Decision` (line 124-131)
- `ActionResult` (line 132-139)
- `CausalDelta` (line 140-149)
- `TickMetrics` (line 151-163)

**目标文件：** `src/openllm/core/models.py`

**main_loop.py改动：**
```python
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult,
                     CausalDelta, TickMetrics)
```

**验收：** pytest全量通过

---

## Phase 2: ISN+章鱼I+IKO提取

### T-2-1: 提取ISN到isn_impl.py

**从main_loop.py提取：**
- `class ISN` (line 1219-1459, ~240行)

**目标文件：** `src/openllm/core/isn_impl.py`

**验收：** pytest全量通过

### T-2-2: 提取章鱼I+左右脑到octopus_impl.py

**从main_loop.py提取：**
- `class 章鱼I` (line 325-445, ~120行)
- `class _LeftBrain` (line 452-520, ~70行)
- `class _RightBrain` (line 523-555, ~33行)

**目标文件：** `src/openllm/core/octopus_impl.py`

**验收：** pytest全量通过

### T-2-3: 提取IKO到iko_impl.py

**从main_loop.py提取：**
- `class IKO` (line 1460-1560, ~100行)

**目标文件：** `src/openllm/core/iko_impl.py`

**验收：** pytest全量通过

---

## Phase 3: ISA+LLMProvider+CLI提取

### T-3-1: 提取ISA到isa_impl.py

**从main_loop.py提取：**
- `class ISA` (line 230-322+进化代码, ~140行)

**目标文件：** `src/openllm/core/isa_impl.py`

**验收：** pytest全量通过

### T-3-2: 提取LLMProvider到provider_impl.py

**从main_loop.py提取：**
- `class LLMProvider` (line 165-223, ~59行)

**目标文件：** `src/openllm/core/provider_impl.py`

**验收：** pytest全量通过

### T-3-3: main_loop.py瘦身到<300行

**提取完成后main_loop.py只剩：**
- imports (~30行)
- `class Agent` (~200行)
- CLI入口 (~50行)

**验收：** `wc -l src/openllm/core/main_loop.py` < 300

---

## Phase 4: 端到端验证

### T-4-1: 全量pytest
### T-4-2: Agent.run_once()端到端测试
### T-4-3: 三碑同落

---

## 成本估算

| Phase | 工时 | 依赖 |
|:-----:|:----:|:----:|
| Phase 1 | 1h | 无 |
| Phase 2 | 1h | Phase 1 |
| Phase 3 | 0.5h | Phase 2 |
| Phase 4 | 0.5h | Phase 3 |
| **总计** | **3h** | |

## 验证铁律

每个Phase完成后：
1. `python3 -m pytest tests/ -q` 必须全量通过
2. `from openllm.core.xxx_impl import Xxx` 必须成功
3. `wc -l src/openllm/core/main_loop.py` 必须递减

---

*石碑清单: jiak + RECALL + 本PAL*
