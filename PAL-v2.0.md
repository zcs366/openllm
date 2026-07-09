# PAL v2.0 · openLLM六体整合 · stub进化计划

> 军师祭酒 · 2026-07-10
> 工程法典已注入。七神启示已消化。代码已亲手读过。

## 一句话

将main_loop.py的5个stub类（~500行）逐步进化为真正实现（126个文件30778行），保持stub接口不变，内部委托给真正实现。

## Non-Goal（本次不做的三件事）

1. **不改main_loop.py的_execute_tick()结构** — 11阶段心跳不动，只替换stub类的内部实现
2. **不删loop.py** — 标记deprecated但保留，RuleRouter提取到独立文件后loop.py不再被import
3. **不造新框架** — 不写orchestrator.py，不写新的HeartbeatContext dataclass。用已有的Context（七要素）和Agent类

## 架构约束（七神+工程法典）

| 约束 | 来源 | 具体要求 |
|------|------|---------|
| stub保留接口 | 狄俄尼索斯 | 每个stub类的公开方法签名不变 |
| 内部委托 | 狄俄尼索斯 | stub内部import真正实现，不删stub |
| try/except降级 | 阿瑞斯 | 替换失败→fallback到stub逻辑 |
| 链路顺序 | 阿瑞斯 | ISA→IOS→ISN→章鱼I→IKO（上游先换） |
| 先验证import | 阿波罗 | Day 1跑全量import测试 |
| 效果对比 | 雅典娜 | 每替换一个体→对比替换前后输出 |
| stub提取 | 赫菲斯托斯 | 5个stub类从main_loop.py提取到stubs/目录 |

## 阶段计划

### Phase 0: 地基（Day 1）— 调查+提取

| 编号 | 任务 | 负责人 | 交付物 | 验收标准 |
|------|------|:------:|--------|---------|
| T-0-1 | 全量import测试 | 军师 | import_test_report.txt | 从main_loop.py出发，逐个import外部模块，记录成功/失败 |
| T-0-2 | stub提取到stubs/ | 鲁班 | src/openllm/stubs/（5个文件） | main_loop.py减~500行，from .stubs.xxx import替代内嵌class |
| T-0-3 | loop.py退役标记 | 鲁班 | loop.py头部加deprecated注释 | 不影响任何现有import |

### Phase 1: ISA进化（Day 2-3）— 让系统记得我

| 编号 | 任务 | 负责人 | 交付物 | 验收标准 |
|------|------|:------:|--------|---------|
| T-1-1 | ISA stub进化→unified_memory | 萧何 | stubs/isa.py内部委托 | build_context()调用unified_memory.py |
| T-1-2 | evidence_replay集成 | 萧何 | isa进化后的build_context包含证据回放 | Phase 2.5的evidence_replay不再需要单独import |
| T-1-3 | ISA替换对比测试 | 雅典娜 | 对比报告 | 替换前后build_context()输出的memory字段更丰富 |

### Phase 2: IOS进化（Day 4-5）— 治理层接入

| 编号 | 任务 | 负责人 | 交付物 | 验收标准 |
|------|------|:------:|--------|---------|
| T-2-1 | IOS stub进化→governance/ | 鲁班 | stubs/ios.py内部委托 | risk_check()调用governance/reliability.py |
| T-2-2 | rejection集成 | 鲁班 | ios进化后的arbitrate()支持拒绝权 | 可以reject一个proposal |
| T-2-3 | IOS替换对比测试 | 雅典娜 | 对比报告 | 风险评估从LLM调用→规则引擎+LLM混合 |

### Phase 3: ISN进化（Day 6-7）— 执行体激活

| 编号 | 任务 | 负责人 | 交付物 | 验收标准 |
|------|------|:------:|--------|---------|
| T-3-1 | ISN stub进化→tools/executor | 萧何 | stubs/isn.py内部委托 | execute()调用ToolRegistry |
| T-3-2 | Hermes ISN模式接入 | 赫尔墨斯 | isn.py参考Hermes已验证接口 | ISN的skill注册与Hermes兼容 |
| T-3-3 | ISN替换对比测试 | 雅典娜 | 对比报告 | 工具执行从stub→真正ToolRegistry |

### Phase 4: 章鱼I进化（Day 8-9）— 感知体激活

| 编号 | 任务 | 负责人 | 交付物 | 验收标准 |
|------|------|:------:|--------|---------|
| T-4-1 | 章鱼I stub进化→iai/+hemispheres | 韩信 | stubs/octopus.py内部委托 | reason()调用真正的左右脑 |
| T-4-2 | prediction/slow_path集成 | 韩信 | octopus进化后的predict使用真实因果预测 | Phase 3预测从LLM→规则+LLM |
| T-4-3 | 章鱼I替换对比测试 | 雅典娜 | 对比报告 | 推理质量提升可度量 |

### Phase 5: IKO进化（Day 10-11）— 输出体优化

| 编号 | 任务 | 负责人 | 交付物 | 验收标准 |
|------|------|:------:|--------|---------|
| T-5-1 | IKO stub进化→iko/output_router | 子产 | stubs/iko.py内部委托 | process_output()调用output_router |
| T-5-2 | IKO七因子管线升级 | 子产 | iko进化后的输出经过完整管线 | 输出格式化+度量+审计 |
| T-5-3 | IKO替换对比测试 | 雅典娜 | 对比报告 | 输出质量提升可度量 |

### Phase 6: 端到端+效果验证（Day 12-14）

| 编号 | 任务 | 负责人 | 交付物 | 验收标准 |
|------|------|:------:|--------|---------|
| T-6-1 | 六体全替换端到端测试 | 全员 | 端到端测试通过 | 用户说话→六体全用真正实现→有输出 |
| T-6-2 | 效果对比报告 | 雅典娜 | 替换前vs替换后对比 | 每体替换前后输出质量量化对比 |
| T-6-3 | 三碑同落 | 全员 | jiak+RECALL+备忘录 | 所有交付物入法体系 |

## 成本估算

| 项目 | 估算 | 说明 |
|------|------|------|
| 工时 | 14天×1h/天 = 14h | 张成市时间 |
| API | ~¥15 | deepseek-chat 14h |
| 硬件 | ¥0 | 现有WSL环境 |
| **总计** | **¥15 + 14h** | |

## 风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|:----:|:----:|------|
| 外部模块import失败 | 高 | 中 | Phase 0提前发现，写adapter |
| stub接口与真正实现不匹配 | 高 | 低 | stub保留原接口，内部adapter转换 |
| 替换后心跳跑不通 | 中 | 高 | try/except+stub fallback |
| 效果反而变差 | 低 | 中 | 对比测试及时发现，回滚到上一Phase |

## 石碑清单

- [ ] jiak卡片: openllm-six-body-evolution-v2.0
- [ ] RECALL: PAL v2.0创建记录
- [ ] 备忘录: ~/hermes/output/极大/openllm-six-body-evolution-pal.md

## Changelog

| 版本 | 日期 | 变动 | 原因 |
|------|------|------|------|
| v2.0 | 2026-07-10 | 初版。stub进化替代stub替换 | 七神启示+工程法典+代码审计 |
