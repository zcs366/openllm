# Agent Harness工程方法论 v1.0
# 从 arXiv:2607.01087 提取 · 核战队联席会议通过 · 2026-07-04

## 定义

Agent Harness工程方法论 = **将Agent高速执行暴露的结构性失败转化为持久治理机制，使治理复合增长的工程方法。**

来源: Davis et al., "Cheap Code, Costly Judgment: A Case Study on Governable Agentic Software Engineering", arXiv:2607.01087, Jul 2026.

## 核心循环

```
速度暴露失败 → 人类/架构师分类(局部/结构) → 治理转换(编码为控制) → 后续Agent继承更窄空间 → 治理复合增长
```

这是非终止循环——速度越高，暴露的失败类越多，治理越厚，速度越可持续。

## 七条定律

### 定律一：治理转换定律（核心）
> 治理不是设计出来的——是被失败"发现"的。每次结构性失败都是一次治理升级机会。

论文数据: 12周、88份田野笔记、55次治理转换（35次控制+20次架构）。
关键区分: **ex-ante治理**（已知义务→推导控制）不够——必须有 **ex-post治理**（从失败中发现控制）。

### 定律二：治理基质定律
> 治理代码量 ≥ 2× 产品代码量

论文数据: 1.16MLOC治理 / 420KLOC产品 = 2.75×。
治理基质包括: 静态分析(238KLOC) + 动态测试(405KLOC) + Agent文档(247KLOC) + Agent基础设施(110KLOC) + 工具(162KLOC)。

### 定律三：Agent可读定律
> 治理机制必须同时人类可读和Agent可执行

论文数据: "The same legibility that makes these mechanisms usable by agents also made it tractable to identify and categorize them for analysis."
含义: 纯人类文档（如AGENTS.md）在高速Agent工作中会饱和失效。治理必须是类型化的、可查询的、机器可验证的。

### 定律四：Soft→Deterministic转换定律
> 隐性约定在高速Agent工作中必然饱和，必须升级为确定性控制

论文数据: "weak forms of governance fail with project and agent scale... a low-probability agent harness violation becomes a certainty."
含义: Iam原则（soft control）→ 可验证断言（deterministic control）的转换是必经之路。

### 定律五：复合增长定律
> 治理有复利——早期治理投资让后续治理更便宜、更快、更准

论文数据: "Repeated conversions increase the environment's capacity to absorb future work."
含义: 第1次治理转换可能需要1天，第10次可能只需10分钟。

### 定律六：人类判断不可替代定律
> 当实现变得充足，判断变得稀缺。人类工作从"写代码"转向"定义什么值得治理"

论文数据: "Now I feel like an engineer, and more than an engineer."
含义: Agent不能自己决定"这个失败是不是结构性的"——这需要架构师的判断。

### 定律七：约束即能力定律
> 更强的Agent需要更强的约束，而不是更少的约束

论文数据: "Subject's method was to use governance to systematically reduce the opportunities for unconstrained reasoning."
含义: 约束减少Agent需要推理的空间，让Agent在正确空间内高效工作。

## 三个关键可测试命题

1. **速度-暴露命题**: 更高的Agent速度应在更短时间内暴露结构性失败类
2. **Soft控制饱和命题**: 随Agent速度增加，基于审查的质量制度应显示逃逸缺陷率上升
3. **治理转换复合命题**: 将失败转换为确定性控制的团队应持续保持速度

## 十族治理机制

| # | 机制族 | 定义 | 代表性机制 |
|---|--------|------|-----------|
| 1 | 文档治理 | Agent可读规则可执行化 | 规则索引+上限lint, 强制snippet表 |
| 2 | 上下文+调度 | Agent知道什么、什么角色、什么约束 | brief-linting, 动态上下文注入, 角色分发 |
| 3 | Agent可观察性 | Agent/worker/deploy状态发射 | Agent注册表, 事件总线, sentinel |
| 4 | 资源中介 | 共享资源串行化 | 测试串行化, 构建串行化, Agent准入协议 |
| 5 | 合入门控 | 分级合入检查 | sentinel首提交早退, pre-commit, merge-train, 部署门控 |
| 6 | 规范接缝 | 类型化边界模型 | 组件目录, 每文档模型, 迭代原语 |
| 7 | 验证+合规 | 确定性检查覆盖不变量 | 规则引擎, 文件损坏检查 |
| 8 | 静态+动态分析 | 分析强制软件不变量 | 项目特定分析, 单元/功能/e2e测试, 属性测试, fuzz |
| 9 | 溯源 | 记录什么变了、为什么、通过什么变异面 | 每mutator归属戳, Changelog |
| 10 | 修复词汇 | 限定修复动作和失败类别为闭合类型集 | 闭合修复动词集, codemod优先阈值 |

## 工程化落地模式

```
┌─────────────────────────────────────────────────────────┐
│  治理转换引擎 (Governance Conversion Engine)              │
│                                                         │
│  Input: Agent执行失败                                    │
│    ↓                                                    │
│  [Step 1] 失败捕获 (Failure Capture)                     │
│    - 从Agent执行trace中提取失败签名                      │
│    - 记录: 失败类型/触发条件/影响范围/上下文             │
│    ↓                                                    │
│  [Step 2] 失败分类 (Failure Classification)              │
│    - 局部缺陷: 个例修复即可                              │
│    - 结构性失败: 需要治理机制干预                        │
│    - 分类信号: 失败频率/失败模式相似度/影响范围          │
│    ↓                                                    │
│  [Step 3] 治理设计 (Governance Design)                   │
│    - 架构响应: 修改系统边界,消除失败类                   │
│    - 控制响应: 添加检测机制,捕获失败实例                 │
│    ↓                                                    │
│  [Step 4] 治理安装 (Governance Installation)             │
│    - 写入治理基质 (IO-S规则/ISN约束/ISA记忆)             │
│    - 更新Agent上下文 (后续Agent自动继承)                 │
│    ↓                                                    │
│  [Step 5] 治理验证 (Governance Verification)             │
│    - 回归测试: 已知失败不再发生                          │
│    - 副作用检查: 治理不引入新失败                        │
│    ↓                                                    │
│  Output: 治理基质增长 + 后续Agent行动空间收窄            │
└─────────────────────────────────────────────────────────┘
```

## 与openLLM五体的映射

| 治理机制族 | openLLM对应 | 当前状态 |
|---|---|---|
| 文档治理 | Iam Harness 26条 + SOUL.md + CODE.md | ⚠️ SOFT |
| 上下文+调度 | ISA四层注入 + IO-S任务调度 + ISN技能路由 | ✅ 已有骨架 |
| Agent可观察性 | 哨兵4 Loop + RECALL链式审计 + checkpoint | ✅ 核心机制已有 |
| 资源中介 | IO-S资源调度 + 锁管理 | ⚠️ 仅单Agent |
| 合入门控 | IO-S五步流水线 + 包拯审计 | ✅ 骨架已有 |
| 规范接缝 | ISN三层分级 + 工具风险分级 | ⚠️ 类型化边界弱 |
| 验证+合规 | 归藏审计 + 包拯九法 | ✅ 审计有,规则引擎缺 |
| 静态+动态分析 | 归藏+包拯 | ❌ 覆盖率不足 |
| 溯源 | jiak RECALL链式审计 + recall_append签名 | ✅ 机制完整 |
| 修复词汇 | ISN工具白名单 + IO-S risk分级 | ✅ 已实现 |
