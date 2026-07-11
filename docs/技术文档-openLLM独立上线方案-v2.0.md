# 技术文档：openLLM 独立上线方案 v2.0

> 基于竞品源码深度分析 + 七神启示 + 五人合议终裁
> 版本：v2.0 | 日期：2026-07-12 | 作者：匠石

---

## 一、项目背景

### 1.1 竞品分析结论

基于11个竞品源码库（CC 249K行/opencode 42K/aider 27K/codex 48K等）的逐行审计：

| 发现 | 证据 |
|------|------|
| 所有竞品都没有因果记忆 | CC/opencode/aider/codex/openhands全部是"看到什么做什么" |
| CC的4层压缩管线是工程壁垒 | 每轮循环都跑Snip→Micro→Collapse→Compact |
| CC的StreamingToolExecutor是速度优势 | 模型输出时就开始执行工具 |
| openLLM的G1-G6治理是独有差异化 | 不可变事件链+防篡改审计，竞品全部缺失 |

### 1.2 openLLM现状审计

| 模块 | 代码行数 | 心跳集成 | 成熟度 |
|------|:--------:|:--------:|:------:|
| 因果记忆 | 537行 | ✅ Phase 3+8 | ⭐⭐⭐ |
| G1-G6治理 | ~3000行 | ⚠️ 只routing check | ⭐⭐⭐ |
| 六体并行 | 1016行 | ❌ 顺序执行 | ⭐⭐ |
| 自修改守卫 | 143行 | ❌ 死代码 | ⭐ |

### 1.3 七神启示（2026-07-12）

七神一致裁决：**方案顺序反了。** 原方案先做CLI（竞品已有），后做护城河（独有价值）。应先证明护城河有用（demo），再让用户能装能跑。

### 1.4 五人合议终裁

| Agent | 判断 |
|-------|------|
| 子产 | 窄做。因果demo是核心决策点 |
| 韩信 | 终局='能解释自己为什么做这个决策的Agent' |
| 鲁班 | 工程可行。P0=30行复用已有代码 |
| 萧何 | P0半天→P1 1.5天→P2 5天→P3持续 |
| 子贡 | P0鲁班执行→P1并行窗口→P2关键路径 |

---

## 二、技术方案

### 2.1 架构决策

**核心决策**：先让用户看见灵魂（因果demo），再让用户装上身体（pip install）。

**技术选型**：
- 语言：Python（已有695测试）
- 入口：`openllm` CLI（pyproject.toml已定义）
- 默认Provider：Ollama（零配置可用）
- 依赖最小化：requests + rich

### 2.2 P0：因果记忆demo

**目标**：30行代码证明"没有因果记忆时Agent犯错，有了时不错"

**实现路径**：
```
demo_causal.py
  ├── import causal_memory + ios_causal
  ├── 场景A（无因果）：Agent重复犯同一个错
  ├── 场景B（有因果）：Agent因因果记忆选择不同策略
  └── 终端彩色对比输出
```

**复用已有代码**：
- `causal_memory.py`(394行)：CausalMemoryStore + JSONL持久化
- `ios_causal.py`(143行)：learn_causal() + 10种机制分类
- 心跳Phase 3(predict) + Phase 8(causal_compare)已接入

**验收标准**：
- `python demo_causal.py` 能跑
- 输出两组对比（无因果 vs 有因果）
- 无因果时Agent重复犯错，有因果时选择不同策略

### 2.3 P1：可安装+首次体验

**目标**：`pip install openllm && openllm` 能跑

**实现路径**：
1. 修复pyproject.toml依赖
2. 确保`openllm`命令指向正确的main()
3. 添加`--provider`/`--model`/`--api-key`参数
4. demo集成到首次体验

**验收标准**：
- `pip install -e .` 成功
- `openllm` 进入交互对话
- `openllm --once "hello"` 返回结果
- `openllm --test` 12/12通过

### 2.4 P2：护城河完整激活

**目标**：因果机制分类升级 + G1-G6接入心跳 + 六体并行化

**实现路径**：
1. 因果机制分类从字符串匹配→规则引擎（~100行）
2. G1-G6完整deliberation接入心跳Phase 3.5/5.5（~150行）
3. 六体从顺序→concurrent.futures并行（~20行改动）
4. 文件竞态保护（借鉴opencode，~50行）

**验收标准**：
- `openllm --test` 扩展到20项
- 因果机制自动分类（超时/权限/依赖/配置/逻辑等）
- 高风险决策触发propose→challenge→vote
- 左右脑同时执行（耗时<顺序的60%）

### 2.5 P3：差异化打磨

**目标**：自修改守卫清理 + 自动摘要 + README + tag v0.1.0

---

## 三、Non-Goal

1. **不做Hermes适配层**——独立产品，不寄生
2. **不做TUI/Web UI**——先用cmd.Cmd
3. **不做Gateway**——先做CLI
4. **不引入重型框架**——只有requests+rich

---

## 四、成功标准

| 指标 | 目标 | 衡量方法 |
|------|:----:|---------|
| 安装 | `pip install openllm` 成功 | pip install返回0 |
| 启动 | `openllm` 进入交互对话 | 输入消息有回复 |
| 单次 | `openllm --once "hello"` 返回结果 | 输出非空 |
| 测试 | `openllm --test` 20/20通过 | 全绿 |
| 因果 | Phase 8产生因果对照记录 | causal_memory.jsonl有新条目 |
| 治理 | 高风险决策触发deliberation | governance审计链有新事件 |
| 并行 | 左右脑同时执行 | 耗时<顺序执行的60% |

---

*文档版本：v2.0 | 基于七神启示+五人合议终裁*
