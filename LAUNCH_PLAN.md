# openLLM 独立上线方案

> "拉出去单干，不在hermes里了。" — 张成市 2026-07-12
> 匠石出品 · 三碑同落

---

## 一、现状审计

| 维度 | 状态 | 证据 |
|------|:----:|------|
| 核心模块import | ✅ | `from openllm.core.main_loop import Agent` 成功 |
| 测试 | ✅ 695通过 | `pytest tests/ -x -q` |
| CLI入口 | ⚠️ 有但简陋 | pyproject.toml定义`openllm`命令，main_loop.py有交互+单次模式 |
| 配置系统 | ❌ 缺失 | 没有config.yaml/config.json，provider硬编码 |
| 文件安全 | ❌ 缺失 | 无modTime检查，无竞态保护 |
| 自动压缩 | ❌ 缺失 | 无95%阈值触发 |
| 因果记忆 | ⚠️ 骨架 | Phase 3+8已接入，但机制分类是placeholder |
| G1-G6治理 | ⚠️ 沉睡 | 3000行代码，心跳只调了routing check |
| 六体并行 | ❌ 顺序 | 左右脑是顺序调用，不是concurrent.futures |
| 自修改守卫 | ❌ 死代码 | 143行，0调用者，0测试 |

---

## 二、上线目标

**最小可用产品**：`pip install openllm && openllm`

用户能做的事：
1. `openllm` — 进入交互式Agent对话
2. `openllm --once "写一个快速排序"` — 单次任务
3. `openllm --provider ollama --model qwen2.5:7b` — 选模型
4. `openllm --test` — 跑12项M0自检

---

## 三、分阶段方案

### Phase 0：可安装可运行（1天）

**目标**：`pip install -e . && openllm` 能跑起来

**任务**：
1. 修复pyproject.toml依赖（添加缺失的包）
2. 确保`openllm`命令指向正确的main()
3. 添加`--provider`/`--model`/`--api-key`参数
4. 添加config.json自动发现（`~/.openllm/config.json`）
5. Ollama作为默认provider（零配置可用）

**验证**：`pip install -e . && openllm --test` 12/12通过

### Phase 1：工程补全（3天）

**目标**：补齐竞品都有但openLLM没有的工程基础

**任务**：
1. **文件竞态保护**（借鉴opencode）
   - 全局`file_records`表跟踪读写时间
   - edit/write/patch工具执行前检查modTime
   - ~50行代码

2. **自动摘要触发**（借鉴opencode/CC）
   - token用量达context window 95%时自动触发
   - 调用summarizeProvider生成摘要
   - ~30行代码

3. **Provider多支持完善**
   - Ollama（已有stub）
   - DeepSeek（需更新API key）
   - MiMo（已有）
   - 统一配置格式

**验证**：`openllm --provider ollama --model qwen2.5:7b` 能完成多轮对话

### Phase 2：护城河激活（5天）

**目标**：让四大"护城河"从沉睡中醒来

**任务**：

1. **因果记忆机制分类升级**（2天）
   - `_classify_mechanism`从字符串匹配→规则引擎
   - 定义10种机制类型（超时/权限/依赖/配置/逻辑/...）
   - 基于错误信息+上下文自动分类
   - ~100行代码

2. **G1-G6完整deliberation接入心跳**（2天）
   - Phase 3.5/5.5的governance_check升级为完整流程
   - 高风险决策触发propose→challenge→vote
   - 审计链自动记录
   - ~150行代码

3. **六体真正并行**（1天）
   - `octopus.reason()`改为`concurrent.futures.ThreadPoolExecutor`
   - left.think()和right.critique()同时执行
   - ~20行代码改动

**验证**：`openllm --test` 扩展到20项，覆盖新功能

### Phase 3：差异化打磨（持续）

**目标**：让openLLM的独特价值可感知

**任务**：
1. 自修改守卫接入+测试
2. 心跳状态机（heartbeat.py）从守护模式接入交互模式
3. 输出格式优化（IKO七因子管线）
4. README + 快速开始文档

---

## 四、技术决策

### 4.1 不做什么
- **不做Hermes适配层**——独立产品，不寄生
- **不做Gateway**——先做CLI，Gateway是Phase 2+
- **不做TUI**——先用cmd.Cmd，之后可升级到prompt_toolkit
- **不做Web UI**——先活下来再好看

### 4.2 做什么
- **CLI优先**——`pip install openllm && openllm`是唯一入口
- **Ollama默认**——零配置可用，不需要API key
- **config.json**——`~/.openllm/config.json`管理provider/model/api-key
- **12→20项M0测试**——每次改动都跑测试

### 4.3 依赖最小化
```
dependencies = [
    "requests>=2.28",
    "rich>=13.0",
]
```
不引入LangChain/LlamaIndex等重型框架。

---

## 五、文件变更预估

| 文件 | 变更类型 | 行数 | Phase |
|------|---------|:----:|:-----:|
| pyproject.toml | 修改 | ~10 | P0 |
| core/main_loop.py | 增强main() | ~50 | P0 |
| core/config.py | 新建 | ~80 | P0 |
| core/file_safety.py | 新建 | ~50 | P1 |
| core/auto_summary.py | 新建 | ~30 | P1 |
| core/ios_causal.py | 升级机制分类 | ~100 | P2 |
| core/agent_heartbeat.py | 升级governance | ~150 | P2 |
| core/octopus_impl.py | 并行化reason() | ~20 | P2 |
| tests/test_launch.py | 新建 | ~100 | P0-P2 |
| README.md | 新建 | ~100 | P0 |

**总计**：~690行变更/新建

---

## 六、执行顺序

```
Day 1: Phase 0（可安装可运行）
  → pyproject.toml修复
  → config.py新建
  → main()增强（--provider/--model/--api-key）
  → pip install -e .验证
  → openllm --test 12/12

Day 2-4: Phase 1（工程补全）
  → file_safety.py（文件竞态保护）
  → auto_summary.py（自动摘要）
  → Provider多支持完善
  → 多轮对话验证

Day 5-9: Phase 2（护城河激活）
  → 因果记忆机制分类升级
  → G1-G6 deliberation接入
  → 六体并行化
  → 测试扩展到20项

Day 10+: Phase 3（差异化打磨）
  → 自修改守卫接入
  → README文档
  → 首次git tag v0.1.0
```

---

## 七、成功标准

| 指标 | 目标 | 衡量方法 |
|------|:----:|---------|
| 安装 | `pip install openllm` 成功 | pip install返回0 |
| 启动 | `openllm` 进入交互对话 | 输入消息有回复 |
| 单次 | `openllm --once "hello"` 返回结果 | 输出非空 |
| 测试 | `openllm --test` 20/20通过 | 全绿 |
| Provider | Ollama/DeepSeek/MiMo三选一可用 | 实际对话验证 |
| 因果 | Phase 8产生因果对照记录 | causal_memory.jsonl有新条目 |
| 治理 | 高风险决策触发deliberation | governance审计链有新事件 |
| 并行 | 左右脑同时执行 | 耗时<顺序执行的60% |

---

*方案设计完成。等你终裁，立即开干。*
