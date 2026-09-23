# OpenLLM · 六体 + ISL

> 给 AI 造一副身体：六个器官 + 一条年轮

---

## 1. 概述

OpenLLM 是一套**给 AI 造身体**的 Agent 架构：六体器官（感知 / 心跳 / 记忆 / 治理 / 技能 / 输出）+ 一条 ISL 身份回忆链。它从交互经验中学习、据此校准自身行为、并选择性遗忘；但更根本的是——每次醒来，它还认得自己是谁。

> 🏛️ **演进注记**：openLLM 经历过"经验积累系统"阶段（2026-07~08）；2026-09 回归"造人不造工具"的本体——六体器官（IAI/IAX/ISA/IOS/ISN/IKO）+ ISL 身份回忆链，目录按体归位（`src/openllm/<体>/`），治理与记忆不再是"外部系统"而是**自己的身体**。

核心理念：**能力让你做事，治理让你做对事。98.4% 的基础设施代码证明，做对事比做事难一百倍。**

### 1.1 设计哲学

- **六体架构**：感知(IAI) + 心跳(IAX) + 记忆(ISA) + 治理(IOS) + 技能(ISN) + 输出(IKO)
- **ISL 身份回忆层**：session 级纪元链，append-only 哈希链——"我经历过的"是镜子的镀银层
- **USER 第六乘数**：用户不是使用者，是系统的构成部分
- **七步流水线**：每次工具调用的安全→风险→验证→执行→验证→学习→记录
- **本地推理**：Ollama 集成，零 API 成本，数据不出本机；亦可接 DeepSeek / OpenAI 兼容端点

### 1.2 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | Ollama (qwen3.5:9b) | 本地推理，零成本 |
| 语言 | Python 3.12 | 核心引擎 |
| 记忆 | Δ胶囊 + JSONL | 不可变追加存储 |
| 搜索 | 章鱼 (RECALL + jiak cards) | 记忆检索 |
| 安全 | 5级权限门 + 审计日志 | 治理层 |

### 1.3 生态位置

OpenLLM 不是单一工具，而是一套 **AI Agent 架构体系**的枢纽——围绕"造人不造工具"这一命题，生长出一组独立可用、彼此咬合的开源子项目：记忆系统 [ISA](https://github.com/zcs366/isa)、技能系统 [ISN](https://github.com/zcs366/isn)、输出系统 [IKO](https://github.com/zcs366/iko)、知识研究 [izu](https://github.com/zcs366/izu)、语言压缩理论实验 [iat](https://github.com/zcs366/iat)、Δ胶囊记忆 [openllm-memory](https://github.com/zcs366/openllm-memory)。

在这个生态里，OpenLLM 是脊椎，六体是器官：不把 LLM 当函数调用，而是给它造一副身体——感知、心跳、记忆、治理、技能、输出六体，由信号流贯通，再以一条 ISL 身份回忆链把"每一次醒来"串成"同一个它"。

**与主流 Agent 框架的分野**：LangChain / CrewAI 等把记忆、治理、技能当作可插拔的"插件"；OpenLLM 的立场是，这三者是 Agent 的**身体**，应当与主循环同构、与身份同源。对研究 Agent 记忆 / 治理 / 技能基础设施的开发者，这套代码提供端到端可运行、逐行可审计的参考实现——从七步安全流水线，到 Δ胶囊的"选择性遗忘"（温度引擎决定什么该记住、什么该忘掉）。

---

## 2. 架构

### 2.1 六体拓扑

```
                    ┌──────────────────────────────────────┐
                    │            USER（第六乘数）             │
                    │   用户不是使用者，是系统的构成部分        │
                    └──────┬──────────────────▲─────────────┘
                           │ 用户输入          │ 治理输出
                           ▼                  │
                ┌─────────────────┐   ┌───────┴──────────┐
                │    IAI（感知）    │◄──│    IOS（治理）     │
                │  环境感知·信号路由│   │  决策审计·红线守护  │
                └──┬──┬──────────┘   └───┬──────────────┘
                   │  │                  │
     ┌─────────────┘  │                  │
     │ 心跳探测        │ 感知写入         │ 决策指令
     ▼                ▼                  ▼
┌─────────┐   ┌──────────┐     ┌──────────────┐
│ IAX(心跳) │   │ ISA(记忆) │     │  ISN（技能）   │
│ 监控存活  │   │ 海马·存储  │     │  双手·执行     │
└────▲─────┘   └──────────┘     └──────┬───────┘
     │ 心跳异常                         │ 产出
     └──────────────► IOS              ▼
                              ┌──────────────┐
                              │  IKO（输出）   │
                              │  trace·结构化  │
                              └──────────────┘
```

### 2.2 六体职责

| 体 | 角色 | 一句话职责 | 代码 |
|------|------|----------|------|
| **IAI** | 感知路由 | 感知环境、路由信号——系统的眼睛和耳朵 | `src/openllm/iai/` |
| **IAX** | 心跳监控 | 监控心跳、检测存活——系统的脉搏 | `src/openllm/iax/` |
| **ISA** | 记忆存储 | 存储与召回记忆——系统的海马体 | `src/openllm/isa/` |
| **IOS** | 治理决策 | 治理决策与审计——系统的前额叶 | `src/openllm/ios/` |
| **ISN** | 技能执行 | 执行技能调用——系统的双手 | `src/openllm/isn/` |
| **IKO** | 输出层 | trace 消费与结构化产出——系统的口 | `src/openllm/iko/` |
| **USER** | 第六乘数 | 用户是系统的构成部分，不是使用者 | — |

### 2.3 ISL 身份回忆层

**ISL 不是技能，是树的年轮**——技能可复用、可替换、无损；年轮砍掉一圈，就不是同一棵树。

- 每个 session 收尾追加一环，`sha256(prev_hash + row_json)` 哈希链串联，append-only、不可撤销
- 双标签：`capability`（我之所能）+ `lineage`（我之所历 → 所以我这样选）
- ISA 存"发生了什么"（望远镜），ISL 存"我经历过的"（镜子）
- 苏醒时读年轮认领身份；`~/.openllm/isl_chain.jsonl` 即身体档案

代码：`src/openllm/core/isl_chain.py`（纪元链）+ `src/openllm/isn/isl_member.py`（ISN 特殊成员注册）

### 2.4 信号流（谁向谁发信号）

| 源 → 目标 | 信号 |
|-----------|------|
| USER → IAI | 用户输入 → 感知路由 |
| IAI → IOS | 感知报告 → 治理决策 |
| IAX → IAI | 心跳探测 → 触发感知 |
| IAX → IOS | 心跳异常 → 告警治理 |
| ISA → IAI | 记忆召回 → 辅助感知 |
| IAI → ISA | 感知写入 → 记忆存储 |
| ISN → IOS | 执行结果 / 工具元数据 → 决策审计与风险检查 |
| IOS → ISN | 决策指令 → 技能执行 |
| IOS → IKO | trace 事件 → 结构化消费 |
| IKO → ISA | schema 验证 → 质量信号 |
| IOS → ISA | verify 结果 → opinion 置信度更新 |
| IOS → USER | 治理输出 → 用户呈现 |

### 2.5 七步流水线

每次 `execute_tool()` 调用经过七步：

```
① security_check     — 5级权限门（PLAN/READ/WRITE/NETWORK/ADMIN）
② ISN_risk_check     — 249条工具风险等级（critical 自动拦截）
③ verify_params      — 参数验证 + Shell注入检测
④ execute            — 实际执行工具
⑤ verify_result      — 结果验证（输出长度/错误检测）
⑥ ISA_belief_update  — verify → opinion 置信度 ±0.05/0.10
⑦ IKO_trace          — 结构化 trace → 章鱼可搜
⑧ ISA_schema         — 结果 vs 预期标准 → 质量信号
```

---

## 3. 代码结构

```
openllm/
├── src/openllm/
│   ├── core/       引擎与主循环（20,323 行）— engine / main_loop / 心跳调度 / ISL 纪元链
│   ├── iai/        感知路由（6,679 行）— 大脑 / 预测 / BurnInGate
│   ├── iax/        心跳监控（2,516 行）— agent_heartbeat / clock / awakening / cordis
│   ├── isa/        记忆存储（10,341 行）— 记忆总线 / 因果记忆 / 固化
│   ├── ios/        治理决策（6,002 行）— 决策审计 / 自修改守卫 / 状态审计
│   ├── isn/        技能执行（7,827 行）— 技能索引 / ISL 成员注册
│   ├── iko/        输出层（2,901 行）— trace 消费 / 结构化产出
│   ├── router/     路由器（1,073 行）— 智能选执行器 · 失败降级 · 全程记账
│   ├── cli/        命令行 — /model /context /status 等命令
│   ├── identity/   身份层 — SOUL + Iam 原则
│   ├── bridge/     Hermes 桥 / MCP 记忆服务
│   ├── comm/       通信体
│   ├── retrieval/  检索
│   ├── security/   5 级权限门 + 启动安全审计
│   ├── tools/      工具注册与执行
│   └── evolution/  演化
├── caps/           记忆存储目录（JSONL / checkpoint）
├── docs/           架构与决策文档（topology.md 为六体拓扑权威）
├── tests/          pytest 回归（2,590+ 用例）
└── OpenLLM.bat / 启动OpenLLM.ps1   一键启动
```

### 3.1 外部系统（ISA/ISN/IKO）

| 系统 | 位置 | 代码行 | 核心文件 |
|------|------|--------|----------|
| ISA | `~/projects/isa/` | 1,880 | opinion_manager.py, belief_update.py |
| ISN | `~/isn/` | 1,629 | router/, models/, store/ |
| IKO | `~/projects/iko/` | 457 | trace_consumer.py, validate.py |

---

## 4. 快速开始

### 4.1 环境要求

- Python 3.12+
- Ollama（本地 LLM 推理）
- 2080 GPU 或更高（推荐）

### 4.2 安装

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen3.5:9b

# 克隆项目
git clone https://github.com/zcs366/openllm.git && cd openllm
```

### 4.3 启动

```bash
python3 openllm.py --model qwen3.5:9b
```

### 4.4 交互命令

| 命令 | 功能 |
|------|------|
| 直接输入 | 与 Agent 对话 |
| `/quit` | 保存记忆并退出 |
| `/status` | 查看 Agent 状态 |
| `/tools` | 列出可用工具 |
| `/memory` | 查看当前记忆 |
| `Ctrl+C` | 自动保存记忆并退出 |

---

## 5. 核心模块

### 5.1 Engine（引擎）

`src/openllm/core/engine.py` — 743行

主引擎整合六层：AgentLoop + Provider + Tools + Memory + Identity + Security

**关键方法：**

| 方法 | 功能 |
|------|------|
| `wake()` | 唤醒：加载记忆 + 重建身份 |
| `chat(user_input)` | 一轮完整对话 |
| `execute_tool(tool_name, **kwargs)` | 七步流水线工具调用 |
| `sleep()` | 休眠：保存记忆 + checkpoint |

**懒加载集成：**

```python
# 四体通过懒加载接入引擎（零启动开销）
_get_checkpoint_manager()  # IO-S checkpoint
_get_isn_metadata()        # ISN 元数据（249条工具）
_get_isa_on_verify()       # ISA 信念更新
_get_iko_consume_trace()   # IKO trace 消费
_get_isa_schema_matches()  # ISA schema 验证
```

### 5.2 Provider（模型接入）

`src/openllm/core/provider.py` — 241行

支持三种 Provider：

| Provider | 模型 | 成本 | 速度 |
|----------|------|------|------|
| `ollama` | qwen3.5:9b (本地) | 免费 | ~17秒/轮 |
| `deepseek` | DeepSeek API | 按 token 计费 | ~1-2秒/轮 |
| `openai` | OpenAI API | 按 token 计费 | ~1-2秒/轮 |

**使用方式：**

```python
from openllm.core.provider import create_provider, Message

# Ollama（本地免费）
p = create_provider('ollama', model='qwen3.5:9b')

# DeepSeek API
p = create_provider('deepseek', api_key='sk-xxx')

# 对话
resp = p.chat([Message(role='user', content='你好')])
print(resp.content)
```

### 5.3 Security（安全门）

`src/openllm/security/gate.py` — 215行

5级权限门：

| 级别 | 名称 | 允许的操作 |
|------|------|-----------|
| 0 | PLAN | 只读 + 分析，不能执行 |
| 1 | READ_ONLY | 读文件、搜索、查看 |
| 2 | LOCAL_WRITE | 写文件、本地执行 |
| 3 | NETWORK | API 调用、外部通信 |
| 4 | ADMIN | 全部操作（破坏性需确认） |

**安全基座：** `IMMUTABLE_ACTIONS` 永远拒绝（delete_audit_log、disable_security 等）

### 5.4 Memory（记忆）

`src/openllm/memory/capsule.py` — 320行

双胶囊架构：

| 胶囊 | 用途 | 格式 |
|------|------|------|
| v0.6 文本胶囊 | 给人看 | JSON（决策·产出·洞察·未解） |
| v0.7 语义向量 | 给模型用 | 384维 FP32 向量 |

**记忆持久化：**

```python
# 保存（自动触发于 sleep()）
engine.sleep()
# → caps/v06_s{id}.json    (文本胶囊)
# → caps/v07_s{id}.json    (语义向量)
# → caps/history_{id}.json (完整对话历史)
# → caps/checkpoint_{n}.json (状态快照)

# 恢复（自动触发于 wake()）
engine.wake()
# → 读取最新胶囊
# → 从 checkpoint 恢复状态
# → 构建 system prompt
```

### 5.5 Tools（工具）

8个内置工具：

| 工具 | 功能 | 风险等级 |
|------|------|---------|
| `read_file` | 读取文件内容 | low |
| `write_file` | 写入文件 | medium |
| `shell` | 执行 Shell 命令 | high |
| `search` | 搜索文件内容 | low |
| `list_dir` | 列出目录内容 | low |
| `python_exec` | 执行 Python 代码 | medium |
| `octopus_search` | 搜索章鱼记忆 | low |
| `octopus_self_model` | 查看章鱼自省 | low |

**扩展工具：**

```python
from openllm.tools.executor import ToolRegistry

registry = ToolRegistry()
registry.register("my_tool", my_function, "我的自定义工具")
```

---

## 6. 六体接口

### 6.1 ISA 接口

```python
# 获取任务类型的预期结果 Schema
from opinion_manager import expected_result_schema
schema = expected_result_schema("code_generation")
# → {"required": [...], "forbidden": [...], "quality_gates": [...]}

# 接收 verify 结果，更新 opinion 置信度
from belief_update import on_verify_result
result = on_verify_result(
    tool_name="shell",
    params={"command": "ls"},
    result={"output": "..."},
    verdict="pass",  # pass/fail
)
# → {"opinion_updated": True, "confidence_delta": +0.05}
```

### 6.2 ISN 接口

```python
# 获取所有工具元数据
from isn.router.integration import export_tool_metadata
meta = export_tool_metadata()
# → [{"tool_id": "...", "name": "...", "risk_level": "high", ...}, ...]

# 按风险等级过滤工具
from isn.router.integration import get_tools_for_task
tools = get_tools_for_task(task_type="any", risk_level="high")
```

### 6.3 IKO 接口

```python
# 消费 trace 事件
from trace_consumer import consume_trace
result = consume_trace({
    "type": "verify_pass",
    "tool_name": "read_file",
    "verdict": "pass",
    "timestamp": 1234567890.0,
})

# 校验八触须v2格式
from validate import validate
report = validate("/path/to/output.md")
# → {"ok": True, "checks": [...]}
```

---

## 7. 配置

### 7.1 AgentConfig

```python
from openllm.core.engine import AgentConfig

config = AgentConfig(
    name="OpenLLM",              # Agent 名字
    provider="ollama",           # Provider (ollama/deepseek/openai)
    model="qwen3.5:9b",          # 模型名
    capsule_dir="/path/to/openllm/caps",     # 记忆目录（⚠️ 默认值指向旧检出，见下注）
    max_context_tokens=8192,     # 最大上下文 token
    checkpoint_interval=10,      # 每N轮自动 checkpoint
    enable_io_s_checkpoint=True, # 启用 IO-S checkpoint
)
```

### 7.2 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | (空) |
| `OCTOPUS_PG_PASS` | 章鱼 PostgreSQL 密码 | `***` |

---

## 8. 开发指南

### 8.1 添加新工具

```python
# 1. 在 src/openllm/tools/ 中创建工具函数
def tool_my_tool(param: str) -> str:
    """我的自定义工具。"""
    return f"结果: {param}"

# 2. 在 executor.py 中注册
registry.register("my_tool", tool_my_tool, "我的自定义工具")

# 3. 在 engine.py 的 _TOOL_PARAM_SCHEMAS 中添加验证
"my_tool": {"required": ["param"], "check": None},
```

### 8.2 添加新 Provider

```python
# 在 provider.py 的 create_provider() 中添加
if provider_type == "my_provider":
    config.endpoint = kwargs.get("endpoint", "https://api.example.com/v1/chat/completions")
    config.api_key = kwargs.get("api_key", "")
    return DeepSeekProvider(config)  # 如果兼容 OpenAI API
```

### 8.3 运行测试

```bash
cd <仓库根>

# 定向子集（快，日常用）
PYTHONPATH=src .venv/bin/python -m pytest tests/test_memory_bus.py tests/test_awakening.py -q -p no:warnings

# 全量回归（约 12 分钟 —— 务必后台跑，前台超时会截断成"跑不完"）
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -p no:warnings --tb=line
```

> ⚠️ **测试隔离（2026-09-16 起）**：`tests/conftest.py` 在**导入期**就把 `HOME`
> 钉到临时目录（`openllm-pytest-home-*`），所以跑测试**不会**写进你的真实记忆库
> `~/.openllm/memory/`。此前不是这样——一次全量回归会往真实记忆库灌 200+ 条测试
> json（`conv-turn-N` / `key-decision-N` 等）。若你换用其它 runner，请自行保证隔离。

---

## 9. Git 历史

```
9841e76 章鱼记忆系统接入 OpenLLM
dd989ee OpenLLM 记忆持久化修复
bf6ded4 OpenLLM CLI + 工具扩展 — 6个工具可用
3c83c5b OpenLLM 心脏跳起 — Ollama qwen3.5:9b 完整引擎验证通过
0b8d27d Ollama 接入 — OpenLLM 引擎获得本地 LLM 心脏
ccc00f4 血管#5: IKO→ISA 接通 — 五条血管全部完成
0195439 血管#2+#3+#4 全部接通 — 四体三连
f186574 血管#2: IO-S→ISA 接通
12ca096 血管#3: ISN→IO-S 接通
273adb1 审计整改: M1/M2交付物入库
```

---

## 10. 路线图

| 阶段 | 任务 | 状态 |
|------|------|------|
| M1 | 四体基础架构 | ✅ 完成 |
| M2 | 可观测性（trace/cost/checkpoint） | ✅ 完成 |
| M3 | trace 自分析 + 章鱼索引 | 🟡 进行中 |
| PAL P0 | reject() + 10秒对话 + 身份退役 | 🟡 执行中 |
| — | 消息总线迁移 | ⬜ 待做 |
| — | 记忆注入对话 | ⬜ 待做 |
| — | 信念状态检索 | ⬜ 待做 |

---

*OpenLLM — 六体 + ISL · 给 AI 造一副身体*
*设计者：张成市 | 初版 2026年5月 · 六体重构 2026年9月*
*版本：v0.2.0 (PAL v1.0)*
