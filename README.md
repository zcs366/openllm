# OpenLLM 技术文档

> 面向 LLM 的原生 Harness Agent — 让 AI 从每次交互中学习

---

## 1. 概述

OpenLLM 是一个面向大语言模型的原生 Agent 框架。它不是更聪明的模型——是**能从自己的经历中学习的有机体**。

核心理念：**能力让你做事，治理让你做对事。98.4% 的基础设施代码证明，做对事比做事难一百倍。**

### 1.1 设计哲学

- **四体架构**：记忆(ISA) + 治理(IO-S) + 技能(ISN) + 输出(IKO)
- **五条血管**：四体之间的数据流，让系统从"四个零件"变成"一台发动机"
- **七步流水线**：每次工具调用的安全→风险→验证→执行→验证→学习→记录
- **本地推理**：Ollama 集成，零 API 成本，数据不出本机

### 1.2 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | Ollama (qwen3.5:9b) | 本地推理，零成本 |
| 语言 | Python 3.12 | 核心引擎 |
| 记忆 | Δ胶囊 + JSONL | 不可变追加存储 |
| 搜索 | 章鱼 (RECALL + jiak cards) | 记忆检索 |
| 安全 | 5级权限门 + 审计日志 | 治理层 |

---

## 2. 架构

### 2.1 四体架构

```
┌─────────────────────────────────────────────────┐
│                   OpenLLM Engine                 │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │   ISA    │  │   IO-S   │  │   ISN    │  │   IKO    │
│  │  记忆    │  │   治理   │  │   技能   │  │   输出   │
│  │ 11,151行 │  │ 2,548行  │  │ 1,629行  │  │  457行   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘
│       │              │              │              │
│       └──────────────┴──────────────┴──────────────┘
│                      五条血管
└─────────────────────────────────────────────────┘
```

### 2.2 四体职责

| 系统 | 代号 | 职责 | 核心能力 |
|------|------|------|----------|
| **ISA** | 记忆系统 | "我经历了什么" | opinion 置信度、矛盾检测、章鱼搜索 |
| **IO-S** | 治理系统 | "我该遵守什么" | verify 钩子、权限门、checkpoint |
| **ISN** | 技能系统 | "我有什么能力" | 249条工具索引、风险等级、元数据 |
| **IKO** | 输出系统 | "我产出什么" | trace 消费、八触须v2格式、校验 |

### 2.3 五条血管

| # | 连接 | 数据流 | 状态 |
|---|------|--------|------|
| 1 | ISA→IO-S | 好结果标准 → verify 判断 | 接口已定义 |
| 2 | IO-S→ISA | verify 结果 → opinion 置信度更新 | ✅ 已接通 |
| 3 | ISN→IO-S | 工具元数据 → 风险检查 | ✅ 已接通 |
| 4 | IO-S→IKO | trace 事件 → 结构化消费 | ✅ 已接通 |
| 5 | IKO→ISA | schema 验证 → 质量信号 | ✅ 已接通 |

### 2.4 七步流水线

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
~/projects/openllm/
├── openllm.py                  # CLI 入口
├── src/openllm/
│   ├── core/
│   │   ├── engine.py           # 主引擎（743行）— 四体集成 + 七步流水线
│   │   ├── loop.py             # Agent 循环（123行）— plan→act→observe→reflect
│   │   ├── provider.py         # 模型接入（241行）— Ollama/DeepSeek/OpenAI
│   │   ├── meta.py             # 元认知（268行）— 认知仪表盘 + 自救
│   │   ├── tool_scope.py       # 工具作用域（125行）— 4种预定义作用域
│   │   └── trace.py            # 追踪系统（182行）— 结构化 trace
│   ├── security/
│   │   └── gate.py             # 权限门（215行）— 5级权限 + 审计日志
│   ├── tools/
│   │   ├── executor.py         # 工具注册（216行）— 8个内置工具
│   │   └── octopus.py          # 章鱼搜索（91行）— RECALL + jiak cards
│   ├── memory/
│   │   └── capsule.py          # Δ胶囊（320行）— 双胶囊 + 检查点
│   └── identity/
│       └── soul.py             # 身份层 — SOUL + Iam 原则
├── caps/                       # 记忆存储目录
│   ├── v06_*.json              # 文本胶囊（给人看）
│   ├── v07_*.json              # 语义向量（给模型用）
│   ├── history_*.json          # 完整对话历史
│   └── checkpoint_*.json       # 状态快照
└── src/openllm/core/trace.py   # 追踪系统
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
cd ~/projects/openllm
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

## 6. 四体接口

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
    capsule_dir="~/projects/openllm/caps",  # 记忆目录
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
cd ~/projects/openllm

# 测试 Ollama 连接
python3 test_ollama.py

# 测试章鱼搜索
python3 test_octopus.py

# 测试完整引擎
python3 test_engine.py
```

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
| — | 更多 LLM（DeepSeek API、其他 Ollama 模型） | ⬜ 待做 |
| — | Web UI | ⬜ 待做 |
| — | 多 Agent 协作 | ⬜ 待做 |

---

*OpenLLM — 面向 LLM 的原生 Harness Agent*
*设计者：张成市 | 2026年5月*
*版本：v0.2.0*
