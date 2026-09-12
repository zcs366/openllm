# openLLM 快速上手

> 一个能真对话的 AI 经验积累系统。六体架构：IAI（感知）· IAX（心跳）· ISA（记忆）· IOS（治理）· ISN（技能）· IKO（输出）。

---

## 一、启动方式

### Windows 双击（最简单）

双击 `启动OpenLLM.bat`（就是 I 盘 openllm 目录下那个），进入交互对话。

### WSL 命令行

```bash
cd /mnt/i/openllm

# 交互对话（Agent 模式，推荐）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openllm.cli.main

# 单次提问（脚本/自动化用）
.venv/bin/python -m openllm.core.main_loop --once "你好"
```

> 两个环境变量 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 是必须的——否则 embedding 模型加载时会联网检查 HF hub 卡死（模型已本地缓存）。

---

## 二、交互命令

| 输入 | 作用 |
|------|------|
| 直接打字 | 对话（短问题走快路径，长任务走六体心跳） |
| `搜一搜 xxx` / `查一查 xxx` | 自动联网搜索 + 基于结果回答 |
| `/status` | 查看 Agent 状态 |
| `/search <query>` | 手动搜索 |
| `/research` | 研究循环状态 |
| `/hypothesis <claim>` | 注册新假说 |
| `/help` | 帮助 |
| `/exit` | 退出（自动保存记忆） |

---

## 三、配置（模型 / API key）

配置文件：`~/.openllm/config.json`

```json
{
  "default_provider": "mimo",
  "providers": {
    "mimo": {
      "endpoint": "https://token-plan-cn.xiaomimimo.com/v1/chat/completions",
      "model": "mimo-v2.5",
      "api_key_ref": "keyvault"
    },
    "deepseek": {
      "endpoint": "https://api.deepseek.com/v1/chat/completions",
      "model": "deepseek-v4-flash",
      "api_key_ref": "keyvault"
    }
  }
}
```

- **API key 存在密钥库**（`~/.openllm/vault/`，DPAPI 加密），不是明文。
- 切换模型：改 `default_provider`（mimo / deepseek / ollama 本地）。
- key 取用顺序：环境变量 → 密钥库 → config 明文。

---

## 四、六体架构

```
src/openllm/
├── iai/   感知（事件总线 + 因果预测 + 左右脑）
├── iax/   心跳（5阶段 PERCEIVE→DECIDE→EXECUTE→LEARN→FEEDBACK）
├── isa/   记忆（统一记忆 + Δ胶囊 + 因果记忆 + 温度遗忘）
├── ios/   治理（权限门 + 自我修改守卫 + 验证账本）
├── isn/   技能（工具执行 + 沙箱 + 技能生命周期）
└── iko/   输出（trace + 质量分 + 输出路由）
```

六体之间不直接互相 import，通过 HeartbeatContext 通信。

---

## 五、常见问题

**启动卡住不动？** 大概率是 embedding 模型联网检查——确保带了 `HF_HUB_OFFLINE=1`（.bat 已内置）。

**回复是 `[模拟LLM]`？** 说明 API key 没读到——检查 `~/.openllm/vault/` 里有没有 `mimo_api_key.dpapi`，或重装 editable install（`cd /mnt/i/openllm && .venv/bin/python -m pip install -e .`）。

**想换成本地模型？** 设 `default_provider: "ollama"`（需 Ollama 运行在 localhost:11434）。
