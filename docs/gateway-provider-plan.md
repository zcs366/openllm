# openLLM 网关接入方案

> 调研时间：2026-08-30 | 调研者：mimo subagent
> 状态：**方案完成，待实施**

---

## 一、现状调研

### 1.1 ProviderType 枚举（provider.py L33-39）

```python
class ProviderType(Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"
    OLLAMA = "ollama"
```

注意：factory 中还有 `mimo` 和 `qwen` 两个 provider_type，但**未加入 ProviderType 枚举**——它们是硬编码在 `create_provider()` 工厂函数里的隐式成员。

### 1.2 Provider 实现类（provider.py）

| 类 | 行号 | 协议 | 用途 |
|---|---|---|---|
| `DeepSeekProvider` | L84-234 | OpenAI 兼容（/v1/chat/completions） | 默认 provider，也被 ollama/mimo/qwen/openai 复用 |
| `AnthropicProvider` | L239-330 | Anthropic 原生（/v1/messages） | Claude 系列 |
| `GeminiProvider` | L335-431 | Google 原生（:generateContent） | Gemini 系列 |

**关键发现：** `DeepSeekProvider` 本质上是**通用 OpenAI 兼容客户端**——endpoint 可配、auth header 是 `Bearer token`、payload/response 都是 OpenAI 格式。`ollama`、`mimo`、`qwen`、`openai` 四种 provider_type 全都复用它。

### 1.3 provider_impl.py（75行）——独立的简化封装

`LLMProvider` 类，从 `~/.openllm/config.json` 读 `default_provider` 配置，提供更简单的 `chat(messages)` 接口。**这是一个独立的 provider 封装**，与 `provider.py` 的 `create_provider()` 体系并行。用于一些不需要完整 Agent 引擎的轻量场景。

### 1.4 模型调用入口全集

| 文件 | 函数/方法 | 行号 | 调用链 |
|---|---|---|---|
| `provider.py` | `DeepSeekProvider.chat()` | L95 | 核心调用，同步+流式 |
| `provider.py` | `create_provider()` | L436 | 工厂，创建 provider 实例 |
| `engine.py` | `OpenLLMEngine._init_provider()` | L238 | 从 config+env 初始化 provider |
| `engine.py` | `OpenLLMEngine.oneshot()` | L322 | 轻量级调用（委托 ones.py） |
| `oneshot.py` | `oneshot()` | L23 | 直接调 `create_provider` + `chat()` |
| `provider_impl.py` | `LLMProvider.chat()` | L35 | 独立封装，从 config.json 读配置 |
| `memory_evaluator.py` | 直接调 `create_provider` | L293 | 记忆评估用 mimo |

### 1.5 API Key 获取机制

**三条路径并存：**

1. **env 变量**（provider.py L55-62）：`ModelConfig.__post_init__` 读 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`
2. **config.json**（engine.py L244-268）：`~/.openllm/config.json` → `providers.{name}.api_key`，用于 ollama/mimo/qwen
3. **engine_utils.py**（L28-44）：`get_api_key(provider)` 根据 provider 名选择 env 或 config.json

**网关 key 来源：** `~/one-api/.gateway_key`（文件权限 600），内容为 `sk-wKk...he__`。

### 1.6 凭据防火墙白名单（credential_firewall.py L49-56）

```python
DEFAULT_WHITELIST = {
    'api.deepseek.com', 'api.openai.com', 'api.anthropic.com',
    'generativelanguage.googleapis.com', 'localhost', '127.0.0.1',
}
```

`127.0.0.1` **已在白名单中**——网关地址 `http://127.0.0.1:13000` 无需额外配置。

### 1.7 已有 config.json 配置

```json
{
  "default_provider": "mimo",
  "providers": {
    "deepseek": { "endpoint": "https://api.deepseek.com/v1/chat/completions", ... },
    "mimo": { "endpoint": "https://token-plan-cn.xiaomimimo.com/v1/chat/completions", ... },
    "lmstudio": { "endpoint": "http://localhost:11234/v1/chat/completions", ... },
    "ollama": { "endpoint": "http://localhost:11434/v1/chat/completions", ... }
  }
}
```

---

## 二、方案选择：新增 gateway 还是复用 DeepSeekProvider

### 结论：**复用 DeepSeekProvider + 新增 gateway provider_type**

理由：
1. `DeepSeekProvider` 已经是通用 OpenAI 兼容客户端——endpoint 可配、协议完全一致
2. 不需要新类，只需要在工厂函数里加一个分支
3. 最小改动量：~20 行代码 + config.json 一个条目

### 不选的方案

| 方案 | 否决原因 |
|---|---|
| 新建 OpenAICompatProvider 类 | 和 DeepSeekProvider 重复，增加维护成本 |
| 用现有 openai provider_type | openai 的默认 endpoint 指向 api.openai.com，需要改代码逻辑 |
| 只改 config.json 加 endpoint | 没有处理 key 来源（文件 vs env），也没有注册到 ProviderType |

---

## 三、改动清单

### 3.1 provider.py —— 枚举 + 工厂

**改动 A：ProviderType 枚举加 GATEWAY（+1行）**

```python
class ProviderType(Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"
    OLLAMA = "ollama"
    GATEWAY = "gateway"          # ← 新增
```

**改动 B：create_provider 工厂加 gateway 分支（+15行）**

在 `ollama` 分支之后、`mimo` 分支之前插入：

```python
if provider_type == "gateway":
    config.endpoint = kwargs.get("endpoint", "http://127.0.0.1:13000/v1/chat/completions")
    # key 从文件读取（优先 kwargs > 文件 > env）
    if not config.api_key:
        gateway_key_path = Path.home() / "one-api" / ".gateway_key"
        if gateway_key_path.exists():
            config.api_key = gateway_key_path.read_text().strip()
    if not config.api_key:
        config.api_key = os.environ.get("OPENLLM_GATEWAY_KEY", "")
    if not config.api_key:
        raise ValueError("gateway需要api_key参数或 ~/one-api/.gateway_key 文件")
    config.model = kwargs.get("model", "gemini-3.6-flash")
    logger.info(f"gateway provider: {config.model} @ {config.endpoint}")
    return DeepSeekProvider(config)  # 复用 OpenAI 兼容客户端
```

### 3.2 engine.py —— _init_provider 支持 gateway

**改动 C：_init_provider 加 gateway 分支（+12行）**

在 `elif p == "mimo":` 之前插入：

```python
elif p == "gateway":
    gateway_key_path = Path.home() / "one-api" / ".gateway_key"
    api_key = gateway_key_path.read_text().strip() if gateway_key_path.exists() else ""
    import json as _json
    cfg_path = Path.home() / ".openllm" / "config.json"
    if cfg_path.exists():
        cfg = _json.loads(cfg_path.read_text())
        gw = cfg.get("providers", {}).get("gateway", {})
        if not api_key:
            api_key = gw.get("api_key", "")
        endpoint = gw.get("endpoint", "")
        cfg_model = gw.get("model", "")
        if cfg_model:
            self.config.model = cfg_model
    if not endpoint:
        endpoint = "http://127.0.0.1:13000/v1/chat/completions"
```

### 3.3 engine_utils.py —— get_api_key 支持 gateway

**改动 D：get_api_key 加 gateway 分支（+5行）**

```python
def get_api_key(provider: str) -> str:
    if provider == "ollama":
        return "ollama"
    elif provider == "gateway":
        gateway_key_path = Path.home() / "one-api" / ".gateway_key"
        if gateway_key_path.exists():
            return gateway_key_path.read_text().strip()
        return os.environ.get("OPENLLM_GATEWAY_KEY", "")
    elif provider == "anthropic":
        ...
```

**改动 E：get_endpoint 支持 gateway（+3行）**

```python
def get_endpoint(provider: str) -> str:
    if provider in ("ollama", "mimo", "qwen"):
        ...
    elif provider == "gateway":
        return "http://127.0.0.1:13000/v1/chat/completions"
    return ""
```

### 3.4 config.json —— 新增 gateway provider

```json
"gateway": {
    "endpoint": "http://127.0.0.1:13000/v1/chat/completions",
    "model": "gemini-3.6-flash",
    "api_key": ""
}
```

key 不放 config.json（明文），运行时从 `~/one-api/.gateway_key` 读取。

### 3.5 provider_impl.py —— LLMProvider 支持 gateway

**改动 F：_load_config 增加 gateway key 文件读取（可选，+5行）**

当前 `LLMProvider` 从 config.json 读 api_key。如果 config.json 里 gateway 的 api_key 为空，应 fallback 读文件：

```python
# 在 _load_config 之后
if not self.api_key and self.endpoint and "127.0.0.1" in self.endpoint:
    gw_key = Path.home() / "one-api" / ".gateway_key"
    if gw_key.exists():
        self.api_key = gw_key.read_text().strip()
```

---

## 四、代码骨架（完整可插入片段）

### 4.1 provider.py — gateway 分支完整代码

```python
# 在 create_provider() 中，ollama 分支之后插入：

if provider_type == "gateway":
    config.endpoint = kwargs.get("endpoint", "http://127.0.0.1:13000/v1/chat/completions")
    # 三级 key 获取：kwargs > 文件 > env
    if not config.api_key:
        gw_key_path = Path.home() / "one-api" / ".gateway_key"
        if gw_key_path.exists():
            config.api_key = gw_key_path.read_text().strip()
    if not config.api_key:
        config.api_key = os.environ.get("OPENLLM_GATEWAY_KEY", "")
    if not config.api_key:
        raise ValueError("gateway需要api_key或~/one-api/.gateway_key文件")
    config.model = kwargs.get("model", "gemini-3.6-flash")
    logger.info(f"gateway provider: {config.model} @ {config.endpoint}")
    return DeepSeekProvider(config)
```

### 4.2 使用方式

```python
from openllm.core.provider import create_provider

# 方式1：指定 provider="gateway"
gw = create_provider("gateway", model="gemini-3.6-flash")

# 方式2：通过 config.json default_provider 切换
# config.json: "default_provider": "gateway"

# 方式3：oneshot 快速调用
from openllm.core.oneshot import oneshot
result = oneshot("hello", provider="gateway", model="gemini-3.6-flash")
```

---

## 五、风险与验证

### 5.1 风险清单

| 风险 | 等级 | 应对 |
|---|---|---|
| gateway key 文件权限泄露 | 中 | 保持 600，运行时读取，不写入 config.json |
| one-api 网关宕机导致 agent 不可用 | 中 | 网关是 fallback，不影响 primary provider |
| 模型名不匹配（网关注册的名 vs 代码传的名） | 低 | config.json 可配 model，默认 gemini-3.6-flash |
| 与现有 openai provider_type 混淆 | 低 | gateway 指向本地，openai 指向 api.openai.com |
| credential_firewall 误拦截 | 无 | 127.0.0.1 已在白名单 |

### 5.2 验证步骤

```bash
# 1. 确认网关存活
curl -s http://127.0.0.1:13000/v1/models | python3 -m json.tool

# 2. 用 gateway key 测试
GW_KEY=$(cat ~/one-api/.gateway_key)
curl -s http://127.0.0.1:13000/v1/chat/completions \
  -H "Authorization: Bearer $GW_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.6-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'

# 3. Python 代码验证
python3 -c "
from openllm.core.provider import create_provider
gw = create_provider('gateway', model='gemini-3.6-flash')
from openllm.core.provider import ChatMessage
resp = gw.chat([ChatMessage(role='user', content='say hello in 5 words')])
print(resp.content, f'({resp.latency_ms:.0f}ms)')
"

# 4. oneshot 验证
python3 -c "
from openllm.core.oneshot import oneshot
r = oneshot('说你好', provider='gateway', model='gemini-3.6-flash')
print(r)
"
```

### 5.3 回滚方案

改动全部集中在 4 个文件的 if 分支新增，回滚 = 删除 gateway 分支即可。config.json 新增的 gateway 条目可保留不影响其他 provider。

---

## 六、改动汇总

| 文件 | 改动行数 | 类型 |
|---|---|---|
| `src/openllm/core/provider.py` | +16 行 | 枚举 + 工厂分支 |
| `src/openllm/core/engine.py` | +12 行 | _init_provider 分支 |
| `src/openllm/core/engine_utils.py` | +8 行 | get_api_key + get_endpoint |
| `src/openllm/core/provider_impl.py` | +5 行 | LLMProvider fallback |
| `~/.openllm/config.json` | +4 行 | gateway provider 条目 |
| **总计** | **~45 行** | 纯增量，无删除 |

---

## 附录：完整调用链路图

```
用户/Agent
    │
    ├─ OpenLLMEngine._init_provider()     ← engine.py L238
    │   ├─ 读 config.json / env
    │   └─ create_provider("gateway", ...) ← provider.py L436
    │       ├─ 读 ~/one-api/.gateway_key
    │       └─ return DeepSeekProvider(config)  ← 复用 OpenAI 兼容客户端
    │
    ├─ OpenLLMEngine.run_once() / chat()
    │   └─ self.provider.chat(messages)    ← provider.py L95
    │       ├─ POST http://127.0.0.1:13000/v1/chat/completions
    │       ├─ Authorization: Bearer <gateway_key>
    │       └─ 解析 OpenAI 格式响应
    │
    └─ oneshot() / classify() / extract() ← oneshot.py L23
        └─ create_provider("gateway", ...)
            └─ DeepSeekProvider.chat()
```
