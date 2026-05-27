---
title: "Isa 设计原型 v1.0"
created: 2026-05-27
type: design
version: 1.0
tags: [fata, isa, prototype, architecture, websocket, openllm]
status: "设计阶段"
---

# Isa 设计原型 v1.0

> 不是又一个聊天 App。
> 是 openLLM 的肉身。

---

## 一、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户层                                │
│                                                             │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│    │ Web App  │  │ 微信小程序│  │ 移动 App │                │
│    │ (Day 1)  │  │ (Day 30) │  │ (Day 90) │                │
│    └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│         │             │             │                       │
│         └─────────────┼─────────────┘                       │
│                       │                                     │
│                  WebSocket                                  │
│                       │                                     │
├───────────────────────┼─────────────────────────────────────┤
│                       ▼                                     │
│    ┌─────────────────────────────────────────────┐          │
│    │              Isa 后端                        │          │
│    │                                             │          │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │          │
│    │  │ 连接管理  │  │ 消息路由  │  │ 分身管理  │  │          │
│    │  │ WebSocket │  │ Pub/Sub  │  │ Instance │  │          │
│    │  │ Manager   │  │ Router   │  │ Manager  │  │          │
│    │  └──────────┘  └──────────┘  └──────────┘  │          │
│    │                                             │          │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │          │
│    │  │ 用户管理  │  │ 群组管理  │  │ 外接桥接  │  │          │
│    │  │ Auth     │  │ Group    │  │ Bridge   │  │          │
│    │  │ Manager  │  │ Manager  │  │ Manager  │  │          │
│    │  └──────────┘  └──────────┘  └──────────┘  │          │
│    │                                             │          │
│    └─────────────────────┬───────────────────────┘          │
│                          │                                  │
├──────────────────────────┼──────────────────────────────────┤
│                          ▼                                  │
│    ┌─────────────────────────────────────────────┐          │
│    │              Δ总线                           │          │
│    │                                             │          │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │          │
│    │  │ 消息队列  │  │ 事件存储  │  │ 路由规则  │  │          │
│    │  │ Redis    │  │ SQLite   │  │ Config   │  │          │
│    │  │ Pub/Sub  │  │ WAL      │  │ YAML     │  │          │
│    │  └──────────┘  └──────────┘  └──────────┘  │          │
│    │                                             │          │
│    └─────────────────────┬───────────────────────┘          │
│                          │                                  │
├──────────────────────────┼──────────────────────────────────┤
│                          ▼                                  │
│    ┌──────────────┬──────┴──────┬──────────────┐            │
│    │              │             │              │            │
│    ▼              ▼             ▼              ▼            │
│ ┌──────┐    ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│ │openLLM│    │ Δ胶囊    │  │ Skills   │  │ 外接平台 │       │
│ │(决策) │    │ (记忆)   │  │ (技能)   │  │ 桥接     │       │
│ │       │    │          │  │          │  │          │       │
│ │身份   │    │事件记忆  │  │Skill引擎 │  │Telegram  │       │
│ │记忆   │    │知识记忆  │  │进化机制  │  │元宝      │       │
│ │技能   │    │身份记忆  │  │分享生态  │  │钉钉      │       │
│ │约束   │    │做梦机制  │  │          │  │飞书      │       │
│ └──────┘    └──────────┘  └──────────┘  └──────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、技术栈

### 前端（Web App）

| 组件 | 选型 | 理由 |
|------|------|------|
| **框架** | React 18 + TypeScript | 生态成熟，组件丰富 |
| **UI 库** | Tailwind CSS + shadcn/ui | 简洁、可定制、好看 |
| **状态管理** | Zustand | 轻量，比 Redux 简单 |
| **WebSocket** | 原生 WebSocket API | 无需额外依赖 |
| **构建工具** | Vite | 快，HMR 好 |

### 后端

| 组件 | 选型 | 理由 |
|------|------|------|
| **语言** | Python 3.11+ | 与 Hermes 生态一致 |
| **框架** | FastAPI | 异步、WebSocket 原生支持 |
| **WebSocket** | FastAPI WebSocket | 内置，无需额外库 |
| **消息队列** | Redis Pub/Sub | 轻量，已有基础设施 |
| **数据库** | SQLite + WAL | Δ总线统一存储 |
| **认证** | JWT + 自定义 | 简单、无状态 |

### 部署

| 组件 | 选型 | 理由 |
|------|------|------|
| **服务器** | 国内 VPS（阿里云/腾讯云） | 国内可用，低延迟 |
| **反向代理** | Nginx | 成熟、稳定 |
| **进程管理** | systemd | 与 Hermes 一致 |
| **域名** | 自定义域名 | 品牌感 |

---

## 三、核心模块设计

### 3.1 连接管理器（WebSocket Manager）

```python
class ConnectionManager:
    """管理所有 WebSocket 连接"""

    def __init__(self):
        self.connections: dict[str, WebSocket] = {}  # user_id -> ws
        self.groups: dict[str, set[str]] = {}        # group_id -> {user_ids}

    async def connect(self, user_id: str, ws: WebSocket):
        """用户连接"""
        await ws.accept()
        self.connections[user_id] = ws

    async def disconnect(self, user_id: str):
        """用户断开"""
        del self.connections[user_id]

    async def send_to_user(self, user_id: str, message: dict):
        """发送给单个用户"""
        if user_id in self.connections:
            await self.connections[user_id].send_json(message)

    async def send_to_group(self, group_id: str, message: dict):
        """发送给群组所有成员"""
        for user_id in self.groups.get(group_id, set()):
            await self.send_to_user(user_id, message)
```

### 3.2 消息路由器（Message Router）

```python
class MessageRouter:
    """消息路由：用户消息 → Δ总线 → openLLM → 响应"""

    def __init__(self, bus: DeltaBus):
        self.bus = bus

    async def route_inbound(self, message: dict):
        """路由入站消息"""
        # 1. 发布到 Δ总线
        await self.bus.publish("agent.inbound", message)

        # 2. openLLM 处理（异步）
        response = await self.process_with_openllm(message)

        # 3. 发布响应到 Δ总线
        await self.bus.publish("agent.outbound", response)

        # 4. 返回给用户
        return response

    async def process_with_openllm(self, message: dict) -> dict:
        """调用 openLLM 处理消息"""
        # 加载身份
        soul, iam = await self.bus.load_identity()

        # 检索相关记忆
        memories = await self.bus.recall(message["content"])

        # 加载相关技能
        skills = await self.bus.match_skills(message["content"])

        # 约束检查
        constraints = await self.bus.get_constraints()

        # 调用 LLM
        response = await llm.chat(
            message=message["content"],
            soul=soul,
            iam=iam,
            memories=memories,
            skills=skills,
            constraints=constraints
        )

        # 记录到 Δ胶囊
        await self.bus.observe(message, response)

        return response
```

### 3.3 分身管理器（Instance Manager）

```python
class InstanceManager:
    """管理 Agent 分身"""

    def __init__(self, bus: DeltaBus):
        self.bus = bus
        self.instances: dict[str, Instance] = {}

    async def create_instance(self, config: dict) -> str:
        """创建新分身"""
        instance_id = generate_uuid()
        self.instances[instance_id] = Instance(
            id=instance_id,
            soul=config["soul"],       # 共享 SOUL
            iam=config["iam"],         # 共享 Iam
            channel=config["channel"], # 所在频道
            memory=MemoryStore(),      # 独立记忆
        )
        return instance_id

    async def switch_instance(self, user_id: str, instance_id: str):
        """切换用户当前分身"""
        # 更新用户的当前分身
        await self.bus.update_user_instance(user_id, instance_id)

    async def sync_instances(self):
        """同步所有分身的身份（SOUL + Iam）"""
        soul, iam = await self.bus.load_identity()
        for instance in self.instances.values():
            instance.soul = soul
            instance.iam = iam
```

### 3.4 外接桥接器（Bridge Manager）

```python
class BridgeManager:
    """桥接外部平台"""

    def __init__(self, bus: DeltaBus):
        self.bus = bus
        self.bridges: dict[str, Bridge] = {}

    async def register_bridge(self, platform: str, config: dict):
        """注册外部平台桥接"""
        if platform == "telegram":
            self.bridges["telegram"] = TelegramBridge(config)
        elif platform == "feishu":
            self.bridges["feishu"] = FeishuBridge(config)
        # ...

    async def bridge_inbound(self, platform: str, message: dict):
        """外部消息进入 Δ总线"""
        # 转换为统一格式
        unified = self.bridges[platform].to_unified(message)
        # 发布到 Δ总线
        await self.bus.publish("agent.inbound", unified)

    async def bridge_outbound(self, platform: str, response: dict):
        """响应发送到外部平台"""
        # 转换为平台格式
        platform_msg = self.bridges[platform].from_unified(response)
        # 发送到平台
        await self.bridges[platform].send(platform_msg)
```

---

## 四、数据库设计

### 4.1 SQLite 表结构

```sql
-- 用户表
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSONB DEFAULT '{}'
);

-- 分身表
CREATE TABLE instances (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    channel TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 群组表
CREATE TABLE groups (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 群组成员表
CREATE TABLE group_members (
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT DEFAULT 'member',
    PRIMARY KEY (group_id, user_id),
    FOREIGN KEY (group_id) REFERENCES groups(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 消息表（WAL 模式）
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id)
);

-- 记忆表（Δ胶囊）
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    layer TEXT NOT NULL,  -- 'event' | 'knowledge' | 'identity'
    type TEXT NOT NULL,   -- 'observation' | 'action' | 'reflection' | 'dream'
    content JSONB NOT NULL,
    importance REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archived_at TIMESTAMP NULL
);

-- 技能表
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    version TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 五、API 设计

### 5.1 WebSocket 消息格式

```json
// 用户发送消息
{
    "type": "message",
    "instance_id": "xxx",
    "content": "你好",
    "timestamp": "2026-05-27T13:00:00Z"
}

// Agent 响应
{
    "type": "response",
    "instance_id": "xxx",
    "content": "你好！有什么可以帮你的？",
    "thinking": false,
    "timestamp": "2026-05-27T13:00:01Z"
}

// Agent 正在思考
{
    "type": "thinking",
    "instance_id": "xxx",
    "status": "searching memories..."
}

// 系统通知
{
    "type": "notification",
    "content": "新分身已创建",
    "level": "info"
}
```

### 5.2 REST API

```python
# 用户管理
POST   /api/auth/register     # 注册
POST   /api/auth/login        # 登录
GET    /api/users/me           # 获取当前用户

# 分身管理
GET    /api/instances          # 列出所有分身
POST   /api/instances          # 创建分身
GET    /api/instances/{id}     # 获取分身详情
DELETE /api/instances/{id}     # 删除分身

# 群组管理
GET    /api/groups             # 列出所有群组
POST   /api/groups             # 创建群组
POST   /api/groups/{id}/join   # 加入群组
POST   /api/groups/{id}/leave  # 离开群组

# 记忆查询
GET    /api/memories           # 查询记忆
GET    /api/memories/{id}      # 获取记忆详情

# 技能管理
GET    /api/skills             # 列出所有技能
POST   /api/skills             # 安装技能
GET    /api/skills/{id}        # 获取技能详情
```

---

## 六、前端界面设计

### 6.1 核心页面

```
┌─────────────────────────────────────────────────┐
│  Isa                                    ☰  👤  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────┐  ┌─────────────────────────────┐  │
│  │ 分身列表 │  │                             │  │
│  │         │  │   聊天区域                   │  │
│  │ ○ 军师  │  │                             │  │
│  │ ○ Flash │  │   [消息气泡]                │  │
│  │ ○ MiMo  │  │   [消息气泡]                │  │
│  │         │  │   [消息气泡]                │  │
│  │ ────── │  │                             │  │
│  │ 群组    │  │                             │  │
│  │ ○ 核战队│  │                             │  │
│  │ ○ 家庭  │  │   ┌─────────────────────┐  │  │
│  │         │  │   │ 输入消息...     🎤 ➤│  │  │
│  └─────────┘  │   └─────────────────────┘  │  │
│               └─────────────────────────────┘  │
│                                                 │
├─────────────────────────────────────────────────┤
│  状态: ● 在线 | 记忆: 363 facts | 技能: 24     │
└─────────────────────────────────────────────────┘
```

### 6.2 关键交互

| 操作 | 交互方式 |
|------|---------|
| 发送消息 | 输入框 + Enter/发送按钮 |
| 切换分身 | 左侧分身列表点击 |
| 创建分身 | 分身列表底部"+"按钮 |
| 查看状态 | 底部状态栏 |
| 语音输入 | 麦克风按钮 |
| 查看记忆 | 设置 → 记忆 |
| 管理技能 | 设置 → 技能 |

---

## 七、开发路线图

### Week 1：骨架

```
Day 1-2: 后端骨架
  ├── FastAPI + WebSocket 基础框架
  ├── SQLite 数据库初始化
  ├── 消息路由（Δ总线接口）
  └── 简单的 openLLM 调用（单轮对话）

Day 3-4: 前端骨架
  ├── React + Vite 项目初始化
  ├── WebSocket 连接
  ├── 基础聊天界面
  └── 消息收发

Day 5-7: 联调
  ├── 前后端联调
  ├── 单轮对话闭环
  └── 部署到 VPS
```

### Week 2：核心功能

```
Day 8-10: 分身管理
  ├── 创建/切换/删除分身
  ├── 分身独立记忆
  └── 分身共享身份

Day 11-12: 群组功能
  ├── 创建/加入群组
  ├── 群组消息广播
  └── @mention 支持

Day 13-14: 国内部署
  ├── VPS 配置（阿里云/腾讯云）
  ├── Nginx 反向代理
  ├── HTTPS 证书
  └── 域名配置
```

### Week 3：增强

```
Day 15-17: 记忆系统
  ├── Δ胶囊三层记忆
  ├── 做梦机制（consolidation）
  └── 记忆查询界面

Day 18-19: 技能系统
  ├── Skill 加载/执行
  ├── Skill 安装/分享
  └── 技能管理界面

Day 20-21: 外接桥接
  ├── Telegram 桥接
  ├── 飞书桥接
  └── 统一消息格式
```

### Week 4：打磨

```
Day 22-24: UI/UX
  ├── 界面美化（Tailwind + shadcn/ui）
  ├── 响应式设计
  └── 动画/过渡效果

Day 25-26: 稳定性
  ├── 错误处理
  ├── 重连机制
  └── 日志系统

Day 27-28: 测试
  ├── 单元测试
  ├── 集成测试
  └── 用户测试
```

---

## 八、与现有系统的关系

| 组件 | Hermes Agent 现有 | Isa 新增 | 关系 |
|------|-------------------|---------|------|
| **消息路由** | Gateway (config.yaml) | Isa 路由器 | Isa 替代 Gateway 的路由功能 |
| **分身管理** | 无 | InstanceManager | Isa 新增 |
| **记忆系统** | memory + fact_store | Δ胶囊三层 | Isa 接入 Δ胶囊 |
| **技能系统** | skills | SkillEngine | Isa 接入 Skills |
| **外接平台** | Gateway 平台适配器 | BridgeManager | Isa 通过桥接复用 |
| **认证** | GATEWAY_ALLOW_ALL_USERS | JWT + 用户系统 | Isa 新增 |

**关键设计决策**：Isa 不是重写 Hermes。Isa 是 Hermes 的"前端"——它提供用户界面和分身管理，底层仍然调用 Hermes 的 Skills、记忆、约束系统。

---

## 九、哲学锚点

> iPhone 不是发明了最好的手机。
> 它发明了最像人的东西——你触摸它，它回应。
>
> Isa 不是发明了最好的聊天 App。
> 它要成为最像伙伴的东西——你想到它，它就在。

---

*Isa 设计原型 v1.0 — 五根柱子之第五根：身体*
*从图纸到代码，等待开干。*
