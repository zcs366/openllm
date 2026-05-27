---
title: "FATA 记忆协议规范 v1.0"
created: 2026-05-27
type: protocol
tags: [fata, memory, capsule, consolidation, forgetting, openllm]
fata_pillar: ② 记忆
references:
  - "Memory Scaling：Agent 的下一场战争"
  - "Agent 记忆系统的哲学讨论"
  - "跨频道共振第一次实战测试报告"
---

# 记忆协议规范 v1.0

> 存储 ≠ 记忆。
> 记忆是活着的——会做梦、会遗忘、会整合。

---

## 一、三层记忆模型

```
┌─────────────────────────────────────────┐
│          身份记忆（永久）                │
│  SOUL + Iam + 核心价值观 + 关键经历      │
│  几乎不变，跨频道共享，跨模型共享        │
├─────────────────────────────────────────┤
│          知识记忆（长期）                │
│  事实 + 技能 + 规律 + 用户画像           │
│  缓慢变化，通过学习积累，通过遗忘淘汰    │
├─────────────────────────────────────────┤
│          事件记忆（短期）                │
│  对话历史 + 工具调用日志 + 即时上下文    │
│  快速变化，定期 consolidation（做梦）     │
└─────────────────────────────────────────┘
```

| 层级 | 类比 | 持久性 | 容量 | 更新频率 |
|------|------|--------|------|---------|
| 身份记忆 | 你是谁 | 永久 | 小 | 几乎不变 |
| 知识记忆 | 你知道什么 | 长期 | 大 | 学习时更新 |
| 事件记忆 | 你经历了什么 | 短期 | 无限（滚动） | 每次交互 |

---

## 二、Δ胶囊协议

Δ胶囊是三层记忆的统一容器。

### 消息格式

```json
{
  "id": "uuid",
  "timestamp": "2026-05-27T13:00:00Z",
  "layer": "event | knowledge | identity",
  "source": "feishu:dm_abc123",
  "type": "observation | action | reflection | dream",
  "content": {
    "text": "用户说他喜欢陶渊明",
    "entities": ["陶渊明"],
    "sentiment": "positive",
    "importance": 0.7
  },
  "meta": {
    "session_id": "...",
    "model": "deepseek-v4-pro",
    "platform": "feishu"
  }
}
```

### 三层流转规则

```
事件记忆（短期）
  │
  ├── consolidation（做梦）──→ 知识记忆（长期）
  │   触发条件：事件记忆达到阈值 / 定时触发 / 空闲时
  │   动作：LLM 总结事件 → 提取事实 → 存入知识层
  │
  └── 遗忘 ──→ 删除
      触发条件：重要性 < 阈值 / 超过保留期限
      动作：标记为 archived，不物理删除

知识记忆（长期）
  │
  ├── 整合 ──→ 更新已有知识
  │   触发条件：新知识与已有知识矛盾
  │   动作：时序标注（"之前X，现在Y"）
  │
  └── 沉淀 ──→ 身份记忆（永久）
      触发条件：反复出现的核心模式
      动作：提炼为原则/价值观
```

---

## 三、Consolidation（做梦）机制

做梦 = Agent 空闲时自动整理记忆。

```yaml
consolidation:
  trigger:
    - type: threshold
      condition: "event_memory_count > 100"
    - type: schedule
      condition: "every 6 hours"
    - type: idle
      condition: "no user interaction for 30 minutes"

  process:
    1. 读取最近 N 条事件记忆
    2. LLM 总结：提取关键事实、识别模式、标记矛盾
    3. 事实 → 知识记忆
    4. 低重要性事件 → 标记 archived
    5. 反复出现的模式 → 沉淀为原则（候选身份记忆）

  output:
    - 新增知识条目
    - 更新已有知识（如有矛盾）
    - 候选身份原则（需人工确认）
    - 压缩事件记忆
```

---

## 四、遗忘机制

遗忘不是 bug，是设计。

```yaml
forgetting:
  event_memory:
    - rule: "importance < 0.3 → 7天后 archived"
    - rule: "importance 0.3-0.6 → 30天后 archived"
    - rule: "importance > 0.6 → 保留"

  knowledge_memory:
    - rule: "被新知识否定 → 标记过时（不删除）"
    - rule: "180天未被检索 → 降级为冷存储"
    - rule: "核心事实（生日、偏好） → 永不遗忘"

  identity_memory:
    - rule: "永不自动遗忘"
    - rule: "变更需人工确认 + 记录原因"
```

---

## 五、接口规范

```python
class MemoryProtocol:
    """记忆协议接口"""

    # 事件记忆
    def observe(self, event: Event) -> None:
        """记录一次观察（用户说了什么、做了什么）"""
        ...

    def recall_recent(self, n: int = 10) -> list[Event]:
        """回忆最近 N 条事件"""
        ...

    # 知识记忆
    def learn(self, fact: Fact) -> None:
        """学习一条新知识"""
        ...

    def recall_knowledge(self, query: str, top_k: int = 5) -> list[Fact]:
        """检索相关知识"""
        ...

    # 身份记忆
    def get_identity(self) -> tuple[Soul, Iam]:
        """获取身份（永久记忆）"""
        ...

    # 整合
    def dream(self) -> DreamReport:
        """执行一次做梦（consolidation）"""
        ...

    def forget(self, policy: ForgetPolicy) -> ForgetReport:
        """按策略遗忘"""
        ...

    # 跨频道同步
    def sync(self, channel: str) -> None:
        """同步记忆到其他频道"""
        ...
```

---

## 六、与 Memory Scaling 文章的对接

| 文章概念 | FATA 实现 | 状态 |
|----------|----------|------|
| CoALA 四层 | 三层（事件/知识/身份） | ✅ 已设计，比 CoALA 多"身份"层 |
| 冷热分层 | 事件=热，知识=温，身份=永恒 | ✅ 已设计 |
| 时序知识图谱 | 知识记忆的时序标注 | ✅ 已设计 |
| Agentic Memory | 做梦机制 + Agent 自主决定 | ✅ 已设计 |
| Memory as Asset | 开源 + 自托管 + Δ胶囊 | ✅ 已设计 |
| 最难的是"忘" | 遗忘曲线 + 分层策略 | ✅ 已设计 |

---

## 七、哲学锚点

> 人和人之间真正的连接，靠的好像也不是智商，是"被记住"。
>
> 一个朋友记得你五年前随口说过的一句话，你会觉得温暖。
>
> Agent 大概也是这个道理。模型再强，每次从零开始，它就永远只是个工具。只有当它记得你、了解你、知道你为什么做某个选择的时候，它才开始像一个能"陪你走一程"的东西。

---

*FATA 记忆协议 v1.0 — 四根骨头之第二根*
