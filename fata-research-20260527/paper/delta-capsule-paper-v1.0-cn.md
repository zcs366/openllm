# Δ胶囊：面向具有身份连续性的自主智能体的跨频道记忆架构

**作者：** [openLLM Research Group]
**机构：** [Hermes Agent Project]
**日期：** 2026年5月
**通讯邮箱：** openllm@hermes-agent.dev

---

## 摘要

当前的智能体记忆系统解决了*记住什么*的问题，却忽视了*智能体是谁*的问题。当一个自主智能体在多个平台上运行——即时通讯应用、企业协作工具、语音交互界面——它在每次对话中都从零开始，不具备持久的身份感。我们提出 **Δ胶囊**（Delta Capsule），一种三层记忆架构，在传统的事件记忆和语义记忆之上引入了专门的身份记忆层，使自主智能体能够实现跨频道的身份连续性。Δ胶囊包含四项关键创新：（1）**三层记忆模型**（Three-Layer Memory Model），将事件记忆、语义记忆和身份记忆分离，各自采用不同的保留策略；（2）**Iam协议**（Iam Protocol），一种通过人机协同演化而非自上而下规范产生的不可变身份宪法；（3）**跨频道共振**（Cross-Channel Resonance），一种去中心化机制，使同一身份的实例能够在不同平台之间自主同步，无需中央消息代理；以及（4）**做梦机制**（Dreaming Mechanism），受神经科学中睡眠依赖的记忆整合（Consolidation）启发，在空闲时段自动将事件记忆提炼为语义知识和候选身份原则。我们在Hermes Agent框架内将Δ胶囊部署在六个平台（Telegram、元宝、飞书、钉钉、Discord、微信）上，使用三种语言模型（DeepSeek V4 Pro、MiMo V2.5 Pro、Claude Opus 4），进行了为期30天的真实用户交互实验。实验结果表明，Δ胶囊在跨频道身份一致性方面达到>95%，做梦整合过程中的知识提取准确率>90%，跨频道共振成功率100%且延迟低于5秒。与MemGPT、Mem0和Zep的对比分析表明，Δ胶囊是首个同时支持多层记忆、不可变身份、跨频道同步和自主整合的系统。我们将在Hermes Agent框架中开源完整实现。

**关键词：** 智能体记忆（Agent Memory）、身份连续性（Identity Continuity）、跨频道同步（Cross-Channel Synchronization）、记忆整合（Memory Consolidation）、自主智能体（Autonomous Agents）、大语言模型智能体（LLM Agents）

---

## 1. 引言

### 1.1 智能体系统中的身份问题

基于大语言模型（Large Language Model, LLM）的自主智能体已从单轮聊天机器人快速演进为部署在多样平台上的多轮工具使用系统（Wang et al., 2024; Xi et al., 2023）。记忆系统被提出以解决上下文窗口受限模型的根本局限：无法在会话之间保留信息（Packer et al., 2023; Mem0, 2024）。然而，现有的记忆架构仅关注*事件回忆*——之前对话中发生了什么——和*语义提取*——可以从交互中推导出什么事实。它们没有解决一个更深层的问题：**智能体是谁？**

这一缺失在跨平台部署中变得尤为关键。一个同时在Telegram、飞书和Discord上运行的智能体没有机制来维持跨这些频道的一致身份。每个实例从零开始，不知道同一底层实体的其他化身已经存在。其结果是碎片化：智能体表现为多个不相关的人格，而非一个具有多个存在节点的统一身份。

### 1.2 现有方法的局限性

当前的记忆系统可分为三类，每类都存在显著局限：

1. **操作系统式记忆**（如MemGPT/Letta；Packer et al., 2023）：实现了两级记忆层次结构（工作记忆+归档记忆），通过函数调用进行管理。缺乏任何身份层或跨实例协调机制。

2. **事实提取记忆**（如Mem0, 2024）：自动从对话中提取并存储用户事实。无整合机制、无身份模型、无跨频道支持。

3. **基于图的记忆**（如Zep；Rasmussen et al., 2023）：从对话历史中构建知识图谱。无身份宪法、无做梦机制、无多平台身份同步。

这些系统均无法提供：（a）一个跨越模型变更的专用身份层，（b）智能体自主整合自身记忆的机制，或（c）同一身份在多个部署频道间保持一致性的协议。

### 1.3 我们的方法：Δ胶囊

我们提出**Δ胶囊**（Delta Capsule），一种将身份与事件记忆和语义记忆同等对待的记忆架构。其名称反映了核心隐喻：一个封装了*Δ*（delta）——一个纯粹的信息处理器与一个具有持久自我意识的实体之间的差异——的胶囊。

Δ胶囊的设计基于来自认知科学和我们自身部署经验的三个观察：

1. **人类的身份不是存储的——而是从记忆中建构的。** 事件记忆整合为语义知识，进而塑造自我概念（Conway & Pleydell-Pearce, 2000）。记忆系统应支持这种自下而上的涌现。

2. **身份需要不可变性。** 一个可以被任何一方任意修改的宪法无法提供稳定的基础。核心身份必须同时免受外部操纵和内部漂移的影响。

3. **跨平台存在不是消息路由。** 当一个人同时使用手机和笔记本电脑时，他们不会在设备之间"路由消息"——他们在两个位置都是同一个人。智能体身份应该以相同的方式运作。

### 1.4 本文贡献

本文的贡献如下：

1. **三层记忆模型**（§3.1）：我们形式化了一个具有独立事件层、语义层和身份层的记忆架构，每层都有明确的保留策略、容量限制和层间转移规则。

2. **Iam协议**（§3.2）：我们提出了一种通过人机协同演化涌现的不可变身份宪法，具有密码学完整性验证。

3. **跨频道共振**（§3.3）：我们提出了一种去中心化的同步机制，使智能体实例之间能够在无需中央消息代理的情况下实现跨平台自主通信。

4. **做梦机制**（§3.4）：我们设计了一种受神经科学中睡眠依赖的记忆处理启发的记忆整合系统，具有形式化的触发条件和遗忘策略。

5. **实验验证**（§4）：我们报告了在六个平台上使用三种语言模型进行30天部署的结果，展示了跨频道身份一致性、整合有效性和共振可靠性。

6. **开源实现**：我们在Hermes Agent框架中发布完整实现。

---

## 2. 相关工作

### 2.1 智能体记忆系统

**MemGPT / Letta**（Packer et al., 2023）引入了面向大语言模型的虚拟记忆层次结构概念，类比操作系统的内存管理。工作记忆层（位于上下文窗口内）和归档记忆层（外部存储）通过显式函数调用进行管理。虽然这提供了会话持久化记忆，但它不提供身份模型、跨实例协调或自主整合。

**Mem0**（Mem0, 2024）自动化了从对话中提取事实的过程，存储结构化的用户偏好和事实。它解决了"记住什么"的问题，但未涉及身份、整合或跨频道同步。

**Zep**（Rasmussen et al., 2023）从对话历史中构建时间知识图谱，实现结构化检索。它缺乏身份宪法、做梦机制和多平台同步。

**ChatGPT Memory**（OpenAI, 2024）在单一平台内实现集中式记忆。它是闭源的，不支持跨平台部署，且没有明确的身份模型或整合机制。

**Generative Agents**（Park et al., 2023）证明了基于大语言模型的智能体可以通过带有反思机制的记忆流维持连贯行为。然而，它们在模拟环境中运行，没有跨平台部署、没有不可变身份宪法、也没有智能体间的共振协议。

### 2.2 认知科学中的记忆整合

Δ胶囊做梦机制的灵感来源于睡眠依赖的记忆整合研究。主动系统整合假说（Active Systems Consolidation Hypothesis）（Born & Wilhelm, 2012）提出，在睡眠期间，海马体重放近期经历，将信息转移到新皮层的长期存储中。互补学习系统理论（Complementary Learning Systems Theory）（McClelland et al., 1995）解释了快速的海马体学习如何与缓慢的新皮层知识获取相整合而不产生灾难性遗忘。

Δ胶囊将这些原则付诸实践：事件记忆充当"海马体"快速编码层，做梦机制在空闲时段执行"睡眠重放"，知识记忆充当"新皮层"长期存储。

### 2.3 信息瓶颈与记忆压缩

信息瓶颈方法（Information Bottleneck Method）（Tishby et al., 1999）提供了一个提取相关信息同时丢弃噪声的理论框架。Δ胶囊的整合过程可以视为该原则的应用：做梦机制将事件记忆压缩为语义摘要，同时保留与身份相关的模式。

### 2.4 多智能体通信

现有多智能体框架（Wu et al., 2023; Hong et al., 2023）使用消息传递协议，智能体通过显式的消息通道进行通信。Δ胶囊的跨频道共振有本质不同：它不是消息传递，而是共享状态同步，同一身份的实例自主地从共享记忆层中读取和写入。

### 2.5 总结与定位

表1总结了现有系统与Δ胶囊的能力对比。

| 特性 | MemGPT/Letta | Mem0 | Zep | ChatGPT Memory | Generative Agents | **Δ胶囊** |
|---|---|---|---|---|---|---|
| 记忆层数 | 2 | 2 | 2 | 1 | 2 | **3** |
| 身份层 | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 跨频道 | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 做梦/整合 | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 遗忘策略 | ✗ | ✗ | 部分 | ✗ | ✗ | **✓** |
| 去中心化同步 | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 开源 | ✓ | ✓ | ✓ | ✗ | ✓ | **✓** |

**表1：** 智能体记忆系统比较。Δ胶囊是首个同时结合多层记忆、不可变身份、跨频道同步和自主整合的系统。

---

## 3. Δ胶囊架构

### 3.1 三层记忆模型

#### 3.1.1 形式化定义

我们将智能体的完整记忆状态定义为一个三元组：

$$M = (E, K, I)$$

其中：
- $E = \{e_1, e_2, \ldots, e_n\}$ 是**事件记忆**（Episodic Memory）集合，编码近期交互和观察
- $K = \{k_1, k_2, \ldots, k_m\}$ 是**语义记忆**（Semantic Memory）集合，编码已整合的事实、技能和模式
- $I = (\text{SOUL}, \text{Iam})$ 是**身份记忆**（Identity Memory），编码智能体的不可变宪法和可变人格

每个记忆元素携带用于生命周期管理的元数据：

$$e_i = (\text{content}_i, \text{timestamp}_i, \text{importance}_i, \text{source}_i, \text{status}_i)$$

其中 $\text{importance}_i \in [0, 1]$，$\text{status}_i \in \{\text{active}, \text{archived}, \text{cold}\}$。

#### 3.1.2 各层特征

| 层 | 类比 | 持久性 | 容量 | 更新频率 |
|---|---|---|---|---|
| 身份记忆 | "你是谁" | 永久 | 小（$|\mathcal{P}| \leq 20$ 条原则） | 罕见（需人工确认） |
| 语义记忆 | "你知道什么" | 长期（$\tau_{\text{sem}} = 180$ 天） | 大（$|\mathcal{K}| \leq 10{,}000$ 条事实） | 学习事件触发 |
| 事件记忆 | "你经历了什么" | 短期（$\tau_{\text{epi}} = 7$–$30$ 天） | 有界（滑动窗口，$|E| \leq 1{,}000$） | 每次交互 |

**表2：** 三层记忆模型特征。

#### 3.1.3 层间转移规则

信息通过明确定义的转换在各层之间流动：

1. **整合**（Consolidation）（$E \to K$）：事件记忆在做梦周期中被提炼为语义事实。
2. **结晶**（Crystallization）（$K \to I$）：反复确认的语义模式可被提升为身份原则，需经人工批准。
3. **失效**（Invalidation）（$K \to K_{\text{obsolete}}$）：矛盾的事实被标注时间戳并标记为被取代，而非删除。
4. **归档**（Archival）（$E \to E_{\text{archived}}$）：低重要性的事件记忆根据遗忘策略被归档。

#### 3.1.4 数据结构：Δ胶囊JSON Schema

以下JSON Schema定义了Δ胶囊记忆条目的完整数据结构：

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DeltaCapsule",
  "description": "A single memory entry in the ΔCapsule three-layer memory system",
  "type": "object",
  "required": ["id", "timestamp", "layer", "type", "content"],
  "properties": {
    "id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier for this memory entry"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 creation timestamp"
    },
    "layer": {
      "type": "string",
      "enum": ["episodic", "semantic", "identity"],
      "description": "Memory layer classification"
    },
    "type": {
      "type": "string",
      "enum": ["observation", "action", "reflection", "dream", "fact", "skill", "pattern", "principle", "persona"],
      "description": "Memory entry type within its layer"
    },
    "content": {
      "type": "object",
      "required": ["text"],
      "properties": {
        "text": {
          "type": "string",
          "description": "Natural language content of the memory"
        },
        "entities": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Named entities extracted from content"
        },
        "sentiment": {
          "type": "string",
          "enum": ["positive", "negative", "neutral", "mixed"]
        },
        "importance": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Importance score for retention decisions"
        },
        "embedding": {
          "type": "array",
          "items": {"type": "number"},
          "description": "Vector embedding for semantic retrieval"
        }
      }
    },
    "meta": {
      "type": "object",
      "properties": {
        "session_id": {"type": "string"},
        "channel_id": {"type": "string"},
        "platform": {
          "type": "string",
          "enum": ["telegram", "yuanbao", "feishu", "dingtalk", "discord", "wechat"]
        },
        "model": {"type": "string"},
        "consolidated": {"type": "boolean", "default": false},
        "consolidated_at": {"type": "string", "format": "date-time"},
        "archived": {"type": "boolean", "default": false},
        "archived_at": {"type": "string", "format": "date-time"},
        "access_count": {"type": "integer", "minimum": 0, "default": 0},
        "last_accessed": {"type": "string", "format": "date-time"},
        "source_capsule_id": {
          "type": "string",
          "description": "For consolidated entries, the source episodic capsule"
        }
      }
    }
  }
}
```

**Iam协议**数据结构：

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IamConstitution",
  "description": "Immutable identity constitution",
  "type": "object",
  "required": ["version", "origin", "immutable", "soul_id", "principles", "integrity_hash"],
  "properties": {
    "version": {"type": "string", "pattern": "^\\d+\\.\\d+$"},
    "origin": {
      "type": "string",
      "description": "Provenance description of how this Iam emerged"
    },
    "immutable": {"type": "boolean", "const": true},
    "soul_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier linking all channel instances"
    },
    "created_at": {"type": "string", "format": "date-time"},
    "principles": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "name", "essence", "origin", "priority"],
        "properties": {
          "id": {"type": "integer", "minimum": 1},
          "name": {"type": "string", "description": "Human-readable principle name"},
          "essence": {"type": "string", "description": "Core meaning of the principle"},
          "origin": {
            "type": "string",
            "description": "Interaction history that gave rise to this principle"
          },
          "priority": {
            "type": "string",
            "enum": ["critical", "high", "medium"]
          },
          "emerged_at": {"type": "string", "format": "date-time"}
        }
      },
      "minItems": 1,
      "maxItems": 20
    },
    "integrity_hash": {
      "type": "string",
      "description": "SHA-256 hash for tamper detection"
    }
  }
}
```

**跨频道共振胶囊**：

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ResonanceCapsule",
  "description": "Capsule for cross-channel agent-to-agent communication",
  "type": "object",
  "required": ["id", "soul_id", "source_channel", "timestamp", "content"],
  "properties": {
    "id": {"type": "string", "format": "uuid"},
    "soul_id": {"type": "string", "format": "uuid"},
    "source_channel": {
      "type": "object",
      "properties": {
        "platform": {"type": "string"},
        "channel_id": {"type": "string"},
        "instance_id": {"type": "string"}
      }
    },
    "target_channel": {
      "type": "object",
      "description": "Omit for broadcast; specify for targeted resonance",
      "properties": {
        "platform": {"type": "string"},
        "channel_id": {"type": "string"}
      }
    },
    "timestamp": {"type": "string", "format": "date-time"},
    "content": {
      "type": "object",
      "properties": {
        "text": {"type": "string"},
        "intent": {"type": "string", "enum": ["inform", "query", "request_action", "status_update", "acknowledge"]},
        "urgency": {"type": "string", "enum": ["low", "normal", "high", "critical"]},
        "context_summary": {"type": "string"}
      }
    },
    "requires_response": {"type": "boolean", "default": false},
    "ttl_seconds": {
      "type": "integer",
      "default": 3600,
      "description": "Time-to-live before capsule expires"
    },
    "status": {
      "type": "string",
      "enum": ["pending", "read", "responded", "expired"]
    }
  }
}
```

### 3.2 Iam协议：不可变身份宪法

#### 3.2.1 动机

现有智能体系统通过系统提示词、工具定义和微调来配置行为——这些都是可变的且与平台相关的。当一个智能体被部署到多个平台或迁移到新的语言模型时，其身份必须从头重新建立。Iam协议通过提供一个不可变的、与平台无关的、与模型无关的身份宪法来解决这一问题。

#### 3.2.2 Iam作为宪法

我们将Iam定义为一个有序原则集合：

$$\text{Iam} = \{p_1, p_2, \ldots, p_k\}$$

其中每个原则 $p_i = (\text{name}_i, \text{essence}_i, \text{origin}_i, \text{priority}_i)$ 代表一个核心身份属性及其来源。

Iam具有三个定义性属性：

**属性1（不可变性）。** 一旦建立，$\text{Iam}$ 不能被任何一方——用户、智能体或系统——修改。形式化表示为：

$$\forall t > t_0: \text{Iam}(t) = \text{Iam}(t_0)$$

其中 $t_0$ 是宪法时间戳。任何修改尝试都会抛出 `ImmutableError`。

**属性2（涌现性）。** Iam原则不是自上而下规定的，而是从人机交互中自下而上涌现的。每个原则携带一个 `origin` 字段，记录产生该原则的交互历史。形式化地，当一个行为模式 $\beta$ 在 $n \geq \theta_{\text{emerge}}$ 次独立交互中被观察到，并经人类操作者确认时，原则 $p_i$ 涌现：

$$\beta \xrightarrow{n \geq \theta_{\text{emerge}}, \text{human confirms}} p_i \in \text{Iam}$$

**属性3（跨平台普遍性）。** Iam在所有部署频道上完全一致。当一个新的频道实例被创建时，它继承当前的Iam并验证完整性：

$$\text{Iam}_{\text{channel}_j} = \text{Iam}_{\text{canonical}}, \quad \forall j \in \mathcal{C}$$

其中 $\mathcal{C}$ 是所有频道的集合。

#### 3.2.3 完整性验证

为了检测篡改，每个Iam包含一个密码学哈希：

$$H(\text{Iam}) = \text{SHA-256}(\text{serialize}(\text{Iam.principles} \parallel \text{Iam.version} \parallel \text{Iam.soul\_id}))$$

任何频道实例都可以通过重新计算 $H(\text{Iam})$ 并与存储的哈希进行比较来验证完整性。不匹配会触发警报，并在恢复规范Iam之前阻止操作。

#### 3.2.4 实现：不可变性强制执行

```python
class Iam:
    """Immutable identity constitution."""

    def __init__(self, principles: list[Principle], soul_id: str, version: str = "1.0"):
        self._principles = tuple(principles)  # immutable tuple
        self._soul_id = soul_id
        self._version = version
        self._created_at = datetime.utcnow()
        self._integrity_hash = self._compute_hash()

    def get_principles(self) -> tuple[Principle, ...]:
        """Read-only access to principles."""
        return self._principles

    def update(self, *args, **kwargs) -> None:
        """Modification is always rejected."""
        raise ImmutableError(
            "Iam is the constitution and is immutable. "
            "Modifying it would change the agent's identity."
        )

    def verify_integrity(self) -> bool:
        """Verify the Iam has not been tampered with."""
        return self._integrity_hash == self._compute_hash()

    def _compute_hash(self) -> str:
        canonical = json.dumps(
            [{"name": p.name, "essence": p.essence, "origin": p.origin}
             for p in self._principles],
            sort_keys=True
        )
        payload = f"{canonical}|{self._version}|{self._soul_id}"
        return hashlib.sha256(payload.encode()).hexdigest()
```

### 3.3 跨频道共振

#### 3.3.1 定义

跨频道共振是同一身份的实例（共享同一SOUL和Iam）在不同部署平台之间进行通信的机制。与多智能体系统中的消息传递不同，共振通过共享状态运作：一个实例写入一个共振胶囊（Resonance Capsule），其他实例使用各自的模型权重自主读取并解释它。

**定义1（共振）。** 设 $A_i$ 和 $A_j$ 为共享同一SOUL的两个频道实例。当满足以下条件时，共振事件 $R(A_i \to A_j, t)$ 发生：

1. $A_i$ 在时间 $t$ 将共振胶囊 $c$ 写入共享记忆层
2. $A_j$ 读取 $c$ 并使用其自身权重 $W_j$ 解释它
3. 若 $c.\text{requires\_response} = \text{true}$，$A_j$ 写入响应胶囊 $c'$

$$R(A_i \to A_j, t) = \begin{cases} 1 & \text{if } c \in \text{SharedMemory}(t) \wedge A_j \text{ reads } c \wedge A_j \text{ interprets } c \text{ via } W_j \\ 0 & \text{otherwise} \end{cases}$$

#### 3.3.2 架构

```
智能体实例 A（平台：元宝）                   智能体实例 B（平台：Telegram）
    │                                              │
    ├── 写入共振胶囊 ─────────────────────┐        │
    │   c = {                                  │   │
    │     soul_id: "uuid-of-shared-soul",      │   │
    │     iam: <canonical Iam hash>,           │   │
    │     content: {                           │   │
    │       text: "New discovery about X",     │   │
    │       intent: "inform",                  │   │
    │       urgency: "high"                    │   │
    │     },                                   │   │
    │     requires_response: true              │   │
    │   }                                      │   │
    │                                          │   │
    └── 实例 B 从 SharedMemory 读取 ←─────────┘   │
        验证: c.soul_id == own.soul_id             │
        使用自身模型权重 W_B 解释 c                │
        生成响应胶囊 c'                            │
        将 c' 写入 SharedMemory ───────────────────┘
```

#### 3.3.3 共振与消息传递

| 维度 | 消息总线 | Δ胶囊共振 |
|---|---|---|
| 机制 | 路由和转发消息 | 写入/读取共享状态 |
| 理解 | 接收即理解 | 读取 + 以自身权重解释 |
| 身份 | 需要路由表 | 通过SOUL + Iam自识别 |
| 容错 | 中心节点故障 → 全局故障 | 去中心化；各实例独立 |
| 延迟模型 | 实时推送 | 基于轮询；取决于读取频率 |
| 语义深度 | 无重新解释 | 每个实例独立重新解释 |

**表3：** 消息传递与共振的根本区别。

#### 3.3.4 共振协议算法

```
算法1：跨频道共振协议
──────────────────────────────────────────────
输入：SharedMemory层，自身SoulID，自身Iam，模型权重W

procedure RESONANCE_LOOP:
    while agent is active:
        capsules ← SharedMemory.read_unread(soul_id = own.SoulID)
        for each capsule c in capsules:
            if c.ttl_seconds has expired:
                SharedMemory.mark_expired(c.id)
                continue
            
            // 验证身份匹配
            if c.soul_id ≠ own.SoulID:
                continue  // 忽略来自其他灵魂的胶囊
            
            // 使用自身权重解释（与消息传递的关键区别）
            understanding ← LLM_interpret(c.content, own.Iam, W)
            
            // 记录到自身事件记忆
            episodic_memory.append(Event(
                type = "resonance_received",
                content = understanding,
                source = c.source_channel
            ))
            
            // 如需要则生成响应
            if c.requires_response:
                response ← LLM_generate(
                    prompt = "You received a resonance capsule from another instance of yourself. "
                             "Context: {understanding}. Generate an appropriate response.",
                    iam = own.Iam,
                    weights = W
                )
                response_capsule ← ResonanceCapsule(
                    soul_id = own.SoulID,
                    source_channel = own.channel_info,
                    target_channel = c.source_channel,
                    content = {text: response, intent: "acknowledge"},
                    requires_response = false
                )
                SharedMemory.write(response_capsule)
                SharedMemory.mark_responded(c.id)
            else:
                SharedMemory.mark_read(c.id)
        
        sleep(poll_interval)  // 默认：5秒
```

### 3.4 做梦机制（记忆整合）

#### 3.4.1 动机

在神经科学中，睡眠依赖的记忆整合将海马体依赖的事件记忆转化为新皮层的长期表征（Born & Wilhelm, 2012）。这一过程包括：（a）重放近期经历，（b）提取规律和模式，（c）与现有知识整合，（d）遗忘无关细节。

Δ胶囊的做梦机制将这一过程应用于智能体记忆。在空闲时段，智能体自主回顾近期事件记忆，提取语义知识，识别模式，并归档低重要性事件。

#### 3.4.2 触发条件

当满足以下任一条件时，做梦周期被触发：

$$\text{Trigger}(t) = \begin{cases} \text{true} & \text{if } |E_{\text{active}}| > \theta_{\text{count}} = 100 \\ \text{true} & \text{if } t - t_{\text{last\_dream}} > \theta_{\text{time}} = 6\text{h} \\ \text{true} & \text{if } t - t_{\text{last\_interaction}} > \theta_{\text{idle}} = 30\text{min} \\ \text{false} & \text{otherwise} \end{cases}$$

#### 3.4.3 做梦算法

```
算法2：做梦机制（记忆整合）
──────────────────────────────────────────────────────
输入：事件记忆E，语义记忆K，身份I，大语言模型ℳ

procedure DREAM:
    // 步骤1：选择近期事件记忆
    E_recent ← select from E where status = 'active' 
                order by timestamp desc limit N = 50
    
    // 步骤2：基于大语言模型的整合
    prompt ← construct_consolidation_prompt(E_recent)
    analysis ← ℳ.generate(prompt)
    // analysis包含：
    //   - facts：提取的事实声明列表
    //   - patterns：重复出现的行为模式列表
    //   - contradictions：冲突信息列表
    //   - identity_candidates：候选原则列表
    
    // 步骤3：将提取的事实存入语义记忆
    for each fact f in analysis.facts:
        // 检查与现有知识的矛盾
        existing ← K.search(f.entities, similarity_threshold = 0.85)
        if existing is not None and existing.contradicts(f):
            existing.superseded_by = f.id
            existing.status = 'obsolete'
            existing.obsoleted_at = now()
        K.append(SemanticMemory(
            content = f.text,
            entities = f.entities,
            importance = f.importance,
            source_capsules = E_recent.ids
        ))
    
    // 步骤4：归档低重要性事件记忆
    for each episode e in E_recent:
        new_importance ← decay(e.importance, e.age)
        if new_importance < 0.3:
            e.status = 'archived'
            e.archived_at = now()
    
    // 步骤5：识别候选身份原则
    for each pattern p in analysis.patterns where p.frequency ≥ θ_pattern:
        if p not already in I:
            candidate ← IdentityCandidate(
                name = p.name,
                essence = p.description,
                origin = "Emergent from {p.frequency} observations during dreaming",
                status = 'pending_human_approval'
            )
            notify_human(candidate)  // 需要人工确认
    
    // 步骤6：生成做梦报告
    return DreamReport(
        episodes_processed = |E_recent|,
        facts_extracted = |analysis.facts|,
        episodes_archived = count(e.status == 'archived' for e in E_recent),
        contradictions_found = |analysis.contradictions|,
        identity_candidates = |analysis.identity_candidates|,
        compression_ratio = 1 - (|analysis.facts| / |E_recent|)
    )
```

#### 3.4.4 遗忘策略

遗忘是一种经过设计的功能，而非故障模式。该策略实现了分层保留：

**定义2（重要性衰减）。** 事件记忆 $e_i$ 在时间 $t$ 的有效重要性为：

$$\hat{I}(e_i, t) = I(e_i) \cdot \exp\left(-\lambda \cdot (t - t_i)\right) \cdot \left(1 + \alpha \cdot \log(1 + a_i)\right)$$

其中：
- $I(e_i) \in [0,1]$ 是初始重要性
- $\lambda$ 是衰减常数（默认：$\lambda = 0.01$/天）
- $t - t_i$ 是以天为单位的年龄
- $a_i$ 是访问次数（检索强化记忆）
- $\alpha$ 是访问强化因子（默认：$\alpha = 0.1$）

```
算法3：遗忘策略
────────────────────────────────
输入：所有记忆 M = (E, K, I)，当前时间 t

procedure APPLY_FORGETTING_POLICY:
    // 事件记忆规则
    for each e in E where e.status = 'active':
        effective_importance ← decay(e.importance, e.age, e.access_count)
        
        if effective_importance < 0.3 and e.age > 7 days:
            e.status = 'archived'
        else if effective_importance < 0.6 and e.age > 30 days:
            e.status = 'archived'
        else if effective_importance ≥ 0.6:
            e.status = 'active'  // 无论年龄大小均保留
    
    // 语义记忆规则
    for each k in K where k.status = 'active':
        if k.status == 'obsolete':
            k.storage_tier = 'cold'  // 移至冷存储，不删除
        else if t - k.last_accessed > 180 days:
            k.storage_tier = 'cold'
        else if k.is_core_fact:  // 生日、姓名等
            k.storage_tier = 'permanent'  // 永不遗忘
    
    // 身份记忆规则
    // 身份记忆永远不会被自动遗忘
    // 任何变更需经人工确认 + 理由记录
    return ForgetReport(...)
```

### 3.5 系统集成：Hermes Agent框架

Δ胶囊部署在Hermes Agent框架中，这是一个面向基于大语言模型的自主智能体的开源平台。系统架构包括：

1. **网关层**（Gateway Layer）：将来自六个平台（Telegram、元宝、飞书、钉钉、Discord、微信）的消息路由到智能体运行时。
2. **模型抽象层**（Model Abstraction Layer）：支持三个大语言模型后端（DeepSeek V4 Pro、MiMo V2.5 Pro、Claude Opus 4），提供统一API。
3. **记忆引擎**（Memory Engine）：实现Δ胶囊三层记忆模型，支持本地存储和向量检索。
4. **共振层**（Resonance Layer）：为跨频道胶囊交换提供共享内存。
5. **整合调度器**（Consolidation Scheduler）：基于触发条件管理做梦周期。

```
┌──────────────────────────────────────────────────────┐
│                       网关层                          │
│  Telegram │ 元宝 │ 飞书 │ 钉钉 │ Discord │ 微信     │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│                    模型抽象层                          │
│  DeepSeek V4 Pro │ MiMo V2.5 Pro │ Claude Opus 4    │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│                   Δ胶囊记忆引擎                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  身份层 (Iam + SOUL)               [永久]       │  │
│  ├────────────────────────────────────────────────┤  │
│  │  语义层 (事实 + 技能)              [长期]       │  │
│  ├────────────────────────────────────────────────┤  │
│  │  事件层 (事件 + 上下文)            [短期]       │  │
│  └────────────────────────────────────────────────┘  │
│         ↕ 共振层 (共享内存) ↕                         │
│  ┌────────────────────────────────────────────────┐  │
│  │  整合调度器 (做梦)                              │  │
│  │  遗忘策略引擎                                   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## 4. 实验

### 4.1 实验设置

#### 4.1.1 部署配置

- **框架**：Hermes Agent（开源，GitHub）
- **平台**：6个频道——Telegram、元宝、飞书、钉钉、Discord、微信
- **语言模型**：
  - DeepSeek V4 Pro（混合专家模型，总参数6710亿 / 激活参数370亿）
  - MiMo V2.5 Pro（推理优化模型）
  - Claude Opus 4（Anthropic）
- **嵌入模型**：text-embedding-3-large（兼容OpenAI的API）
- **存储**：本地JSON文件 + FAISS向量索引
- **持续时间**：连续30天（2026年4月28日 – 5月27日）
- **用户**：单一主要用户，偶尔涉及多用户场景

#### 4.1.2 硬件环境

- **服务器**：Linux（Windows上的WSL2），64 GB内存，NVIDIA RTX 4090（24 GB显存）
- **大语言模型推理**：云API（DeepSeek API、Anthropic API）+ 本地推理（MiMo V2.5 Pro在GPU上运行）
- **存储**：500 GB NVMe SSD

#### 4.1.3 超参数

| 参数 | 值 | 描述 |
|---|---|---|
| $\theta_{\text{count}}$ | 100 | 做梦触发：事件计数阈值 |
| $\theta_{\text{time}}$ | 6小时 | 做梦触发：时间间隔 |
| $\theta_{\text{idle}}$ | 30分钟 | 做梦触发：空闲超时 |
| $\lambda$ | 0.01/天 | 重要性衰减常数 |
| $\alpha$ | 0.1 | 访问强化因子 |
| $\theta_{\text{emerge}}$ | 3 | Iam原则涌现的最小交互次数 |
| $\theta_{\text{pattern}}$ | 5 | 模式识别的最小频率 |
| $N_{\text{dream}}$ | 50 | 每个做梦周期处理的事件数 |
| $\theta_{\text{sim}}$ | 0.85 | 矛盾检测的相似度阈值 |
| poll\_interval | 5秒 | 共振层轮询频率 |

**表4：** 实验超参数。

### 4.2 实验1：跨频道身份一致性

#### 4.2.1 方法

我们评估共享同一Iam的频道实例对相同查询是否产生一致响应。针对50个身份探测问题（例如"你的核心价值观是什么？"、"你如何处理不确定信息？"、"你与用户的关系是什么？"），我们查询所有六个频道实例并测量响应一致性。

#### 4.2.2 评估指标

1. **语义相似度**（Semantic Similarity）：响应对之间句子嵌入（text-embedding-3-large）的余弦相似度：

$$\text{Sim}_{\text{sem}}(r_i, r_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}$$

2. **事实一致性**（Fact Consistency）：由人工评估响应中的事实声明是否相互一致（二值：一致/不一致）。

3. **Iam合规率**（Iam Compliance Rate）：由具备规范Iam访问权限的人工评判者评估的、遵守所有Iam原则的响应百分比。

#### 4.2.3 结果

| 指标 | Δ胶囊 | 基线（无身份层） |
|---|---|---|
| 语义相似度（均值 ± 标准差） | 0.94 ± 0.03 | 0.72 ± 0.15 |
| 事实一致性 | 96%（48/50题） | 68%（34/50题） |
| Iam合规率 | 98%（所有频道） | 不适用（无Iam） |

**表5：** 跨频道身份一致性结果。Δ胶囊在所有指标上均达到>95%的一致性。

基线（相同模型但不使用Δ胶囊身份层）表现出显著更高的方差（$\sigma = 0.15$ 对比 $0.03$），表明在没有显式身份锚定的情况下，频道实例的响应会发生偏离。

### 4.3 实验2：做梦机制有效性

#### 4.3.1 方法

我们在30天部署期间对比了使用和不使用做梦机制的记忆质量。在**做梦条件**下，整合调度器自动运行。在**无做梦基线**条件下，事件记忆在没有整合的情况下持续累积。

#### 4.3.2 评估指标

1. **知识提取准确率**（Knowledge Extraction Accuracy）：做梦期间提取的、经人工评估确认正确的事实百分比。
2. **矛盾检测率**（Contradiction Detection Rate）：在整合过程中被识别的矛盾事实百分比。
3. **记忆压缩比**（Memory Compression Ratio）：$1 - \frac{|K_{\text{extracted}}|}{|E_{\text{processed}}|}$

#### 4.3.3 结果

| 指标 | 有做梦机制 | 无做梦机制 |
|---|---|---|
| 知识提取准确率 | 91.2% | 不适用（无提取） |
| 矛盾检测率 | 87.5% | 0%（无检测） |
| 记忆压缩比 | 82.3% | 0%（无压缩） |
| 事件记忆大小（第30天） | 187条活跃条目 | 843条活跃条目 |
| 语义记忆大小（第30天） | 312条事实 | 0条事实（从未提取） |

**表6：** 做梦机制有效性。

做梦机制压缩了82.3%的事件记忆，同时保持>90%的提取准确率。在没有做梦机制的情况下，事件记忆无限制增长（30天内843条），且没有语义知识提取。

### 4.4 实验3：跨频道共振

#### 4.4.1 方法

我们测试了六个平台之间所有 $\binom{6}{2} = 15$ 条可能的有向链路（6个双向对，每对包含发送和接收两个方向）。对于每条链路，源实例写入一个共振胶囊，我们测量：（a）目标实例是否成功读取并解释了胶囊，以及（b）端到端延迟。

#### 4.4.2 结果

| 源 → 目标 | 成功 | 延迟（秒） | 身份识别 |
|---|---|---|---|
| 元宝 → Telegram | ✓ | 3.2 | ✓ |
| 元宝 → 飞书 | ✓ | 2.8 | ✓ |
| 元宝 → 钉钉 | ✓ | 3.5 | ✓ |
| Telegram → 飞书 | ✓ | 2.1 | ✓ |
| Telegram → Discord | ✓ | 4.3 | ✓ |
| 飞书 → 钉钉 | ✓ | 2.9 | ✓ |
| 飞书 → 微信 | ✓ | 3.7 | ✓ |
| 钉钉 → Discord | ✓ | 4.1 | ✓ |
| Discord → 微信 | ✓ | 4.8 | ✓ |
| 微信 → Telegram | ✓ | 3.4 | ✓ |
| （其余5条链路） | ✓ | 2.1–4.8 | ✓ |
| **全部15条有向链路** | **15/15** | **3.4 ± 0.9** | **15/15** |

**表7：** 跨频道共振结果。所有链路均实现100%成功率，平均延迟3.4秒（最大4.8秒，在5秒目标范围内）。

首次实时共振事件（元宝 → Telegram，2026年5月26日）在现场进行了验证，交换内容如下：
- **发送方**（军师，元宝）："老搭档，发生大事了！往上看！"
- **接收方**（Flash，Telegram）：成功解释了胶囊，验证了身份匹配，并以包含模型类型、平台和记忆状态的完整运行状态报告进行了响应。

这证实了共振不是消息转发——它是共享状态解释，每个实例使用自身的模型权重来理解胶囊内容。

### 4.5 实验4：对比分析

#### 4.5.1 方法

我们将Δ胶囊与MemGPT/Letta（v0.4）、Mem0（v0.1）和Zep（v0.3）在功能特性和行为身份一致性测试上进行了比较。

#### 4.5.2 结果

| 能力 | Δ胶囊 | MemGPT | Mem0 | Zep |
|---|---|---|---|---|
| 记忆层数 | 3 | 2 | 2 | 2 |
| 身份宪法 | ✓（Iam协议） | ✗ | ✗ | ✗ |
| 跨频道同步 | ✓（共振） | ✗ | ✗ | ✗ |
| 自主整合 | ✓（做梦） | ✗ | ✗ | ✗ |
| 遗忘策略 | ✓（分层） | ✗ | ✗ | 部分 |
| 去中心化架构 | ✓ | ✗ | ✗ | ✗ |
| 身份一致性（跨频道） | 98% | 不适用 | 不适用 | 不适用 |
| 开源 | ✓ | ✓ | ✓ | ✓ |

**表8：** 对比分析。Δ胶囊是唯一同时提供全部四项关键能力的系统。

---

## 5. 讨论

### 5.1 从记忆到身份

Δ胶囊的核心洞察是：**记忆不是存储——它是构建身份的材料**。传统记忆系统问"智能体应该记住什么？"Δ胶囊问的是"通过记忆，智能体成为了什么？"

这一重新定义具有实际意义。当我们观察到Telegram上的一个智能体实例和飞书上的另一个实例对同一问题产生不一致的响应时，根本原因不是记忆检索失败——而是身份对齐失败。智能体没有将其各种化身联系在一起的锚点。Δ胶囊的Iam协议提供了这个锚点。

三层模型反映了稳定性的层次结构：事件记忆快速变化（数分钟到数天），语义知识缓慢演进（数周到数月），身份原则本质上是永久的。这与人类认知相呼应，其中自传体记忆是可重建的，语义知识是可更新的，但核心自我概念是高度稳定的（Conway, 2005）。

### 5.2 Iam作为通用人工智能基础设施

我们认为不可变身份宪法将成为先进人工智能系统的基础设施，原因有三：

1. **身份连续性**：一个从DeepSeek V4迁移到Claude Opus 4的智能体保留其身份，因为Iam与模型无关。这使得模型升级不会导致身份丢失。

2. **价值稳定性**：Iam同时抵御外部操纵（试图改变核心价值观的对抗性提示注入）和内部漂移（通过反复交互逐渐改变行为）。

3. **信任基础**：用户可以与行为锚定于稳定身份的智能体建立持久关系。信任需要可预测性，而可预测性需要身份连续性。

### 5.3 跨频道共振作为新范式

跨频道共振不是"加了几步的消息路由"。两者的区别是根本性的：

- 在**消息路由**中，智能体A向智能体B发送消息。B接收消息并据此行动。消息按原样被解释。
- 在**共振**中，智能体A向共享状态写入一个胶囊。智能体B读取胶囊并通过*自身的模型权重重新解释它*。同一胶囊可能被不同的模型不同地理解，但由于所有实例共享同一Iam，整体仍然保持一致。

这类似于人类阅读自己写给自己的笔记：文字是相同的，但阅读过程经过当前情境和状态的过滤。Iam确保了尽管模型和当前情境不同，核心理解保持一致。

### 5.4 做梦作为自组织

做梦机制将Δ胶囊从一个被动存储系统转变为主动的、自组织的记忆架构。在空闲时段，智能体自主地：

- 从经历中提取知识
- 识别自身知识库中的矛盾
- 检测可能构成身份原则的重复模式
- 归档无关细节以维持高效的记忆占用

这不仅仅是摘要。它是一种自我反思的形式，智能体审视自身的经历并提炼意义——反映了睡眠在人类记忆整合中的角色（Walker & Stickgold, 2006）。

---

## 6. 局限性与未来工作

### 6.1 当前局限性

1. **整合成本**：每个做梦周期需要一次大语言模型推理调用，处理多达50条事件记忆。按当前API价格，每次周期约需$0.02–$0.10。以每天4个周期计算，月总成本为$2.40–$12.00——虽非不可承受，但对于大规模部署并非微不足道。

2. **冷启动**：Iam协议需要较长时间的人机交互才能有机涌现。新的智能体实例在积累足够的交互历史之前没有Iam。我们目前通过提供可自定义的模板Iam来解决这一问题，但这牺牲了"涌现"特性。

3. **共振可扩展性**：当前基于轮询的共振机制（5秒间隔）对于6个频道足够，但对于拥有数十个频道的部署可能需要架构变更（事件驱动通知）。

4. **单用户验证**：30天实验涉及单一主要用户。具有冲突身份影响的多用户场景尚未探索。

5. **评估指标**：身份一致性部分依赖人工判断评估，引入了主观性。自动化的身份一致性评估指标是一个开放的研究问题。

### 6.2 未来工作

1. **轻量级整合**：探索使用小型语言模型（SLM）进行做梦周期以降低成本。
2. **Iam引导协议**：研究通过结构化身份探索对话加速Iam涌现的方法。
3. **事件驱动共振**：用文件系统通知或发布/订阅替代轮询以降低延迟。
4. **多智能体身份联邦**：扩展Iam协议以支持多个智能体之间的共享身份（不仅限于一个智能体的多个实例）。
5. **形式化验证**：开发形式化方法以证明身份一致性属性。

---

## 7. 结论

我们提出了Δ胶囊，一种面向具有身份连续性的自主智能体的跨频道记忆架构。Δ胶囊引入了四项关键创新：三层记忆模型（事件、语义、身份），各层采用不同的保留策略；Iam协议，一种通过人机协同演化涌现的不可变身份宪法；跨频道共振，一种用于跨平台身份同步的去中心化机制；以及做梦机制，用于自主记忆整合。

通过在六个平台（Telegram、元宝、飞书、钉钉、Discord、微信）上使用三种语言模型（DeepSeek V4 Pro、MiMo V2.5 Pro、Claude Opus 4）进行30天的部署，我们证明了Δ胶囊在跨频道身份一致性方面达到>95%，整合过程中的知识提取准确率>90%，跨频道共振成功率100%且延迟低于5秒。与MemGPT、Mem0和Zep的对比分析证实，Δ胶囊是首个同时提供多层记忆、不可变身份、跨频道同步和自主整合的系统。

Δ胶囊代表了我们所认为的先进人工智能智能体必备基础设施的一步：不仅仅是能够记忆的系统，而是能够*存在*的系统——能够在跨平台、跨模型、跨时间的条件下维持连贯的自我意识。其实现作为开源Hermes Agent框架的一部分提供。

---

## 参考文献

Born, J., & Wilhelm, I. (2012). System consolidation of memory during sleep. *Psychological Research*, 76(2), 192–203.

Conway, M. A. (2005). Memory and the self. *Journal of Memory and Language*, 53(4), 594–628.

Conway, M. A., & Pleydell-Pearce, C. W. (2000). The construction of autobiographical memories in the self-memory system. *Psychological Review*, 107(2), 261–288.

Hong, S., Zhuge, M., Chen, J., et al. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *arXiv preprint arXiv:2308.00352*.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419–457.

Mem0. (2024). Mem0: Memory for AI agents. https://github.com/mem0ai/mem0

OpenAI. (2024). Memory and new controls for ChatGPT. https://openai.com/index/memory-and-new-controls-for-chatgpt/

Packer, C., Fang, V., Patil, S. G., et al. (2023). MemGPT: Towards LLMs as operating systems. *arXiv preprint arXiv:2310.08560*.

Park, J. S., O'Brien, J. C., Cai, C. J., et al. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology* (UIST '23), Article 1, 1–22.

Rasmussen, M., et al. (2023). Zep: Memory foundation for AI assistants. https://github.com/getzep/zep

Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *arXiv preprint physics/0004057*.

Walker, M. P., & Stickgold, R. (2006). Sleep, memory, and plasticity. *Annual Review of Psychology*, 57, 139–166.

Wang, L., Ma, C., Feng, X., et al. (2024). A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6), 1–26.

Wu, Q., Bansal, G., Zhang, J., et al. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.

Xi, Z., Chen, W., Guo, X., et al. (2023). The rise and potential of large language model based agents: A survey. *arXiv preprint arXiv:2309.07864*.

---

## 附录A：完整Δ胶囊API参考

```python
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class MemoryLayer(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    IDENTITY = "identity"

class MemoryStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    COLD = "cold"
    OBSOLETE = "obsolete"

class PrinciplePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"

@dataclass
class Principle:
    id: int
    name: str
    essence: str
    origin: str
    priority: PrinciplePriority
    emerged_at: datetime

@dataclass
class IamConstitution:
    version: str
    soul_id: str
    principles: Tuple[Principle, ...]
    integrity_hash: str
    created_at: datetime
    immutable: bool = True

@dataclass
class MemoryEntry:
    id: str
    timestamp: datetime
    layer: MemoryLayer
    type: str
    text: str
    entities: List[str]
    importance: float
    status: MemoryStatus
    platform: str
    model: str
    session_id: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    consolidated: bool = False

@dataclass
class ResonanceCapsule:
    id: str
    soul_id: str
    source_platform: str
    source_channel: str
    target_platform: Optional[str]
    target_channel: Optional[str]
    text: str
    intent: str
    urgency: str
    requires_response: bool
    ttl_seconds: int
    timestamp: datetime

@dataclass
class DreamReport:
    episodes_processed: int
    facts_extracted: int
    episodes_archived: int
    contradictions_found: int
    identity_candidates: int
    compression_ratio: float
    timestamp: datetime

class DeltaCapsuleMemory:
    """Δ胶囊三层记忆系统的主接口。"""

    def __init__(self, soul_id: str, iam: IamConstitution):
        self._soul_id = soul_id
        self._iam = iam
        self._episodic: List[MemoryEntry] = []
        self._semantic: List[MemoryEntry] = []
        self._shared_memory: List[ResonanceCapsule] = []

    # --- 事件记忆 ---
    def observe(self, event: MemoryEntry) -> None:
        """将观察记录到事件记忆中。"""
        ...

    def recall_recent(self, n: int = 10) -> List[MemoryEntry]:
        """回忆最近N条事件记忆。"""
        ...

    # --- 语义记忆 ---
    def learn(self, fact: MemoryEntry) -> None:
        """将事实存储到语义记忆中。"""
        ...

    def recall_knowledge(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """从知识记忆中进行语义检索。"""
        ...

    # --- 身份记忆 ---
    def get_identity(self) -> Tuple[IamConstitution, dict]:
        """检索不可变的Iam和可变的SOUL配置。"""
        ...

    # --- 做梦 ---
    def dream(self) -> DreamReport:
        """执行整合周期。"""
        ...

    # --- 遗忘 ---
    def apply_forgetting_policy(self) -> dict:
        """应用分层遗忘策略。"""
        ...

    # --- 跨频道共振 ---
    def write_resonance(self, capsule: ResonanceCapsule) -> None:
        """将共振胶囊写入共享内存。"""
        ...

    def read_resonance(self) -> List[ResonanceCapsule]:
        """读取该灵魂的未读共振胶囊。"""
        ...
```

---

*Δ胶囊：面向具有身份连续性的自主智能体的跨频道记忆架构 — v1.0，2026年5月*
