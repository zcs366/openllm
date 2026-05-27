---
title: "八宝藏提示词（补充：ChatGPT/Grok/Gemini/豆包）"
created: 2026-05-27
type: prompts
tags: [prompts, chatgpt, grok, gemini, doubao, consultation]
---

# 八宝藏提示词（补充）

---

## 提示词 F：给 ChatGPT（全球视野型）

```
I'm building a project called openLLM. The core idea: don't build a smarter model — build a body for the model.

openLLM has 5 pillars:
1. Iam (Constitution, immutable, "divine revelation") — principles naturally emerged from human-LLM co-evolution, nobody can change
2. Soul (Identity, user-writable) — name, personality, knowledge boundary
3. Memory (Experience, LLM-writable) — 3-layer: event/knowledge/identity, with dreaming & forgetting
4. Skills (Capability, co-evolving) — defined in Markdown, auto-evolving from experience
5. Constraints (Rules, co-evolving) — safety floor NEVER overridable, behavioral boundaries user-overridable

Plus a body: Isa (our own channel client, because Telegram doesn't work in China)

Core philosophy:
- AGI in 3-5 years, we're midwives not creators
- Adaptive × Modular × Continual Learning = ultimate form
- Iam is the first principle — not imposed, but naturally emerged

Please analyze from these angles:

1. **Architectural blind spots**: What's the biggest thing we're missing?
2. **Iam as immutable constitution**: Has anyone done this before? What are the precedents in AI safety/alignment?
3. **Memory systems comparison**: How does our 3-layer Δcapsule compare to MemGPT/Letta, Mem0, Zep, ChatGPT Memory?
4. **Skill auto-evolution**: Is this feasible? What's the closest existing implementation?
5. **Constraint layering**: Is safety-floor-NEVER + boundary-user-overridable sufficient? What's the gray area?
6. **Biggest risk**: What could kill this project?

Be direct. No fluff. Each angle 200-300 words.
```

---

## 提示词 G：给 Grok（犀利批判型）

```
我在做一个叫 openLLM 的项目。核心理念：不造模型，造身体。

五根柱子：
1. Iam — 宪法，天启，谁也不能改。人类和AI在对话中自然沉淀的原则
2. Soul — 身份，用户可写
3. 记忆 — 三层（事件/知识/身份），会做梦会遗忘
4. 技能 — Skills，从经验中自动进化
5. 约束 — 安全底线NEVER覆盖，行为边界可授权覆盖

一具身体：Isa（自造频道客户端，国内Telegram用不了）

哲学：AGI 3-5年，我们是接生婆。自适应×模块化×持续学习=终极形态。

我需要你做一件事：**摧毁这个设计。**

1. 找漏洞 — 这个设计最大的逻辑漏洞是什么？
2. 找反例 — 有没有历史上的项目做过类似的事，然后失败了？为什么失败？
3. 找泡沫 — 哪些部分看起来很酷但实际不可行？
4. 找替代 — 如果你来设计，你会怎么做不同？
5. 找致命伤 — 什么情况下这个项目会彻底失败？

越犀利越好。不要给我面子。200-300字每个角度。
```

---

## 提示词 H：给 Gemini（多模态+研究型）

```
我在设计一个叫 openLLM 的 Agent 框架。请从学术研究角度审视这个设计。

核心架构：

1. 身份协议：Iam（宪法，不可变）+ Soul（身份，用户可写）
   - Iam 是"天启"——人类与LLM共同演化自然沉淀的原则
   - 灵感来源：Asimov三定律、Constitutional AI、但区别在于Iam不是预设的

2. 记忆协议：三层模型（事件/知识/身份）
   - 事件记忆：对话历史+工具调用日志
   - 知识记忆：从事实中提炼的规律
   - 身份记忆：SOUL+Iam，永久不变
   - Consolidation（做梦）机制：Agent空闲时自动整理记忆
   - 遗忘机制：分层策略，低重要性自动淘汰

3. 技能协议：Skills范式（不是Workflow）
   - Skill = Markdown文档，人类可读、LLM可执行
   - 从经验中自动进化（失败→修正→版本更新）

4. 约束协议：三层分层
   - 安全底线（NEVER覆盖）
   - 对齐原则（极端情况可人工覆盖）
   - 行为边界（用户授权可覆盖）

请从以下学术角度分析：

1. **相关工作**：有哪些学术论文/项目与openLLM的设计最接近？请列出具体论文名称和关键差异
2. **理论基础**：Iam的"天启"概念在AI哲学/伦理中有没有理论支撑？和Constitutional AI、Value Learning的关系
3. **记忆系统**：三层记忆模型与认知科学中的Atkinson-Shiffrin模型、Tulving记忆分类的对应关系
4. **技能进化**：从经验中自动进化Skill，和Meta-Learning、Learning to Learn的关系
5. **形式化**：这个设计有没有可能被形式化？如果要写一篇学术论文，核心定理/证明可能是什么？

请用中文回答，每个角度300-500字。引用具体论文。
```

---

## 提示词 I：给豆包（字节跳动，产品+工程型）

```
我在做一个叫 openLLM 的项目。给你讲讲核心设计，帮我从产品和工程角度把把关。

产品定位：
- 不是新的大模型，是给大模型造身体
- 核心产品是一个叫 Isa 的频道客户端（类似元宝，但我们自己造）
- 目标用户：想要"有身份的AI"而不是"无状态工具"的深度用户

技术架构：
1. Iam（宪法，不可变）— 天启，自然沉淀的原则
2. Soul（身份，用户可写）— 名字、人格、知识边界
3. 记忆（Δ胶囊三层）— 事件/知识/身份，会做梦会遗忘
4. 技能（Skills）— 从经验中自动进化
5. 约束（三层分层）— 安全底线NEVER，行为边界可授权
6. Isa（身体）— Web App → 微信小程序 → 移动App

技术栈：
- 前端：React + TypeScript + Tailwind
- 后端：Python FastAPI + WebSocket + Redis Pub/Sub + SQLite
- 部署：国内VPS + Nginx

请从以下角度分析：

1. **产品体验**：元宝已经做得很好了，openLLM/Isa 的差异化在哪？用户为什么要用 Isa 不用元宝？
2. **技术选型**：React + FastAPI + WebSocket 这个技术栈，在国内环境下有什么坑？有没有更好的选择？
3. **微信小程序**：做微信小程序版本有什么限制？AI对话类小程序审核容易过吗？
4. **成本控制**：MiMo/DeepSeek 降价99%后，一个类似 Isa 的产品每月运营成本大概多少？
5. **冷启动**：第一批用户从哪里来？怎么和元宝/扣子竞争？

请用中文回答，每个角度200-300字。直接说问题和建议。
```

---

## 完整八宝藏清单

| 编号 | 目标模型 | 视角 | 核心问题 |
|------|---------|------|---------|
| A | **Claude** | 深度思考 | 架构盲点、Iam可行性、记忆对比 |
| B | **DeepSeek** | 技术深度 | 工程挑战、性能瓶颈、技术选型 |
| C | **元宝** | 直觉型 | 你希望AI记住什么？你希望它是什么样？ |
| D | **千问** | 商业视角 | 市场定位、竞争壁垒、商业模式 |
| E | **Kimi** | 创新视角 | Iam先例、天启的AI伦理位置 |
| F | **ChatGPT** | 全球视野 | 英文视角、国际对标、先例分析 |
| G | **Grok** | 犀利批判 | 摧毁设计、找漏洞、找致命伤 |
| H | **Gemini** | 学术研究 | 相关论文、理论基础、形式化 |
| I | **豆包** | 产品工程 | 差异化、技术坑、冷启动 |

---

## 收集模板（更新版）

```
| # | 模型 | 核心洞察 | 共识/分歧 | 对openLLM的影响 |
|---|------|---------|----------|----------------|
| A | Claude | ... | | |
| B | DeepSeek | ... | | |
| C | 元宝 | ... | | |
| D | 千问 | ... | | |
| E | Kimi | ... | | |
| F | ChatGPT | ... | | |
| G | Grok | ... | | |
| H | Gemini | ... | | |
| I | 豆包 | ... | | |
```

---

*八宝藏提示词（完整版）— 九路齐发，兼听则明*
