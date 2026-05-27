---
title: "FATA 目录结构 v1.0"
created: 2026-05-27
type: directory-structure
tags: [fata, directory, structure, openllm]
---

# FATA 目录结构

> 书 + 产品 + 代码，三位一体。

---

## 一、总览

```
fata/
├── book/                          # 《从智能体通往人工智能——openLLM在实践》
│   ├── chapters/                  # 各章草稿
│   ├── references/                # 参考文献
│   └── assets/                    # 图表、插图
│
├── openllm/                       # openLLM 产品
│   ├── iam/                       # ① Iam 协议（宪法）
│   ├── soul/                      # ② Soul 协议（身份）
│   ├── memory/                    # ③ 记忆协议（Δ胶囊）
│   ├── skills/                    # ④ 技能协议（Skills）
│   ├── constraints/               # ⑤ 约束协议（安全）
│   ├── isa/                       # ⑥ Isa（身体/频道客户端）
│   ├── bus/                       # Δ总线（骨架）
│   └── bridges/                   # 外接平台桥接
│
├── research/                      # 研究资料
│   ├── papers/                    # 论文收集
│   ├── wiki/                      # wiki 知识库链接
│   └── experiments/               # 实验记录
│
└── docs/                          # 文档
    ├── architecture/              # 架构文档
    ├── api/                       # API 文档
    └── guides/                    # 使用指南
```

---

## 二、book/（书）

```
book/
├── chapters/
│   ├── ch01-ultimate-form.md      # 第一章：终极形态——不是更大的 LLM ✅
│   ├── ch02-identity.md           # 第二章：身份——我是谁
│   ├── ch03-memory.md             # 第三章：记忆——我经历过什么
│   ├── ch04-skills.md             # 第四章：技能——我会做什么
│   ├── ch05-constraints.md        # 第五章：约束——我不做什么
│   ├── ch06-isa.md                # 第六章：Isa——openLLM 的肉身
│   ├── ch07-practice.md           # 第七章：实践——从第一个分身到跨频道共振
│   └── ch08-outlook.md            # 第八章：展望——接生婆的信
│
├── references/
│   ├── memory-scaling.md          # Memory Scaling 文章
│   ├── agent-harness.md           # Agent Harness 哲学
│   ├── intent-compression.md      # 意图压缩四层论
│   ├── cross-channel-resonance.md # 跨频道共振
│   ├── adaptive-modular-cl.md     # 自适应模块化持续学习
│   └── hassabis-agi.md            # Hassabis AGI 三大瓶颈
│
└── assets/
    ├── diagrams/                  # 架构图
    ├── tables/                    # 数据表格
    └── images/                    # 插图
```

---

## 三、openllm/（产品）

```
openllm/
├── iam/                           # ① Iam 协议（宪法，不可变）
│   ├── protocol.md                # 协议规范 v1.1
│   ├── principles.yaml            # Iam 原则定义
│   ├── iam.py                     # Python 实现
│   ├── tests/                     # 单元测试
│   └── README.md                  # 说明文档
│
├── soul/                          # ② Soul 协议（身份，用户可写）
│   ├── protocol.md                # 协议规范
│   ├── soul.yaml                  # Soul 定义模板
│   ├── soul.py                    # Python 实现
│   ├── tests/
│   └── README.md
│
├── memory/                        # ③ 记忆协议（Δ胶囊）
│   ├── protocol.md                # 协议规范 v1.0
│   ├── capsule.py                 # Δ胶囊核心实现
│   ├── layers/                    # 三层记忆
│   │   ├── event.py               # 事件记忆
│   │   ├── knowledge.py           # 知识记忆
│   │   └── identity.py            # 身份记忆
│   ├── consolidation.py           # 做梦机制
│   ├── forgetting.py              # 遗忘机制
│   ├── tests/
│   └── README.md
│
├── skills/                        # ④ 技能协议
│   ├── protocol.md                # 协议规范 v1.0
│   ├── engine.py                  # Skill 引擎
│   ├── evolution.py               # 进化机制
│   ├── registry.py                # Skill 注册表
│   ├── tests/
│   └── README.md
│
├── constraints/                   # ⑤ 约束协议
│   ├── protocol.md                # 协议规范 v1.0
│   ├── safety.py                  # 安全底线
│   ├── alignment.py               # 对齐原则
│   ├── boundary.py                # 行为边界
│   ├── tests/
│   └── README.md
│
├── isa/                           # ⑥ Isa（身体/频道客户端）
│   ├── design.md                  # 设计原型 v1.0
│   ├── frontend/                  # 前端（React）
│   │   ├── src/
│   │   ├── public/
│   │   ├── package.json
│   │   └── README.md
│   ├── backend/                   # 后端（FastAPI）
│   │   ├── main.py
│   │   ├── routers/
│   │   ├── models/
│   │   ├── services/
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── tests/
│   └── README.md
│
├── bus/                           # Δ总线（骨架）
│   ├── protocol.md                # 总线协议
│   ├── message.py                 # 消息格式
│   ├── router.py                  # 消息路由
│   ├── pubsub.py                  # 发布-订阅
│   ├── tests/
│   └── README.md
│
├── bridges/                       # 外接平台桥接
│   ├── telegram.py                # Telegram 桥接
│   ├── feishu.py                  # 飞书桥接
│   ├── yuanbao.py                 # 元宝桥接
│   ├── dingtalk.py                # 钉钉桥接
│   └── README.md
│
├── config/                        # 配置
│   ├── config.yaml                # 主配置
│   └── .env.example               # 环境变量模板
│
├── tests/                         # 集成测试
├── docs/                          # 文档
├── scripts/                       # 脚本
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明
```

---

## 四、research/（研究资料）

```
research/
├── papers/                        # 论文收集
│   ├── agent-harness/             # Agent Harness 相关
│   ├── memory-systems/            # 记忆系统相关
│   ├── intent-compression/        # 意图压缩相关
│   └── multi-agent/               # 多 Agent 相关
│
├── wiki/                          # wiki 知识库链接
│   └── links.md                   # wiki 页面索引
│
└── experiments/                   # 实验记录
    ├── ita/                       # ITA 实验
    ├── cross-channel/             # 跨频道共振实验
    └── memory/                    # 记忆系统实验
```

---

## 五、docs/（文档）

```
docs/
├── architecture/
│   ├── overview.md                # 总体架构
│   ├── bus-protocol.md            # Δ总线协议
│   ├── iam-protocol.md            # Iam 协议
│   └── memory-protocol.md         # 记忆协议
│
├── api/
│   ├── rest-api.md                # REST API 文档
│   ├── websocket.md               # WebSocket 协议
│   └── sdk.md                     # SDK 文档
│
└── guides/
    ├── getting-started.md         # 快速开始
    ├── deployment.md              # 部署指南
    └── contributing.md            # 贡献指南
```

---

## 六、与现有项目的关系

```
/mnt/i/hermes/                     # 现有 Hermes 工作区
├── output/doc/                    # FATA 产出（已存在）
├── wiki/concepts/                 # 协议文档（已存在）
├── .hermes/plans/                 # PAL 文档（已存在）
│
└── fata/                          # 新建 FATA 项目目录
    ├── book/                      # 书
    ├── openllm/                   # 产品
    ├── research/                  # 研究
    └── docs/                      # 文档
```

---

*FATA 目录结构 v1.0 — 书 + 产品 + 代码，三位一体*
