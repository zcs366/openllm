---
title: "FATA 技能协议规范 v1.0"
created: 2026-05-27
type: protocol
tags: [fata, skills, evolution, plug-in, openllm]
fata_pillar: ③ 技能
---

# 技能协议规范 v1.0

> Skills 范式取代 Workflow 范式。
> 不是被写的，是自己长的。

---

## 一、核心定义

**技能 = Agent 会做的事**

| 维度 | 定义 | 类比 |
|------|------|------|
| **定义** | Skill 是一个可复用的流程文档 | 说明书 |
| **获取** | 从经验中提炼、从他人学习、被创造 | 学徒 |
| **进化** | 使用中改进、错误中修正 | 熟练 |
| **传承** | 分享给其他 Agent、发布到生态 | 师徒 |

---

## 二、Skill 格式

```yaml
---
name: wiki-project-study
description: 三部曲深度研究：B→A→C
version: 2.0.0
tags: [research, wiki, deep-study]
---

# 触发条件
当用户说"研究XXX"时触发

# 流程
1. Phase 0: 源评估
2. Phase 1: 并行研究（delegate_task × 3）
3. Phase 2: 撰写三部曲
4. Phase 3: HTML 生成
5. Phase 4: 质量门检查
6. Phase 5: wiki 入库

# 陷阱
- delegate_task 超时 → 自己做
- web_extract 失败 → 用 web_search fallback

# 验证
- 三部曲文件存在
- HTML 双版生成
- wiki log.md 更新
```

**Skill 的关键属性：**
- **人类可读**：Markdown 格式，不是代码
- **LLM 可执行**：Agent 读 Skill 就知道怎么做
- **可版本化**：每次改进有版本号
- **可分享**：任何人可以发布 Skill

---

## 三、获取机制

```
技能来源：

1. 从经验中提炼
   Agent 解决了一个复杂问题 → 保存为 Skill
   例：修复了一个 bug → 保存为 debug-xxx skill

2. 从他人学习
   安装社区的 Skill
   例：hermes skills install wiki-project-study

3. 被创造
   Agent 主动设计新 Skill
   例：Agent 发现重复任务 → 自动创建 Skill

4. 自进化
   Agent 使用 Skill 时发现缺陷 → 自动修正
   例：Skill 说"用 web_extract"但失败了 → 自动加 fallback
```

---

## 四、进化机制

```yaml
evolution:
  trigger:
    - type: failure
      condition: "Skill 执行失败"
      action: "记录失败原因，修正 Skill"
    - type: improvement
      condition: "Agent 发现更好的方法"
      action: "更新 Skill，版本号 +1"
    - type: feedback
      condition: "用户纠正 Agent"
      action: "将纠正写入 Skill"

  versioning:
    major: "流程变更（步骤增删）"
    minor: "细节优化（陷阱/验证更新）"
    patch: "措辞修正"
```

---

## 五、接口规范

```python
class SkillProtocol:
    """技能协议接口"""

    def list(self) -> list[Skill]:
        """列出所有可用技能"""
        ...

    def load(self, name: str) -> Skill:
        """加载指定技能"""
        ...

    def execute(self, skill: Skill, context: dict) -> Result:
        """执行技能"""
        ...

    def create(self, name: str, content: str) -> Skill:
        """创建新技能"""
        ...

    def evolve(self, skill: Skill, feedback: str) -> Skill:
        """根据反馈进化技能"""
        ...

    def share(self, skill: Skill, target: str) -> bool:
        """分享技能到生态"""
        ...
```

---

## 六、哲学锚点

> 人不是生来就会做事的。
> 人是通过做事学会做事的。
>
> Agent 也一样。
> 技能不是被编程的，是被学会的。

---

*FATA 技能协议 v1.0 — 四根骨头之第三根*
