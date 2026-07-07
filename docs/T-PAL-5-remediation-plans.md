# MemoryBus整改预案

> 立碑日期：2026-07-07
> 关联预测：P1-P5（见prediction_tracker.py）
> 触发条件：对应预测被证伪时自动触发

---

## P1失败：检索质量不如旧ICE

### 根因
关键词匹配 ≠ 语义匹配。"数据库连不上"搜不到"连接超时"。

### 整改方案：Embedding Reranker

```
当前流程：
  Query → Provider.search(关键词匹配) → top-K

整改后流程：
  Query → Provider.search(关键词匹配) → 候选池(扩大到top-20)
       → EmbeddingReranker(语义排序) → top-K
```

**核心改动：**
1. `memory_bus.py` 新增 `EmbeddingReranker` 类
2. `query()` 方法在merge-sort后加rerank步骤
3. reranker可插拔——有embedding用embedding，没有用原score

**预估工作量：** ~200行
**涉及文件：** `memory_bus.py` + 新增 `reranker.py`
**验证标准：** recall@5从<0.6提升到≥0.7

---

## P2失败：jiak recall@5 < 0.6

### 根因
jiak card keywords覆盖面不足 + query扩展缺失。

### 整改方案：Query扩展 + 混合排序

```
当前：query.text → tokenize → 关键词匹配
整改：query.text → tokenize → 同义词扩展 → 混合排序
```

**核心改动：**
1. `jiak_provider.py` 新增 `_expand_query()` — 调用 `belief_update.py` 的 `SYNONYM_MAP`
2. 搜索分数改为混合：`keyword_score×0.4 + importance×0.3 + temperature×0.3`
3. 新增 `_partial_match()` — 子串匹配（"连接"匹配"数据库连接超时"）

**预估工作量：** ~150行
**涉及文件：** `jiak_provider.py`
**验证标准：** recall@5从<0.6提升到≥0.6

---

## P3失败：跨provider写入重复

### 根因
写入路由是"第一个接受者"，无跨provider去重。

### 整改方案：写入去重层

```
当前：WriteRequest → 第一个provider接受 → 完成
整改：WriteRequest → 所有provider询问 → 去重后存储
```

**核心改动：**
1. `memory_bus.py` `write()` 方法改为收集所有provider的store结果
2. 新增 `_dedup_writes()` — 对content做Jaccard相似度检测
3. Jaccard>0.8的视为重复，只保留第一个

**预估工作量：** ~100行
**涉及文件：** `memory_bus.py`
**验证标准：** write_log中跨provider Jaccard>0.8的记录为0

---

## P4失败：100 provider性能退化

### 根因
串行查询所有provider。

### 整改方案：分组并行

```
当前：provider1 → provider2 → ... → providerN（串行）
整改：
  Phase 1: top-3 priority串行（快速返回高优先级）
  Phase 2: 其余并行（ThreadPoolExecutor）
  完成即返回（不等所有provider）
```

**核心改动：**
1. `memory_bus.py` `query()` 改为两阶段：串行top-3 + 并行其余
2. 新增 `max_wait_ms` 参数——超时即返回已有结果
3. 用 `concurrent.futures.ThreadPoolExecutor`

**预估工作量：** ~150行
**涉及文件：** `memory_bus.py`
**验证标准：** 100 provider下query延迟<50ms

---

## P5失败：30天内<3个Hermes插件使用

### 根因
集成摩擦太高/开发者不知道MemoryBus存在。

### 整改方案：降低集成门槛

```
1. Hermes插件模板集成
   - 新建plugin时默认 import ISA + bus_write
   - 5分钟上手教程

2. Bypass模式
   - 不改现有代码，只加一行包装
   - from openllm.memory.isa import ISA; isa = ISA(); isa.bus_write(...)

3. 文档推广
   - 在HERMES.md/CLAUDE.md中加入MemoryBus使用指南
   - 示例代码片段
```

**预估工作量：** 文档为主，~100行示例代码
**验证标准：** grep bus_write/bus_query调用者≥3

---

## 通用整改流程

任何预测被证伪时：

```
1. 记录：prediction_tracker.py --record PX fail "原因"
2. 分析：读对应整改方案
3. 执行：按方案编码
4. 验证：重跑检查，确认通过
5. 立碑：jiak卡片 + RECALL + 备忘录
6. 记录：prediction_tracker.py --record PX pass "整改后验证"
```

**铁律：整改不等不靠。证伪→立即执行预案→验证→立碑。**
