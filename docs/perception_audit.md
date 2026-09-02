# openLLM 感知源审计报告（②多眼律 · PAL T-F-8）

> 审计日期：2026-09-02
> 审计范围：`core/octopus_impl.py`（薄转发壳）→ `iai/octopus.py`（章鱼I）+ `core/tentacle.py`（触手脑）+ `iai/event_bus.py`（EventBus+SignalBridge）+ `iai/active_sampler.py`（主动采样）+ `core/isa_impl.py`（ISA build_context）+ `core/agent_heartbeat.py`（_perceive融合）
> 蜻蜓复眼隐喻：28000小眼 + 360°视野 → openLLM的"眼睛"=触手阵列+EventBus+搜索+采样

---

## 一、现有感知源清单

### 1.1 触手脑阵列（TentacleBrain系列）

| 感知源 | 文件位置 | 感知什么 | 数据流向 | 频率 | 治理级 |
|--------|---------|---------|---------|------|--------|
| **FileWatcherBrain**（文件脑） | `core/tentacle.py:111` | 文件系统变化（新增/修改/删除） | 扫描`~/.openllm/output/`→mtime快照对比→TentacleReport→pending_reports | 被动触发（scan()调用时） | L0自治 |
| **IndexBrain**（索引脑） | `core/tentacle.py:209` | 文件内容（markdown/jsonl全文索引） | rglob扫描→倒排索引→search()→Context.search_results | build()时构建，search()时查询 | L1上报 |

**集成点**：`octopus.py:71-75` 章鱼I构造时注册：
```python
self.tentacles = {
    "file_watcher": FileWatcherBrain(),
    "index": IndexBrain(),
}
```

### 1.2 EventBus事件总线

| 感知源 | 文件位置 | 感知什么 | 数据流向 | 频率 | 通信模式 |
|--------|---------|---------|---------|------|---------|
| **EventBus** | `iai/event_bus.py:45` | 五体（IAI/IAX/ISA/IOS/ISN/IKO）状态变化 | BaseEventEmitter.emit_event()→publish()→Subscriber回调+JSONL持久化 | 事件驱动 | pub/sub+JSONL+3维过滤(source/type/brain_id) |
| **SignalBridge** | `iai/event_bus.py:162` | 文件系统IPC信号（core/signal.py信号文件） | signal_recv()轮询(5s)→转换为Event→EventBus.publish() | 每5秒轮询（daemon线程） | 桥接：文件系统→EventBus |

**集成点**：`iai/__init__.py:8` 导出EventBus, Event, Subscriber, BaseEventEmitter, SignalBridge

### 1.3 主动采样器

| 感知源 | 文件位置 | 感知什么 | 数据流向 | 频率 | 约束 |
|--------|---------|---------|---------|------|------|
| **ActiveSampler** | `iai/active_sampler.py:21` | 不直接感知外部，控制感知策略 | sample()→mark_used()→日志写入~/.openllm/iai/sample_log.jsonl | 每轮最多3次（赫淮斯托斯硬上限） | 预算硬上限+反预测扰动(explore) |

**集成点**：`iai/active_sampler.py:11` 注释"章鱼I感知阶段可选接入（sampler is None时零开销）"——**当前未实际接入主循环**。

### 1.4 ISA感知层

| 感知源 | 文件位置 | 感知什么 | 数据流向 | 频率 |
|--------|---------|---------|---------|------|
| **MemoryBus记忆检索** | `isa_impl.py:93` | 四源记忆（Δ胶囊+jiak+RECALL+因果） | BusQuery→bus.query()→Context.memory["recalled"] | build_context()时 |
| **因果记忆检索** | `isa_impl.py:136` | 历史因果教训（CausalMemoryStore） | store.search()→Context.causal_hints | build_context()时 |
| **证据回放** | `isa_impl.py:184` | 历史证据（evidence_replay） | create_replay_for_context()→Context.search_results | build_context()时 |
| **D₀感知** | `isa_impl.py:151` | 认知深度（API可用性+health+工具数） | octopus.d0_snapshot()→Context.d0_report | build_context()时 |
| **IoR厌倦过滤** | `isa_impl.py:172` | 已处理主题（抑制重复） | ior.suppress()→过滤search_results | build_context()时 |

### 1.5 心跳感知融合（_perceive）

| 步骤 | 函数位置 | 感知源 | 注入目标 |
|------|---------|--------|---------|
| 1.1 记忆上下文 | `agent_heartbeat.py:98` | ISA.build_context() | ctx.memory, ctx.identity |
| 1.1b 因果疤 | `agent_heartbeat.py:114` | causal.to_context_block() | ctx.causal_block, ctx.search_results |
| 1.1c 苏醒读钟 | `agent_heartbeat.py:128` | clock.now_status() | ctx.search_results |
| 1.2 章鱼索引搜索 | `agent_heartbeat.py:144` | index_brain.search() | ctx.search_results |
| 1.3 证据回放 | `agent_heartbeat.py:147` | evidence_replay | ctx.search_results |
| 1.4 因果预测 | `agent_heartbeat.py:151` | predict_consequences() | hc.prediction |
| 1.5 漂移检测 | `agent_heartbeat.py:159` | drift_detector | turn.trace |

---

## 二、360°覆盖度分析

### 已覆盖方向（9/14）

| 方向 | 感知源 | 覆盖度 |
|------|--------|--------|
| 用户输入 | user_message直接传递 | ✅ 完整 |
| 搜索/信息检索 | IndexBrain + MemoryBus | ✅ 完整 |
| 文件系统监控 | FileWatcherBrain | ✅ 完整 |
| 事件流 | EventBus pub/sub | ✅ 完整 |
| 文件系统IPC信号 | SignalBridge桥接 | ✅ 完整 |
| 时间/时钟 | Clock + timestamp | ✅ 完整 |
| 因果历史 | CausalMemoryStore | ✅ 完整 |
| 记忆召回 | MemoryBus/UnifiedMemory | ✅ 完整 |
| 认知深度 | D₀感知(d0_snapshot) | ✅ 完整 |

### 缺失方向（5/14）

| 方向 | 缺失描述 | 风险等级 | 推荐优先级 |
|------|---------|---------|-----------|
| **资源/环境监控** | CPU/内存/GPU/磁盘使用率——Agent不知道自己是否在"饥饿"运行 | 中 | P1 |
| **外部世界变化** | Web API/新闻/RSS/GitHub事件——Agent是信息孤岛 | 中 | P2 |
| **日志/异常监控** | 系统日志/错误/异常——Agent不知道自己哪里出错 | 低 | P2 |
| **用户行为模式** | 用户习惯/偏好/活跃时段——Agent不知道用户何时活跃 | 低 | P3 |
| **跨Agent感知** | 其他Agent状态/联邦学习信号——当前只有单Agent | 低 | P3 |

---

## 三、感知融合逻辑现状

### 3.1 融合架构（_perceive函数）

```
用户消息(msg)
    │
    ├──→ ISA.build_context()
    │       ├── MemoryBus四源检索 → memory["recalled"]
    │       ├── CausalMemoryStore → causal_hints
    │       ├── D₀感知 → d0_report
    │       └── IndexBrain搜索 → search_results[0..n]
    │
    ├──→ 苏醒协议注入 → search_results[0]
    ├──→ 因果疤注入 → search_results.append()
    ├──→ 时钟读取 → search_results.append()
    ├──→ 章鱼索引搜索 → search_results.append()
    ├──→ 证据回放 → search_results.append()
    │
    ├──→ predict_consequences() → hc.prediction
    └──→ drift_detector → turn.trace
```

### 3.2 关键观察

1. **search_results是感知融合的"集散地"**——所有感知源最终汇聚到`ctx.search_results`列表（目前7种来源）
2. **ActiveSampler未接入**——有完整的采样器实现，但未在_perceive()中实际调用
3. **EventBus未接入_perceive**——EventBus是独立的事件通道，不参与build_context()融合
4. **SignalBridge是daemon线程**——独立轮询，不阻塞主循环，但其事件未直接注入感知上下文
5. **FileWatcherBrain扫描不自动触发**——需要手动调用scan()，_perceive()中未调用

---

## 四、扩展点定义

### 4.1 接入模式（基于现有架构）

所有新感知源应遵循：
- **继承TentacleBrain**（L0自治级，不上报大脑即独立运行）
- **结果注入ctx.search_results**（通过`_perceive()`的扩展点）
- **通过EventBus发布事件**（可选，用于跨体通信）
- **采样预算控制**（接入ActiveSampler的预算机制）

### 4.2 推荐扩展源

#### 扩展源1：ResourceMonitor（资源感知）

**动机**：Agent不知道自己是否在"饥饿"运行——CPU>90%时应降级，磁盘满时应告警。

**接入模式**：
```
class ResourceMonitor(TentacleBrain):
    """L0自治：资源超阈值时自动降级"""
    - scan(): 读取/proc/stat, /proc/meminfo, GPU状态
    - 自治决策：CPU>90% → report("resource_pressure", 0.8, {...})
    - 集成点：_perceive()中调用monitor.scan() → 结果追加到search_results
```

**数据流向**：/proc/* → scan() → TentacleReport → ctx.search_results.append("[resource] CPU=95%...")
**频率**：每轮1次（_perceive中调用）
**治理级**：L0自治（资源压力时自动触发降级）

#### 扩展源2：LogWatcher（日志感知）

**动机**：Agent不知道自己哪里出错——错误日志是第一手诊断信号。

**接入模式**：
```
class LogWatcher(TentacleBrain):
    """L0自治：监控日志文件变化，提取关键错误"""
    - 监控：~/.openllm/logs/*.log + Python logging
    - 自治决策：ERROR/WARNING → report("log_anomaly", 0.6, {...})
    - 集成点：_perceive()中调用watcher.scan() → 结果追加到search_results
```

**数据流向**：日志文件 → tail -n → 关键词匹配 → TentacleReport → ctx.search_results
**频率**：每轮1次（_perceive中调用）
**治理级**：L0自治（错误日志自动上报）

#### 扩展源3：ExternalSignalBridge（外部信号感知）

**动机**：Agent是信息孤岛——不知道外部世界变化（GitHub PR/新论文/新闻）。

**接入模式**：
```
class ExternalSignalBridge(TentacleBrain):
    """L1上报：外部API变化→EventBus→Subscriber"""
    - 轮询源：GitHub API / arXiv RSS / 新闻API
    - 事件发布：通过EventBus.publish()发布到事件总线
    - 集成点：daemon线程（类似SignalBridge）→ 事件→Subscriber回调
```

**数据流向**：外部API → poll() → Event(source="EXTERNAL") → EventBus → Subscriber → ctx.search_results
**频率**：可配置轮询间隔（默认30分钟）
**治理级**：L1上报（外部变化需大脑确认）

---

## 五、审计发现

### 5.1 已发现问题

| # | 问题 | 位置 | 等级 | 建议 |
|---|------|------|------|------|
| 1 | **ActiveSampler未接入主循环** | `active_sampler.py:11` | 中 | 在_perceive()中调用sampler.sample()控制感知频率 |
| 2 | **FileWatcherBrain未自动扫描** | `agent_heartbeat.py:_perceive()` | 中 | 在_perceive()中调用watcher.scan()检测文件变化 |
| 3 | **EventBus未注入感知上下文** | `event_bus.py` | 低 | 可通过Subscriber将事件注入ctx.search_results |
| 4 | **search_results是混合列表** | `models.py:36` | 低 | 建议改为typed列表（区分搜索结果/日志/事件/资源） |

### 5.2 架构优势

- **TentacleBrain通信协议成熟**：L0自治+L1上报+TentacleReport标准格式
- **EventBus pub/sub灵活**：3维过滤(source/type/brain_id)，JSONL持久化
- **SignalBridge桥接设计优秀**：文件系统→EventBus无缝转换
- **感知融合点明确**：`_perceive()`是唯一的感知入口，扩展点清晰
- **采样预算控制就绪**：ActiveSampler的赫淮斯托斯硬上限防止API成本膨胀

---

## 六、总结

### 现状：9个感知源（5个成熟+4个在ISA层）

| 类别 | 数量 | 状态 |
|------|------|------|
| 触手脑（TentacleBrain） | 2 | 成熟可用 |
| EventBus+SignalBridge | 2 | 成熟可用 |
| ActiveSampler | 1 | 就绪但未接入 |
| ISA感知层 | 4 | 集成在build_context()中 |
| **总计** | **9** | **覆盖9/14方向** |

### 蜻蜓复眼对标

- 蜻蜓28000小眼 → openLLM当前9个感知源
- 蜻蜓360°视野 → openLLM覆盖9/14方向（64%覆盖度）
- 缺口：资源监控、外部世界、日志异常、用户行为、跨Agent

### 下一步

1. **P0**：将ActiveSampler接入_perceive()（1行代码+预算控制）
2. **P1**：实现ResourceMonitor（最小可用，监控CPU/内存）
3. **P2**：实现LogWatcher（监控错误日志）
4. **P3**：实现ExternalSignalBridge（外部世界感知）

---

*审计完成 · 军规三·可审计：每项结论指向具体代码行*
