# PAL v1.0 — ISA记忆系统通电验收 + 综述v2（2026-08-14）

> 前期项目摘要（PAL继承）：
> - 上次PAL：`f3847fe` PAL v1.0 包拯审计整改（2026-07-09）
> - 背景：综述《openllm-isa-survey-20260814》判定ISA"80%未通电"（21/50分）→ 同日工作区完成"通电Phase 1"（未提交）→ 本PAL负责**通电验收 + 修测试断裂 + 综述v2**
> - **执行结果（2026-08-14）：6任务全完成，1414 passed，三碑同落（jiak: isa-power-on-pal-20260814）**

## 一、资产清查（已实测验证✅）

| 资产 | 状态 | 实测证据 |
|------|------|---------|
| MemoryBus 6 Provider注册 | ✅ 已通电（未提交） | `bus.get_providers()`=6个，total_records=2512 |
| build_context 走 MemoryBus | ✅ 已通电（未提交） | `isa_impl.py:67-87` 统一检索+降级 |
| 温度公式 causal_delta×3.0 | ✅ 已改（未提交） | `causal_memory.py` + `memory_os.py:44` |
| scan_for_eviction 扫描淘汰 | ✅ 已通电（未提交） | `temperature_engine.py:204` 实测0候选（BASE_HEAT=3.0架空） |
| 免疫检查接线 | ❌ 断线 | `memory_bus.py` docstring宣称→免疫，write()直通provider |
| Update操作 | ❌ 缺失 | memory_bus无update方法，unified只有delete |
| 4个测试collection error | ❌ 断裂 | test_iai_router/slow_path/topology/user_state import已删模块 |
| 42个memory单测 | ✅ 全绿 | pytest 42 passed |
| 综述文件 | ⚠️ 过时 | 记录"6 Provider只注册1个"，实际已6个 |
| 未提交改动 | ⚠️ 84文件±27K行 | +2461/-24903，含iai 4文件废弃删除 |

## 二、PAL任务表

### P0（必须做——通电验收的根基）

**T-ISA-1 修复4个测试collection error** 🔴 可并行：⚠️ 预估：1h
- 事实：`iai/__init__.py` 合并说明明确"4个已deprecated（router/slow_path/user_state/topology），路由迁移至octopus.reason()"
- 4个测试文件（未跟踪）import被删模块 → collection失败
- 处理：**删除4个测试文件**（测的是废弃模块；core.router已有test_routing_regression.py覆盖，无需保留）
- 交付物：`tests/test_iai_router.py`等4文件删除 + 全量collection通过
- 验收：`pytest --co` 零error

**T-ISA-2 全量测试回归** 🔴 依赖T-ISA-1 预估：1h
- 交付物：全量pytest结果（当前370 collected）
- 验收：除已知废弃外全部通过，输出统计
- 若发现其他断裂 → 记录修复

**T-ISA-3 综述v2更新** 🔴 依赖T-ISA-2 预估：2h
- 事实：综述是"通电前"快照，6处结论已过时（Provider数量/build_context/遗忘扫描/入口文件/数据量/免疫接线）
- 处理：v2追加「通电Phase 1实录」章节，修正5操作评分（Storage 8→8.5、Retrieval 4→6.5、Update 1→1、Compression 5→5、Forgetting 3→3.5），新增前沿对标（Oracle生命周期/memorywire/记忆安全六阶段/Synthius-Mem）
- 交付物：`~/hermes/output/openllm-isa-survey-20260814-v2.md`
- 验收：新旧状态并列呈现，标注「综述快照vs当前代码」

### P1（支撑——通电后的真实gap）

**T-ISA-4 免疫检查接线** 🟡 依赖T-ISA-2 预估：2h
- 事实：memory_bus.py docstring写"→免疫检查→"，write()直通provider.store()
- 处理：write()路径插入immune检查（ImportError静默降级，不阻断写入）
- 交付物：memory_bus.py write()改造 + 单测
- 验收：写入带untrusted来源时触发L1拦截

**T-ISA-5 score归一化** 🟡 预估：2h
- 事实：causal用temperature当score(0-3)、capsule用关键词分(0-2)、jiak用BM25——直接sort是"苹果+橘子"
- 处理：MemoryBus.query()内按provider分桶→各自归一化→合并排序
- 交付物：memory_bus.py query()改造 + 测试
- 验收：跨provider score可比（0-1区间）

### P2（深化——可砍但建议做）

**T-ISA-6 BASE_HEAT调参验证** 🟢 预估：1h
- 事实：BASE_HEAT=3.0 > EVICT_THRESHOLD=0.3 → scan_for_eviction永远0候选 → 遗忘形同虚设
- 处理：实验验证（BASE_HEAT=0.5/1.0/2.0下的淘汰候选数），出数据不直接改
- 交付物：实验报告（不改代码）

**T-ISA-7 jiak生命周期API** 🟢 预估：4h（可另立PAL）
- 07-18精读就提出的write→consolidate→revise→evict→summarize→remove，27天未落地
- 本PAL只立项，不实施

## 三、全局依赖图

```
T-ISA-1 (删测试) ──→ T-ISA-2 (全量回归) ──→ T-ISA-3 (综述v2)
                        │
                        ├──→ T-ISA-4 (免疫接线)
                        └──→ T-ISA-5 (score归一化)   ← 与T-ISA-4并行

T-ISA-6 (BASE_HEAT实验) ← 独立，随时可跑
T-ISA-7 (立项) ← 不实施
```

## 四、间隙调节

- T-ISA-2回归等待 → 切T-ISA-6跑实验（同环境零冲突）
- T-ISA-3综述写作思路枯竭 → 切T-ISA-4读免疫代码

## 五、今日P0清单（立即开干）

| # | 任务 | 谁做 | 预估 | 交付物 |
|---|------|------|------|--------|
| 1 | T-ISA-1 删4个断裂测试 | 军师 | 0.2h | collection零error |
| 2 | T-ISA-2 全量回归 | 军师 | 1h | 测试统计 |
| 3 | T-ISA-3 综述v2 | 军师 | 2h | survey-v2.md |

> 三任务串行依赖（1→2→3），无并行分叉。T-ISA-4/5/6为今日间隙填充项。
