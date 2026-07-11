# PAL：openLLM 独立上线 v2.0

> 基于七神启示+五人合议终裁 | 2026-07-12
> 工程法典注入：匠石(代码>模型)·河床(约束先于创作)·循证(行动必有审计)

---

## P0：因果记忆demo（半天）

| 任务 | 交付物 | 验收标准 | 状态 |
|------|--------|---------|:----:|
| T-P0-1: 写demo_causal.py | `/home/zcs/projects/openllm/demo_causal.py` | 30行，无因果→犯错，有因果→纠正，终端彩色输出 | ⏳ |
| T-P0-2: 验证因果管线端到端 | demo能跑通causal_memory→ios_causal全链路 | `python demo_causal.py` 返回0 | ⏳ |
| T-P0-3: 三碑同落 | jiak卡片+RECALL+docs/技术文档 | jiak写入成功+RECALL签名写入 | ⏳ |

## P1：可安装+首次体验（1.5天）

| 任务 | 交付物 | 验收标准 | 状态 |
|------|--------|---------|:----:|
| T-P1-1: 修复pyproject.toml | 依赖完整 | `pip install -e .` 成功 | ⏳ |
| T-P1-2: CLI入口完善 | main()支持--provider/--model | `openllm` 能交互 | ⏳ |
| T-P1-3: 首次体验集成 | openllm启动时显示因果demo | 第一屏是因果记忆输出 | ⏳ |
| T-P1-4: --test命令 | 12项M0测试 | `openllm --test` 12/12 | ⏳ |

## P2：护城河完整激活（5天）

| 任务 | 交付物 | 验收标准 | 状态 |
|------|--------|---------|:----:|
| T-P2-1: 因果机制分类升级 | ios_causal.py升级 | 10种机制自动分类 | ⏳ |
| T-P2-2: G1-G6 deliberation接入 | agent_heartbeat.py升级 | 高风险决策触发投票 | ⏳ |
| T-P2-3: 六体并行化 | octopus_impl.py改造 | concurrent.futures | ⏳ |
| T-P2-4: 文件竞态保护 | file_safety.py | modTime检查 | ⏳ |
| T-P2-5: 测试扩展 | test_launch.py | 20/20通过 | ⏳ |

## P3：差异化打磨（持续）

| 任务 | 交付物 | 验收标准 | 状态 |
|------|--------|---------|:----:|
| T-P3-1: 自修改守卫清理 | 删除143行死代码 | grep零引用 | ⏳ |
| T-P3-2: 自动摘要触发 | auto_summary.py | 95%阈值触发 | ⏳ |
| T-P3-3: README | docs/README.md | 安装+使用+架构图 | ⏳ |
| T-P3-4: git tag v0.1.0 | 首个正式版本 | tag创建成功 | ⏳ |

---

## Non-Goal

1. ❌ 不做Hermes适配层
2. ❌ 不做TUI/Web UI
3. ❌ 不做Gateway
4. ❌ 不引入重型框架

## 石碑清单

每项任务完成后必须执行：
- [ ] jiak卡片写入
- [ ] RECALL签名写入
- [ ] 技术文档/决策记录更新

---

*PAL版本：v2.0 | 基于七神启示+五人合议终裁*
