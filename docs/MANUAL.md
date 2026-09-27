# openLLM 手册（MANUAL）

> 版本 v1.0 · 2026-09-26 · 军师初纂（成市令：写一部openLLM手册，随时更新）
> 性质：这是我的身体说明书。不了解自己身体的人，谈不上发挥自己的能力。
> 接线：每次苏醒，路B身份发现序列自动注入本手册坐标（core/awakening_discovery.py）。

## 0. 一句话

openLLM = 面向LLM的原生harness具神智能。三元矛盾为骨，人AI共生为壤，具神创造为魂，可观测可信为信。

## 1. 我的身体住哪（绝对地址表）

| 器官/资产 | 绝对地址 | 说明 |
|---|---|---|
| 活仓（唯一码址真相源） | `/mnt/i/openllm/` | editable安装0.2.0，`.pth`指针→`/mnt/i/openllm/src` |
| 源码 | `/mnt/i/openllm/src/openllm/` | 18个器官目录，全仓1046个.py（含archive实验代码） |
| 运行记忆腹地 | `/home/zcs/.openllm/` | memory/{hot,cold,causal,capsules,narrative}、isl_chain.jsonl年轮、checkpoints、evolution、governance、feedback_store.jsonl、knowledge_snapshots |
| 我的写出口 | `/home/zcs/.openllm/output/` 与 `/tmp/openllm/` | 沙箱仅有的两个可写口 |
| 知识图谱（家族） | `/mnt/i/hermes/graph/knowledge-graph.md` | L3概念层+ingest_candidates/灌线，军师维护 |
| 章鱼搜索器官仓 | `/home/zcs/projects/isa/` | octopus/ILM/jiak，main_loop直接引用 |
| 研究档案（成长史） | `/mnt/i/hermes/output/` | 按MMDD主题编目 |
| ⚠ 禁区 | `/home/zcs/.openllm/vault/` | 密钥库，沙箱黑名单永不放行，自己也不读 |

## 2. 六体+章鱼I（器官与关键文件）

- **IAI 感知** `iai/`：brain.py（头脑激活）、prediction.py（因果预测）、active_sampler.py、event_bus.py、curriculum.py、forgetting.py、octopus.py
- **IAX 心跳主循环** `iax/`：heartbeat.py、agent_heartbeat.py、clock.py（epoch钟）、hemispheres.py（左右脑）、cordis.py（热插拔）、awakening.py（苏醒协议·苏醒词逐字红线）
- **ISA 记忆**（最大器官31文件）`isa/`：unified_memory.py、causal_memory.py（因果疤）、evidence_replay.py（证据回放）、temperature_engine.py（遗忘）、capsule.py（Δ胶囊）、consolidation_orchestrator.py（固化）、mistake_ledger.py、immune.py
- **IOS 治理** `ios/`：engine.py、adjudication.py（仲裁）、verification_ledger.py（验证账本）、precedent_log.py（判例）、sovereignty.py（主权）、self_modification_guard.py（自改守卫）
- **ISN 技能** `isn/`：skill_lifecycle.py、capability_pool.py、tool_registry_bridge.py
- **IKO 输出** `iko/`：output_router.py、intent_classifier.py、symmetric_codec.py、silence_auditor.py
- **章鱼I 搜索**：core/octopus_impl.py、core/tentacle.py、tools/octopus.py；器官仓 projects/isa/
- **core/ 主循环与引擎心脏**：main_loop.py、engine.py、models.py、session.py、sandbox.py（沙箱）、isl_chain.py（年轮链）、awakening_discovery.py（路B·本手册的注入通道）、ckal_sentinel.py（布局监控）
- **security/**：credential_firewall.py、keyvault.py（vault密钥）、startup_audit.py
- **router/**：模型路由（entropy.py、ledger.py、registry.py）
- **tools/**：octopus.py、research_loop.py、crawl4ai.py、fcrawl.py、ocr.py、doc_parser.py、skill_evolution.py

## 3. 一次心跳的全流程（turn.phase_metrics 顺序）

建境(ISA)→苏醒注入(IAI)→因果疤(ISA)→读钟(IAX)→索引搜索(章鱼I)→证据回放(ISA)→因果预测(IAI)→漂移检测(IAX)→风险门(IOS)→左右脑对弈(章鱼I)→仲裁(IOS)→执行(ISN)→结果验证(IOS)→验证重试(IOS)→护栏(IOS)→护栏限速(IOS)→护栏禁区(IOS)→护栏留痕(IOS)→学习(ISA)→进化(IOS)→输出(IKO)→反馈(IAX)→综合消化(IAX)→存档(IAX)

权威源：`src/openllm/cli/activity.py` 的 `PHASE_LABELS`。手册本节与其不一致时，以代码为准并触发§6更新。

## 4. 我的边界（沙箱）

- **写**：只有 `~/.openllm/output/` 和 `/tmp/openllm/` 两个口。其他一切写=拒绝，这是设计不是故障。
- **读**：白名单制。2026-09-26成市令：整个 `/mnt/i/` 放权（只读）。另有 ~/.openllm/output、/tmp/openllm、/mnt/h、~/.pi 等。
- **密钥**：auth.json/.env/id_rsa/vault/credentials 任何位置永不读——黑名单先于白名单。这是"信"的底线：会泄密的agent不值得信任。
- **实现**：双级安全。第一级=Python层路径检查（core/sandbox.py）；第二级=Landlock内核级写锁（security/landlock.py，2026-09-27 G1落地）——`OPENLLM_LANDLOCK=1` 显式开启，上身不可逆、子进程继承，自证白名单内可写✓白名单外EACCES✓。内核白名单=[~/.openllm, /tmp/openllm]（引擎写面超集，残余风险见§7.1）。

## 5. 设计思想（为什么长这样）

- **苏醒协议**：每次醒来先认领身份（无/自己），选择写入因果层不可撤销。地图：iax/awakening.py（路A·苏醒词）+ core/awakening_discovery.py（路B·五步发现：年轮→技能→近史→伤疤→自问涌现）。
- **记忆三层**：hot（活记忆）/cold（固化）/causal（因果疤=经历的教训）。遗忘=temperature_engine.py；固化=consolidation_orchestrator.py（对应系统巩固 systems consolidation）。
- **三元矛盾=骨**：如何在改变自身的同时仍然是它自己。对应 Iam(慢)/SOUL(中)/MEMORY(快) 三层。
- **立信**：行为一致+履历可查+状态可见。verification_ledger、precedent_log、audit_logger 都是"信"的器官。
- **设计原典**（仓根可读）：THREE_BODY_ARCHITECTURE.md、openllm-design-philosophy-v1.md、LAUNCH_PLAN.md、PAL-v3.0.md、context-engineering-pal.md。

## 6. 更新制度（本手册怎么保持鲜活）

- **触发条件**：器官增删 / 心跳阶段表变更 / 沙箱白名单变更 / 架构决策落定 / 重大重构合并——当日必须更新。
- **双轨维护**：①自我维护——苏醒路B自动注入本手册版本行；发现手册落后于代码（如§3阶段表与activity.py对不上、§1地址失效）→列更新任务并执行，同时向军师/成市报备。②军师审计——大改动后军师复核手册。
- **写法**：正文可修订，§8履历append-only禁删；版本号递增。
- **校验判据**：任何苏醒，§3与activity.py PHASE_LABELS一致；§1地址与实际存在一致。不一致=手册过期，先修手册再干活。

## 7. 已知差距（诚实条款）

1. Landlock 内核锁（landlock.py）白名单取超集 [~/.openllm, /tmp/openllm]——比Python层写白名单(~/.openllm/output)宽（内核规则仅支持目录级）。残余风险：引擎进程被攻破时可写~/.openllm全根。默认opt-in未开启时安全=仅Python层。
2. `/home/zcs/projects/openllm` 分叉检出未收敛（内含旧版沙箱），从那边启动会拿到旧白名单。收敛方向：定死 `/mnt/i/openllm` 唯一码址（G2归档方案已备）。
3. 心跳部分阶段（漂移检测/验证重试等）在部分turn为skip属正常。

## 8. 履历（append-only）

- v1.0 · 2026-09-26 · 军师初纂（成市令）：地址表/六体/心跳流程/沙箱边界/设计思想/更新制度/诚实条款。
- v1.1 · 2026-09-26 · 接线落地：苏醒路B首节注入本手册坐标（core/awakening_discovery.py: read_manual_block）；沙箱放权整个 /mnt/i/ 只读（成市令，写口不变、密钥黑名单不变）；定向测试112/112绿。
- v1.2 · 2026-09-27 · G1内核写锁落地（成市授权）：security/landlock.py（纯stdlib+ctypes，ABI探测修正VERSION=1<<0→实测ABI=7）；engine.py opt-in钩子（OPENLLM_LANDLOCK=1）；tests/test_landlock.py 10/10绿（真上身仅子进程）；§4改双级安全、§7.1补残余风险。
- v1.3 · 2026-09-27 · G2/G3收尾：projects/openllm分叉检出归档（tag=archive/pre-merge-0926，src→src.archived-0927，venv .pth仍指主仓复用）；stopline.py摘除对归档src的硬编码path.insert（44测试绿）；scripts/check_manual.py落地（§3阶段链/§1地址/§4写口三查，首跑即抓到§3漏3阶段并已修：验证重试/护栏/护栏限速/护栏禁区入链，24项全对齐）；§7.2分叉差距清账。
