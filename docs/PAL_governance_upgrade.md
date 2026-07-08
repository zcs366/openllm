# openLLM 治理升级 PAL v2（五人合议修正版）
# 核战队联席会议通过 · 军师终裁 · 2026-07-04
# 基于 arXiv:2607.01087 + Agent Harness工程方法论 v1.0

## 变更记录
- v1: 初版（P0-P3四项）
- v2: 五人合议修正——子产砍P1/P2/P3，韩信加第三信号，鲁班审架构细节，萧何排路径，子贡排调度

## 背景
论文核心：治理代码 ≥ 2× 产品代码（2.75×）。治理被失败"发现"而非预先设计。
openLLM现状：10族治理机制 5✅ 4⚠️ 1❌。Phase 8/9骨架有但循环断裂。

## 方法论
Agent Harness工程方法论 v1.0 七条定律（详见 methodology_agent_harness_engineering.md）

## P0: 治理转换引擎（唯一P项）
**目标**: Phase 8/9 从"记录失败"升级为"发现治理需求并转换为持久治理"
**代码量**: ~800行
**时间**: 2周（10个工作日）
**验收标准**: 3次同类失败→自动触发治理转换→输出可验证的GovernanceRule

---

### P0-a: 结构性失败分类器
**文件**: `src/openllm/core/structural_failure_classifier.py` (新建)
**代码量**: ~250行
**时间**: Day 1-3

**设计（鲁班修正版）**:
```python
class StructuralFailureClassifier:
    """区分局部缺陷 vs 结构性失败 vs 待定。"""
    
    def classify(self, sig: FailureSignature, 
                 history: list[FailureSignature]) -> str:
        """返回: 'structural' | 'local' | 'ambiguous'
        
        韩信第三信号: ambiguous → 发GovernanceRequest
        """
        freq_score = self._time_weighted_frequency(sig, history)   # 权重0.4
        spread_score = self._spread_score(sig, history)            # 权重0.3
        ctx_sim = self._context_similarity(sig, history)           # 权重0.3
        
        composite = freq_score * 0.4 + spread_score * 0.3 + ctx_sim * 0.3
        
        if composite >= 0.7: return 'structural'
        elif composite >= 0.4: return 'ambiguous'  # → GovernanceRequest
        else: return 'local'
    
    def _time_weighted_frequency(self, sig, history) -> float:
        """时间衰减频率: 近期权重高，远期衰减。"""
        # 半衰期: 7天
        ...
    
    def _spread_score(self, sig, history) -> float:
        """扩散检测: 影响≥2个不同tool/component。"""
        ...
    
    def _context_similarity(self, sig, history) -> float:
        """上下文相似度: TF-IDF而非纯关键词。"""
        ...
```

**韩信第三信号集成**:
```python
# ambiguous → GovernanceRequest
@dataclass
class GovernanceRequest:
    """Agent主动请求治理干预。"""
    request_id: str
    failure_signature: FailureSignature
    reason: str           # 为什么ambiguous
    timestamp: float
    status: str           # pending | resolved | dismissed
    
# 写入 governance_requests.jsonl
```

---

### P0-b: 治理转换引擎
**文件**: `src/openllm/core/governance_engine.py` (新建)
         `src/openllm/core/governance_rule.py` (新建)
**代码量**: ~400行
**时间**: Day 4-7

**治理规则模型**:
```python
@dataclass
class GovernanceRule:
    rule_id: str
    rule_type: str           # architecture | control | constraint
    family: str              # 10族之一
    target: str              # 作用目标
    condition: str           # 触发条件（可机器验证）
    action: str              # 执行动作
    source_failure: str      # 来源失败signature
    confidence: float
    status: str              # pending | active | superseded | rolled_back
    conflicts: list[str]     # 鲁班: 冲突的已有规则ID
    verification_history: list  # 鲁班: 验证记录
    created_at: float
```

**治理引擎五步管线（鲁班修正版）**:
```python
class GovernanceEngine:
    def capture_failure(self, trace: dict) -> FailureSignature:
        """Step 1: 失败捕获"""
        ...
    
    def classify_failure(self, sig: FailureSignature) -> str:
        """Step 2: 失败分类（调用P0-a分类器）"""
        return self.classifier.classify(sig, self.history)
    
    def design_governance(self, sig: FailureSignature, 
                          classification: str) -> GovernanceRule:
        """Step 3: 治理设计——两种模式"""
        if classification == 'structural':
            return self._design_architecture_response(sig)  # 消除失败类
        else:
            return self._design_control_response(sig)       # 检测失败实例
    
    def install_governance(self, rule: GovernanceRule) -> bool:
        """Step 4: 治理安装——含冲突检测（鲁班要求）"""
        conflicts = self._detect_conflicts(rule)
        if conflicts:
            rule.conflicts = [c.rule_id for c in conflicts]
            rule.status = 'conflicted'
            return False
        # 写入治理基质
        self._write_rule(rule)
        rule.status = 'active'
        return True
    
    def verify_governance(self, rule: GovernanceRule) -> VerificationResult:
        """Step 5: 治理验证——含回滚机制（鲁班要求）"""
        regression = self._run_regression(rule)
        side_effects = self._check_side_effects(rule)
        
        if not regression.passed or side_effects.has_critical:
            self._rollback_governance(rule)  # 回滚！
            rule.status = 'rolled_back'
            return VerificationResult(passed=False)
        return VerificationResult(passed=True)
```

---

### P0-c: 集成到 main_loop.py
**文件**: `src/openllm/core/main_loop.py` (修改)
**改动量**: ~150行
**时间**: Day 8-10

**改动点**:
1. `learn_causal()` (line 536): 接入 StructuralFailureClassifier
   - structural → GovernanceEngine.design_governance()
   - ambiguous → 发GovernanceRequest
   - local → 原逻辑（记录到causal_memory）

2. `evolve()` (line 769): 升级为 GovernanceEngine 的验证+安装
   - 原有机制聚类逻辑保留
   - 新增: 调用GovernanceEngine.install_governance() + verify_governance()

3. 新增 Phase 8.5: `convert_governance()`
   - 在learn_causal和evolve之间插入治理转换步骤

**萧何警告**: 先写集成测试再改main_loop.py！

---

## 验收清单

- [ ] P0-a: 分类器能区分 structural/local/ambiguous（单元测试）
- [ ] P0-a: ambiguous自动发GovernanceRequest（单元测试）
- [ ] P0-a: 时间衰减近期权重>远期（单元测试）
- [ ] P0-b: 治理规则可写入/查询/验证（单元测试）
- [ ] P0-b: 冲突检测能发现与已有规则的冲突（单元测试）
- [ ] P0-b: 验证失败时自动回滚（单元测试）
- [ ] P0-c: 3次同类TOOL_PARAM失败→自动触发治理转换（集成测试）
- [ ] P0-c: 产出的GovernanceRule可被后续Agent继承（集成测试）
- [ ] P0-c: main_loop.py 改动不破坏现有功能（回归测试）

## 执行计划

```
Day 1-3:  P0-a 结构性失败分类器
  Day 1: 写骨架 + 三个评分函数
  Day 2: 写时间衰减 + 上下文相似度 + 测试
  Day 3: 集成到failure_tracker.py + 跑通单测

Day 4-7:  P0-b 治理转换引擎
  Day 4: 写GovernanceRule数据模型
  Day 5: 写GovernanceEngine五步管线骨架
  Day 6: 写冲突检测 + 回滚机制
  Day 7: 写治理设计（架构响应+控制响应）+ 测试

Day 8-10: P0-c 集成到main_loop.py
  Day 8: 写集成测试（模拟3次同类失败→验证治理转换触发）
  Day 9: 改learn_causal()接入分类器
  Day 10: 改evolve()接入治理引擎 + 端到端测试
```

## 投票记录

| 专家 | 权重 | 投票 | 核心意见 |
|---|---|---|---|
| 子产 | 0.20 | 批准（砍P1/P2/P3） | 只做P0，分两步 |
| 韩信 | 0.30 | 批准（加第三信号） | 增加GovernanceRequest |
| 鲁班 | 0.30 | 批准（加架构细节） | 时间衰减+冲突检测+回滚 |
| 萧何 | 0.10 | 批准 | 先测后接，P0-c风险最高 |
| 子贡 | 0.10 | 批准 | 10天执行计划 |
| **军师** | **议长** | **终裁批准** | **按修正后方案执行** |
