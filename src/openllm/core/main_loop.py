#!/usr/bin/env python3
"""
openLLM Agent主循环 — 三体组装层

感知层(ISA+章鱼I) → 决策层(IOS) → 执行层(ISN+IKO)
心跳驱动三层轮流工作。
"""
import json, os, sys, time, uuid, io, contextlib
from pathlib import Path

# 数据模型
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult,
                     CausalDelta, TickMetrics)

# 三层架构
from .perception import ISA, 章鱼I        # 感知层：我知道什么
from .decision import IOS                  # 决策层：我决定什么
from .execution import ISN, IKO            # 执行层：我做什么
from .provider_impl import LLMProvider

# Session
from .session import Session, Turn, TurnStatus, create_session

# 可选依赖
try:
    from .tool_validator_types import ToolCall as _TVToolCall, ValidationResult as _TVResult, validate_tool_result as _tv_validate
    _HAS_TOOL_VALIDATOR = True
except ImportError:
    _HAS_TOOL_VALIDATOR = False

try:
    from .inference_budget import InferenceBudgetManager
    _HAS_BUDGET = True
except ImportError:
    _HAS_BUDGET = False

_REFLECT_SCRIPT = Path.home() / "projects" / "isa" / "octopus" / "scripts" / "octopus_reflect.py"
_HAS_REFLECT = _REFLECT_SCRIPT.exists()

class Agent:
    """openLLM Agent — 独立运行的Agent框架"""
    
    def __init__(self, mode: str = "console"):
        self.mode = mode
        
        # 静默模式：suppress所有print
        if mode == "silent":
            self._suppress = contextlib.redirect_stdout(io.StringIO())
            self._suppress.__enter__()
        else:
            self._suppress = None
        
        # 五体初始化
        self.isa = ISA(mode)
        self.octopus = 章鱼I()
        self.ios = IOS()
        self.isn = ISN()
        self.iko = IKO()
        
        # Session/Turn
        # 六体自监督反馈环
        from ..governance.feedback_loop import FeedbackLoop
        self.feedback_loop = FeedbackLoop()
        self._body_outputs = {}  # 暂存每体最新输出
        
        # 间隙体——空闲期好奇心
        from .idle_wander import IdleWanderer
        self._wanderer = IdleWanderer(threshold=3)

        self.session = create_session(max_context_tokens=self._load_max_context())
        
        # 状态
        self.running = False
        self._tick_count = 0
        self._max_ticks = 100  # 安全上限
        self._last_output = ""
        
        # ── IAX Layer 7 推理预算管理器 ──
        self._budget_manager = InferenceBudgetManager() if _HAS_BUDGET else None
        self._budget_checked_today = False
        
        # ── IOS Layer 15 Token经济（接入v1·2026-07-30） ──
        from .token_economy import TokenEconomy
        self._token_economy = TokenEconomy()
        
        # ── IAX+IOS Layer 11 上下文漂移检测（接入v1·2026-07-30） ──
        from .context_drift_detector import ContextDriftDetector
        self._drift_detector = ContextDriftDetector()
        
        # ── tool_validator 失败追踪 ──
        self._tool_failures: list[dict] = []

        # ── MemoryEvaluator 因果度量（E2 缺口③·2026-08-23） ──
        try:
            from ..memory.memory_evaluator import MemoryEvaluator
            self.memory_evaluator = MemoryEvaluator()
        except Exception:
            self.memory_evaluator = None

        # ── Clock 时钟账本（E2·2026-08-23） ──
        try:
            from .clock import Clock
            self.clock = Clock()
        except Exception:
            self.clock = None

        # ── 研究引擎（实验+论文+研究循环） ──
        from ..tools.research_loop import ResearchLoop
        self.research = ResearchLoop()
    
    def run(self):
        """主循环入口"""
        self.running = True
        print("\n═══ openLLM Agent 启动 ═══")
        print("  模式: 独立运行 · 无外部依赖")
        print("  Session:", self.session.id)
        print("  输入 /quit 退出\n")
        
        while self.running and self._tick_count < self._max_ticks:
            try:
                self._tick()
                self._tick_count += 1
            except KeyboardInterrupt:
                self.iko.trace("shutdown", "ok", detail="用户中断")
                break
            except Exception as e:
                self.iko.trace("fatal", "error", detail=str(e)[:50])
                ok = self.ios.recover(e)
                if not ok:
                    print(f"  🔴 不可恢复错误: {e}")
                    break
        
        self.shutdown()
        if self.mode != "silent":
            print("\n═══ openLLM Agent 关闭 ═══")
    
    def run_once(self, message: str) -> str:
        """
        单次运行（用于测试/非交互模式）
        
        用法：
            agent = Agent(mode="silent")
            result = agent.run_once("你好")
        """
        saved_mode = self.isa.mode
        self.isa.mode = "silent"
        try:
            msg = Message(text=message)
            self._execute_tick(msg)
            return self._clean_output(self._last_output)
        finally:
            self.isa.mode = saved_mode
    
    def shutdown(self):
        """关闭Agent，恢复stdout"""
        if self._suppress:
            self._suppress.__exit__(None, None, None)
            self._suppress = None
        self.iko.shutdown()
        # ISL：session收尾沉淀一环（空环也写，不可撤销）
        try:
            from openllm.core.isl_chain import ISLChain
            ISLChain().append_epoch(session_id=self.session.id)
        except Exception:
            import logging
            logging.getLogger("openllm.isl").exception("ISL epoch写入失败（不阻断关闭）")
        self.session.end()
    

    def _load_max_context(self) -> int:
        """从config.json读取max_context_tokens。"""
        import json
        cfg_path = Path.home() / ".openllm" / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                return cfg.get("agent", {}).get("max_context_tokens", 100000)
            except Exception:
                pass
        return 100000

    def _clean_output(self, raw: str) -> str:
        """从Agent raw output中提取干净回复。过滤内部系统噪声。"""
        if not raw:
            return ""
        lines = raw.split("\n")
        # 内部系统标记
        noise = ["[L1]", "[L2]", "[L3]", "[L4]", "[L5]", "[isa_ice]", "摘要:",
                 "🔴", "🗜️", "[jiak", "🔍", "□ ", "━━", "──", "│ ", "╔", "╚",
                 "关键决策", "关键句", "最近洞察", "[章鱼", "[PLUR", "[openllm",
                 "[jika", "[ambient", "---", "强制读取", "未读不可跳过", "语义场",
                 "铁律", "[上下文压缩", "线索:", "来源:", "写入方式", "意识笔记",
                 "写入", "同时追加", "第一人称", "不要总结", "🐙", "[文档]",
                 "[isa_ice]", "[记忆]", "[openllm-", "[octopus-", "[jiak-",
                 "token:", "[UNTRUSTED]", "搜索词", "搜索经验", "有新洞察"]
        # 从末尾往前找最后一段"干净"文本
        last_clean = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            s = lines[i].strip()
            if not s:
                continue
            if any(n in s for n in noise):
                break
            last_clean = i
        # 取last_clean到末尾
        result = "\n".join(lines[last_clean:]).strip()
        # 去掉空行
        result = "\n".join(l for l in result.split("\n") if l.strip()).strip()
        return result if result else raw.strip()

    def _tick(self):
        """一次心跳"""
        msg = self.isa.listen()
        if msg is None:
            # ── 间隙体：空闲期好奇心 ──
            if self._wanderer.tick_idle():
                self._do_wander()
            else:
                self.running = False
            return
        self._wanderer.tick_active()  # 有输入·重置空闲计数
        self._execute_tick(msg)
    
    def _do_wander(self):
        """散步模式——注意力的自由偏移"""
        discoveries = self._wanderer.wander()
        if discoveries:
            # 散步发现写入日志，不触发决策/执行
            for d in discoveries:
                self.iko.trace("wander", "ok", detail=f"{d.domain}: {d.finding[:40]}")
    
    def _record_inference(self, phase: str):
        """IAX: 从LLMProvider提取最近一次调用的token使用量，记录到预算管理器+Token经济"""
        if not self._budget_manager and not self._token_economy:
            return
        # 从左右脑provider取_last_usage（Phase 3用左脑，Phase 5用左脑+右脑）
        usage = getattr(self.octopus.left.provider, '_last_usage', {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            return  # 模拟模式或无数据，跳过
        try:
            # ── 原有：budget_manager记录 ──
            if self._budget_manager:
                rec = self._budget_manager.record_inference(
                    model=self.octopus.left.provider.model,
                    tokens_in=prompt_tokens,
                    tokens_out=completion_tokens,
                    duration_ms=0,  # 精确计时在phase trace中已有
                )
                self.iko.trace("budget", "ok",
                    detail=f"{phase}: in={prompt_tokens} out={completion_tokens} cost=${rec.cost_usd:.4f}")
                # 每天第一次调用时检查预算
                if not self._budget_checked_today:
                    self._budget_checked_today = True
                    budget = self._budget_manager.check_budget()
                    if budget["over_budget"]:
                        print(f"  ⚠️ IAX 推理预算超限: {budget['total_tokens']}/{budget['limit']} "
                              f"({budget['usage_ratio']:.0%}) 今日成本=${budget['cost_usd']:.4f}")
            # ── 新增：token_economy记录 ──
            if self._token_economy:
                self._token_economy.record_usage(
                    model=self.octopus.left.provider.model,
                    tokens_in=prompt_tokens,
                    tokens_out=completion_tokens,
                    task=phase,
                )
        except Exception:
            pass  # 预算追踪失败不阻塞主循环

    def _execute_tick(self, msg: Message):
        """一次完整的10阶段心跳（委托给agent_heartbeat）"""
        from .agent_heartbeat import execute_tick
        execute_tick(self, msg)


# ═══════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════

def main():
    """CLI入口"""
    args = sys.argv[1:]
    
    if "--agent-mode" in args:
        # Agent模式：静默启动，支持结构化API
        agent = Agent(mode="silent")
        result = None
        try:
            if "--once" in args:
                idx = args.index("--once")
                message = args[idx + 1] if idx + 1 < len(args) else "你好"
                result = agent.run_once(message)
            else:
                agent.run()
        finally:
            agent.shutdown()
        if result is not None:
            print(result)
        return
    
    if "--once" in args:
        # 单次模式
        idx = args.index("--once")
        message = args[idx + 1] if idx + 1 < len(args) else "你好"
        agent = Agent(mode="silent")
        try:
            result = agent.run_once(message)
        finally:
            agent.shutdown()
        print(result)
        return
    
    if "--test" in args:
        # 测试模式：跑一次完整的cycle然后退出
        _run_tests()
        return
    
    # 交互模式（默认）
    agent = Agent(mode="console")
    agent.run()


def _run_tests():
    """运行M0测试"""
    print("=== M0: Agent主循环测试 ===\n")
    
    # 测试1: 初始化
    agent = Agent(mode="silent")
    assert agent.isa is not None
    assert agent.octopus is not None
    assert agent.ios is not None
    assert agent.isn is not None
    assert agent.iko is not None
    print("✅ 测试1: 五体初始化")
    
    # 测试2: run_once
    result = agent.run_once("测试消息")
    assert result is not None
    print(f"✅ 测试2: run_once 输出={result[:30]}")
    
    # 测试3: Session/Turn生命周期
    assert agent.session.start_time is not None
    assert len(agent.session.turns) >= 1
    print(f"✅ 测试3: Session/Turn生命周期 total_turns={len(agent.session.turns)}")
    
    # 测试4: 左右脑对弈
    ctx = agent.isa.build_context(Message("写一段Python代码"))
    proposal, critique = agent.octopus.reason(ctx)
    assert proposal is not None
    assert critique is not None
    print(f"✅ 测试4: 左脑提案={proposal.content[:30]} 右脑批判={critique.verdict}")
    
    # 测试5: IO-S仲裁
    decision = agent.ios.arbitrate(proposal, critique)
    assert decision is not None
    print(f"✅ 测试5: 仲裁={decision.action} 批准={decision.approved}")
    
    # 测试6: 工具执行
    result = agent.isn.execute(decision)
    print(f"✅ 测试6: 执行 success={result.success}")
    
    # 测试7: IKO可观测
    report = agent.iko.report()
    assert report["total_ticks"] >= 2  # run_once + 手动
    print(f"✅ 测试7: 可观测 total_ticks={report['total_ticks']} ok_rate={report['ok_rate']:.0%}")
    
    # 测试8: IO-S自我进化
    print(f"✅ 测试8: 进化日志 {len(agent.ios.evolution_log)}条（>=1）")
    
    # 测试9: 因果预测
    prediction = agent.octopus.predict_consequences(ctx)
    assert prediction is not None
    print(f"✅ 测试9: 因果预测 {prediction.summary_text()[:40]}")
    
    # 测试10: 因果对照
    delta = agent.octopus.compare(prediction, result)
    assert delta is not None
    print(f"✅ 测试10: 因果对照 {delta.summary_text()}")
    
    # 测试11: 风险评估
    risk = agent.ios.risk_check(ctx, prediction)
    assert risk is not None
    print(f"✅ 测试11: 风险评估 level={risk.level}")
    
    # 测试12: 10阶段trace日志
    ticks_file = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
    assert ticks_file.exists()
    with open(ticks_file) as f:
        lines = f.readlines()
    assert len(lines) >= 2  # run_once + 手动
    print(f"✅ 测试12: 10阶段trace日志 ticks={len(lines)}")
    
    print(f"\n全部 12/12 测试通过 ✅")


if __name__ == "__main__":
    main()
