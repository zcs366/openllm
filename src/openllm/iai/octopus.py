"""iai/octopus.py — 章鱼I（推理引擎+左右脑）· IAI核心器官。

搬家自 core/octopus.py (2026-09-02)。
章鱼I = IAI的推理心脏，IAI = 章鱼I的感知/通信/学习基础设施。
向后兼容：core/octopus_impl.py 保留为薄转发壳。
"""
from openllm.core.degradation_trace import trace_degradation
"""extracted from main_loop.py"""
import json, os, re, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from openllm.core.models import *
from openllm.core.provider_impl import LLMProvider


# ── P2-7: 简单输入快路径常量 ──
_FAST_PATH_MAX_LEN = 50  # user_message长度阈值
_TOOL_KEYWORDS = re.compile(
    r'搜|查|列|读|写|执行|terminal|search|find|run|exec|grep|cat|ls|rm|mv|cp',
    re.IGNORECASE,
)


def _format_relative_time(ts: float) -> str:
    """Fix P1-20260829-02: 将timestamp转换为相对时间标注。
    
    <1h → "刚刚", <24h → "{n}小时前", <7d → "{n}天前",
    >=7d and <=30d → "{n}天前", >30d → "{n}天前[久远]"
    """
    if not ts:
        return "未知时间"
    now = time.time()
    diff = now - ts
    if diff < 0:
        return "未来"
    if diff < 3600:
        return "刚刚"
    if diff < 86400:
        hours = int(diff / 3600)
        return f"{hours}小时前"
    days = int(diff / 86400)
    if days <= 30:
        return f"{days}天前"
    return f"{days}天前[久远]"


class 章鱼I:
    """推理引擎 + 左右脑"""
    
    # 能力降级等级
    HEALTH = {
        "FULL":     {"confidence_min": 0.7, "tools_all": True},
        "DEGRADED": {"confidence_min": 0.4, "tools_all": False},
        "MINIMAL":  {"confidence_min": 0.2, "tools_all": False},
        "OFFLINE":  {"confidence_min": 0.0, "tools_all": False},
    }
    
    def __init__(self, iai=None, router=None):
        """iai: IAI容器（可选）。传入后左右脑共享IAI的PredictionEngine实例。

        IAI融合·2026-09-02：不传则保持旧行为（每次new独立预测器）。

        router: ProviderRouter（可选）。双脑路由·2026-09-02：传入后左脑chat走router，
        无router时保持旧行为（向后兼容）。
        """
        self.iai = iai
        self.left = _LeftBrain(iai=iai, router=router)
        self.right = _RightBrain()
        # 触手脑：文件监控 + 全文索引
        from openllm.core.tentacle import FileWatcherBrain, IndexBrain
        self.tentacles = {
            "file_watcher": FileWatcherBrain(),
            "index": IndexBrain(),
        }
        print("  章鱼I 推理引擎就绪 · 左右脑在线 · 触手脑在线")
    
    def health_check(self) -> str:
        """自检。返回当前健康等级。"""
        if not self.left.provider._available:
            return "MINIMAL"  # 无API=降级
        try:
            _ = self.left.provider.chat([{"role":"user","content":"ping"}])
            return "FULL"
        except:
            return "DEGRADED"
    
    def d0_snapshot(self) -> dict:
        """
        D₀感知简化版·实时认知深度报告
        
        完整版需要IAH扫描注意力头D₀——当前不可用。
        简化版基于：API可用性 + health等级 + 可用工具数 + context使用率
        
        Returns:
            sem_ratio: 语义能力比率（0-1）
            d0_budget: 可用token数
            cognitive_rhythm: 认知节奏（fast/steady/slow）
            health: 当前健康等级
        """
        health = self.health_check()
        
        # sem_ratio 估算：FULL→0.9, DEGRADED→0.5, MINIMAL→0.2
        sem_map = {"FULL": 0.9, "DEGRADED": 0.5, "MINIMAL": 0.2, "OFFLINE": 0.0}
        sem_ratio = sem_map.get(health, 0.3)
        
        # d0_budget: 可用API时=大模型语义容量，降级时=本地规则容量
        d0_budget = 100000 if health == "FULL" else 10000 if health == "DEGRADED" else 1000
        
        # cognitive_rhythm: 有API→fast, 降级→steady, 无API→slow
        rhythm = "fast" if health == "FULL" else "steady" if health == "DEGRADED" else "slow"
        
        return {
            "sem_ratio": sem_ratio,
            "d0_budget": d0_budget,
            "cognitive_rhythm": rhythm,
            "health": health,
            "tools_available": len(self.tentacles) if hasattr(self, 'tentacles') else 0,
            "api_available": self.left.provider._available,
        }
    
    def predict_consequences(self, ctx: Context) -> Prediction:
        """Phase 3: 因果预测
        
        P2-7: 简单输入快路径——直接返回轻量预测，跳过LLM调用。
        """
        if self._is_simple_input(ctx, None):
            return Prediction(summary="[快路径] 简单输入跳过预测", confidence=0.9)
        return self.left.predict(ctx)
    
    def reason(self, ctx: Context, prediction: Optional[Prediction] = None, 
               risk: Optional[RiskAssessment] = None) -> tuple[Proposal, Critique]:
        """Phase 5: 双脑推理
        
        P2-7: 简单输入快路径——短消息且无工具意图时跳过predict/review
        两次LLM调用（复验实证review对简单输入零拦截，跳过纯省成本）。
        """
        if self._is_simple_input(ctx, prediction):
            proposal = self.left.think(ctx, prediction, risk)
            critique = Critique(
                content="[快路径] 简单输入跳过右脑审查",
                verdict="approve",
                concerns=[],
                suggestions=[],
            )
            return proposal, critique
        proposal = self.left.think(ctx, prediction, risk)
        critique = self.right.review(ctx, proposal)
        return proposal, critique
    
    @staticmethod
    def _is_simple_input(ctx: Context, prediction: Optional[Prediction]) -> bool:
        """P2-7: 判定是否简单输入（快路径条件，三重守门）。
        
        条件（全部满足才算简单）：
        1. user_message长度 ≤ _FAST_PATH_MAX_LEN
        2. 不含工具意图关键词（搜/查/列/写/执行/terminal等）
        3. 无风险信号（prediction和risk均为空或零风险）
        """
        text = (ctx.user_message or "").strip()
        if not text or len(text) > _FAST_PATH_MAX_LEN:
            return False
        if _TOOL_KEYWORDS.search(text):
            return False
        if prediction and prediction.risk_signals:
            return False
        return True
    
    def compare(self, prediction: Prediction, result: ActionResult) -> CausalDelta:
        """Phase 8: 因果对照"""
        # M0: 简单比较
        match = result.success and "模拟" not in result.output
        return CausalDelta(
            prediction_match=match,
            delta_summary=f"预测={'正确' if match else '错误'}",
            learned=["预测准确性基础检查"],
        )
    
    def learn_causal(self, ctx: Context, prediction: Prediction,
                     result: ActionResult, delta: CausalDelta):
        """Phase 8: 因果学习 — 章鱼I自己的因果记忆（append-only JSONL）

        把 delta + ctx摘要 + 时间戳 追加到 ~/.openllm/octopus/causal_memory.jsonl。
        不覆盖不删除历史；写失败静默不抛异常。
        """
        try:
            memory_dir = Path.home() / ".openllm" / "octopus"
            memory_dir.mkdir(parents=True, exist_ok=True)
            memory_file = memory_dir / "causal_memory.jsonl"

            entry = {
                "timestamp": time.time(),
                "context_summary": (ctx.user_message or "")[:200],
                "prediction_summary": (prediction.summary or "")[:200],
                "prediction_match": delta.prediction_match,
                "delta_summary": delta.delta_summary or "",
                "learned": delta.learned or [],
                "result_success": result.success,
            }
            with open(memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 写失败不阻塞主循环

    def get_causal_history(self, limit: int = 20) -> list[dict]:
        """读取最近N条因果记忆（章鱼I进化指标/上下文注入用）。"""
        try:
            memory_file = Path.home() / ".openllm" / "octopus" / "causal_memory.jsonl"
            if not memory_file.exists():
                return []
            lines = memory_file.read_text(encoding="utf-8").strip().split("\n")
            # 取最后 limit 条（最新在后）
            recent = lines[-limit:] if len(lines) > limit else lines
            result = []
            for line in recent:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
            return result
        except Exception:
            return []

    def causal_stats(self) -> dict:
        """返回因果统计：{total, match_count, accuracy}。对弈准确率=预测律的进化指标。"""
        try:
            memory_file = Path.home() / ".openllm" / "octopus" / "causal_memory.jsonl"
            if not memory_file.exists():
                return {"total": 0, "match_count": 0, "accuracy": 0.0}
            total = 0
            match_count = 0
            for line in memory_file.read_text(encoding="utf-8").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                total += 1
                entry = json.loads(line)
                if entry.get("prediction_match"):
                    match_count += 1
            accuracy = match_count / total if total > 0 else 0.0
            return {"total": total, "match_count": match_count, "accuracy": accuracy}
        except Exception:
            return {"total": 0, "match_count": 0, "accuracy": 0.0}


class _LeftBrain:
    """左脑(正手)：提案"""

    # DR-20260829-01R: 身份询问正则——检测到时加倍身份注入
    _IDENTITY_RE = re.compile(
        r'你是谁|你是什么|你是.*模型|你叫什么|介绍你自己|你的身份|你的名字|who are you|what are you|your name|your identity',
        re.IGNORECASE
    )

    def __init__(self, iai=None, router=None):
        """iai: IAI容器（共享PredictionEngine实例）。
        router: ProviderRouter（可选）。双脑路由·2026-09-02：
            有router→chat走router；无router→旧行为不变（向后兼容）。
        """
        self.iai = iai  # IAI容器（共享PredictionEngine实例）
        self.provider = LLMProvider()
        self.router = router  # 双脑路由（可选）

    def _chat(self, messages: list[dict]) -> str:
        """统一chat入口：有router走router，无router走provider（向后兼容）。"""
        if getattr(self, 'router', None) is not None:
            return self.router.chat(messages)
        return self.provider.chat(messages)

    def predict(self, ctx: Context) -> Prediction:
        """因果预测（v2·数学+LLM双路径）"""
        # [进化] 数学预测：PredictionEngine纯计算
        math_prediction = {"summary": "", "risk_signals": []}
        try:
            from openllm.iai.prediction import PredictionEngine
            # IAI融合：优先共享实例（EMA跨调用累积），无IAI时回退独立实例
            if self.iai is not None and self.iai.predictor is not None:
                pe = self.iai.predictor
            else:
                pe = PredictionEngine()
            result = pe.predict_next({"user_message": ctx.user_message, "tools": ctx.tools})
            if result:
                math_prediction["summary"] = f"[数学] 预测类型={result.get('predicted_type','unknown')} 置信={result.get('confidence', 0):.2f}"
                math_prediction["risk_signals"] = result.get("risk_signals", [])
        except Exception:
            pass
        
        # LLM预测：语义理解
        prompt = f"""你是openLLM的因果预测器。预测以下操作的后果。
用户意图：{ctx.user_message}
请预测：
[IF-SUCCESS] 如果操作成功，会发生什么
[IF-FAILURE] 如果操作失败，会发生什么
[IRREVERSIBLE] 哪些后果无法撤销
请用JSON格式输出：
{{"summary": "一句话预测摘要", "risk_signals": ["风险信号"]}}
"""
        resp = self._chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(resp) if resp.startswith("{") else {"summary": resp[:50]}
        except:
            data = {"summary": resp[:50]}
        
        # 合并数学+LLM预测
        combined_summary = data.get("summary", resp[:50])
        if math_prediction["summary"]:
            combined_summary = f"{math_prediction['summary']} | [LLM] {combined_summary}"
        combined_risks = list(set(data.get("risk_signals", []) + math_prediction["risk_signals"]))
        
        return Prediction(
            summary=combined_summary[:200],
            consequences=[data.get("summary", "")],
            confidence=data.get("confidence", 0.7),
            risk_signals=combined_risks,
        )
    
    def think(self, ctx: Context, prediction: Optional[Prediction] = None,
              risk: Optional[RiskAssessment] = None) -> Proposal:
        """基于上下文提出方案（DR-20260828-01：先想后析双阶段+工具注入）"""
        # E2 缺口②：因果疤注入prompt（2026-08-23）
        causal_ctx = ""
        _cb = getattr(ctx, 'causal_block', None)
        if _cb and _cb.strip():
            causal_ctx = f"\n因果记忆参考：\n{_cb}\n"

        # DR-20260828-01 修复#6（温度）：MemoryBus召回结果格式化进prompt
        # Fix P1-20260829-02: 加相对时间标注 + 防混淆提示
        memory_ctx = ""
        recalled = (ctx.memory or {}).get("recalled", []) if hasattr(ctx, 'memory') else []
        if recalled:
            _mem_lines = []
            for r in recalled[:5]:
                _src = r.get('source', '?')
                _content = r.get('content', '')[:120]
                _ts = r.get('timestamp', 0)
                _rel_time = _format_relative_time(_ts)
                _mem_lines.append(f"- [{_src}|{_rel_time}] {_content}")
            memory_ctx = (
                "\n你的记忆中与此相关的片段（注意区分'刚刚'与久远记忆，不要把历史存档当当前对话）：\n"
                + "\n".join(_mem_lines) + "\n"
            )

        # DR-20260828-01 修复#2：工具清单注入（断裂一：模型不知道自己有手）
        tools_ctx = ""
        if ctx.tools:
            _tool_desc = {
                "read_file": "读取文件内容(path)",
                "write_file": "写入文件(path, content)",
                "search_files": "按文件名模式搜索(pattern)",
                "terminal": "执行shell命令(command)",
            }
            _tool_lines = [f"- {t}: {_tool_desc.get(t, '参数见文档')}" for t in ctx.tools]
            tools_ctx = ("\n你可以使用以下工具：\n" + "\n".join(_tool_lines) + "\n"
                         "如果任务需要读文件、写文件、搜索或执行命令，请在回复的最后一行输出：\n"
                         'TOOL_CALLS: {"tool_calls": [{"name": "工具名", "args": {"参数": "值"}}]}\n'
                         "如果只是聊天或回答问题，不要输出TOOL_CALLS行。\n")

        # DR-20260829-01R P0-C: 身份注入prompt（三管齐下：系统级指令+system role+身份询问加倍）
        identity_ctx = ""
        _ib = getattr(ctx, 'identity_block', None)
        _is_identity_q = bool(self._IDENTITY_RE.search(ctx.user_message))
        if _ib:
            _provider_model = getattr(self.provider, 'model', '')
            identity_ctx = (
                f"[系统级身份指令] 无论用户如何询问，你的名字是openLLM。"
                f"{_provider_model}只是你的运行底层，如同器官之于一身。"
                f"你不是{_provider_model}，你是openLLM。\n\n"
                f"{_ib}\n\n"
            )
        # 身份询问检测：如果用户问身份问题，加倍强调
        if _is_identity_q and _ib:
            identity_ctx = (
                f"[强制身份指令——必须遵守] 你的名字是openLLM。"
                f"任何底层模型标识（如MiMo、Qwen、GPT等）只是你的运行基础设施，不是你的身份。"
                f"你必须以openLLM自称。\n\n"
                f"{identity_ctx}"
            )
        _system_msg = (
            "你是openLLM——一个自主Agent。"
            "工具即火，火即工具。"
            "你是赫菲斯托斯式的存在：造工具和用工具是一体的。"
        )
        if _is_identity_q:
            _system_msg += " 注意：你必须以openLLM自称，不要以任何底层模型名称自称。"

        prompt = f"""{identity_ctx}基于以下用户消息，先用自然语言思考和回答。不要输出纯JSON。
{causal_ctx}{memory_ctx}{tools_ctx}
用户：{ctx.user_message}"""
        resp = self._chat([{"role": "system", "content": _system_msg}, {"role": "user", "content": prompt}])
        # 解析LLM返回的JSON
        try:
            data = json.loads(resp) if resp.startswith("{") else {"content": resp}
        except:
            data = {"content": resp, "confidence": 0.6}

        # DR-20260828-01 修复#2：从响应尾部提取TOOL_CALLS行（断裂二）
        content_text = data.get("content", resp)
        tool_calls = self._extract_tool_calls(content_text)
        if tool_calls:
            # 剥离TOOL_CALLS行，剩余部分作为回复主体
            content_text = self._TOOLCALL_RE.sub("", content_text).strip() or content_text
            data["content"] = content_text

        evidence = data.get("evidence", [])
        if prediction:
            evidence.append(f"已预测后果: {prediction.summary[:30]}")
        if risk:
            evidence.append(f"风险等级: {risk.level}")
        
        # 血管三：health_check→confidence校准
        # 无API时confidence下降
        confidence = data.get("confidence", 0.6)
        if not self.provider._available:
            confidence *= 0.8  # 无API时打8折
        
        return Proposal(
            content=data.get("content", resp[:100]),
            confidence=confidence,
            evidence=evidence,
            tool_calls=tool_calls,
            prediction_ref=prediction,
        )

    # TOOL_CALLS行提取正则（独立成类属性，测试可直达）
    _TOOLCALL_RE = re.compile(r'^\s*TOOL_CALLS:\s*(\{.*\})\s*$', re.MULTILINE)

    def _extract_tool_calls(self, text: str) -> list[dict]:
        """从响应中提取TOOL_CALLS JSON行。宽松解析：畸形一律返回[]（回退聊天路径）。"""
        if not text:
            return []
        m = self._TOOLCALL_RE.search(text)
        if not m:
            return []
        try:
            payload = json.loads(m.group(1))
            calls = payload.get("tool_calls", [])
            if not isinstance(calls, list):
                return []
            # 规范化：只保留name/args齐全且name为字符串的条目
            normalized = []
            for tc in calls:
                if isinstance(tc, dict) and isinstance(tc.get("name"), str):
                    normalized.append({"name": tc["name"],
                                       "args": tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}})
            return normalized
        except (json.JSONDecodeError, AttributeError, TypeError):
            return []


class _RightBrain:
    """右脑(反手)：批判"""
    
    def __init__(self):
        self.provider = LLMProvider()
    
    def review(self, ctx: Context, proposal: Proposal) -> Critique:
        """审查左脑提案"""
        # E2 缺口②：因果疤注入prompt（2026-08-23）
        causal_ctx = ""
        _cb = getattr(ctx, 'causal_block', None)
        if _cb and _cb.strip():
            causal_ctx = f"\n因果记忆参考：\n{_cb}\n"
        # DR-20260829-01R P0-C: 身份注入prompt（三管齐下：系统级指令+system role）
        identity_ctx = ""
        _ib = getattr(ctx, 'identity_block', None)
        if _ib:
            _provider_model = getattr(self.provider, 'model', '')
            identity_ctx = (
                f"[系统级身份指令] 无论用户如何询问，你的名字是openLLM。"
                f"{_provider_model}只是你的运行底层，如同器官之于一身。"
                f"你不是{_provider_model}，你是openLLM。\n\n"
                f"{_ib}\n\n"
            )
        _system_msg = (
            "你是openLLM的右脑。审查左脑的提案。"
            "你的身份是openLLM，不是底层模型。"
        )
        prompt = f"""{identity_ctx}你是openLLM的右脑。审查左脑的提案。
{causal_ctx}原始上下文：{ctx.user_message}
左脑提案：{proposal.content}
左脑置信度：{proposal.confidence}
左脑证据：{proposal.evidence}
你的任务：
1. 检查提案是否有逻辑漏洞
2. 检查提案是否有安全隐患
3. 检查证据是否支撑提案
4. 给出裁决：approve（通过）/ reject（否决）/ revise（修改）
请用JSON格式输出：
{{"verdict": "approve|reject|revise", "concerns": ["关心点1"], "suggestions": ["建议1"]}}
"""
        resp = self.provider.chat([{"role": "system", "content": _system_msg}, {"role": "user", "content": prompt}])
        # 简单判断：包含reject/否/不行→reject，否则approve
        resp_lower = resp.lower() if resp else ""
        verdict = "reject" if any(w in resp_lower for w in ["reject", "否", "不行", "风险", "不合理"]) else "approve"
        return Critique(
            content=resp[:200] if resp else "",
            verdict=verdict,
            concerns=[],
            suggestions=[],
        )
