"""
SessionCausalExtractor — 会话结束后的因果链提取器
====================================================

每次会话结束后，调用LLM从JCoT推理链中提取因果关系。
这是ISA因果记忆链的"离线训练"。

在线：AutoCausalWriter（实时，单条，粗粒度）
离线：SessionCausalExtractor（会话后，批量，细粒度）

流程：
  会话中 → JCoT记录推理链（reasoning_chains表）
  会话后 → 本模块读取JCoT → LLM提取因果链 → 写入CausalMemoryStore
  定期   → memory_extractor从CausalMemoryStore提取高价值 → jage

Usage:
    extractor = SessionCausalExtractor()
    result = extractor.extract_from_session("20260815_111357_d37564")
    print(f"提取了 {result['causal_chains']} 条因果链")
"""

import json
import time
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from .causal_memory import TrustLevel
from dataclasses import dataclass, asdict

logger = logging.getLogger("openllm.session_causal_extractor")


@dataclass
class CausalChain:
    """从会话中提取的因果链"""
    session_id: str
    prediction: str      # 之前预测了什么
    actual: str          # 实际发生了什么
    delta: str           # 差距是什么
    lesson: str          # 下次应该怎么做
    confidence: float    # 0-1
    source_jcot_ids: list  # 来源的JCoT ID
    extracted_at: float


class SessionCausalExtractor:
    """
    会话结束后的因果链提取器。
    
    从JCoT推理链中提取因果关系，写入CausalMemoryStore。
    """
    
    def __init__(self):
        self.jcot_db = Path.home() / ".hermes" / "jiak" / "jcot.db"
        # 延迟导入，避免循环依赖
        self._causal_store = None
    
    def _get_causal_store(self):
        if self._causal_store is None:
            from .causal_memory import get_causal_store
            self._causal_store = get_causal_store()
        return self._causal_store
    
    def get_session_jcot(self, session_id: str) -> List[Dict]:
        """获取指定会话的所有JCoT推理链"""
        if not self.jcot_db.exists():
            logger.warning(f"JCoT数据库不存在: {self.jcot_db}")
            return []
        
        conn = sqlite3.connect(str(self.jcot_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # 查reasoning_chains表
        chains = []
        try:
            cur.execute(
                "SELECT * FROM reasoning_chains WHERE session LIKE ?",
                (f"%{session_id}%",)
            )
            for row in cur.fetchall():
                chains.append(dict(row))
        except Exception as e:
            logger.warning(f"查询reasoning_chains失败: {e}")
        
        # 也查reasoning_paths表
        try:
            cur.execute(
                "SELECT * FROM reasoning_paths WHERE session_id = ?",
                (session_id,)
            )
            for row in cur.fetchall():
                chains.append(dict(row))
        except Exception as e:
            logger.warning(f"查询reasoning_paths失败: {e}")
        
        conn.close()
        return chains
    
    def build_extraction_prompt(self, jcot_chains: List[Dict], session_context: str = "") -> str:
        """构造LLM提取prompt"""
        
        chains_text = ""
        for i, chain in enumerate(jcot_chains[:20], 1):  # 最多20条
            topic = chain.get("topic", chain.get("trigger", "unknown"))
            conclusion = chain.get("conclusion", "")
            reasoning = chain.get("reasoning_path", chain.get("reasoning_steps", ""))
            confidence = chain.get("confidence", 0.5)
            
            if isinstance(reasoning, list):
                reasoning = " → ".join(reasoning)
            
            chains_text += f"""
JCoT #{i}: {topic}
  推理: {str(reasoning)[:300]}
  结论: {str(conclusion)[:200]}
  置信度: {confidence}
"""
        
        prompt = f"""你是一个因果关系提取器。从以下会话的推理链中，提取因果关系。

会话上下文: {session_context[:500] if session_context else "无"}

推理链记录:
{chains_text}

请提取因果关系。每个因果关系必须包含:
1. prediction: 之前预测了什么（一句话）
2. actual: 实际发生了什么（一句话）
3. delta: 预测和实际的差距（一句话）
4. lesson: 下次应该怎么做（一句话）
5. confidence: 0-1的置信度
6. source: 来源的JCoT编号

输出JSON数组，每个元素包含上述字段。只提取有明确因果关系的——如果某个推理链只是"做了X"没有因果含义，跳过。

输出格式:
[
  {{
    "prediction": "...",
    "actual": "...",
    "delta": "...",
    "lesson": "...",
    "confidence": 0.8,
    "source": "JCoT #1, #3"
  }}
]
"""
        return prompt
    
    def extract_with_llm(self, prompt: str) -> List[Dict]:
        """调用LLM提取因果链"""
        # 尝试用openLLM的engine
        try:
            from openllm.core.engine import OpenLLMEngine, AgentConfig
            engine = OpenLLMEngine(AgentConfig(
                provider="qwen",
                name="causal-extractor",
                security_level=3,
                model="qwen3.8-max",
                max_context_tokens=4096,
            ))
            response = engine.chat(prompt)
            
            # 解析JSON响应——从后往前找最后一个完整的JSON数组
            # Qwen的思考过程可能在JSON前后包裹大量文本
            import re
            # 找所有可能的JSON数组
            arrays = re.findall(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            if arrays:
                # 取最后一个（通常是最终输出）
                for arr in reversed(arrays):
                    try:
                        parsed = json.loads(arr)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # 降级：找最后一个 { 开始的JSON对象
            last_brace = response.rfind('{')
            if last_brace >= 0:
                # 尝试从最后一个 { 到结尾
                candidate = response[last_brace:]
                # 找匹配的 ]
                bracket_end = candidate.rfind(']')
                if bracket_end >= 0:
                    candidate = candidate[:bracket_end+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            return parsed
                        elif isinstance(parsed, dict):
                            return [parsed]
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning(f"LLM提取失败: {e}")
        
        # 降级：用规则提取
        return []
    
    def extract_from_session(self, session_id: str, session_context: str = "") -> Dict:
        """
        从指定会话中提取因果链。
        
        Args:
            session_id: 会话ID
            session_context: 会话上下文（可选）
        
        Returns:
            {causal_chains: int, written: int, details: [...]}
        """
        logger.info(f"开始提取会话 {session_id} 的因果链...")
        
        # 1. 读取JCoT
        jcot_chains = self.get_session_jcot(session_id)
        if not jcot_chains:
            logger.info(f"会话 {session_id} 没有JCoT记录")
            return {"causal_chains": 0, "written": 0, "details": []}
        
        logger.info(f"找到 {len(jcot_chains)} 条JCoT记录")
        
        # 2. 构造prompt
        prompt = self.build_extraction_prompt(jcot_chains, session_context)
        
        # 3. 调用LLM
        causal_chains = self.extract_with_llm(prompt)
        if not causal_chains:
            logger.info("LLM未提取到因果链（可能没有明确因果关系）")
            return {"causal_chains": 0, "written": 0, "details": []}
        
        logger.info(f"LLM提取了 {len(causal_chains)} 条因果链")
        
        # 4. 写入CausalMemoryStore
        store = self._get_causal_store()
        written = 0
        details = []
        
        for chain in causal_chains:
            try:
                # 解析source字段获取JCoT ID
                source_str = chain.get("source", "")
                source_ids = []
                for part in source_str.split("#"):
                    part = part.strip().rstrip(",")
                    if part.isdigit():
                        source_ids.append(int(part))
                
                confidence = chain.get("confidence", 0.5)
                store.store(
                    action_signature=f"session:{session_id}",
                    context_features=["session_causal_extract"],
                    prediction=chain.get("prediction", ""),
                    prediction_confidence=confidence,
                    actual_result=chain.get("actual", ""),
                    actual_success=confidence > 0.5,
                    delta=chain.get("delta", ""),
                    delta_magnitude=1.0 - confidence,
                    lesson=chain.get("lesson", ""),
                    source=f"session_causal_extractor:{session_id}",
                    trust_level=TrustLevel.INTERNAL,
                )
                written += 1
                details.append({
                    "prediction": chain.get("prediction", "")[:100],
                    "lesson": chain.get("lesson", "")[:100],
                    "confidence": chain.get("confidence", 0.5),
                })
            except Exception as e:
                logger.warning(f"写入因果链失败: {e}")
        
        result = {
            "causal_chains": len(causal_chains),
            "written": written,
            "details": details,
        }
        
        logger.info(f"会话 {session_id} 因果提取完成: {len(causal_chains)}条提取, {written}条写入")
        return result
    
    def extract_recent(self, hours: int = 24, limit: int = 5) -> List[Dict]:
        """提取最近N小时内的会话的因果链"""
        # 从reasoning_chains中找最近的会话
        if not self.jcot_db.exists():
            return []
        
        conn = sqlite3.connect(str(self.jcot_db))
        cur = conn.cursor()
        
        cutoff = time.time() - hours * 3600
        cur.execute(
            "SELECT DISTINCT session FROM reasoning_chains WHERE timestamp > ? LIMIT ?",
            (str(cutoff), limit)
        )
        sessions = [row[0] for row in cur.fetchall()]
        conn.close()
        
        results = []
        for session_id in sessions:
            result = self.extract_from_session(session_id)
            results.append({"session_id": session_id, **result})
        
        return results


# ── CLI入口 ──

if __name__ == "__main__":
    import sys
    
    extractor = SessionCausalExtractor()
    
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
        result = extractor.extract_from_session(session_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("用法: python session_causal_extractor.py <session_id>")
        print("示例: python session_causal_extractor.py 20260815_111357_d37564")
