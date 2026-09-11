#!/usr/bin/env python3
"""
debate_trace_extractor.py — 题④裁决内化管道第①步：轨迹提取器
从jiak卡片和河床合议文档提取DebateTrace，输出jsonl。

零LLM、纯规则解析。幂等（processed_index防止重复处理）。

数据源：
1. jiak卡片：~/.hermes/jiak/cards/*.json（decisions/insights字段）
2. 河床合议文档：818具神智能研究/ 目录下七神/五人/终裁/合议 md文件

输出格式（DebateTrace）：
{
    trace_id, source_file, topic, date, source_type,
    agents: [{name, verdict, is_final}],
    consensus_ratio, preference_hint, extracted_at
}

用法：
    python debate_trace_extractor.py [--output OUTPUT] [--dry-run]
"""

import json
import hashlib
import re
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── 常量 ────────────────────────────────────────────────────────────

JIAK_DIR = Path.home() / ".hermes" / "jiak" / "cards"
河床_DIRS = [
    Path("/mnt/i/hermes/output/818具神智能研究"),
    Path("/mnt/i/hermes/output/河床0901ILM进IAI本地微调"),
]
默认输出 = Path(__file__).parent / "debate_traces.jsonl"
INDEX_FILE = Path(__file__).parent / ".debate_trace_index.json"

# 合议类关键词
合议关键词 = re.compile(r"七神|五人|终裁|合议|核战", re.IGNORECASE)

# 七神名称映射
七神全名 = {
    "阿波罗": "阿波罗", "阿波罗(真理)": "阿波罗", "阿波罗·真理": "阿波罗",
    "雅典娜": "雅典娜", "雅典娜(战略)": "雅典娜", "雅典娜·战略": "雅典娜",
    "赫淮斯托斯": "赫淮斯托斯", "赫淮斯托斯(锻造)": "赫淮斯托斯", "赫淮斯托斯·锻造": "赫淮斯托斯",
    "克洛诺斯": "克洛诺斯", "克洛诺斯(时效)": "克洛诺斯", "克洛诺斯·时间": "克洛诺斯",
    "阿佛洛狄忒": "阿佛洛狄忒", "阿佛洛狄忒(人本)": "阿佛洛狄忒", "阿佛洛狄忒·温度": "阿佛洛狄忒",
    "赫尔墨斯": "赫尔墨斯", "赫尔墨斯(接口)": "赫尔墨斯", "赫尔墨斯·连接": "赫尔墨斯",
    "德墨忒尔": "德墨忒尔", "德墨忒尔(验伪)": "德墨忒尔", "德墨忒尔·收获": "德墨忒尔",
    "军师": "军师", "军师终裁": "军师",
}

# 日期提取
日期模式 = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


# ─── 工具函数 ─────────────────────────────────────────────────────────

def 规范化神名(raw: str) -> str:
    """将各种格式的神名统一为标准名"""
    raw = raw.strip().rstrip("：:").strip()
    if raw in 七神全名:
        return 七神全名[raw]
    # 模糊匹配
    for k, v in 七神全名.items():
        if raw in k or k in raw:
            return v
    return raw


def 提取日期(text: str) -> str:
    """从文本提取日期，支持YYYY-MM-DD和YYYYMMDD，返回YYYY-MM-DD"""
    m = 日期模式.search(text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # 备选：YYYYMMDD格式（文件名常见）
    m2 = re.search(r"(\d{4})(\d{2})(\d{2})", text)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if 2020 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    return "unknown"


def 生成trace_id(source: str, topic: str) -> str:
    """基于来源+主题生成唯一ID"""
    h = hashlib.md5(f"{source}:{topic}".encode()).hexdigest()[:12]
    return f"dt_{h}"


def 计算共识比例(agents: list) -> float:
    """有结论的agent数 / 总agent数"""
    if not agents:
        return 0.0
    有结论 = sum(1 for a in agents if a.get("verdict", "").strip())
    return round(有结论 / len(agents), 2)


def 估算偏好(agents: list) -> dict:
    """
    粗略估算偏好对（预览用，不是最终DPO对）。
    多数派同向=chosen，被驳回的=rejected。
    简单规则：verdict中含"✅""真""采纳""支持"的归chosen，含"❌""假""否决"的归rejected。
    """
    chosen_count = 0
    rejected_count = 0
    chosen_agents = []
    rejected_agents = []
    for a in agents:
        v = a.get("verdict", "")
        name = a.get("name", "")
        if any(k in v for k in ["✅", "真", "采纳", "支持", "GO"]):
            chosen_count += 1
            chosen_agents.append(name)
        elif any(k in v for k in ["❌", "假", "否决", "砍掉", "不"]):
            rejected_count += 1
            rejected_agents.append(name)
    return {
        "chosen_count": chosen_count,
        "rejected_count": rejected_count,
        "chosen_agents": chosen_agents,
        "rejected_agents": rejected_agents,
    }


# ─── 河床文档解析 ────────────────────────────────────────────────────

def 解析河床表格行(line: str) -> Optional[dict]:
    """
    解析markdown表格行：| 神名 | ... | 裁决 |
    返回 {name, verdict} 或 None
    """
    line = line.strip()
    if not line.startswith("|"):
        return None
    # 跳过分隔行
    if re.match(r"^\|[\s\-|]+\|$", line):
        return None
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if len(cells) < 2:
        return None
    # 找到神名所在的cell
    神名cell = None
    裁决cell = None
    for i, c in enumerate(cells):
        normalized = 规范化神名(c.split("(")[0].split("·")[0])
        if normalized in 七神全名.values():
            神名cell = normalized
            # 裁决通常是最后一列或包含最多内容的列
            裁决cell = cells[-1] if i != len(cells) - 1 else (cells[-2] if len(cells) > 2 else "")
            break
    if not 神名cell:
        return None
    if not 裁决cell:
        return None
    return {"name": 神名cell, "verdict": 裁决cell.strip()}


def 解析河床标题段(content: str) -> list:
    """
    解析以 ### 标题标记的神名段落
    如：### ☀️ 阿波罗 · 真理
    """
    agents = []
    # 匹配 ### 开头的标题行，提取神名
    标题模式 = re.compile(
        r"#{2,3}\s*[☀⚔🔨⏳💎📨🌾]?\s*(\S+)\s*[·・]\s*(\S+)",
        re.MULTILINE
    )
    段落切分 = re.split(r"(?=^#{2,3}\s)", content, flags=re.MULTILINE)

    for section in 段落切分:
        m = 标题模式.search(section[:200])
        if not m:
            continue
        name = 规范化神名(m.group(1))
        if name not in 七神全名.values():
            continue
        # 提取"核心判断"或"裁决"后面的内容
        verdict = ""
        判断模式 = re.compile(r"(?:核心判断|裁决|判断|立场)[：:]\s*(.+?)(?:\n\n|\n###|\n---|\Z)", re.DOTALL)
        jm = 判断模式.search(section)
        if jm:
            verdict = jm.group(1).strip()[:500]  # 截断过长内容
        else:
            # 取标题后第一段
            lines = section.strip().split("\n")
            第一段 = []
            for line in lines[1:]:
                if line.strip().startswith("#") or line.strip().startswith("|"):
                    break
                if line.strip():
                    第一段.append(line.strip())
            verdict = " ".join(第一段)[:500]
        if verdict:
            agents.append({"name": name, "verdict": verdict, "is_final": False})
    return agents


def 解析河床终裁(content: str) -> Optional[dict]:
    """解析军师终裁段落，返回 {name:'军师', verdict, is_final:True}"""
    # 找终裁/军师终裁段落
    终裁模式 = re.compile(
        r"(?:##\s*(?:二、?|三、?)?\s*(?:军师终裁|终裁|军师裁决)|军师终裁[：:])(.+?)(?=\n##\s|\n---|\Z)",
        re.DOTALL
    )
    m = 终裁模式.search(content)
    if not m:
        # 备选：查找"终裁"附近的判断
        终裁行模式 = re.compile(r"终裁[：:].*?[\n](.+?)(?:\n##|\n---|\Z)", re.DOTALL)
        m = 终裁行模式.search(content)
    if m:
        verdict = m.group(1).strip()[:500]
        if verdict:
            return {"name": "军师", "verdict": verdict, "is_final": True}
    return None


def 提取河床文档(filepath: Path) -> Optional[dict]:
    """从一个河床md文件提取DebateTrace"""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    if len(content) < 100:
        return None

    # 提取主题（文件标题 # 行）
    标题模式 = re.compile(r"^#\s+(.+)", re.MULTILINE)
    tm = 标题模式.search(content)
    topic = tm.group(1).strip() if tm else filepath.stem

    date = 提取日期(content)
    source = str(filepath)

    agents = []

    # 策略1：解析表格
    for line in content.split("\n"):
        parsed = 解析河床表格行(line)
        if parsed:
            # 避免重复
            if not any(a["name"] == parsed["name"] for a in agents):
                parsed["is_final"] = False
                agents.append(parsed)

    # 策略2：解析标题段落（补充表格未覆盖的）
    if len(agents) < 3:
        标题agents = 解析河床标题段(content)
        for a in 标题agents:
            if not any(x["name"] == a["name"] for x in agents):
                agents.append(a)

    # 策略3：解析终裁
    终裁 = 解析河床终裁(content)
    if 终裁:
        agents.append(终裁)
    elif "军师" in content and any(k in content for k in ["终裁", "裁决", "GO", "否决"]):
        # 兜底：文档含军师裁决但未匹配到结构化段落
        # 尝试从标题行或首段提取
        终裁行 = re.search(r"终裁一句[：:]\s*\*\*(.+?)\*\*", content)
        if 终裁行:
            agents.append({"name": "军师", "verdict": 终裁行.group(1)[:500], "is_final": True})

    if not agents:
        return None

    return {
        "trace_id": 生成trace_id(source, topic),
        "source_file": source,
        "topic": topic,
        "date": date,
        "source_type": "河床合议文档",
        "agents": agents,
        "consensus_ratio": 计算共识比例(agents),
        "preference_hint": 估算偏好(agents),
        "extracted_at": datetime.now().isoformat(),
    }


# ─── jiak卡片解析 ─────────────────────────────────────────────────────

def 提取jiak卡片(filepath: Path) -> Optional[dict]:
    """从jiak卡片JSON提取DebateTrace（仅合议类卡片）"""
    try:
        card = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception:
        return None

    title = card.get("title", "")
    decisions = card.get("decisions", [])
    insights = card.get("insights", [])
    tags = card.get("tags", [])
    content = card.get("content", "")

    # 判断是否为合议类卡片
    合议信号 = False
    检查文本 = f"{title} {' '.join(tags)} {content}"
    if 合议关键词.search(检查文本):
        合议信号 = True
    # 有多个decisions的卡片也可能是合议产物
    if len(decisions) >= 5 and any(
        k in 检查文本 for k in ["裁决", "七神", "五人", "启示", "核战", "终裁"]
    ):
        合议信号 = True

    if not 合议信号:
        return None

    if not decisions and not insights:
        return None

    date = 提取日期(str(filepath))
    agents = []

    # 尝试从decisions中提取agent归属
    for d in decisions:
        if isinstance(d, str):
            verdict = d.strip()
        elif isinstance(d, dict):
            verdict = d.get("content", d.get("verdict", ""))
        else:
            continue
        if not verdict:
            continue

        # 尝试从文本中识别agent名
        agent_name = None
        for name in 七神全名.values():
            if name in verdict[:30]:
                agent_name = name
                break
        if not agent_name:
            agent_name = "合议组"

        # 检查是否是终裁
        is_final = bool(re.search(r"军师|终裁", verdict[:30]))

        agents.append({
            "name": agent_name,
            "verdict": verdict[:500],
            "is_final": is_final,
        })

    if not agents:
        return None

    return {
        "trace_id": 生成trace_id(str(filepath), title),
        "source_file": str(filepath),
        "topic": title,
        "date": date,
        "source_type": "jiak卡片",
        "agents": agents,
        "consensus_ratio": 计算共识比例(agents),
        "preference_hint": 估算偏好(agents),
        "extracted_at": datetime.now().isoformat(),
    }


# ─── 批量提取 ─────────────────────────────────────────────────────────

def 加载已处理索引() -> set:
    """加载已处理文件的哈希索引"""
    if INDEX_FILE.exists():
        try:
            data = json.loads(INDEX_FILE.read_text())
            return set(data.get("processed", []))
        except Exception:
            pass
    return set()


def 保存索引(index: set):
    """保存已处理文件索引"""
    INDEX_FILE.write_text(json.dumps({"processed": sorted(index)}, ensure_ascii=False, indent=2))


def 扫描河床文档() -> list:
    """扫描所有河床合议文档"""
    results = []
    seen = set()
    for dir_path in 河床_DIRS:
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("*.md"):
            # 文件名必须包含合议关键词
            if not 合议关键词.search(md_file.name):
                continue
            # 排除设计文档本身
            if "裁决内化设计" in md_file.name:
                continue
            if md_file.name in seen:
                continue
            seen.add(md_file.name)
            result = 提取河床文档(md_file)
            if result:
                results.append(result)
    return results


def 扫描jiak卡片() -> list:
    """扫描jiak卡片"""
    results = []
    if not JIAK_DIR.exists():
        return results
    for json_file in JIAK_DIR.glob("*.json"):
        result = 提取jiak卡片(json_file)
        if result:
            results.append(result)
    return results


def 运行(output_path: Optional[str] = None, dry_run: bool = False):
    """主提取流程"""
    output = Path(output_path) if output_path else 默认输出
    已处理 = 加载已处理索引()

    print(f"📂 扫描河床合议文档...")
    河床traces = 扫描河床文档()
    print(f"   → 河床文档提取: {len(河床traces)} 条Trace")

    print(f"📂 扫描jiak卡片...")
    jiaktraces = 扫描jiak卡片()
    print(f"   → jiak卡片提取: {len(jiaktraces)} 条Trace")

    all_traces = 河床traces + jiaktraces

    # 幂等：过滤已处理
    新traces = []
    for t in all_traces:
        if t["trace_id"] not in 已处理:
            新traces.append(t)
            已处理.add(t["trace_id"])

    print(f"\n📊 提取统计:")
    print(f"   处理河床文档: {len(河床traces)} 条")
    print(f"   处理jiak卡片: {len(jiaktraces)} 条")
    print(f"   新增Trace: {len(新traces)} 条")
    print(f"   去重后总计: {len(已处理)} 条（含历史）")

    if 新traces:
        # 计算统计
        agent_counts = [len(t["agents"]) for t in 新traces]
        avg_agents = sum(agent_counts) / len(agent_counts) if agent_counts else 0
        print(f"   平均agents数: {avg_agents:.1f}")
        print(f"   最大agents数: {max(agent_counts) if agent_counts else 0}")
        print(f"   有军师终裁: {sum(1 for t in 新traces if any(a.get('is_final') for a in t['agents']))} 条")

        # 抽样展示
        print(f"\n📋 抽样展示（第1条）:")
        sample = 新traces[0]
        print(f"   trace_id: {sample['trace_id']}")
        print(f"   topic: {sample['topic'][:80]}")
        print(f"   date: {sample['date']}")
        print(f"   source_type: {sample['source_type']}")
        print(f"   agents数: {len(sample['agents'])}")
        print(f"   consensus_ratio: {sample['consensus_ratio']}")
        for a in sample['agents'][:3]:
            final标记 = " [终裁]" if a.get('is_final') else ""
            print(f"     · {a['name']}{final标记}: {a['verdict'][:80]}...")

    if dry_run:
        print("\n🔍 Dry-run模式，不写文件")
        return

    # 写jsonl
    with open(output, "a", encoding="utf-8") as f:
        for t in 新traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 更新索引
    保存索引(已处理)
    print(f"\n✅ 输出: {output} ({len(新traces)} 条新Trace)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从jiak卡片和河床合议文档提取DebateTrace")
    parser.add_argument("--output", "-o", help="输出jsonl路径")
    parser.add_argument("--dry-run", action="store_true", help="只扫描不写文件")
    args = parser.parse_args()
    运行(output_path=args.output, dry_run=args.dry_run)
