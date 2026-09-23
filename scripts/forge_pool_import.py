#!/usr/bin/env python
"""
forge_pool_import — 能力池初始摄入（一次性/可重跑）
===================================================
把 Hermes 全部 skill（rglob 含嵌套分类目录）+ openLLM 全部工具（在册27+库存4）
摄入 ~/.openllm/pool/。池子格式是 openLLM 自己的 PoolEntry，不寄生 Hermes。

用法：
    cd /mnt/i/openllm && OPENLLM_SECURITY_LEVEL=3 .venv/bin/python scripts/forge_pool_import.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENLLM_SECURITY_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openllm.isn.capability_pool import CapabilityPool, PoolEntry

HERMES_SKILLS = Path.home() / ".hermes" / "skills"

# openLLM 在册工具（create_default_tools 27件）—— 描述与 executor 同源
OPENLLM_TOOLS = [
    ("read_file", "读取文件内容"), ("write_file", "写入文件"),
    ("shell", "执行Shell命令"), ("search", "搜索文件内容"),
    ("list_dir", "列出目录内容"), ("python_exec", "执行Python代码"),
    ("octopus_search", "搜索章鱼记忆+外部搜索（v4黑匣子记录）"),
    ("octopus_self_model", "查看章鱼自省状态"),
    ("octopus_search_stats", "搜索黑匣子统计——后端成功率/延迟分布"),
    ("octopus_route", "根据查询推荐最优搜索后端（MAB策略）"),
    ("octopus_health", "章鱼搜索系统健康检查"),
    ("curl_impersonate", "浏览器TLS指纹模拟——绕过反爬检测(Chrome/Firefox/Safari)"),
    ("hermes_search", "Hermes Search v7.0——多源聚合搜索(cnscrape/arXiv/web)"),
    ("tool_failure_log", "工具失败日志——记录失败→统计→需求信号(失败驱动锻造)"),
    ("daily_health_check", "搜索后端每日健康报告"),
    ("tool_hunter", "失败驱动工具猎手——从失败信号搜GitHub找替代品"),
    ("auto_forge", "工厂自动锻造——评估GitHub项目→生成Hermes工具框架"),
    ("resilience", "韧性模块——错误分类+指数退避+URL去重(从Crawlee移植)"),
    ("loop_detector", "循环检测器——防止Agent卡死无限重试(从browser-use移植)"),
    ("snapshot_enhancer", "结构化元素索引——解析ariaSnapshot生成可交互元素列表"),
    ("doc_parser", "文档结构化解析——表格→Markdown、公式→LaTeX、图表→语义描述"),
    ("fcrawl", "Fcrawl网页爬虫引擎——本地免费零API Key。scrape/crawl/map/search/batch/agent"),
    ("crawl4ai", "浏览器级网页抓取——Playwright驱动JS渲染+反爬，输出LLM原生Markdown"),
    ("ocr", "长文档OCR——百度Unlimited-OCR(R-SWA机制)，几十页PDF一次解析"),
    ("memory_write", "记忆写入——写当前会话胶囊"),
    ("memory_read", "记忆读取——读会话记忆"),
    ("memory_search", "记忆搜索——跨胶囊检索"),
]
# 库存武器（2026-09-22 已上膛：inventory_tools.py包成8件工具，tool_load可装载）
INVENTORY_TOOLS = [
    ("pk_search", "检索论文知识库(query)——标题/摘要/概念匹配"),
    ("pk_summary", "论文知识库摘要()——论文数/概念数/主题分布"),
    ("rl_step", "研究循环推进(action,claim,prediction,...)——假设/观察/设计/运行/结论/迭代"),
    ("rl_status", "研究循环状态()——阶段/迭代数/当前假设"),
    ("ee_hypothesis", "实验假设管理(action,id,claim,prediction,...)——add/list/record/summary"),
    ("ee_summary", "实验引擎摘要()——假设数/证实反驳分布"),
    ("se_predict", "登记预测(card_id,prediction,confidence)——技能进化校准链入口"),
    ("se_confidence", "查询进化置信度(card_id)——单卡或全局状态"),
]


def main() -> None:
    pool = CapabilityPool()
    report = {"skills": 0, "tools": 0, "skipped": []}

    # 1. Hermes skills（rglob 含嵌套分类目录，frontmatter name 为准）
    seen = set()
    for skill_md in sorted(HERMES_SKILLS.rglob("SKILL.md")):
        if "__pycache__" in skill_md.parts or "_quarantine" in skill_md.parts:
            continue
        meta = CapabilityPool._parse_frontmatter(
            skill_md.read_text(encoding="utf-8", errors="replace"))
        name = meta.get("name") or skill_md.parent.name
        if not meta.get("description") or name in seen:
            report["skipped"].append(name)
            continue
        seen.add(name)
        tags = meta.get("tags", "")
        tag_list = [t.strip() for t in str(tags).replace("[", "").replace("]", "")
                    .split(",") if t.strip()] if tags else []
        pool.upsert(PoolEntry(
            kind="skill", name=name,
            description=meta["description"][:500],
            version=meta.get("version", "0.0.0"),
            domain_tags=tag_list,
            source_path=str(skill_md.parent),
            source_framework="hermes",
        ))
        report["skills"] += 1

    # 2. openLLM 工具（在册 + 库存）
    for name, desc in OPENLLM_TOOLS:
        pool.upsert(PoolEntry(kind="tool", name=name, description=desc,
                              source_framework="openllm",
                              source_path="/mnt/i/openllm/src/openllm/tools/executor.py",
                              extra={"status": "active"}))
        report["tools"] += 1
    for name, desc in INVENTORY_TOOLS:
        pool.upsert(PoolEntry(kind="tool", name=name, description=desc,
                              source_framework="openllm",
                              source_path=f"/mnt/i/openllm/src/openllm/tools/{name}.py",
                              extra={"status": "inventory未上膛"}))
        report["tools"] += 1

    print(f"skills入池: {report['skills']}  tools入池: {report['tools']}  "
          f"跳过: {len(report['skipped'])}")
    print("池子统计:", pool.stats())
    bad = [s for s in report["skipped"] if ":" in s or "无description" in s]
    if bad:
        print("真跳过(无description等):", bad[:10])


if __name__ == "__main__":
    main()
