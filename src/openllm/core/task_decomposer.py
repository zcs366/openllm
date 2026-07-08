"""task_decomposer.py — IAX任务分解器（Layer 13 PEEU）

将复杂任务按规则拆解为子任务DAG，驱动并行/串行执行。

分解规则（纯规则，零LLM调用）：
  - 含"和"/"以及" → 拆分为并列子任务
  - 含"然后"/"之后" → 建立先后依赖
  - 含"测试"/"验证" → 标记为最后一个子任务（依赖前面全部完成）
  - 无明显拆分信号 → 单任务不拆分

持久化状态到 ~/.hermes/core/task_decomposer_state.json
"""

import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("openllm.task_decomposer")

# ── 状态枚举 ──

class TaskStatus(str, Enum):
    """子任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ── 数据类 ──

@dataclass
class SubTask:
    """分解后的子任务"""
    task_id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None


# ── 分解器 ──

class TaskDecomposer:
    """IAX任务分解器 — 规则驱动，零LLM调用。

    用法::

        td = TaskDecomposer()
        tasks = td.decompose("设计架构 和 编写代码 然后 测试验证")
        batches = td.get_execution_order(tasks)
        for batch in batches:
            # batch 中的任务可并行执行
            ...
            td.record_completion(task_id, result="ok")
        progress = td.get_status()
    """

    _STATE_DIR = Path.home() / ".hermes" / "core"
    _STATE_FILE = _STATE_DIR / "task_decomposer_state.json"

    def __init__(self) -> None:
        self._tasks: dict[str, SubTask] = {}
        self._original_task: str = ""
        self._load_state()

    # ── 公开 API ──

    def decompose(self, complex_task: str, context: dict | None = None) -> list[SubTask]:
        """将复杂任务分解为子任务列表。

        Args:
            complex_task: 原始任务描述
            context: 可选上下文信息（暂未使用，预留扩展）

        Returns:
            子任务列表（已注入依赖关系）
        """
        self._original_task = complex_task
        self._tasks.clear()

        # 拆分并列任务（"和"/"以及"）
        parallel_parts = re.split(r'\s+和\s+|\s+以及\s+', complex_task)

        all_parts: list[tuple[str, list[str]]] = []  # (描述, 依赖前缀)
        for part in parallel_parts:
            # 拆分串行步骤（"然后"/"之后"）
            serial_steps = re.split(r'\s+然后\s+|\s+之后\s+', part.strip())
            for i, step in enumerate(serial_steps):
                step = step.strip()
                if not step:
                    continue
                # 当前步骤依赖同一并列块中它之前的步骤
                deps = []
                if i > 0 and all_parts:
                    # 依赖同块内前一步
                    deps.append(all_parts[-1][0])
                all_parts.append((step, deps))

        # 生成 SubTask
        tasks: list[SubTask] = []
        verify_ids: list[str] = []

        for desc, deps_desc in all_parts:
            tid = f"sub_{uuid4().hex[:8]}"
            task = SubTask(task_id=tid, description=desc)
            tasks.append(task)

            # 记录依赖（用描述匹配，后面绑定ID）
            tasks[-1]._pending_deps = deps_desc  # type: ignore[attr-defined]

            # 标记"测试/验证"类任务
            if re.search(r'测试|验证|check|verify', desc, re.IGNORECASE):
                verify_ids.append(tid)

        # 绑定依赖 ID（用描述查找对应 task_id）
        desc_to_id = {t.description: t.task_id for t in tasks}
        for t in tasks:
            pending = getattr(t, '_pending_deps', [])
            for dep_desc in pending:
                if dep_desc in desc_to_id:
                    t.dependencies.append(desc_to_id[dep_desc])
            if hasattr(t, '_pending_deps'):
                delattr(t, '_pending_deps')

        # 验证类任务依赖所有非验证任务
        if verify_ids:
            non_verify = [t.task_id for t in tasks if t.task_id not in verify_ids]
            for vid in verify_ids:
                task = self._tasks.get(vid) or next((t for t in tasks if t.task_id == vid), None)
                if task:
                    task.dependencies.extend(non_verify)
                    task.dependencies = list(set(task.dependencies))

        # 注册到内部状态
        for t in tasks:
            self._tasks[t.task_id] = t

        self._save_state()
        logger.info("分解完成: %s → %d 个子任务", complex_task[:30], len(tasks))
        return tasks

    def build_dag(self, tasks: list[SubTask] | None = None) -> dict[str, list[str]]:
        """构建依赖关系图。

        Returns:
            {task_id: [依赖的task_id列表]}
        """
        src = tasks or list(self._tasks.values())
        return {t.task_id: list(t.dependencies) for t in src}

    def get_execution_order(self, tasks: list[SubTask] | None = None) -> list[list[str]]:
        """拓扑排序，返回可并行执行的批次。

        Returns:
            批次列表，每个批次内的任务 ID 可并行执行
        """
        src = {t.task_id: t for t in (tasks or list(self._tasks.values()))}
        in_degree: dict[str, int] = {tid: 0 for tid in src}
        dependents: dict[str, list[str]] = defaultdict(list)

        for tid, t in src.items():
            for dep in t.dependencies:
                if dep in src:
                    in_degree[tid] += 1
                    dependents[dep].append(tid)

        batches: list[list[str]] = []
        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)

        while queue:
            batch = list(queue)
            batches.append(batch)
            queue.clear()
            for tid in batch:
                for dep_tid in dependents.get(tid, []):
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        queue.append(dep_tid)

        return batches

    def record_completion(self, task_id: str, result: str = "") -> None:
        """记录子任务完成状态。

        Args:
            task_id: 子任务ID
            result: 完成结果描述
        """
        task = self._tasks.get(task_id)
        if not task:
            logger.warning("未知 task_id: %s", task_id)
            return
        task.status = TaskStatus.DONE
        task.result = result
        self._save_state()
        logger.info("子任务完成: %s → %s", task_id, result[:30] if result else "(无)")

    def record_failure(self, task_id: str, reason: str = "") -> None:
        """记录子任务失败。"""
        task = self._tasks.get(task_id)
        if not task:
            return
        task.status = TaskStatus.FAILED
        task.result = reason
        self._save_state()

    def get_status(self) -> dict:
        """获取整体进度摘要。

        Returns:
            {total, done, failed, pending, percent, batches_remaining}
        """
        all_tasks = list(self._tasks.values())
        total = len(all_tasks)
        done = sum(1 for t in all_tasks if t.status == TaskStatus.DONE)
        failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)
        pending = total - done - failed

        # 计算剩余批次
        remaining = [t for t in all_tasks if t.status == TaskStatus.PENDING]
        remaining_batches = len(self.get_execution_order(remaining)) if remaining else 0

        return {
            "total": total,
            "done": done,
            "failed": failed,
            "pending": pending,
            "percent": round(done / total * 100, 1) if total else 0,
            "batches_remaining": remaining_batches,
        }

    def reset(self) -> None:
        """清空所有子任务状态。"""
        self._tasks.clear()
        self._original_task = ""
        self._save_state()

    # ── 持久化 ──

    def _save_state(self) -> None:
        """持久化到 JSON 文件。"""
        self._STATE_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "original_task": self._original_task,
            "tasks": {
                tid: {
                    "task_id": t.task_id,
                    "description": t.description,
                    "dependencies": t.dependencies,
                    "status": t.status.value,
                    "result": t.result,
                }
                for tid, t in self._tasks.items()
            },
        }
        try:
            self._STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        except OSError as exc:
            logger.error("持久化失败: %s", exc)

    def _load_state(self) -> None:
        """从 JSON 文件恢复状态。"""
        if not self._STATE_FILE.exists():
            return
        try:
            data = json.loads(self._STATE_FILE.read_text())
            self._original_task = data.get("original_task", "")
            for tid, info in data.get("tasks", {}).items():
                self._tasks[tid] = SubTask(
                    task_id=info["task_id"],
                    description=info["description"],
                    dependencies=info.get("dependencies", []),
                    status=TaskStatus(info.get("status", "pending")),
                    result=info.get("result"),
                )
            logger.debug("从 %s 恢复了 %d 个子任务", self._STATE_FILE, len(self._tasks))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("加载状态失败，使用空状态: %s", exc)
