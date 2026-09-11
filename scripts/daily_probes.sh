#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# daily_probes.sh — 三角制衡三探针每日自动 emit
# 依次跑：rule_change_probe → knowledge_monotonic → triad_coupling_probe
# 每步 try/except 包裹，一个挂不影响其他（降级铁律）
# 输出追加到 ~/.openllm/evolution/daily_probes.log
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

PROJ_DIR="/home/zcs/projects/openllm"
PYTHON="${PROJ_DIR}/.venv/bin/python"
export PYTHONPATH="${PROJ_DIR}/src:${PYTHONPATH:-}"

LOG_DIR="${HOME}/.openllm/evolution"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/daily_probes.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "═══ daily_probes.sh START $(ts) ═══" | tee -a "${LOG}"

FAILURES=0

# ── Probe 1: rule_change_probe (R度量) ──
echo "[1/3] rule_change_probe (R度量) ..." | tee -a "${LOG}"
if ${PYTHON} -m openllm.evolution.rule_change_probe --days 7 --emit 2>&1 | tee -a "${LOG}"; then
    echo "[1/3] rule_change_probe: OK" | tee -a "${LOG}"
else
    echo "[1/3] rule_change_probe: FAILED (exit $?)" | tee -a "${LOG}"
    FAILURES=$((FAILURES + 1))
fi

# ── Probe 2: knowledge_monotonic (K度量) ──
# 无 CLI main()，内联 Python 调用
echo "[2/3] knowledge_monotonic (K度量) ..." | tee -a "${LOG}"
if ${PYTHON} -c "
from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
probe = KnowledgeMonotonicProbe()
snap = probe.take_snapshot()
report = probe.check_monotonic()
result = probe.emit_report()
print(f'K度量: verdict={report.verdict}')
print(f'  capacity_delta={report.capacity_delta}')
print(f'  recall={report.recall} accuracy={report.accuracy_sample}')
print(f'  依据: {report.detail}')
print(f'  emit: {result}')
" 2>&1 | tee -a "${LOG}"; then
    echo "[2/3] knowledge_monotonic: OK" | tee -a "${LOG}"
else
    echo "[2/3] knowledge_monotonic: FAILED (exit $?)" | tee -a "${LOG}"
    FAILURES=$((FAILURES + 1))
fi

# ── Probe 3: triad_coupling_probe (三角制衡 T4) ──
# 消费前两者的当日输出，最后跑
echo "[3/3] triad_coupling_probe (三角制衡 T4) ..." | tee -a "${LOG}"
if ${PYTHON} -m openllm.evolution.triad_coupling_probe --days 30 --emit 2>&1 | tee -a "${LOG}"; then
    echo "[3/3] triad_coupling_probe: OK" | tee -a "${LOG}"
else
    echo "[3/3] triad_coupling_probe: FAILED (exit $?)" | tee -a "${LOG}"
    FAILURES=$((FAILURES + 1))
fi

# ── 汇总 ──
echo "═══ daily_probes.sh END $(ts) | failures=${FAILURES}/3 ═══" | tee -a "${LOG}"
exit ${FAILURES}
