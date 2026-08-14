# openLLM Codebase Audit — Peer Review Report

**Reviewer:** Senior Code Auditor (independent verification)  
**Auditor under review:** 军师 (original 17-issue audit)  
**Date:** 2026-07-22  
**Scope:** Verify 11 P0/P1 claims + independent security review

---

## Summary

| Rating | Count |
|--------|-------|
| CONFIRMED | 8 |
| DOWNGRADED | 2 |
| DISPUTED | 0 |
| NEW (missed by auditor) | 5 |

**Overall verdict:** The original audit is solid. 8/11 findings confirmed. The auditor correctly identified the most critical security issues. However, 2 findings were overstated and 5 significant issues were missed.

---

## Finding-by-Finding Verification

### Finding #1 — Right brain verdict uses string matching instead of JSON parsing
**octopus_impl.py:207-208** (auditor cited line 188 — off by ~20 lines)  
**Rating: CONFIRMED**

The right brain prompt explicitly asks for JSON output:
```
请用JSON格式输出：
{"verdict": "approve|reject|revise", "concerns": [...], "suggestions": [...]}
```

But the response handling (lines 207-208) ignores the JSON entirely:
```python
resp_lower = resp.lower() if resp else ""
verdict = "reject" if any(w in resp_lower for w in ["reject", "否", "不行", "风险", "不合理"]) else "approve"
```

**Additional sub-issue:** The code asks for 3 verdicts (approve/reject/revise) but can only return 2 (approve/reject). The "revise" path is dead code. Concerns and suggestions are always empty lists despite being in the prompt.

**Impact:** LLM response parsing is completely wasted. The right brain's structured output is reduced to keyword matching.

---

### Finding #2 — _terminal_real() has no dangerous command check
**isn_impl.py:233-241**  
**Rating: CONFIRMED (escalated to NEW-P0: worse than claimed)**

`_terminal_real()` runs `subprocess.run(command, shell=True)` with **zero** safety checks — no dangerous command list, no sandbox path validation.

The auditor claimed the verify hook mitigates this, but **the verify hook is dead code**:

1. ISN registers `_terminal_real` in ToolRegistry (line 32-33)
2. ToolRegistry._requires_verify = `{"write_file", "shell"}` (executor.py:42)
3. The tool is registered as `"terminal"`, NOT `"shell"`
4. Therefore: `tool_name in self._requires_verify` → **False** for "terminal"
5. The verify hook **never fires** for terminal commands

Furthermore, ISN._execute_action (line 128) calls `self.tools[tool_name]` directly, not `self._tool_registry.execute()`. The entire ToolRegistry path is bypassed.

**Verify hook itself is also weaker than _terminal():**
| Check | _terminal() | _verify_before_complete |
|-------|-------------|------------------------|
| rm | ✅ blocks "rm" | ❌ only blocks "rm -rf" |
| redirect | ✅ blocks "> " | ❌ only blocks "> /dev" |
| dd | ✅ blocks "dd" | ❌ only blocks "dd if=" |

**Security impact:** HIGH — via ToolRegistry (if ever used), `_terminal_real` can execute arbitrary shell commands with no safety checks. The `_terminal` function (which IS used) has basic checks but still uses `shell=True`.

---

### Finding #3 — _search_files() searches entire HOME directory
**isn_impl.py:182-190**  
**Rating: CONFIRMED**

```python
result = subprocess.run(
    ["find", str(Path.home()), "-name", pattern, "-maxdepth", "4"],
    capture_output=True, text=True, timeout=10
)
```

Searches the user's entire home directory (4 levels deep) without any sandbox restriction. The sandbox.check_path() is never called. The `pattern` argument is user-controlled but since this uses list-form subprocess (not shell=True), shell injection isn't possible — however, the search scope is a privacy concern.

**Impact:** MEDIUM — can enumerate any file in the user's home directory tree.

---

### Finding #4 — build_context() creates new UnifiedMemory/CausalMemoryStore every call
**isa_impl.py:47-56, 72-79**  
**Rating: CONFIRMED (escalated — worse than claimed)**

Each `build_context()` call creates:
- `UnifiedMemory()` — creates 4 directories + loads ALL memories from 3 disk layers (hot/warm/cold)
- `CausalMemoryStore()` — creates directories + loads ALL causal memories and patterns from disk

These are **expensive operations involving disk I/O** (reading JSON files, glob patterns, directory creation). And `build_context()` is called **3 times per tick** (PERCEIVE, DECIDE, LEARN phases).

**Total per-tick cost:** 3 × (UnifiedMemory init + CausalMemoryStore init) = 6 expensive disk I/O operations just for context building.

**Impact:** HIGH — performance degradation that scales with memory size. Should use singleton/cached instances.

---

### Finding #5 — health_check() calls LLM ping every tick
**octopus_impl.py:36**  
**Rating: CONFIRMED**

```python
_ = self.left.provider.chat([{"role":"user","content":"ping"}])
```

Every health_check makes a real LLM API call. No caching, no TTL, no rate limiting.

**Impact:** MEDIUM — unnecessary API cost on every tick.

---

### Finding #6 — health_check() + d0_snapshot() = 2 API calls per tick
**main_loop.py:92**  
**Rating: DOWNGRADED**

The auditor claims 2 API calls from this combination, but `d0_snapshot()` internally calls `health_check()`, so it's 1 API call, not 2.

**However, the real picture is worse:** health_check is called via d0_snapshot from:
1. build_context (PERCEIVE) → d0_snapshot → health_check
2. build_context (DECIDE) → d0_snapshot → health_check  
3. d0_snapshot directly in _decide
4. build_context (LEARN) → d0_snapshot → health_check

**Total health_check API calls per tick: 4** (not 2 as claimed).

**Impact:** The auditor's specific number is wrong but the underlying issue is actually more severe. Should be escalated.

---

### Finding #7 — Tool list hardcoded, not from ISN.tools
**isa_impl.py:59-61**  
**Rating: CONFIRMED**

```python
tools = ["read_file", "search_files"]
if octopus and octopus.left.provider._available:
    tools.extend(["write_file", "terminal"])
```

ISN.tools has the actual tool registry (`{"read_file", "write_file", "search_files", "terminal"}`). ISA doesn't consult it. If ISN adds new tools, ISA won't know about them.

**Impact:** LOW-MEDIUM — architectural coupling issue. Tools listed in context don't match actual capabilities.

---

### Finding #8 — _load_learned_skills() reads 3 JSONL files every init
**isn_impl.py:42-103**  
**Rating: DOWNGRADED**

Three files are read: learned_skills.jsonl, harness_proposals.jsonl, governance_rules.jsonl. However, this only happens at ISN.__init__, which runs once per Agent lifecycle (not per tick). For a long-running agent, this is negligible.

**Impact:** LOW — one-time init cost, not per-tick.

---

### Finding #9 — compare() uses string matching for simulation detection
**octopus_impl.py:89**  
**Rating: CONFIRMED**

```python
match = result.success and "模拟" not in result.output
```

Any output containing the Chinese word "模拟" (simulation) would be treated as simulated, regardless of context. This is fragile and could produce false negatives (if the agent discusses simulation in legitimate output).

**Impact:** MEDIUM — incorrect causal learning results.

---

### Finding #10 — DECIDE calls build_context() again (redundant with PERCEIVE)
**agent_heartbeat.py:105**  
**Rating: CONFIRMED**

PERCEIVE calls `agent.isa.build_context(msg, ...)` and stores the result in `ctx`. DECIDE calls `agent.isa.build_context(msg, ...)` again, creating a fresh context and discarding the PERCEIVE result.

**Impact:** MEDIUM — doubles the cost of Finding #4 (which is already expensive).

---

### Finding #11 — LEARN calls build_context() third time
**agent_heartbeat.py:178**  
**Rating: CONFIRMED**

```python
ctx = agent.isa.build_context(
    Message(text=hc.user_message), agent.session, agent.octopus, agent.ios)
```

Third invocation of build_context in the same tick. All data from previous calls is discarded.

**Impact:** MEDIUM — triples the cost of Finding #4.

---

## NEW Issues (Missed by Auditor)

### NEW-P0-1: ToolRegistry verify hook is dead code
**isn_impl.py:32-39, tools/executor.py:42,78**

The verify hook (`_verify_before_complete`) is registered on ToolRegistry but **never triggers** for "terminal" because:
1. ToolRegistry._requires_verify = `{"write_file", "shell"}` — doesn't include "terminal"
2. ISN._execute_action bypasses ToolRegistry entirely, calling `self.tools[name]` directly

The entire verify-before-complete safety mechanism is non-functional.

**Severity: P0** — false sense of security.

### NEW-P0-2: Landlock sandbox is not actually applied
**sandbox.py:85-101**

```python
def apply(self) -> bool:
    if not self._landlock_available:
        return False
    try:
        # 尝试使用prctl应用Landlock
        # 注意：实际Landlock API需要更复杂的设置
        # 这里是简化版，用于演示
        return True  # <-- returns True WITHOUT applying anything
    except Exception:
        return False
```

The code comments claim "Landlock内核级文件隔离" but the `apply()` method is a stub that returns True without applying any kernel-level restrictions. The kernel-level isolation is completely absent.

**Severity: P0** — advertised security feature doesn't exist.

### NEW-P0-3: _terminal_real() is unreachable but more dangerous than _terminal()
**isn_impl.py:233-241 vs 192-220**

Two terminal execution paths exist:
- `_terminal()` (line 192): Has dangerous command check + sandbox path check. **This one is actually used.**
- `_terminal_real()` (line 233): Zero safety checks. Registered with ToolRegistry but never called.

If anyone wires up ToolRegistry.execute() in the future, `_terminal_real` would be the entry point — with no safety checks at all. This is a latent vulnerability.

**Severity: P1** — latent security risk.

### NEW-P0-4: _RightBrain.review can never return "revise" verdict
**octopus_impl.py:208**

The LLM is prompted for 3 verdict types but the code only returns "approve" or "reject":
```python
verdict = "reject" if any(w in resp_lower for w in ["reject", "否", "不行", "风险", "不合理"]) else "approve"
```

"revise" is a dead code path. The right brain cannot request modifications — it can only approve or reject.

**Severity: P1** — limits the right brain's expressive capability.

### NEW-P1-5: shell=True used in multiple subprocess calls
**isn_impl.py:216, 237; tools/executor.py:157-164**

Both `_terminal()` and `_terminal_real()` use `subprocess.run(command, shell=True)`. The `tool_shell()` in executor.py also uses `shell=True`. While `_terminal()` has a basic dangerous command check, `shell=True` allows shell metacharacter injection (e.g., backticks, $(), semicolons) that bypass simple string matching.

Example: `echo hello; rm -rf ~/important` — the "rm" is detected, but `echo hello && rm -rf ~/important` with `&&` might bypass depending on string matching.

**Severity: P1** — shell injection vectors exist despite basic checks.

---

## Per-Finding Summary Table

| # | Claim | Line Ref | Rating | Severity |
|---|-------|----------|--------|----------|
| 1 | Right brain string matching | octopus:207 | CONFIRMED | P1 |
| 2 | _terminal_real no check | isn:233 | CONFIRMED | P0 |
| 3 | _search_files HOME dir | isn:182 | CONFIRMED | P1 |
| 4 | build_context creates memory stores | isa:47-79 | CONFIRMED | P0 |
| 5 | health_check LLM ping | octopus:36 | CONFIRMED | P1 |
| 6 | 2 API calls per tick | main_loop:92 | DOWNGRADED | P1 (actually 4) |
| 7 | Hardcoded tools | isa:59 | CONFIRMED | P2 |
| 8 | 3 JSONL files per init | isn:42-103 | DOWNGRADED | P2 |
| 9 | String matching simulation | octopus:89 | CONFIRMED | P1 |
| 10 | DECIDE redundant build_context | heartbeat | CONFIRMED | P1 |
| 11 | LEARN triple build_context | heartbeat | CONFIRMED | P1 |

---

## Top 5 Recommended Fixes (Priority Order)

1. **Fix _terminal_real() security gap** — Add dangerous command check + sandbox path check identical to _terminal(), OR remove _terminal_real entirely since it's dead code.

2. **Fix build_context() object creation** — Use singleton/cached instances of UnifiedMemory and CausalMemoryStore. Eliminate redundant calls (call once per tick, pass context forward).

3. **Fix health_check() API waste** — Add TTL-based caching (e.g., re-check health only every 60 seconds). Remove the 4x-per-tick redundant calls.

4. **Fix ToolRegistry verify hook** — Either add "terminal" to _requires_verify, or ensure ISN._execute_action routes through ToolRegistry.execute() so the hook actually fires.

5. **Fix Right Brain verdict parsing** — Parse the LLM's JSON response instead of keyword matching. Handle all 3 verdict types (approve/reject/revise).

---

*End of peer review.*
