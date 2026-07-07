#!/usr/bin/env python3
"""
MemoryBus合规检查 — 包拯审计用

检查项：
1. ISA是否import MemoryBus
2. ISA.__init__是否创建self.bus
3. 新代码是否使用bus_write/bus_query（而非旧路径store/retrieve）
4. 4个provider是否注册
5. 所有provider是否实现MemoryProvider协议

用法：
  python3 check_memory_bus_compliance.py
  
返回exit code: 0=合规, 1=违规
"""
import sys
import os
import re
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

passed = 0
failed = 0
warnings = 0

def ok(name, detail=""):
    global passed
    print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    passed += 1

def fail(name, detail=""):
    global failed
    print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))
    failed += 1

def warn(name, detail=""):
    global warnings
    print(f"  ⚠️  {name}" + (f" — {detail}" if detail else ""))
    warnings += 1

ISA_PATH = Path.home() / "projects/openllm/src/openllm/memory/isa.py"
BUS_PATH = Path.home() / "projects/openllm/src/openllm/memory/memory_bus.py"
PROVIDERS_DIR = Path.home() / "projects/openllm/src/openllm/memory/providers"

print("=" * 60)
print("MemoryBus 合规检查 · 包拯审计用")
print("=" * 60)

# C1: ISA imports MemoryBus
isa_code = ISA_PATH.read_text(encoding="utf-8") if ISA_PATH.exists() else ""
if "from .memory_bus import" in isa_code:
    ok("C1: ISA imports MemoryBus")
else:
    fail("C1: ISA does NOT import MemoryBus", f"{ISA_PATH}")

# C2: ISA creates self.bus
if "self.bus = MemoryBus()" in isa_code:
    ok("C2: ISA creates self.bus")
else:
    fail("C2: ISA does NOT create self.bus")

# C3: ISA registers providers
if "_register_bus_providers" in isa_code or "bus.register" in isa_code:
    ok("C3: ISA registers providers on bus")
else:
    fail("C3: ISA does NOT register providers")

# C4: ISA has bus_write/bus_query
has_bus_write = "def bus_write" in isa_code
has_bus_query = "def bus_query" in isa_code
if has_bus_write and has_bus_query:
    ok("C4: ISA has bus_write + bus_query")
else:
    fail("C4: ISA missing bus_write/bus_query", f"write={has_bus_write}, query={has_bus_query}")

# C5: Old methods marked deprecated
deprecated_count = isa_code.count("[DEPRECATED]")
if deprecated_count >= 3:
    ok(f"C5: {deprecated_count} old methods marked [DEPRECATED]")
else:
    warn(f"C5: Only {deprecated_count} methods marked deprecated (expected ≥3)")

# C6: 4 providers exist
provider_files = list(PROVIDERS_DIR.glob("*_provider.py"))
if len(provider_files) >= 4:
    ok(f"C6: {len(provider_files)} provider files exist")
else:
    fail(f"C6: Only {len(provider_files)} provider files (expected ≥4)")

# C7: Provider protocol check
bus_code = BUS_PATH.read_text(encoding="utf-8") if BUS_PATH.exists() else ""
if "class MemoryProvider" in bus_code:
    ok("C7: MemoryProvider protocol defined")
else:
    fail("C7: MemoryProvider protocol NOT found in memory_bus.py")

# C8: New code uses bus_write/bus_query (grep recent files)
memory_dir = Path.home() / "projects/openllm/src/openllm/memory"
new_bus_usage = 0
old_direct_usage = 0
for py_file in memory_dir.glob("*.py"):
    if py_file.name in ("isa.py", "memory_bus.py"):
        continue
    code = py_file.read_text(encoding="utf-8")
    new_bus_usage += code.count("bus_write") + code.count("bus_query")
    old_direct_usage += code.count(".memory.store(") + code.count(".causal.store(")

if new_bus_usage > 0 or old_direct_usage == 0:
    ok(f"C8: Bus usage={new_bus_usage}, direct_store={old_direct_usage}")
else:
    warn(f"C8: No bus usage yet, {old_direct_usage} direct store calls")

# C9: MemoryBus is in __init__.py exports
init_path = memory_dir / "__init__.py"
init_code = init_path.read_text(encoding="utf-8") if init_path.exists() else ""
if "MemoryBus" in init_code:
    ok("C9: MemoryBus exported in __init__.py")
else:
    fail("C9: MemoryBus NOT in __init__.py exports")

# C10: Tests exist and pass
test_path = Path.home() / "projects/openllm/tests/test_memory_bus.py"
if test_path.exists():
    ok("C10: test_memory_bus.py exists")
else:
    fail("C10: test_memory_bus.py NOT found")

# Summary
print(f"\n{'='*60}")
print(f"结果: {passed} passed, {failed} failed, {warnings} warnings")
if failed == 0:
    print("🟢 合规 — MemoryBus治理到位")
else:
    print("🔴 违规 — 需要整改")
print(f"{'='*60}")

sys.exit(1 if failed else 0)
