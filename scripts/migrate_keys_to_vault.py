#!/usr/bin/env python3
"""把 config.json 里的明文 API key 迁进密钥库（keyvault），并从 config.json 摘掉。

2026-09-10 定：启动审计一直在喊"明文 API key"，今天兑现。

用法：
    python scripts/migrate_keys_to_vault.py --dry-run   # 只看会做什么
    python scripts/migrate_keys_to_vault.py             # 真迁

行为（顺序不可换）：
  1. 每个 provider 的 api_key 写入密钥库，**写完立刻读回校验**——不等号就整体中止，
     config.json 一个字都不动（先写后删，绝不出现"删了但没存上"的窗口）
  2. 校验通过才从 config.json 摘掉明文，留下 "api_key_ref": "keyvault" 标记
  3. 原 config.json 备份为 config.json.bak-<时间戳>（0600），出错可整体回滚
  4. 幂等：重复跑不会覆盖已迁移的配置
  5. 密钥库后端不安全（file）时仍然迁，但明确告警——"挪出 config.json" 与 "被加密"
     是两件事，不许混为一谈

不清除、不打印任何密钥本体（只打长度与尾号）。

退出码：0 成功/无需迁移 · 2 找不到配置 · 3 读回校验失败（已回滚、未改配置）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openllm.security.keyvault import default_vault, mask  # noqa: E402

MIN_KEY_LEN = 8


def plan(config: dict) -> list[tuple[str, str]]:
    """挑出需要迁移的 (provider, key)。已经是 $ENV_VAR 引用的不算明文。"""
    todo = []
    for name, prov in (config.get("providers") or {}).items():
        if not isinstance(prov, dict):
            continue
        key = prov.get("api_key", "") or ""
        if key and not key.startswith("$") and len(key) > MIN_KEY_LEN:
            todo.append((name, key))
    return todo


def migrate(home: Path, dry_run: bool = False, vault=None) -> int:
    """执行迁移。vault 可注入（测试用），默认取全局密钥库单例。"""
    cfg_path = home / "config.json"
    if not cfg_path.exists():
        print(f"✗ 找不到 {cfg_path}")
        return 2

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    todo = plan(config)
    vault = vault or default_vault()
    st = vault.status()

    print(f"密钥库后端: {st['backend']}  安全={st['secure']}  ({st['detail']})")
    if not st["secure"]:
        print("  ⚠️  后端不安全：密钥仍是明文（只是换个文件放）。仍可迁移，但请知情。")
    print(f"待迁移: {len(todo)} 个 provider")
    for name, key in todo:
        print(f"  · {name:<10} {mask(key)}")

    if not todo:
        print("✓ 无需迁移（config.json 已无明文 key）")
        return 0
    if dry_run:
        print("\n[dry-run] 未做任何改动")
        return 0

    # ① 逐条写入并**读回校验**——校验不过就整体中止，不动 config.json
    written = []
    for name, key in todo:
        vault.set(name, key)
        back = vault.get(name)
        if back != key:
            print(f"✗ {name} 读回校验失败（写入 {mask(key)}，读回 {mask(back)}）"
                  "——中止，config.json 未改")
            for n in written:
                vault.delete(n)
            return 3
        written.append(name)
        print(f"✓ {name} 已入密钥库并校验一致 {mask(key)}")

    # ② 备份
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = cfg_path.with_name(f"config.json.bak-{stamp}")
    shutil.copy2(cfg_path, backup)
    backup.chmod(0o600)

    # ③ 摘掉明文，留标记
    for name, _key in todo:
        prov = config["providers"][name]
        prov.pop("api_key", None)
        prov["api_key_ref"] = "keyvault"

    tmp = cfg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(cfg_path)

    print(f"\n✓ 已迁移 {len(todo)} 个 key；备份: {backup}")
    print(f"✓ config.json 权限: {oct(cfg_path.stat().st_mode)[-3:]}")
    print("  验证：引擎下次启动应能从密钥库取到 key")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写")
    ap.add_argument("--home", default=None, help="配置根目录（默认 ~/.openllm）")
    args = ap.parse_args()
    home = Path(args.home) if args.home else Path.home() / ".openllm"
    return migrate(home, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
