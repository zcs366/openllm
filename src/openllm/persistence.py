"""
DiskPersistence — 磁盘持久化层

JSON-backed dict persistence for compaction results and state snapshots.
Atomic writes (write-to-temp + rename) prevent corruption on crash.

Usage:
    dp = DiskPersistence()
    dp.save({"key": "value"}, "state/compacted.json")
    data = dp.load("state/compacted.json")
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional


class DiskPersistence:
    """JSON file-backed disk persistence for dicts."""

    def save(self, data: dict, path: str) -> None:
        """
        Atomically write a dict to disk as JSON.

        Creates parent directories if needed. Uses write-to-temp + rename
        for crash safety — readers never see a half-written file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")  # trailing newline for clean diffs
            os.replace(tmp_path, str(target))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, path: str) -> Optional[dict]:
        """
        Load a dict from a JSON file.

        Returns None if the file doesn't exist or is empty.
        Raises json.JSONDecodeError if the file is malformed.
        """
        target = Path(path)
        if not target.exists():
            return None
        if target.stat().st_size == 0:
            return None
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
