#!/usr/bin/env python3
"""
DiskPersistence tests — write, read, directory auto-creation
"""

import sys
import os
import tempfile
import shutil
import json

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.persistence import DiskPersistence


def test_save_and_load_basic():
    """Test 1: save a dict, load it back, verify round-trip."""
    dp = DiskPersistence()
    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "test_state.json")
        original = {"model": "gpt2", "layer": 4, "d0": 0.85, "tags": ["test"]}

        dp.save(original, path)
        loaded = dp.load(path)

        assert loaded is not None, "load() returned None for existing file"
        assert loaded == original, f"Round-trip failed: {loaded} != {original}"
        print("  PASS: save/load round-trip correct")
    finally:
        shutil.rmtree(tmp_dir)


def test_load_nonexistent_returns_none():
    """Test 2: loading a file that doesn't exist returns None."""
    dp = DiskPersistence()
    result = dp.load("/tmp/openllm_nonexistent_test_99999.json")
    assert result is None, f"Expected None, got {result}"
    print("  PASS: nonexistent file returns None")


def test_directory_auto_creation():
    """Test 3: save auto-creates deeply nested parent directories."""
    dp = DiskPersistence()
    tmp_dir = tempfile.mkdtemp()
    try:
        nested_path = os.path.join(tmp_dir, "a", "b", "c", "deep_state.json")
        data = {"status": "persisted", "depth": 3}

        dp.save(data, nested_path)

        # Directory was created
        assert os.path.isdir(os.path.dirname(nested_path)), \
            "Parent directory was not auto-created"

        # File is valid JSON
        with open(nested_path, "r") as f:
            loaded = json.load(f)
        assert loaded == data

        print("  PASS: deep directory auto-creation works")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    print("Running DiskPersistence tests...")
    test_save_and_load_basic()
    test_load_nonexistent_returns_none()
    test_directory_auto_creation()
    print("\n✅ All 3 tests passed.")
