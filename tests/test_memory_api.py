#!/usr/bin/env python3
"""
MemoryStore tests — write/read, search, TTL expiration, concurrent access.
"""

import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.memory_api import MemoryStore


def test_write_and_read():
    """Test 1: write an entry, read it back, verify round-trip."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = MemoryStore(tmp_dir)
        store.write("doc1", "neural networks are powerful", {"tags": ["ml"], "source": "test"})

        entry = store.read("doc1")
        assert entry is not None, "read() returned None for existing key"
        assert entry["key"] == "doc1"
        assert entry["value"] == "neural networks are powerful"
        assert entry["metadata"]["tags"] == ["ml"]
        assert entry["metadata"]["source"] == "test"
        assert entry["expires_at"] is None  # no TTL set
        print("  PASS: write/read round-trip correct")
    finally:
        shutil.rmtree(tmp_dir)


def test_search():
    """Test 2: write multiple entries, search returns relevant results ranked by score."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = MemoryStore(tmp_dir)
        store.write("a", "transformer attention heads are key to LLM reasoning", {"type": "note"})
        store.write("b", "gradient descent optimizes loss functions in neural networks", {"type": "note"})
        store.write("c", "the cat sat on the mat enjoying the warm sunlight", {"type": "poem"})
        store.write("d", "attention mechanisms allow models to focus on relevant tokens", {"type": "note"})

        results = store.search("attention transformer")
        assert len(results) > 0, "search() returned empty results"
        # Entry 'a' and 'd' should be top hits (contain both terms)
        top_keys = [r["key"] for r in results[:2]]
        assert "a" in top_keys, f"Expected 'a' in top 2, got {top_keys}"
        assert "d" in top_keys, f"Expected 'd' in top 2, got {top_keys}"
        # Verify scores are positive
        for r in results:
            assert r["score"] > 0, f"Score should be positive: {r['score']}"
        # Verify structure
        assert "key" in results[0]
        assert "value" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]
        print("  PASS: search returns relevant results with correct ranking")
    finally:
        shutil.rmtree(tmp_dir)


def test_ttl_expiration():
    """Test 3: entries with short TTL become unreadable after expiration."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = MemoryStore(tmp_dir)
        store.write("ephemeral", "this will expire", {}, ttl=0.1)
        store.write("permanent", "this stays", {}, ttl=None)

        # Immediately readable
        assert store.read("ephemeral") is not None, "Should be readable before TTL"
        assert store.read("permanent") is not None, "Permanent should always be readable"

        # Wait for TTL to expire
        time.sleep(0.3)

        # Expired entry returns None
        result = store.read("ephemeral")
        assert result is None, f"Expired entry should return None, got {result}"

        # Permanent entry still readable
        assert store.read("permanent") is not None, "Permanent entry should survive"
        print("  PASS: TTL expiration works correctly")
    finally:
        shutil.rmtree(tmp_dir)


def test_concurrent_access():
    """Test 4: concurrent writes from multiple threads don't corrupt data."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = MemoryStore(tmp_dir)
        n_threads = 8
        writes_per_thread = 20
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(writes_per_thread):
                    key = f"t{thread_id}_item{i}"
                    store.write(key, f"content from thread {thread_id} item {i}", {"thread": thread_id})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes raised errors: {errors}"

        expected_total = n_threads * writes_per_thread
        all_keys = store.list_keys()
        assert len(all_keys) == expected_total, \
            f"Expected {expected_total} keys, got {len(all_keys)}"

        # Verify a sample of entries are intact
        for t in range(n_threads):
            for i in range(writes_per_thread):
                key = f"t{t}_item{i}"
                entry = store.read(key)
                assert entry is not None, f"Missing entry: {key}"
                assert entry["metadata"]["thread"] == t
        print("  PASS: concurrent access preserves data integrity")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    print("Running MemoryStore tests...")
    test_write_and_read()
    test_search()
    test_ttl_expiration()
    test_concurrent_access()
    print("\n✅ All 4 tests passed.")
