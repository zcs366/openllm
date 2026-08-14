"""
MemoryStore — Persistent key-value memory with TTL and semantic search.

JSON-backed storage with atomic writes. Supports time-to-live expiration
and TF-IDF keyword search across stored values and metadata.

Usage:
    store = MemoryStore("/tmp/memory")
    store.write("doc1", "neural networks are powerful", {"tags": ["ml"]})
    entry = store.read("doc1")
    results = store.search("neural")
"""

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _tokenize(text: str) -> List[str]:
    """Lowercase split on non-alphanumeric, drop short tokens."""
    tokens = []
    for ch in text.lower():
        if not ch.isalnum():
            tokens.append(" ")
        else:
            tokens.append(ch)
    return [t for t in "".join(tokens).split() if len(t) > 1]


def _tf_idf_score(query_tokens: List[str], doc_tokens: Counter, idf: Dict[str, float]) -> float:
    """Score a document against query tokens using TF-IDF."""
    score = 0.0
    total = sum(doc_tokens.values()) or 1
    for token in query_tokens:
        tf = doc_tokens.get(token, 0) / total
        weight = idf.get(token, 0.0)
        score += tf * weight
    return score


class MemoryStore:
    """Persistent memory store with TTL and semantic search.

    Parameters
    ----------
    path : str
        Directory path for storing memory files.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._path / "_index.json"
        self._lock = threading.Lock()
        self._index: Dict[str, str] = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """Load key-to-file mapping from index."""
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Atomically save the index."""
        fd, tmp = tempfile.mkstemp(dir=str(self._path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(self._index_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _entry_path(self, key: str) -> Path:
        """Return the file path for a given key."""
        # Use hex hash of key for safe filenames
        safe = hashlib.md5(key.encode()).hexdigest()
        return self._path / f"{safe}.json"

    def _save_entry(self, key: str, entry: dict) -> None:
        """Atomically write one entry to disk."""
        target = self._entry_path(key)
        fd, tmp = tempfile.mkstemp(dir=str(self._path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(target))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load_entry(self, key: str) -> Optional[dict]:
        """Load one entry from disk. Returns None if missing."""
        target = self._entry_path(key)
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    # -- public API ---------------------------------------------------------

    def write(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None,
              ttl: Optional[float] = None) -> None:
        """Write a key-value pair with optional metadata and TTL.

        Parameters
        ----------
        key : str
            Unique identifier for this memory entry.
        value : str
            The text content to store.
        metadata : dict, optional
            Arbitrary metadata (tags, source, etc.).
        ttl : float, optional
            Time-to-live in seconds. None means no expiration.
        """
        now = time.time()
        expires_at = now + ttl if ttl is not None else None
        entry: Dict[str, Any] = {
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "created_at": now,
            "expires_at": expires_at,
        }
        with self._lock:
            self._save_entry(key, entry)
            self._index[key] = str(self._entry_path(key))
            self._save_index()

    def read(self, key: str) -> Optional[dict]:
        """Read an entry by key. Returns None if missing or expired."""
        with self._lock:
            entry = self._load_entry(key)
        if entry is None:
            return None
        # Check TTL expiration
        expires_at = entry.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            self.delete(key)
            return None
        return entry

    def delete(self, key: str) -> bool:
        """Delete an entry. Returns True if it existed."""
        with self._lock:
            target = self._entry_path(key)
            if target.exists():
                target.unlink()
            if key in self._index:
                del self._index[key]
                self._save_index()
                return True
            return False

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search entries using TF-IDF keyword matching.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Matching entries sorted by relevance score (descending).
            Each dict has 'key', 'value', 'metadata', 'score'.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Collect all non-expired entries
        entries: List[dict] = []
        with self._lock:
            keys = list(self._index.keys())
        for key in keys:
            entry = self._load_entry(key)
            if entry is None:
                continue
            expires_at = entry.get("expires_at")
            if expires_at is not None and time.time() > expires_at:
                self.delete(key)
                continue
            entries.append(entry)

        if not entries:
            return []

        # Build document frequency
        doc_token_lists = []
        df: Counter = Counter()
        for entry in entries:
            text = entry["value"] + " " + json.dumps(entry.get("metadata", {}))
            tokens = _tokenize(text)
            doc_token_lists.append(Counter(tokens))
            for t in set(tokens):
                df[t] += 1

        n_docs = len(entries)
        idf: Dict[str, float] = {}
        for term, freq in df.items():
            idf[term] = math.log((n_docs + 1) / (freq + 1)) + 1

        # Score and rank
        scored: List[dict] = []
        for entry, doc_counts in zip(entries, doc_token_lists):
            score = _tf_idf_score(query_tokens, doc_counts, idf)
            if score > 0:
                scored.append({
                    "key": entry["key"],
                    "value": entry["value"],
                    "metadata": entry.get("metadata", {}),
                    "score": round(score, 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def list_keys(self) -> List[str]:
        """Return all non-expired keys."""
        with self._lock:
            keys = list(self._index.keys())
        result = []
        for key in keys:
            entry = self._load_entry(key)
            if entry is None:
                continue
            expires_at = entry.get("expires_at")
            if expires_at is not None and time.time() > expires_at:
                self.delete(key)
                continue
            result.append(key)
        return sorted(result)

    def count(self) -> int:
        """Return number of non-expired entries."""
        return len(self.list_keys())
