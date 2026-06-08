"""Disk-backed KV-Cache persistence for cross-session prefix reuse.

Implements local SSD-based KV-Cache serialization, enabling prefix
restoration without re-prefilling or network connectivity. This is the
edge-adapted version of Mooncake's prefix-hash matching, operating
entirely on-device.

Architecture
------------
Storage layout::

    <cache_dir>/
      index.json            # Global cache index with metadata
      <cache_id>.state      # Raw llama.cpp state bytes (KV-Cache tensors)
      <cache_id>.tokens     # Token IDs as JSON (for prefix matching)

Two restore strategies:
  1. **Fast path** (default): ``Llama.load_state()`` — restores the full
     KV-Cache tensor state from serialized bytes. Near-zero eval cost.
  2. **Fallback path**: Token ID replay via ``compiler.eval()`` — works
     when state format is incompatible (e.g. after model update).

Usage::

    from core.kvcache_persist import DiskKVCache

    cache = DiskKVCache("./kv_cache_store")

    # Session 1: prefill prefix, then save
    compiler.eval(prefix_ids)
    cache.save(compiler, prefix_ids, tag="system_prompt")

    # Session 2: load from disk (no re-prefill needed)
    restored_ids = cache.load(compiler, tag="system_prompt")
"""

import hashlib
import json
import os
import shutil
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .compiler import MeaningCompiler


@dataclass
class CacheEntry:
    """Metadata for a single cached prefix entry."""
    cache_id: str
    tag: str
    model_path: str
    model_hash: str
    n_tokens: int
    n_ctx: int
    token_ids: List[int]
    token_hash: str
    state_size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    compressed: bool = False

    def to_dict(self) -> Dict:
        d = {
            "cache_id": self.cache_id,
            "tag": self.tag,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "n_tokens": self.n_tokens,
            "n_ctx": self.n_ctx,
            "token_ids": self.token_ids,
            "token_hash": self.token_hash,
            "state_size_bytes": self.state_size_bytes,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }
        if self.compressed:
            d["compressed"] = True
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "CacheEntry":
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)


def _hash_token_ids(token_ids: List[int]) -> str:
    data = struct.pack(f"<{len(token_ids)}I", *token_ids)
    return hashlib.sha256(data).hexdigest()[:16]


def _hash_model_path(model_path: str) -> str:
    return hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:12]


class DiskKVCache:
    """Disk-backed KV-Cache persistence manager.

    Provides cross-session prefix reuse by serializing the full KV-Cache
    tensor state to local SSD. Each cached prefix is stored as a separate
    binary file with a JSON metadata index for fast lookup.

    Args:
        cache_dir:  Directory for cache storage.
        max_entries: Maximum number of cache entries (LRU eviction).
        max_bytes:  Maximum total cache size in bytes (0 = unlimited).
    """

    INDEX_FILENAME = "index.json"

    def __init__(
        self,
        cache_dir: str = ".kv_cache_store",
        max_entries: int = 64,
        max_bytes: int = 0,
        compress: bool = False,
        compress_level: int = 6,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.compress = compress
        self.compress_level = compress_level

        self._index_path = self.cache_dir / self.INDEX_FILENAME
        self._index: Dict[str, CacheEntry] = {}
        self._stats = {
            "saves": 0,
            "loads": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "bytes_written": 0,
            "bytes_read": 0,
        }
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cid, entry_dict in data.items():
                self._index[cid] = CacheEntry.from_dict(entry_dict)

    def _save_index(self):
        data = {cid: entry.to_dict() for cid, entry in self._index.items()}
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _state_path(self, cache_id: str) -> Path:
        return self.cache_dir / f"{cache_id}.state"

    def _tokens_path(self, cache_id: str) -> Path:
        return self.cache_dir / f"{cache_id}.tokens"

    def _compute_model_hash(self, compiler: MeaningCompiler) -> str:
        return _hash_model_path(compiler.llm.model_path)

    def _evict_if_needed(self):
        if self.max_entries <= 0 and self.max_bytes <= 0:
            return

        entries = sorted(
            self._index.values(), key=lambda e: e.last_accessed
        )

        while self.max_entries > 0 and len(self._index) > self.max_entries:
            self._remove_entry(entries.pop(0))

        if self.max_bytes > 0:
            total = sum(e.state_size_bytes for e in self._index.values())
            while self.max_bytes > 0 and total > self.max_bytes and entries:
                removed = entries.pop(0)
                total -= removed.state_size_bytes
                self._remove_entry(removed)

    def _remove_entry(self, entry: CacheEntry):
        state_p = self._state_path(entry.cache_id)
        tokens_p = self._tokens_path(entry.cache_id)
        if state_p.exists():
            state_p.unlink()
        if tokens_p.exists():
            tokens_p.unlink()
        self._index.pop(entry.cache_id, None)
        self._stats["evictions"] += 1

    def save(
        self,
        compiler: MeaningCompiler,
        token_ids: List[int],
        tag: str = "",
    ) -> str:
        """Save the current KV-Cache state to disk.

        Args:
            compiler:  The active MeaningCompiler with populated KV-Cache.
            token_ids: Token IDs that were evaluated into the cache.
            tag:       Optional human-readable label.

        Returns:
            cache_id: Unique identifier for the saved entry.
        """
        state = compiler.llm.save_state()
        now = time.time()
        model_hash = self._compute_model_hash(compiler)
        token_hash = _hash_token_ids(token_ids)

        existing_id = self.find_by_prefix(token_ids)
        if existing_id and existing_id in self._index:
            old = self._index[existing_id]
            cache_id = existing_id
        else:
            cache_id = f"{model_hash}_{token_hash}_{int(now)}"

        state_path = self._state_path(cache_id)
        tokens_path = self._tokens_path(cache_id)

        state_data = state.llama_state
        if self.compress:
            state_data = zlib.compress(state_data, self.compress_level)
        with open(state_path, "wb") as f:
            f.write(state_data)

        with open(tokens_path, "w", encoding="utf-8") as f:
            json.dump(token_ids, f)

        entry = CacheEntry(
            cache_id=cache_id,
            tag=tag or cache_id,
            model_path=compiler.llm.model_path,
            model_hash=model_hash,
            n_tokens=state.n_tokens,
            n_ctx=compiler.n_ctx,
            token_ids=token_ids,
            token_hash=token_hash,
            state_size_bytes=len(state_data),
            created_at=now,
            last_accessed=now,
            access_count=0,
        )
        entry.compressed = self.compress

        self._index[cache_id] = entry
        self._stats["saves"] += 1
        self._stats["bytes_written"] += len(state_data)
        self._evict_if_needed()
        self._save_index()
        return cache_id

    def load(
        self,
        compiler: MeaningCompiler,
        tag: Optional[str] = None,
        cache_id: Optional[str] = None,
        fallback_replay: bool = True,
    ) -> Optional[List[int]]:
        """Load a cached KV-Cache state from disk.

        Args:
            compiler:       The MeaningCompiler to restore into.
            tag:            Look up by tag (first match).
            cache_id:       Look up by exact cache_id (overrides tag).
            fallback_replay: If True, replay token IDs when state load fails.

        Returns:
            The restored token IDs, or None if not found/failed.
        """
        entry = self._resolve_entry(tag, cache_id)
        if entry is None:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1

        current_hash = self._compute_model_hash(compiler)
        if entry.model_hash != current_hash:
            if fallback_replay:
                return self._replay_tokens(compiler, entry)
            return None

        state_path = self._state_path(entry.cache_id)
        if not state_path.exists():
            if fallback_replay:
                return self._replay_tokens(compiler, entry)
            return None

        try:
            with open(state_path, "rb") as f:
                state_bytes = f.read()

            if getattr(entry, "compressed", False):
                state_bytes = zlib.decompress(state_bytes)

            tokens_path = self._tokens_path(entry.cache_id)
            if tokens_path.exists():
                with open(tokens_path, "r", encoding="utf-8") as f:
                    token_ids = json.load(f)
            else:
                token_ids = entry.token_ids

            from llama_cpp.llama import LlamaState
            import numpy as np

            state = LlamaState(
                input_ids=np.array(token_ids, dtype=np.intc),
                scores=np.zeros(len(token_ids), dtype=np.float32),
                n_tokens=entry.n_tokens,
                llama_state=state_bytes,
                llama_state_size=len(state_bytes),
                seed=0,
            )

            compiler.reset_cache()
            compiler.llm.load_state(state)
            compiler.set_n_tokens(entry.n_tokens)

            entry.last_accessed = time.time()
            entry.access_count += 1
            self._stats["loads"] += 1
            self._stats["bytes_read"] += len(state_bytes)
            self._save_index()
            return token_ids

        except Exception:
            if fallback_replay:
                return self._replay_tokens(compiler, entry)
            return None

    def _replay_tokens(
        self, compiler: MeaningCompiler, entry: CacheEntry
    ) -> List[int]:
        """Fallback: reconstruct KV-Cache by replaying token eval."""
        tokens_path = self._tokens_path(entry.cache_id)
        if tokens_path.exists():
            with open(tokens_path, "r", encoding="utf-8") as f:
                token_ids = json.load(f)
        else:
            token_ids = entry.token_ids

        compiler.rebuild_cache(token_ids)

        entry.last_accessed = time.time()
        entry.access_count += 1
        self._stats["loads"] += 1
        self._save_index()
        return token_ids

    def _resolve_entry(
        self, tag: Optional[str], cache_id: Optional[str]
    ) -> Optional[CacheEntry]:
        if cache_id and cache_id in self._index:
            return self._index[cache_id]
        if tag:
            for entry in self._index.values():
                if entry.tag == tag:
                    return entry
        return None

    def find_by_prefix(self, token_ids: List[int]) -> Optional[str]:
        """Find a cache entry matching the given prefix token IDs.

        Uses SHA-256 hash for O(1) lookup, with token_ids collision guard.
        """
        th = _hash_token_ids(token_ids)
        for cid, entry in self._index.items():
            if entry.token_hash == th and entry.token_ids == token_ids:
                return cid
        return None

    def list_caches(self) -> List[CacheEntry]:
        return sorted(
            self._index.values(), key=lambda e: e.last_accessed, reverse=True
        )

    def delete(self, tag: Optional[str] = None, cache_id: Optional[str] = None) -> bool:
        entry = self._resolve_entry(tag, cache_id)
        if entry is None:
            return False
        self._remove_entry(entry)
        self._save_index()
        return True

    def clear(self):
        for entry in list(self._index.values()):
            self._remove_entry(entry)
        self._save_index()

    def get_stats(self) -> Dict:
        total_bytes = sum(e.state_size_bytes for e in self._index.values())
        return {
            **self._stats,
            "entries": len(self._index),
            "total_bytes": total_bytes,
            "total_bytes_mb": round(total_bytes / (1024 * 1024), 2),
        }

    def get_disk_usage(self) -> Dict[str, int]:
        total = 0
        for p in self.cache_dir.iterdir():
            if p.is_file() and p.name != self.INDEX_FILENAME:
                total += p.stat().st_size
        return {"bytes": total, "mb": round(total / (1024 * 1024), 2)}

    def export_entry(
        self, cache_id: str, dest_dir: str
    ) -> Optional[str]:
        entry = self._index.get(cache_id)
        if entry is None:
            return None
        dest = Path(dest_dir) / f"{cache_id}.bundle"
        dest.mkdir(parents=True, exist_ok=True)
        state_p = self._state_path(cache_id)
        tokens_p = self._tokens_path(cache_id)
        if state_p.exists():
            shutil.copy2(state_p, dest / "state.bin")
        if tokens_p.exists():
            shutil.copy2(tokens_p, dest / "tokens.json")
        with open(dest / "meta.json", "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, indent=2)
        return str(dest)

    def import_entry(self, bundle_dir: str) -> Optional[str]:
        src = Path(bundle_dir)
        meta_p = src / "meta.json"
        if not meta_p.exists():
            return None
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        entry = CacheEntry.from_dict(meta)
        state_src = src / "state.bin"
        tokens_src = src / "tokens.json"
        if state_src.exists():
            shutil.copy2(state_src, self._state_path(entry.cache_id))
        if tokens_src.exists():
            shutil.copy2(tokens_src, self._tokens_path(entry.cache_id))
        self._index[entry.cache_id] = entry
        self._save_index()
        return entry.cache_id
