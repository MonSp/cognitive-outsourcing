import sys
import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kvcache_persist import (
    DiskKVCache,
    CacheEntry,
    _hash_token_ids,
    _hash_model_path,
)


class TestHashing(unittest.TestCase):

    def test_token_hash_deterministic(self):
        ids = [101, 202, 303, 404, 505]
        h1 = _hash_token_ids(ids)
        h2 = _hash_token_ids(ids)
        self.assertEqual(h1, h2)

    def test_token_hash_different_for_different_ids(self):
        h1 = _hash_token_ids([1, 2, 3])
        h2 = _hash_token_ids([1, 2, 4])
        self.assertNotEqual(h1, h2)

    def test_token_hash_empty(self):
        h = _hash_token_ids([])
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 16)

    def test_token_hash_length(self):
        h = _hash_token_ids([100, 200, 300])
        self.assertEqual(len(h), 16)

    def test_model_hash_deterministic(self):
        h1 = _hash_model_path("/path/to/model.gguf")
        h2 = _hash_model_path("/path/to/model.gguf")
        self.assertEqual(h1, h2)

    def test_model_hash_different_paths(self):
        h1 = _hash_model_path("/path/a.gguf")
        h2 = _hash_model_path("/path/b.gguf")
        self.assertNotEqual(h1, h2)


class TestCacheEntry(unittest.TestCase):

    def test_to_dict_roundtrip(self):
        entry = CacheEntry(
            cache_id="test_001",
            tag="system_prompt",
            model_path="/models/test.gguf",
            model_hash="abc123",
            n_tokens=128,
            n_ctx=8192,
            token_ids=[101, 202, 303],
            token_hash="deadbeef",
            state_size_bytes=4096,
            created_at=1000.0,
            last_accessed=1001.0,
            access_count=3,
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        self.assertEqual(restored.cache_id, entry.cache_id)
        self.assertEqual(restored.tag, entry.tag)
        self.assertEqual(restored.token_ids, entry.token_ids)
        self.assertEqual(restored.n_tokens, entry.n_tokens)
        self.assertEqual(restored.access_count, entry.access_count)

    def test_from_dict_defaults(self):
        d = {
            "cache_id": "x",
            "tag": "y",
            "model_path": "m",
            "model_hash": "h",
            "n_tokens": 10,
            "n_ctx": 100,
            "token_ids": [1, 2],
            "token_hash": "th",
            "state_size_bytes": 0,
            "created_at": 0.0,
            "last_accessed": 0.0,
        }
        entry = CacheEntry.from_dict(d)
        self.assertEqual(entry.access_count, 0)


class TestDiskKVCache(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="kvcache_test_")
        self.cache_dir = os.path.join(self.tmpdir, "store")
        self.cache = DiskKVCache(self.cache_dir, max_entries=5)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mock_compiler(self, model_path="/models/test.gguf", n_ctx=8192):
        compiler = MagicMock()
        compiler.llm.model_path = model_path
        compiler.n_ctx = n_ctx
        compiler.n_tokens = 0
        mock_state = MagicMock()
        mock_state.n_tokens = 128
        mock_state.llama_state = b"\x00" * 4096
        compiler.llm.save_state.return_value = mock_state
        compiler.llm.load_state.return_value = None
        return compiler

    def test_save_creates_files(self):
        compiler = self._mock_compiler()
        token_ids = [101, 202, 303, 404, 505]
        cache_id = self.cache.save(compiler, token_ids, tag="test_prefix")

        state_path = Path(self.cache_dir) / f"{cache_id}.state"
        tokens_path = Path(self.cache_dir) / f"{cache_id}.tokens"
        index_path = Path(self.cache_dir) / "index.json"
        self.assertTrue(state_path.exists())
        self.assertTrue(tokens_path.exists())
        self.assertTrue(index_path.exists())

        with open(tokens_path, "r") as f:
            saved_tokens = json.load(f)
        self.assertEqual(saved_tokens, token_ids)

    def test_load_restores_state(self):
        compiler = self._mock_compiler()
        token_ids = [101, 202, 303]
        self.cache.save(compiler, token_ids, tag="my_prefix")

        compiler2 = self._mock_compiler()
        compiler2.n_tokens = 0
        restored = self.cache.load(compiler2, tag="my_prefix")
        self.assertEqual(restored, token_ids)
        compiler2.reset_cache.assert_called()
        compiler2.llm.load_state.assert_called_once()
        compiler2.set_n_tokens.assert_called_once()

    def test_load_miss_returns_none(self):
        compiler = self._mock_compiler()
        result = self.cache.load(compiler, tag="nonexistent")
        self.assertIsNone(result)

    def test_find_by_prefix(self):
        compiler = self._mock_compiler()
        token_ids = [10, 20, 30]
        cache_id = self.cache.save(compiler, token_ids, tag="pfx")
        found = self.cache.find_by_prefix(token_ids)
        self.assertEqual(found, cache_id)

    def test_find_by_prefix_miss(self):
        compiler = self._mock_compiler()
        self.cache.save(compiler, [10, 20], tag="p1")
        found = self.cache.find_by_prefix([99, 99])
        self.assertIsNone(found)

    def test_list_caches(self):
        compiler = self._mock_compiler()
        self.cache.save(compiler, [1], tag="a")
        self.cache.save(compiler, [2], tag="b")
        self.cache.save(compiler, [3], tag="c")
        entries = self.cache.list_caches()
        self.assertEqual(len(entries), 3)
        tags = {e.tag for e in entries}
        self.assertEqual(tags, {"a", "b", "c"})

    def test_delete(self):
        compiler = self._mock_compiler()
        self.cache.save(compiler, [1, 2], tag="del_me")
        self.assertTrue(self.cache.delete(tag="del_me"))
        self.assertIsNone(self.cache.find_by_prefix([1, 2]))

    def test_delete_nonexistent(self):
        self.assertFalse(self.cache.delete(tag="nope"))

    def test_clear(self):
        compiler = self._mock_compiler()
        self.cache.save(compiler, [1], tag="a")
        self.cache.save(compiler, [2], tag="b")
        self.cache.clear()
        self.assertEqual(len(self.cache.list_caches()), 0)

    def test_lru_eviction(self):
        small_cache = DiskKVCache(
            os.path.join(self.tmpdir, "small"), max_entries=3
        )
        compiler = self._mock_compiler()
        small_cache.save(compiler, [1], tag="first")
        small_cache.save(compiler, [2], tag="second")
        small_cache.save(compiler, [3], tag="third")
        self.assertEqual(len(small_cache.list_caches()), 3)
        small_cache.save(compiler, [4], tag="fourth")
        self.assertEqual(len(small_cache.list_caches()), 3)

    def test_stats_tracking(self):
        compiler = self._mock_compiler()
        self.cache.save(compiler, [10, 20], tag="s1")
        self.cache.load(compiler, tag="s1")
        self.cache.load(compiler, tag="missing")
        stats = self.cache.get_stats()
        self.assertEqual(stats["saves"], 1)
        self.assertEqual(stats["loads"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["entries"], 1)

    def test_index_persistence(self):
        compiler = self._mock_compiler()
        self.cache.save(compiler, [100], tag="persistent")
        del self.cache

        cache2 = DiskKVCache(self.cache_dir, max_entries=5)
        entries = cache2.list_caches()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].tag, "persistent")

    def test_overwrite_same_prefix(self):
        compiler = self._mock_compiler()
        token_ids = [10, 20, 30]
        id1 = self.cache.save(compiler, token_ids, tag="v1")
        id2 = self.cache.save(compiler, token_ids, tag="v2")
        self.assertEqual(id1, id2)
        entries = self.cache.list_caches()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].tag, "v2")

    def test_disk_usage(self):
        compiler = self._mock_compiler()
        self.cache.save(compiler, [1, 2, 3], tag="disk_test")
        usage = self.cache.get_disk_usage()
        self.assertGreater(usage["bytes"], 0)

    def test_export_import(self):
        compiler = self._mock_compiler()
        token_ids = [50, 60, 70]
        cache_id = self.cache.save(compiler, token_ids, tag="export_me")

        export_dir = os.path.join(self.tmpdir, "exports")
        os.makedirs(export_dir, exist_ok=True)
        bundle_path = self.cache.export_entry(cache_id, export_dir)
        self.assertIsNotNone(bundle_path)
        self.assertTrue(os.path.exists(os.path.join(bundle_path, "state.bin")))
        self.assertTrue(os.path.exists(os.path.join(bundle_path, "tokens.json")))
        self.assertTrue(os.path.exists(os.path.join(bundle_path, "meta.json")))

        import_dir = os.path.join(self.tmpdir, "imports")
        import_cache = DiskKVCache(import_dir, max_entries=5)
        imported_id = import_cache.import_entry(bundle_path)
        self.assertIsNotNone(imported_id)
        restored = import_cache.load(compiler, cache_id=imported_id)
        self.assertEqual(restored, token_ids)

    def test_empty_cache_operations(self):
        self.assertIsNone(self.cache.find_by_prefix([]))
        self.assertEqual(self.cache.list_caches(), [])
        self.assertEqual(self.cache.get_stats()["entries"], 0)

    def test_fallback_replay_on_missing_state(self):
        compiler = self._mock_compiler()
        token_ids = [10, 20, 30]
        self.cache.save(compiler, token_ids, tag="fallback_test")

        cache_id = self.cache.find_by_prefix(token_ids)
        state_path = Path(self.cache_dir) / f"{cache_id}.state"
        state_path.unlink()

        compiler2 = self._mock_compiler()
        compiler2.n_tokens = 0
        restored = self.cache.load(compiler2, tag="fallback_test")
        self.assertEqual(restored, token_ids)
        compiler2.rebuild_cache.assert_called_once_with(token_ids)


if __name__ == "__main__":
    unittest.main()
