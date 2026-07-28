"""
test_rf6_versioning.py — Unit tests for RF6 smart-money versioning.

Tests cover:
  - get_loaded_version() returns None when no file exists
  - get_loaded_version() returns version from file
  - get_metadata() returns metadata from paired .metadata.json
  - check_smart_money_versioned() caches per-version wallet set
  - SMART_MONEY_PINNED_VERSION config paths vN.json
  - No-overwrite guarantee in backfill_smart_wallets._next_version()
"""

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class TestGetLoadedVersion(unittest.TestCase):
    """get_loaded_version() with no file → None; with file → version from data."""

    def test_no_file_returns_none(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("research.smart_wallets.SMART_WALLETS_PATH", tmp_path / "smart_wallets.json"), \
                 patch("research.smart_wallets._loaded", False), \
                 patch("research.smart_wallets._wallets", set()), \
                 patch("research.smart_wallets._loaded_version", None), \
                 patch("research.smart_wallets._metadata", None):
                import research.smart_wallets as sw
                sw._loaded = False
                sw._wallets = set()
                sw._loaded_version = None
                sw._metadata = None
                # Point the config to the empty temp dir
                with patch("research.smart_wallets._load_pinned_version",
                           return_value=(set(), None, None)):
                    sw._loaded = False
                    version = sw.get_loaded_version()
        self.assertIsNone(version)

    def test_version_from_file(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "smart_wallets_v3.json"
            p.write_text(json.dumps({"version": 3, "wallets": {"AAA": {}, "BBB": {}}}))
            with patch("research.smart_wallets._load_pinned_version",
                       return_value=({"AAA", "BBB"}, 3, None)):
                import research.smart_wallets as sw
                sw._loaded = False
                sw._loaded_version = None
                sw._wallets = set()
                sw._metadata = None
                v = sw.get_loaded_version()
        self.assertEqual(v, 3)


class TestGetMetadata(unittest.TestCase):
    """get_metadata() returns the paired .metadata.json content."""

    def test_metadata_returned(self):
        meta = {"version": 2, "created_at": "2026-07-28T00:00:00Z", "source_data_hash": "abc"}
        with patch("research.smart_wallets._load_pinned_version",
                   return_value=(set(), 2, meta)):
            import research.smart_wallets as sw
            sw._loaded = False
            sw._loaded_version = None
            sw._wallets = set()
            sw._metadata = None
            m = sw.get_metadata()
        self.assertEqual(m["version"], 2)
        self.assertEqual(m["source_data_hash"], "abc")

    def test_metadata_none_when_file_missing(self):
        with patch("research.smart_wallets._load_pinned_version",
                   return_value=(set(), None, None)):
            import research.smart_wallets as sw
            sw._loaded = False
            sw._loaded_version = None
            sw._wallets = set()
            sw._metadata = None
            m = sw.get_metadata()
        self.assertIsNone(m)


class TestCheckSmartMoneyVersioned(unittest.TestCase):
    """check_smart_money_versioned() caches by version and scores correctly."""

    def _write_wallet_file(self, tmp: Path, version: int, wallets: dict) -> Path:
        p = tmp / f"smart_wallets_v{version}.json"
        p.write_text(json.dumps({"version": version, "wallets": wallets}))
        return p

    def test_hit_returns_true(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wallet_file(tmp_path, 1, {"WALLET_A": {"score": 0.9}})
            import research.smart_wallets as sw
            sw._version_cache.clear()
            with patch("research.config.SMART_WALLETS_BASE_DIR", tmp_path):
                hit, count = sw.check_smart_money_versioned(["WALLET_A", "WALLET_X"], 1)
        self.assertTrue(hit)
        self.assertEqual(count, 1)

    def test_miss_returns_false(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wallet_file(tmp_path, 1, {"WALLET_B": {}})
            import research.smart_wallets as sw
            sw._version_cache.clear()
            with patch("research.config.SMART_WALLETS_BASE_DIR", tmp_path):
                hit, count = sw.check_smart_money_versioned(["WALLET_X", "WALLET_Y"], 1)
        self.assertFalse(hit)
        self.assertEqual(count, 0)

    def test_cache_is_populated_after_first_call(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wallet_file(tmp_path, 2, {"W1": {}})
            import research.smart_wallets as sw
            sw._version_cache.clear()
            with patch("research.config.SMART_WALLETS_BASE_DIR", tmp_path):
                # Must pass non-empty buyers — empty list returns early before caching
                sw.check_smart_money_versioned(["SOME_WALLET"], 2)
                self.assertIn(2, sw._version_cache)

    def test_different_versions_independent(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wallet_file(tmp_path, 1, {"WALLET_V1": {}})
            self._write_wallet_file(tmp_path, 2, {"WALLET_V2": {}})
            import research.smart_wallets as sw
            sw._version_cache.clear()
            with patch("research.config.SMART_WALLETS_BASE_DIR", tmp_path):
                hit1, _ = sw.check_smart_money_versioned(["WALLET_V1"], 1)
                hit2, _ = sw.check_smart_money_versioned(["WALLET_V2"], 2)
                miss1, _ = sw.check_smart_money_versioned(["WALLET_V2"], 1)
        self.assertTrue(hit1)
        self.assertTrue(hit2)
        self.assertFalse(miss1)

    def test_thread_safe_concurrent_calls(self):
        """Multiple threads calling versioned check should not corrupt the cache."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_wallet_file(tmp_path, 5, {"SHARED_W": {}})
            import research.smart_wallets as sw
            sw._version_cache.clear()
            results = []

            def _call():
                with patch("research.config.SMART_WALLETS_BASE_DIR", tmp_path):
                    hit, _ = sw.check_smart_money_versioned(["SHARED_W"], 5)
                    results.append(hit)

            threads = [threading.Thread(target=_call) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(results), 10)
        self.assertTrue(all(results))


class TestNoOverwriteGuarantee(unittest.TestCase):
    """backfill_smart_wallets._next_version() never overwrites an existing file."""

    def test_next_version_skips_existing(self):
        from research.backfill_smart_wallets import _next_version
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Write v1 and v2 — _next_version should return 3
            (tmp_path / "smart_wallets_v1.json").write_text("{}")
            (tmp_path / "smart_wallets_v2.json").write_text("{}")
            v = _next_version(tmp_path)
        self.assertEqual(v, 3)

    def test_next_version_starts_at_1_when_empty(self):
        from research.backfill_smart_wallets import _next_version
        with TemporaryDirectory() as tmp:
            v = _next_version(Path(tmp))
        self.assertEqual(v, 1)

    def test_next_version_gaps_are_skipped(self):
        """If v1 and v3 exist but not v2, still returns next after max (4)."""
        from research.backfill_smart_wallets import _next_version
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "smart_wallets_v1.json").write_text("{}")
            (tmp_path / "smart_wallets_v3.json").write_text("{}")
            v = _next_version(tmp_path)
        self.assertEqual(v, 4)


if __name__ == "__main__":
    unittest.main()
