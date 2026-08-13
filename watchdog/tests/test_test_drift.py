"""watchdog/tests/test_test_drift.py — W8 fault injection.

Includes a regression guard for a real false positive found live while
building this: a defensive `try: from x import y / except ImportError: y
= None` pattern (memecoin/journal_reconciler.py's read_sol_delta) was
invisible to a naive top-level-only AST scan."""

import tempfile
import unittest
from pathlib import Path

from watchdog.checks import STATUS_OK, STATUS_WARN
from watchdog.checks.test_drift import check_stale_mocks, check_test_collection


class TestStaleMockFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._tmp.name)
        self.pkg_dir = self.repo_root / "memecoin"
        self.pkg_dir.mkdir()
        (self.pkg_dir / "__init__.py").write_text("")
        self.test_dir = self.repo_root / "tests"
        self.test_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _write_module(self, name: str, source: str) -> Path:
        p = self.pkg_dir / name
        p.write_text(source)
        return p

    def _write_test(self, name: str, source: str) -> Path:
        p = self.test_dir / name
        p.write_text(source)
        return p

    def test_genuinely_removed_symbol_is_flagged(self):
        # check_stale_mocks() resolves modules against the real REPO_ROOT
        # constant, not an injectable one -- so this exercises the
        # resolution helpers directly against our temp package instead.
        from watchdog.checks.test_drift import _find_patch_targets, _resolve_module_file, _module_defines_symbol
        self._write_module("mod.py", "def still_here():\n    pass\n")
        self._write_test("test_x.py",
                          'from unittest.mock import patch\n'
                          'with patch("memecoin.mod.removed_function"):\n'
                          '    pass\n')
        source = (self.test_dir / "test_x.py").read_text()
        targets = _find_patch_targets(source)
        self.assertEqual(targets, ["memecoin.mod.removed_function"])
        module_file = _resolve_module_file("memecoin.mod", self.repo_root)
        self.assertIsNotNone(module_file)
        self.assertFalse(_module_defines_symbol(module_file, "removed_function"))

    def test_try_except_defensive_import_is_not_a_false_positive(self):
        """Regression: the exact real pattern found live in
        journal_reconciler.py -- must not be flagged."""
        from watchdog.checks.test_drift import _resolve_module_file, _module_defines_symbol
        self._write_module("mod2.py",
                            "try:\n"
                            "    from memecoin.other import read_sol_delta\n"
                            "except ImportError:\n"
                            "    read_sol_delta = None\n")
        module_file = _resolve_module_file("memecoin.mod2", self.repo_root)
        self.assertTrue(_module_defines_symbol(module_file, "read_sol_delta"))

    def test_conditional_if_else_module_level_def_is_not_a_false_positive(self):
        from watchdog.checks.test_drift import _resolve_module_file, _module_defines_symbol
        self._write_module("mod3.py",
                            "import sys\n"
                            "if sys.platform == 'win32':\n"
                            "    def helper(): pass\n"
                            "else:\n"
                            "    def helper(): pass\n")
        module_file = _resolve_module_file("memecoin.mod3", self.repo_root)
        self.assertTrue(_module_defines_symbol(module_file, "helper"))

    def test_third_party_target_is_skipped_not_guessed(self):
        from watchdog.checks.test_drift import _resolve_module_file
        self.assertIsNone(_resolve_module_file("requests.get", self.repo_root))

    def test_no_test_dirs_present_is_ok(self):
        results = check_stale_mocks(test_dirs=[self.repo_root / "nonexistent"])
        self.assertEqual(results[0].status, STATUS_OK)


class TestCollectionFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.test_dir = Path(self._tmp.name) / "tests"
        self.test_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_clean_collection_is_ok(self):
        (self.test_dir / "test_ok.py").write_text(
            "def test_trivial():\n    assert True\n")
        results = check_test_collection(test_dirs=[self.test_dir])
        self.assertEqual(results[0].status, STATUS_OK)

    def test_import_error_is_warn_not_silently_passed(self):
        (self.test_dir / "test_broken.py").write_text(
            "import this_module_does_not_exist_anywhere\n"
            "def test_trivial():\n    assert True\n")
        results = check_test_collection(test_dirs=[self.test_dir])
        self.assertEqual(results[0].status, STATUS_WARN)

    def test_nonexistent_dir_is_unknown(self):
        from watchdog.checks import STATUS_UNKNOWN
        results = check_test_collection(test_dirs=[self.test_dir / "nope"])
        self.assertEqual(results[0].status, STATUS_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
