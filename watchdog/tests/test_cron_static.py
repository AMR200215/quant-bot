"""watchdog/tests/test_cron_static.py — W19 fault-injection #1 + baseline
sanity: does the parser flag the exact real-world bug (orphaned backslash
continuation in /etc/cron.d), and does it leave genuinely valid files
alone?
"""

import tempfile
import unittest
from pathlib import Path

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN
from watchdog.checks.cron_static import check_cron_static, parse_cron_d_file

_JOB = {"id": "k5_nightly", "cron_file": "quantbot-v8-inputs", "severity": "CRITICAL"}


class TestCronStaticFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cron_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> Path:
        p = self.cron_dir / name
        p.write_text(content)
        return p

    def test_1_backslash_continuation_is_detected(self):
        """The exact real incident: /etc/cron.d does not support shell-style
        line continuation, so the second physical line is parsed by cron as
        its own (invalid) entry."""
        self._write("quantbot-v8-inputs",
                     "15 0 * * * root cd /root/quant-bot && set -a && . .env && set +a && \\\n"
                     "  .venv/bin/python -m research.scripts.v8_inputs_nightly >> logs/x.log 2>&1\n")
        has_entry, malformed, readable = parse_cron_d_file(self.cron_dir / "quantbot-v8-inputs")
        self.assertTrue(readable)
        self.assertTrue(malformed, "backslash-continuation file must be flagged malformed")
        self.assertIn("orphaned continuation", malformed[0])

        results = check_cron_static([_JOB], cron_dir=str(self.cron_dir), journal_lines=[])
        r = next(x for x in results if x.check_id == "cron_static.k5_nightly")
        self.assertEqual(r.status, STATUS_CRITICAL)

    def test_valid_single_line_entry_is_not_flagged(self):
        self._write("quantbot-v8-inputs",
                     "15 0 * * * root cd /root/quant-bot && "
                     ".venv/bin/python -m research.scripts.v8_inputs_nightly >> logs/x.log 2>&1\n")
        results = check_cron_static([_JOB], cron_dir=str(self.cron_dir), journal_lines=[])
        r = next(x for x in results if x.check_id == "cron_static.k5_nightly")
        self.assertEqual(r.status, STATUS_OK, r.reason)

    def test_valid_file_with_comments_and_env_lines_is_not_flagged(self):
        self._write("quantbot-v8-inputs",
                     "# a comment\n"
                     "SHELL=/bin/sh\n"
                     "PATH=/usr/bin:/bin\n"
                     "\n"
                     "15 0 * * * root cd /root/quant-bot && .venv/bin/python -m x >> logs/x.log 2>&1\n")
        results = check_cron_static([_JOB], cron_dir=str(self.cron_dir), journal_lines=[])
        r = next(x for x in results if x.check_id == "cron_static.k5_nightly")
        self.assertEqual(r.status, STATUS_OK, r.reason)

    def test_missing_file_is_critical(self):
        results = check_cron_static([_JOB], cron_dir=str(self.cron_dir), journal_lines=[])
        r = next(x for x in results if x.check_id == "cron_static.k5_nightly")
        self.assertEqual(r.status, STATUS_CRITICAL)
        self.assertIn("missing", r.reason)

    def test_severity_is_capped_by_job_registry_ceiling(self):
        self._write("quantbot-v8-inputs", "not a valid cron line at all\n")
        warn_job = dict(_JOB, severity="WARN")
        results = check_cron_static([warn_job], cron_dir=str(self.cron_dir), journal_lines=[])
        r = next(x for x in results if x.check_id == "cron_static.k5_nightly")
        self.assertEqual(r.status, STATUS_WARN, "CRITICAL finding must cap at the job's WARN ceiling")

    def test_journal_fetch_unavailable_yields_unknown_not_ok(self):
        self._write("quantbot-v8-inputs",
                     "15 0 * * * root cd /root/quant-bot && .venv/bin/python -m x >> logs/x.log 2>&1\n")
        results = check_cron_static([_JOB], cron_dir=str(self.cron_dir),
                                     journal_lines=None, journal_fetch_failed=True)
        r = next(x for x in results if x.check_id == "cron_static.k5_nightly")
        self.assertEqual(r.status, STATUS_UNKNOWN,
                          "missing evidence must never silently pass as OK")

    def test_journal_error_line_overrides_clean_parse(self):
        """Our own parser can be wrong; if cron's own daemon logged a real
        parser error against this exact file, trust the daemon."""
        self._write("quantbot-v8-inputs",
                     "15 0 * * * root cd /root/quant-bot && .venv/bin/python -m x >> logs/x.log 2>&1\n")
        journal = ["Aug 09 18:46:01 host cron[1]: Error: bad minute; while reading "
                   "/etc/cron.d/quantbot-v8-inputs"]
        results = check_cron_static([_JOB], cron_dir=str(self.cron_dir), journal_lines=journal)
        r = next(x for x in results if x.check_id == "cron_static.k5_nightly")
        self.assertEqual(r.status, STATUS_CRITICAL)

    def test_unmanaged_schedule_detected(self):
        self._write("quantbot-v8-inputs",
                     "15 0 * * * root cd /root/quant-bot && .venv/bin/python -m x >> logs/x.log 2>&1\n")
        self._write("quantbot-mystery-job",
                     "0 12 * * * root echo hi\n")
        results = check_cron_static([_JOB], cron_dir=str(self.cron_dir), journal_lines=[])
        unmanaged = [r for r in results if r.check_id == "cron_static.unmanaged"]
        self.assertEqual(len(unmanaged), 1)
        self.assertIn("quantbot-mystery-job", unmanaged[0].reason)


if __name__ == "__main__":
    unittest.main()
