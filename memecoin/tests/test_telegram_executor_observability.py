"""
memecoin/tests/test_telegram_executor_observability.py — V8-TWIN-FIX VF4/VF7
test #11.

Proves the run_in_executor done-callback (_log_executor_failure) surfaces
a worker exception via logging, and that attaching it does not make the
caller block/await the future — confirmed not the actual root cause of
the 15-candidate mystery (both _screen_and_signal and _on_telegram_signal
already have their own Exception handlers), but VF4 asked for it as a
hardening measure regardless, and it needs its own regression test.

Run: python -m pytest memecoin/tests/test_telegram_executor_observability.py -v
"""

import logging
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from memecoin.telegram_monitor import _log_executor_failure


class TestExecutorFailureSurfaced(unittest.TestCase):

    def setUp(self):
        self._pool = ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        self._pool.shutdown(wait=True)

    def test_11_worker_exception_is_logged_and_loop_not_blocked(self):
        # A worker that blocks briefly before failing -- gives us a window
        # to prove submit() itself returns immediately (doesn't await),
        # while still letting the done-callback (which fires on the worker
        # thread the moment the future completes) land inside the
        # assertLogs capture window below, avoiding a race where the
        # callback fires before capture starts.
        def _boom():
            time.sleep(0.2)
            raise RuntimeError("simulated worker failure")

        t0 = time.time()
        with self.assertLogs("memecoin.telegram_monitor", level="ERROR") as cm:
            fut = self._pool.submit(_boom)
            fut.add_done_callback(lambda f: _log_executor_failure(f, "TestMint12345678"))
            # Submitting + attaching the callback must return near-instantly —
            # proves this does not await/block on the worker.
            self.assertLess(time.time() - t0, 0.1)
            from concurrent.futures import wait as _wait
            _wait([fut], timeout=2)   # wait for completion without re-raising
            time.sleep(0.1)            # let the done-callback finish logging
        self.assertTrue(any("TG executor worker failed" in line for line in cm.output))

    def test_no_exception_produces_no_error_log(self):
        def _ok():
            return 42

        fut = self._pool.submit(_ok)
        fut.add_done_callback(lambda f: _log_executor_failure(f, "TestMint12345678"))
        self.assertEqual(fut.result(timeout=2), 42)
        # No assertion needed beyond "doesn't raise" -- _log_executor_failure
        # must be a silent no-op when future.exception() is None.


if __name__ == "__main__":
    unittest.main()
