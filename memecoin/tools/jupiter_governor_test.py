"""
jupiter_governor_test.py — CLI test for the Jupiter request governor.

Simulates:
  - 100 BACKGROUND requests (bucket drains, excess throttled)
  - 5   EXIT requests (all succeed — independent bucket)
  - 2   EMERGENCY requests (all succeed — independent bucket)

Expected:
  - BACKGROUND gets throttled after burst capacity exhausted
  - EXIT still has full reserved capacity after BACKGROUND storm
  - EMERGENCY still has reserved capacity
  - No infinite loops
  - No uncontrolled request storm (total real HTTP calls = 0, all mocked)

Run:
    python -m memecoin.tools.jupiter_governor_test
"""

import sys
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

_pass_count = 0
_fail_count = 0


def _report(name: str, ok: bool, detail: str = ""):
    global _pass_count, _fail_count
    if ok:
        _pass_count += 1
        print(f"  {_GREEN}PASS{_RESET}  {name}")
    else:
        _fail_count += 1
        print(f"  {_RED}FAIL{_RESET}  {name}" + (f"\n       {detail}" if detail else ""))


def _head(title: str):
    print(f"\n{_BOLD}{'─'*60}{_RESET}")
    print(f"{_BOLD}{title}{_RESET}")
    print(f"{'─'*60}")


def _make_resp(status: int, json_body=None):
    import requests as req
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body or {}
    if status >= 400:
        r.raise_for_status.side_effect = req.exceptions.HTTPError(f"HTTP {status}")
    else:
        r.raise_for_status.return_value = None
    return r


# ── Import governor with fast-test config ────────────────────────────────────
# Override config so buckets are tiny and tests don't have to wait 60+ seconds.

import types, sys as _sys

_fake_cfg = types.ModuleType("memecoin.config")
_fake_cfg.JUPITER_GOVERNOR_ENABLED  = True
_fake_cfg.JUPITER_BACKGROUND_RPM    = 60    # 1 token/second in test
_fake_cfg.JUPITER_EXIT_RPM          = 60
_fake_cfg.JUPITER_EMERGENCY_RPM     = 60
_fake_cfg.JUPITER_MAX_RETRIES       = 3
_fake_cfg.JUPITER_BACKOFF_BASE_MS   = 1     # 1ms — fast test
_fake_cfg.JUPITER_BACKOFF_MAX_MS    = 5
_fake_cfg.JUPITER_JITTER_MS         = 0     # deterministic

_sys.modules["memecoin.config"] = _fake_cfg

from memecoin.jupiter_governor import (   # noqa: E402
    JupiterGovernor, Purpose, _TokenBucket,
    EC_BUCKET_EMPTY, EC_429_RETRYING, EC_429_EXHAUSTED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: BACKGROUND bucket drains; EXIT and EMERGENCY remain unaffected
# ─────────────────────────────────────────────────────────────────────────────

def test_bucket_isolation():
    _head("1. BUCKET ISOLATION — 100 BACKGROUND drain vs EXIT/EMERGENCY")

    gov = JupiterGovernor()
    # Drain BACKGROUND bucket immediately (no waiting for refill)
    bg_burst = int(gov._buckets[Purpose.BACKGROUND].capacity)
    consumed = 0
    skipped  = 0
    for _ in range(100):
        if gov._buckets[Purpose.BACKGROUND].try_consume():
            consumed += 1
        else:
            skipped += 1

    _report(
        f"BACKGROUND burst consumed ({bg_burst} tokens, rest throttled)",
        consumed == bg_burst and skipped == 100 - bg_burst,
        f"consumed={consumed} skipped={skipped} burst={bg_burst}",
    )

    # EXIT bucket must be untouched
    exit_tok = gov.tokens_remaining(Purpose.EXIT)
    _report(
        "EXIT bucket unaffected after BACKGROUND drain",
        exit_tok >= gov._buckets[Purpose.EXIT].capacity - 0.01,
        f"exit_tokens={exit_tok}  capacity={gov._buckets[Purpose.EXIT].capacity}",
    )

    # EMERGENCY bucket must be untouched
    emg_tok = gov.tokens_remaining(Purpose.EMERGENCY)
    _report(
        "EMERGENCY bucket unaffected after BACKGROUND drain",
        emg_tok >= gov._buckets[Purpose.EMERGENCY].capacity - 0.01,
        f"emg_tokens={emg_tok}  capacity={gov._buckets[Purpose.EMERGENCY].capacity}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: BACKGROUND skip returns RuntimeError, no blocking
# ─────────────────────────────────────────────────────────────────────────────

def test_background_skip_nonblocking():
    _head("2. BACKGROUND SKIP — non-blocking when empty")

    gov = JupiterGovernor()
    # Drain all BACKGROUND tokens
    while gov._buckets[Purpose.BACKGROUND].try_consume():
        pass

    t0 = time.monotonic()
    raised = False
    error_class_ok = False
    try:
        gov.request(
            purpose="BACKGROUND", endpoint="quote",
            fn=lambda **kw: _make_resp(200),
            mint="TESTMINT",
        )
    except RuntimeError as e:
        raised = True
        error_class_ok = EC_BUCKET_EMPTY in str(e)
    elapsed = time.monotonic() - t0

    _report(
        "BACKGROUND empty → raises RuntimeError(bucket_empty) immediately",
        raised and error_class_ok,
        f"raised={raised} error_class_ok={error_class_ok}",
    )
    _report(
        "BACKGROUND skip is non-blocking (< 200ms)",
        elapsed < 0.2,
        f"elapsed={elapsed*1000:.1f}ms",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: EXIT bucket has capacity after BACKGROUND storm
# ─────────────────────────────────────────────────────────────────────────────

def test_exit_capacity_after_bg_storm():
    _head("3. EXIT REQUESTS — succeed after BACKGROUND storm drains bg bucket")

    gov = JupiterGovernor()

    # Drain BACKGROUND fully
    while gov._buckets[Purpose.BACKGROUND].try_consume():
        pass

    # Now fire 5 EXIT requests — each should acquire immediately
    ok_resp = _make_resp(200, {"outAmount": "99"})
    success_count = 0
    for i in range(5):
        # Each EXIT call consumes from EXIT bucket (pre-filled to burst=4 + 1 extra possible)
        if gov._buckets[Purpose.EXIT].try_consume():
            success_count += 1

    exit_cap = int(gov._buckets[Purpose.EXIT].capacity)
    # All up to burst capacity should succeed
    _report(
        f"EXIT: {min(5, exit_cap)}/5 requests succeed (up to burst={exit_cap})",
        success_count == min(5, exit_cap),
        f"success_count={success_count} exit_burst={exit_cap}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: EMERGENCY capacity reserved and independent
# ─────────────────────────────────────────────────────────────────────────────

def test_emergency_capacity():
    _head("4. EMERGENCY CAPACITY — reserved after BG + EXIT drain")

    gov = JupiterGovernor()

    # Drain BACKGROUND and EXIT
    while gov._buckets[Purpose.BACKGROUND].try_consume():
        pass
    while gov._buckets[Purpose.EXIT].try_consume():
        pass

    emg_burst = int(gov._buckets[Purpose.EMERGENCY].capacity)
    emg_tok = gov.tokens_remaining(Purpose.EMERGENCY)
    _report(
        "EMERGENCY bucket full after BG+EXIT drain",
        emg_tok >= emg_burst - 0.01,
        f"emg_tokens={emg_tok}  burst={emg_burst}",
    )

    # Consume 2 EMERGENCY tokens
    consumed = sum(1 for _ in range(2) if gov._buckets[Purpose.EMERGENCY].try_consume())
    _report(
        f"EMERGENCY: {min(2, emg_burst)}/2 tokens consumed",
        consumed == min(2, emg_burst),
        f"consumed={consumed} burst={emg_burst}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: governor.request() succeeds on 200
# ─────────────────────────────────────────────────────────────────────────────

def test_request_success():
    _head("5. governor.request() — success path")

    gov = JupiterGovernor()
    ok_resp = _make_resp(200, {"outAmount": "1234"})

    result = gov.request(
        purpose=Purpose.EXIT,
        endpoint="quote",
        fn=lambda **kw: ok_resp,
        mint="ABCDEFGH",
    )
    _report(
        "governor.request(EXIT) returns response on 200",
        result is ok_resp,
        f"result={result}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: 429 retries with backoff, then succeeds
# ─────────────────────────────────────────────────────────────────────────────

def test_429_retry_then_success():
    _head("6. 429 RETRY — retries with backoff, succeeds on 3rd attempt")

    gov = JupiterGovernor()
    ok_resp    = _make_resp(200, {"swapTransaction": "abc"})
    r429       = _make_resp(429)
    call_log   = []

    def _fn(**kw):
        n = len(call_log)
        call_log.append(n)
        if n < 2:
            return r429
        return ok_resp

    with patch("memecoin.jupiter_governor.time.sleep") as mock_sleep:
        result = gov.request(
            purpose=Purpose.EXIT,
            endpoint="swap",
            fn=_fn,
            mint="RETRY_TEST",
        )

    _report(
        "429 retry: succeeded after 2× 429 (3 total calls)",
        result is ok_resp and len(call_log) == 3,
        f"calls={len(call_log)}",
    )
    _report(
        "429 retry: sleep called for each backoff (2 backoffs)",
        mock_sleep.call_count == 2,
        f"sleep_calls={mock_sleep.call_count}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: 429 exhausted after max_retries — raises HTTPError
# ─────────────────────────────────────────────────────────────────────────────

def test_429_exhausted():
    _head("7. 429 EXHAUSTED — raises after JUPITER_MAX_RETRIES attempts")

    import requests as req
    gov    = JupiterGovernor()
    r429   = _make_resp(429)
    calls  = []

    def _fn(**kw):
        calls.append(1)
        return r429

    with patch("memecoin.jupiter_governor.time.sleep"):
        raised = False
        try:
            gov.request(purpose=Purpose.EXIT, endpoint="quote",
                        fn=_fn, mint="EXHAUST")
        except req.exceptions.HTTPError:
            raised = True

    max_r = gov._max_retries
    _report(
        f"429 exhausted: raises HTTPError after {max_r} attempts",
        raised and len(calls) == max_r,
        f"raised={raised} calls={len(calls)} max_retries={max_r}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: no uncontrolled request storm (hard cap on calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_request_storm():
    _head("8. NO REQUEST STORM — calls never exceed JUPITER_MAX_RETRIES")

    import requests as req
    gov   = JupiterGovernor()
    calls = []

    def _always_429(**kw):
        calls.append(1)
        return _make_resp(429)

    with patch("memecoin.jupiter_governor.time.sleep"):
        try:
            gov.request(purpose=Purpose.EMERGENCY, endpoint="quote",
                        fn=_always_429, mint="STORM_TEST")
        except Exception:
            pass

    _report(
        f"No storm: total calls ≤ JUPITER_MAX_RETRIES ({gov._max_retries})",
        len(calls) <= gov._max_retries,
        f"calls={len(calls)}  max={gov._max_retries}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: governor disabled (pass-through) — always acquires
# ─────────────────────────────────────────────────────────────────────────────

def test_governor_disabled():
    _head("9. GOVERNOR DISABLED — pass-through mode")

    gov = JupiterGovernor()
    gov._enabled = False

    # Drain all tokens first
    for p in [Purpose.BACKGROUND, Purpose.EXIT, Purpose.EMERGENCY]:
        while gov._buckets[p].try_consume():
            pass

    # With governor disabled, acquire should still return True
    ok = gov._acquire_bucket(Purpose.BACKGROUND)
    _report(
        "GOVERNOR_ENABLED=False → _acquire_bucket always returns True",
        ok,
        f"acquired={ok}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: token bucket refills over time
# ─────────────────────────────────────────────────────────────────────────────

def test_bucket_refills():
    _head("10. TOKEN BUCKET REFILL — tokens accumulate at correct rate")

    # 600 RPM = 10 tokens/second.  Sleep 0.1s → ~1 token added.
    bucket = _TokenBucket(rpm=600, burst=10)
    # Drain completely
    while bucket.try_consume():
        pass
    assert bucket.tokens_remaining == 0.0

    time.sleep(0.12)   # ~1.2 tokens at 10 tok/s
    after = bucket.tokens_remaining
    _report(
        "Bucket refills at configured rate (0.12s ≈ 1-2 tokens at 600 RPM)",
        0.9 <= after <= 2.5,
        f"tokens_after_0.12s={after:.2f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: CSV log written on request
# ─────────────────────────────────────────────────────────────────────────────

def test_csv_log_written():
    _head("11. CSV LOG — entry written on each request")

    import csv, tempfile
    from pathlib import Path
    from unittest.mock import patch as _patch

    gov = JupiterGovernor()
    ok  = _make_resp(200, {"ok": True})

    # Redirect log path to temp file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    tmp_path.unlink()  # delete so _write_log creates it fresh with header

    with _patch("memecoin.jupiter_governor._LOG_PATH", tmp_path), \
         _patch("memecoin.jupiter_governor._log_lock", threading.Lock()):
        gov.request(purpose=Purpose.EXIT, endpoint="quote",
                    fn=lambda **kw: ok, mint="LOGTEST")

    rows = []
    try:
        with open(tmp_path) as f:
            rows = list(csv.DictReader(f))
    finally:
        tmp_path.unlink(missing_ok=True)

    _report(
        "CSV log: 1 row written on successful request",
        len(rows) == 1 and rows[0].get("success") == "True",
        f"rows={len(rows)} row={rows[0] if rows else '—'}",
    )
    if rows:
        has_fields = all(
            f in rows[0]
            for f in ["purpose", "endpoint", "mint", "latency_ms", "bucket_tokens_remaining"]
        )
        _report("CSV log: required fields present", has_fields,
                f"keys={list(rows[0].keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: governor.status() returns all three buckets
# ─────────────────────────────────────────────────────────────────────────────

def test_status():
    _head("12. governor.status() — returns all bucket token counts")

    gov = JupiterGovernor()
    s   = gov.status()
    _report(
        "status() returns BACKGROUND, EXIT, EMERGENCY keys",
        set(s.keys()) == {Purpose.BACKGROUND, Purpose.EXIT, Purpose.EMERGENCY},
        f"keys={set(s.keys())}",
    )
    _report(
        "status() values are numeric",
        all(isinstance(v, (int, float)) for v in s.values()),
        f"values={s}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print(f"{_BOLD}JUPITER REQUEST GOVERNOR — TESTS{_RESET}")
    print(f"{'═'*60}")
    print(f"  Simulates: 100 BACKGROUND / 5 EXIT / 2 EMERGENCY requests")
    print(f"  All HTTP calls are mocked — no real network traffic\n")

    test_bucket_isolation()
    test_background_skip_nonblocking()
    test_exit_capacity_after_bg_storm()
    test_emergency_capacity()
    test_request_success()
    test_429_retry_then_success()
    test_429_exhausted()
    test_no_request_storm()
    test_governor_disabled()
    test_bucket_refills()
    test_csv_log_written()
    test_status()

    total = _pass_count + _fail_count
    print(f"\n{'═'*60}")
    print(f"{_BOLD}RESULTS  {_pass_count}/{total} passed{_RESET}")
    print(f"{'═'*60}\n")
    if _fail_count > 0:
        sys.exit(1)
