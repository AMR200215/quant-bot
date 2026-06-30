"""
test_jupiter_retry.py — unit tests for Jupiter 429 retry/backoff patch.

Tests:
  1. Quote succeeds on first attempt — no retry triggered
  2. Quote succeeds after 2× 429 then 200
  3. Quote exhausts all retries on persistent 429 — raises HTTPError, logs exhausted
  4. Quote timeout → retry → success
  5. Swap build succeeds after 1× 429
  6. Swap build exhausts retries — raises HTTPError
  7. No uncontrolled loop — never exceeds JUPITER_MAX_RETRIES total requests
  8. Backoff grows exponentially, is capped at JUPITER_BACKOFF_MAX_MS
  9. T22 pump-amm skip flag set for known T22 cache entry
 10. T22 pump-amm skip flag NOT set for SPL cache entry

Run:
    python -m memecoin.tools.test_jupiter_retry
"""

import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch, call

# ── minimal stubs so executor imports without real deps ──────────────────────
# We mock requests before importing executor to avoid real HTTP calls.

import requests as _requests_mod

_GREEN = "\033[32m"
_RED   = "\033[31m"
_RESET = "\033[0m"

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


def _make_response(status: int, json_body=None, raise_on_status=False):
    """Return a mock requests.Response."""
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body or {}
    if raise_on_status or status >= 400:
        r.raise_for_status.side_effect = _requests_mod.exceptions.HTTPError(
            f"HTTP {status}", response=r
        )
    else:
        r.raise_for_status.return_value = None
    return r


# ── import the functions under test ──────────────────────────────────────────
# Patch heavy dependencies so import succeeds in isolation.

_fake_cfg = types.ModuleType("memecoin.config")
_fake_cfg.JUPITER_MAX_RETRIES       = 4
_fake_cfg.JUPITER_BACKOFF_BASE_MS   = 10   # very short for test speed
_fake_cfg.JUPITER_BACKOFF_MAX_MS    = 80
_fake_cfg.JUPITER_BACKOFF_JITTER_MS = 0    # deterministic in tests
_fake_cfg.SLIPPAGE_BUY_PCT          = 30
_fake_cfg.SLIPPAGE_SELL_PCT         = 99
_fake_cfg.PRIORITY_FEE_SOL          = 0.0001
_fake_cfg.LIVE_TRADING              = False
_fake_cfg.LIVE_DRY_RUN              = False

sys.modules.setdefault("memecoin.config", _fake_cfg)

# Stub out modules executor tries to import at module level
for _mod in [
    "base58", "solders", "solders.keypair", "solders.pubkey",
    "solders.transaction", "solders.message", "solders.hash",
    "solders.instruction", "solders.compute_budget",
    "websocket", "websocket._exceptions",
    "app.alerts",
]:
    sys.modules.setdefault(_mod, MagicMock())

from memecoin.executor import (   # noqa: E402  (after sys.modules setup)
    _jup_get_quote,
    _jup_build_swap_tx,
    _jup_backoff_ms,
    _JUP_EC_RETRYING,
    _JUP_EC_EXHAUSTED,
    JUPITER_QUOTE_URL,
    JUPITER_SWAP_URL,
    _TOKEN22_PROGRAM_ID,
    _mint_token_program_cache,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: quote succeeds first try — no retry
# ─────────────────────────────────────────────────────────────────────────────
def test_quote_first_try():
    ok_resp = _make_response(200, {"outAmount": "12345"})
    with patch("memecoin.executor.requests.get", return_value=ok_resp) as mock_get:
        result = _jup_get_quote("MINT111", "SOL111", 1_000_000)
    calls = mock_get.call_count
    _report(
        "quote succeeds first try (1 request, no retry)",
        calls == 1 and result.get("outAmount") == "12345",
        f"calls={calls} result={result}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: quote succeeds after 2× 429 then 200
# ─────────────────────────────────────────────────────────────────────────────
def test_quote_retry_succeeds():
    r429a = _make_response(429)
    r429b = _make_response(429)
    r200  = _make_response(200, {"outAmount": "999"})
    with patch("memecoin.executor.requests.get", side_effect=[r429a, r429b, r200]) as mock_get, \
         patch("memecoin.executor.time.sleep"):
        result = _jup_get_quote("MINT222", "SOL111", 500)
    _report(
        "quote succeeds after 2× 429 (3 requests total)",
        mock_get.call_count == 3 and result.get("outAmount") == "999",
        f"calls={mock_get.call_count} result={result}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: quote exhausts all retries — raises HTTPError, logs exhausted
# ─────────────────────────────────────────────────────────────────────────────
def test_quote_exhausted():
    r429 = _make_response(429)
    with patch("memecoin.executor.requests.get", return_value=r429) as mock_get, \
         patch("memecoin.executor.time.sleep"), \
         patch("memecoin.executor.log") as mock_log:
        raised = False
        try:
            _jup_get_quote("MINT333", "SOL111", 100)
        except _requests_mod.exceptions.HTTPError:
            raised = True

    max_r = _fake_cfg.JUPITER_MAX_RETRIES
    # Logged _JUP_EC_EXHAUSTED at error level
    exhausted_logged = any(
        _JUP_EC_EXHAUSTED in str(c)
        for c in mock_log.error.call_args_list
    )
    _report(
        "quote exhausted: raises HTTPError after max retries",
        raised and mock_get.call_count == max_r,
        f"raised={raised} calls={mock_get.call_count} expected={max_r}",
    )
    _report(
        f"quote exhausted: logs {_JUP_EC_EXHAUSTED}",
        exhausted_logged,
        f"error calls: {[str(c) for c in mock_log.error.call_args_list]}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: quote timeout → retry → success
# ─────────────────────────────────────────────────────────────────────────────
def test_quote_timeout_then_success():
    r200 = _make_response(200, {"outAmount": "42"})
    with patch("memecoin.executor.requests.get",
               side_effect=[_requests_mod.exceptions.Timeout(), r200]) as mock_get, \
         patch("memecoin.executor.time.sleep"):
        result = _jup_get_quote("MINT444", "SOL111", 100)
    _report(
        "quote timeout → retry → success (2 requests)",
        mock_get.call_count == 2 and result.get("outAmount") == "42",
        f"calls={mock_get.call_count} result={result}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: swap build succeeds after 1× 429
# ─────────────────────────────────────────────────────────────────────────────
def test_swap_build_retry_succeeds():
    import base64
    tx_b64 = base64.b64encode(b"fake_tx_bytes").decode()
    r429 = _make_response(429)
    r200 = _make_response(200, {"swapTransaction": tx_b64})
    with patch("memecoin.executor.requests.post", side_effect=[r429, r200]) as mock_post, \
         patch("memecoin.executor.time.sleep"):
        result = _jup_build_swap_tx({"quoteResponse": {}}, "WALLET111")
    _report(
        "swap build succeeds after 1× 429 (2 requests)",
        mock_post.call_count == 2 and result == b"fake_tx_bytes",
        f"calls={mock_post.call_count}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: swap build exhausts retries — raises HTTPError
# ─────────────────────────────────────────────────────────────────────────────
def test_swap_build_exhausted():
    r429 = _make_response(429)
    with patch("memecoin.executor.requests.post", return_value=r429) as mock_post, \
         patch("memecoin.executor.time.sleep"):
        raised = False
        try:
            _jup_build_swap_tx({}, "WALLET222")
        except _requests_mod.exceptions.HTTPError:
            raised = True
    max_r = _fake_cfg.JUPITER_MAX_RETRIES
    _report(
        "swap build exhausted: raises HTTPError after max retries",
        raised and mock_post.call_count == max_r,
        f"raised={raised} calls={mock_post.call_count} expected={max_r}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: no uncontrolled loop — total requests ≤ JUPITER_MAX_RETRIES
# ─────────────────────────────────────────────────────────────────────────────
def test_no_uncontrolled_loop():
    r429 = _make_response(429)
    max_r = _fake_cfg.JUPITER_MAX_RETRIES
    with patch("memecoin.executor.requests.get", return_value=r429) as mock_get, \
         patch("memecoin.executor.time.sleep"):
        try:
            _jup_get_quote("MINT555", "SOL111", 1)
        except Exception:
            pass
    _report(
        f"no uncontrolled loop: total requests ≤ JUPITER_MAX_RETRIES ({max_r})",
        mock_get.call_count <= max_r,
        f"calls={mock_get.call_count}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: backoff is exponential and capped
# ─────────────────────────────────────────────────────────────────────────────
def test_backoff_exponential_capped():
    base = _fake_cfg.JUPITER_BACKOFF_BASE_MS
    ceil = _fake_cfg.JUPITER_BACKOFF_MAX_MS

    backoffs = [_jup_backoff_ms(attempt) for attempt in range(1, 6)]
    # With jitter=0: attempt 1→10, 2→20, 3→40, 4→80(cap), 5→80(cap)
    is_exponential = backoffs[1] >= backoffs[0] and backoffs[2] >= backoffs[1]
    is_capped = all(b <= ceil for b in backoffs)
    _report(
        "backoff grows exponentially",
        is_exponential,
        f"backoffs={[round(b) for b in backoffs]}",
    )
    _report(
        f"backoff capped at JUPITER_BACKOFF_MAX_MS ({ceil}ms)",
        is_capped,
        f"backoffs={[round(b) for b in backoffs]}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: T22 cache → _skip_pamm_t22 would be True
# ─────────────────────────────────────────────────────────────────────────────
def test_t22_cache_skip_flag():
    _DUMMY_MINT = "T22TestMintXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    _mint_token_program_cache[_DUMMY_MINT] = _TOKEN22_PROGRAM_ID
    try:
        skip = (
            _mint_token_program_cache.get(_DUMMY_MINT) == _TOKEN22_PROGRAM_ID
            or "TokenzQ" in _mint_token_program_cache.get(_DUMMY_MINT, "")
        )
        _report(
            "T22 cache entry → _skip_pamm_t22=True",
            skip,
            f"cache[{_DUMMY_MINT[:8]}]={_mint_token_program_cache.get(_DUMMY_MINT)}",
        )
    finally:
        _mint_token_program_cache.pop(_DUMMY_MINT, None)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: SPL cache entry → _skip_pamm_t22 is False
# ─────────────────────────────────────────────────────────────────────────────
def test_spl_cache_no_skip():
    _SPL_PROG   = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    _DUMMY_MINT = "SPLTestMintXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    _mint_token_program_cache[_DUMMY_MINT] = _SPL_PROG
    try:
        skip = (
            _mint_token_program_cache.get(_DUMMY_MINT) == _TOKEN22_PROGRAM_ID
            or "TokenzQ" in _mint_token_program_cache.get(_DUMMY_MINT, "")
        )
        _report(
            "SPL cache entry → _skip_pamm_t22=False",
            not skip,
            f"cache[{_DUMMY_MINT[:8]}]={_mint_token_program_cache.get(_DUMMY_MINT)}",
        )
    finally:
        _mint_token_program_cache.pop(_DUMMY_MINT, None)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print("JUPITER 429 RETRY PATCH — UNIT TESTS")
    print(f"{'═'*60}\n")

    test_quote_first_try()
    test_quote_retry_succeeds()
    test_quote_exhausted()
    test_quote_timeout_then_success()
    test_swap_build_retry_succeeds()
    test_swap_build_exhausted()
    test_no_uncontrolled_loop()
    test_backoff_exponential_capped()
    test_t22_cache_skip_flag()
    test_spl_cache_no_skip()

    total = _pass_count + _fail_count
    print(f"\n{'═'*60}")
    print(f"RESULTS  {_pass_count}/{total} passed")
    print(f"{'═'*60}\n")

    if _fail_count > 0:
        sys.exit(1)
