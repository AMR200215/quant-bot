"""
jupiter_governor.py — Token-bucket rate limiter for all Jupiter API calls.

Three independent priority tiers (highest → lowest):
    EMERGENCY  reserved quota for emergency sell retries
    EXIT       reserved for live exit signals only
    BACKGROUND scanner warming / non-urgent refreshes

Isolation guarantee:
    BACKGROUND callers can NEVER consume EXIT or EMERGENCY tokens.
    Scanner / route warmer MUST use Purpose.BACKGROUND.
    Exit path uses Purpose.EXIT; escalates to Purpose.EMERGENCY if EXIT 429-exhausted.
    EMERGENCY tokens can ONLY be consumed by callers that explicitly request EMERGENCY.

Behavior per tier when bucket is empty:
    BACKGROUND  → returns immediately (skip/defer) — caller decides
    EXIT        → waits up to EXIT_WAIT_S for refill, then returns failure
    EMERGENCY   → waits up to EMERGENCY_WAIT_S for refill, then returns failure

On HTTP 429:
    exponential backoff with jitter (config-driven)
    logs each attempt with: purpose / endpoint / mint / attempt / backoff_ms
    never spins — minimum sleep between retries is JUPITER_BACKOFF_BASE_MS

Integration note (2026-06-30):
    The governor is BUILT but not yet wired into executor.py sell paths.
    Wire-in happens in a separate step after real-wallet simulation passes.
    Scanner Jupiter calls (buy pre-flight quote) should use Purpose.BACKGROUND
    once wired. The executor's exit path will use Purpose.EXIT / EMERGENCY.
"""

import csv
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Any

import requests as _requests

log = logging.getLogger(__name__)

# ── Purpose enum ──────────────────────────────────────────────────────────────

class Purpose(str, Enum):
    BACKGROUND = "BACKGROUND"
    EXIT       = "EXIT"
    EMERGENCY  = "EMERGENCY"


# ── Error classes ─────────────────────────────────────────────────────────────

EC_BUCKET_EMPTY    = "jupiter_bucket_empty"
EC_429_RETRYING    = "jupiter_429_retrying"
EC_429_EXHAUSTED   = "jupiter_429_exhausted"
EC_TIMEOUT         = "jupiter_timeout"
EC_HTTP_ERROR      = "jupiter_http_error"
EC_OK              = ""


# ── Config loader ─────────────────────────────────────────────────────────────

def _cfg(attr: str, default):
    try:
        import memecoin.config as _c
        return getattr(_c, attr, default)
    except Exception:
        return default


# ── CSV log ───────────────────────────────────────────────────────────────────

_LOG_PATH      = Path(__file__).parent.parent / "logs" / "jupiter_requests.csv"
_LOG_FIELDNAMES = [
    "timestamp", "purpose", "endpoint", "mint",
    "success", "http_status", "error_class",
    "latency_ms", "retry_count",
    "bucket_tokens_remaining", "was_429", "backoff_ms",
]
_log_lock = threading.Lock()


def _write_log(row: dict) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _LOG_PATH.exists()
        with _log_lock:
            with open(_LOG_PATH, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_LOG_FIELDNAMES, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerow(row)
    except Exception as e:
        log.debug("jupiter_governor: CSV log write failed: %s", e)


# ── Token bucket ──────────────────────────────────────────────────────────────

class _TokenBucket:
    """
    Thread-safe leaky token bucket.

    Tokens refill continuously at `rate` tokens/second up to `capacity`.
    Starts full (capacity tokens available immediately).
    """

    def __init__(self, rpm: int, burst: int):
        self.capacity   = float(burst)
        self.tokens     = float(burst)   # start full
        self.rate       = rpm / 60.0     # tokens per second
        self._last      = time.monotonic()
        self._lock      = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

    def try_consume(self) -> bool:
        """Non-blocking. Returns True and deducts 1 token, or False if empty."""
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def wait_and_consume(self, timeout_s: float) -> bool:
        """
        Blocking. Waits up to timeout_s for a token to become available.
        Returns True if a token was consumed, False if timed out.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                # Compute how long until next token arrives
                if self.rate > 0:
                    wait_s = min((1.0 - self.tokens) / self.rate, 0.1)
                else:
                    wait_s = 0.1

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(wait_s, remaining))

    @property
    def tokens_remaining(self) -> float:
        with self._lock:
            self._refill()
            return round(self.tokens, 2)


# ── Backoff helper ────────────────────────────────────────────────────────────

def _backoff_ms(attempt: int) -> float:
    """Exponential backoff with jitter, capped at JUPITER_BACKOFF_MAX_MS."""
    base   = _cfg("JUPITER_BACKOFF_BASE_MS",  250)
    ceil   = _cfg("JUPITER_BACKOFF_MAX_MS",   2500)
    jitter = _cfg("JUPITER_JITTER_MS",         100)
    raw    = base * (2 ** (attempt - 1))   # 250, 500, 1000, 2000, …
    capped = min(raw, ceil)
    return capped + random.uniform(0, jitter)


# ── Bucket wait timeouts ──────────────────────────────────────────────────────

_EXIT_WAIT_S      = 5.0   # wait up to 5s for an EXIT token
_EMERGENCY_WAIT_S = 10.0  # wait up to 10s for an EMERGENCY token


# ── Governor ─────────────────────────────────────────────────────────────────

class JupiterGovernor:
    """
    Single shared governor instance.  All Jupiter callers go through
    governor.request() or governor.acquire() + governor.release().

    Buckets are independent — consuming BACKGROUND tokens never reduces
    EXIT or EMERGENCY capacity.
    """

    def __init__(self):
        self._enabled = _cfg("JUPITER_GOVERNOR_ENABLED", True)
        self._buckets: dict[str, _TokenBucket] = {
            Purpose.BACKGROUND: _TokenBucket(
                rpm=_cfg("JUPITER_BACKGROUND_RPM", 20),
                burst=4,
            ),
            Purpose.EXIT: _TokenBucket(
                rpm=_cfg("JUPITER_EXIT_RPM", 15),
                burst=4,
            ),
            Purpose.EMERGENCY: _TokenBucket(
                rpm=_cfg("JUPITER_EMERGENCY_RPM", 6),
                burst=2,
            ),
        }
        self._max_retries = _cfg("JUPITER_MAX_RETRIES", 3)

    def _acquire_bucket(self, purpose: str) -> bool:
        """
        Acquire one token from the purpose bucket.
        BACKGROUND: non-blocking.
        EXIT: waits up to _EXIT_WAIT_S.
        EMERGENCY: waits up to _EMERGENCY_WAIT_S.
        Returns True if acquired.
        """
        if not self._enabled:
            return True   # governor off — always allow

        bucket = self._buckets[purpose]
        if purpose == Purpose.BACKGROUND:
            return bucket.try_consume()
        elif purpose == Purpose.EXIT:
            return bucket.wait_and_consume(_EXIT_WAIT_S)
        else:  # EMERGENCY
            return bucket.wait_and_consume(_EMERGENCY_WAIT_S)

    def tokens_remaining(self, purpose: str) -> float:
        """Current token count for a bucket (thread-safe snapshot)."""
        return self._buckets[purpose].tokens_remaining

    def request(
        self,
        purpose: str,
        endpoint: str,
        fn: Callable,
        mint: str = "",
        **kwargs,
    ) -> _requests.Response:
        """
        Execute a governed Jupiter HTTP request with retry on 429.

        Args:
            purpose:  Purpose.BACKGROUND / EXIT / EMERGENCY
            endpoint: "quote" or "swap" (for logging)
            fn:       requests.get or requests.post (the HTTP callable)
            mint:     token mint address (for logging)
            **kwargs: passed verbatim to fn()

        Returns:
            requests.Response on success (status < 400, or non-429 error
            re-raised immediately without retry)

        Raises:
            requests.exceptions.HTTPError  on final 429 exhaustion or non-429 HTTP error
            requests.exceptions.Timeout    if all retries time out
            RuntimeError("jupiter_bucket_empty") if BACKGROUND bucket is empty

        Logs every attempt to logs/jupiter_requests.csv.
        """
        # ── Acquire bucket slot ────────────────────────────────────────────────
        acquired = self._acquire_bucket(purpose)
        tokens_snap = self.tokens_remaining(purpose)

        if not acquired:
            _write_log({
                "timestamp":              datetime.now(timezone.utc).isoformat(),
                "purpose":                purpose,
                "endpoint":               endpoint,
                "mint":                   mint[:16] if mint else "",
                "success":                False,
                "http_status":            0,
                "error_class":            EC_BUCKET_EMPTY,
                "latency_ms":             0,
                "retry_count":            0,
                "bucket_tokens_remaining": tokens_snap,
                "was_429":                False,
                "backoff_ms":             0,
            })
            if purpose == Purpose.BACKGROUND:
                log.debug(
                    "jupiter_governor: BACKGROUND bucket empty — skipping  endpoint=%s  mint=%s",
                    endpoint, mint[:8] if mint else "",
                )
                raise RuntimeError(EC_BUCKET_EMPTY)
            else:
                # EXIT / EMERGENCY waited and still couldn't get a token
                log.error(
                    "jupiter_governor: %s bucket timed out waiting for token  endpoint=%s  mint=%s",
                    purpose, endpoint, mint[:8] if mint else "",
                )
                raise RuntimeError(EC_BUCKET_EMPTY)

        # ── Request loop with retry on 429 ────────────────────────────────────
        t0         = time.time()
        last_exc   = None
        total_bkoff= 0.0
        was_429    = False

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = fn(**kwargs)
            except _requests.exceptions.Timeout as exc:
                last_exc   = exc
                bkoff      = _backoff_ms(attempt)
                total_bkoff += bkoff
                log.warning(
                    "jupiter_governor: timeout  purpose=%s  endpoint=%s  attempt=%d/%d"
                    "  backoff_ms=%.0f  mint=%s",
                    purpose, endpoint, attempt, self._max_retries, bkoff, mint[:8] if mint else "",
                )
                _write_log({
                    "timestamp":              datetime.now(timezone.utc).isoformat(),
                    "purpose":                purpose,
                    "endpoint":               endpoint,
                    "mint":                   mint[:16] if mint else "",
                    "success":                False,
                    "http_status":            0,
                    "error_class":            EC_TIMEOUT,
                    "latency_ms":             round((time.time() - t0) * 1000, 1),
                    "retry_count":            attempt,
                    "bucket_tokens_remaining": self.tokens_remaining(purpose),
                    "was_429":                False,
                    "backoff_ms":             round(bkoff, 1),
                })
                if attempt < self._max_retries:
                    time.sleep(bkoff / 1000)
                continue

            latency_ms = round((time.time() - t0) * 1000, 1)

            if resp.status_code == 429:
                was_429   = True
                bkoff     = _backoff_ms(attempt)
                total_bkoff += bkoff
                ec        = EC_429_RETRYING if attempt < self._max_retries else EC_429_EXHAUSTED
                log.warning(
                    "jupiter_governor: 429  error_class=%s  purpose=%s  endpoint=%s"
                    "  attempt=%d/%d  backoff_ms=%.0f  mint=%s",
                    ec, purpose, endpoint, attempt, self._max_retries, bkoff,
                    mint[:8] if mint else "",
                )
                _write_log({
                    "timestamp":              datetime.now(timezone.utc).isoformat(),
                    "purpose":                purpose,
                    "endpoint":               endpoint,
                    "mint":                   mint[:16] if mint else "",
                    "success":                False,
                    "http_status":            429,
                    "error_class":            ec,
                    "latency_ms":             latency_ms,
                    "retry_count":            attempt,
                    "bucket_tokens_remaining": self.tokens_remaining(purpose),
                    "was_429":                True,
                    "backoff_ms":             round(bkoff, 1),
                })
                if attempt < self._max_retries:
                    time.sleep(bkoff / 1000)
                    continue
                # Final attempt exhausted — raise so caller can handle
                resp.raise_for_status()

            # Non-429 error: raise immediately (no retry)
            if resp.status_code >= 400:
                _write_log({
                    "timestamp":              datetime.now(timezone.utc).isoformat(),
                    "purpose":                purpose,
                    "endpoint":               endpoint,
                    "mint":                   mint[:16] if mint else "",
                    "success":                False,
                    "http_status":            resp.status_code,
                    "error_class":            EC_HTTP_ERROR,
                    "latency_ms":             latency_ms,
                    "retry_count":            attempt,
                    "bucket_tokens_remaining": self.tokens_remaining(purpose),
                    "was_429":                was_429,
                    "backoff_ms":             round(total_bkoff, 1),
                })
                resp.raise_for_status()

            # Success
            _write_log({
                "timestamp":              datetime.now(timezone.utc).isoformat(),
                "purpose":                purpose,
                "endpoint":               endpoint,
                "mint":                   mint[:16] if mint else "",
                "success":                True,
                "http_status":            resp.status_code,
                "error_class":            EC_OK,
                "latency_ms":             latency_ms,
                "retry_count":            attempt,
                "bucket_tokens_remaining": self.tokens_remaining(purpose),
                "was_429":                was_429,
                "backoff_ms":             round(total_bkoff, 1),
            })
            return resp

        # Exhausted via timeout path — no response obtained
        log.error(
            "jupiter_governor: all retries exhausted  purpose=%s  endpoint=%s"
            "  attempts=%d  mint=%s  last_exc=%s",
            purpose, endpoint, self._max_retries, mint[:8] if mint else "", last_exc,
        )
        raise last_exc or RuntimeError(EC_429_EXHAUSTED)

    def status(self) -> dict:
        """Return a snapshot of all bucket token counts (for health checks / logs)."""
        return {p: self._buckets[p].tokens_remaining for p in self._buckets}


# ── Module-level singleton ────────────────────────────────────────────────────

governor = JupiterGovernor()
