"""
execution_rpc.py — Multi-RPC failover client for execution-critical Solana RPC calls.

Used exclusively by jupiter_rescue.py for sendTransaction, getSignatureStatuses,
getTokenAccountsByOwner, and getTransaction.  Jupiter API calls go through
jupiter_governor instead.

Failover order:
  1. Try each URL in EXECUTION_RPC_URLS (primary list) in order.
  2. On 429 / 503 / timeout: skip to next.
  3. If all primaries fail: try EXECUTION_RPC_FALLBACK_URLS.
  4. If all exhausted: raise RuntimeError.

Config keys (memecoin.config):
  EXECUTION_RPC_URLS          — list[str]  primary endpoints
  EXECUTION_RPC_FALLBACK_URLS — list[str]  fallback endpoints
  EXECUTION_RPC_TIMEOUT_MS    — int        per-request timeout (default 3000)
  EXECUTION_RPC_MAX_RETRIES   — int        total URL attempts across primary+fallback (default 2)
"""

import logging

import requests as _requests

log = logging.getLogger(__name__)

_RETRY_STATUS_CODES = {429, 503}
_DEFAULT_TIMEOUT_SEC = 3.0
_DEFAULT_MAX_RETRIES = 2


def _load_config() -> tuple:
    try:
        from memecoin.config import (
            EXECUTION_RPC_URLS,
            EXECUTION_RPC_FALLBACK_URLS,
            EXECUTION_RPC_TIMEOUT_MS,
            EXECUTION_RPC_MAX_RETRIES,
        )
        return (
            list(EXECUTION_RPC_URLS),
            list(EXECUTION_RPC_FALLBACK_URLS),
            EXECUTION_RPC_TIMEOUT_MS / 1000.0,
            int(EXECUTION_RPC_MAX_RETRIES),
        )
    except (ImportError, AttributeError):
        # Fall back to the solana RPC in CHAINS if available
        try:
            from memecoin.config import CHAINS
            primary = [CHAINS.get("solana", {}).get("rpc", "https://api.mainnet-beta.solana.com")]
        except Exception:
            primary = ["https://api.mainnet-beta.solana.com"]
        return primary, [], _DEFAULT_TIMEOUT_SEC, _DEFAULT_MAX_RETRIES


class ExecutionRpcClient:
    """
    Failover JSON-RPC client for Solana execution-critical calls.

    Attributes
    ----------
    last_used_url : str
        URL that returned the last successful response.
    """

    def __init__(
        self,
        primary_urls: list | None = None,
        fallback_urls: list | None = None,
        timeout_sec: float | None = None,
        max_retries: int | None = None,
    ):
        cfg_primary, cfg_fallback, cfg_timeout, cfg_retries = _load_config()
        self._primary_urls  = primary_urls  if primary_urls  is not None else cfg_primary
        self._fallback_urls = fallback_urls if fallback_urls is not None else cfg_fallback
        self._timeout_sec   = timeout_sec   if timeout_sec   is not None else cfg_timeout
        self._max_retries   = max_retries   if max_retries   is not None else cfg_retries
        self.last_used_url: str = ""

    def all_urls(self) -> list:
        """Primary URLs followed by fallback URLs."""
        return list(self._primary_urls) + list(self._fallback_urls)

    def post(self, payload: dict, timeout_override: float | None = None) -> dict:
        """
        POST a JSON-RPC payload to the first responsive URL.

        Parameters
        ----------
        payload : dict
            JSON-RPC payload (must contain "method").
        timeout_override : float | None
            Per-call timeout override in seconds.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        RuntimeError
            When all URLs are exhausted.
        """
        urls    = self.all_urls()
        timeout = timeout_override if timeout_override is not None else self._timeout_sec
        # Cap total attempts: at most len(primary) + max_retries, but never exceed url count
        max_tries = min(len(urls), len(self._primary_urls) + self._max_retries)
        last_exc: Exception | None = None

        for idx, url in enumerate(urls):
            if idx >= max_tries:
                break
            try:
                resp = _requests.post(url, json=payload, timeout=timeout)
                if resp.status_code in _RETRY_STATUS_CODES:
                    log.warning(
                        "execution_rpc: %s → HTTP %s — rotating to next RPC",
                        url, resp.status_code,
                    )
                    last_exc = RuntimeError(f"{url} returned HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                self.last_used_url = url
                return resp.json()
            except _requests.exceptions.Timeout:
                log.warning("execution_rpc: timeout on %s — rotating", url)
                last_exc = TimeoutError(f"timeout: {url}")
            except Exception as exc:
                log.warning("execution_rpc: error on %s: %s — rotating", url, exc)
                last_exc = exc

        raise RuntimeError(
            f"execution_rpc: all {max_tries} RPC(s) exhausted. Last error: {last_exc}"
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_singleton: ExecutionRpcClient | None = None


def get_client() -> ExecutionRpcClient:
    """Return (or lazily create) the module-level singleton ExecutionRpcClient."""
    global _singleton
    if _singleton is None:
        _singleton = ExecutionRpcClient()
    return _singleton


def reset_client() -> None:
    """Force re-creation of the singleton.  Use in tests or after config reload."""
    global _singleton
    _singleton = None


def rpc_post(payload: dict, timeout_override_sec: float | None = None) -> dict:
    """Convenience wrapper: post through the module singleton."""
    return get_client().post(payload, timeout_override=timeout_override_sec)
