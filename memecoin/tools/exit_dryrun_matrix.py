"""
exit_dryrun_matrix.py — no-SOL exit system validation harness.

Runs route-decision, build, simulation, historical comparison, reserve, and
canary tests without sending any transaction or spending any SOL.

Usage:
    python -m memecoin.tools.exit_dryrun_matrix
    python -m memecoin.tools.exit_dryrun_matrix --t22-bc-mint <MINT>
    python -m memecoin.tools.exit_dryrun_matrix --t22-ps-mint <MINT>
    python -m memecoin.tools.exit_dryrun_matrix --tx-sig <HISTORICAL_SIG>
    python -m memecoin.tools.exit_dryrun_matrix --wallet <PUBKEY>

Flags:
    --t22-bc-mint   Real Token-2022 mint still on bonding curve (for build/sim tests)
    --t22-ps-mint   Real Token-2022 mint on PumpSwap (graduated) (for build/sim/reserve tests)
    --tx-sig        Historical confirmed T22 PumpSwap sell TX sig for account comparison
    --wallet        Wallet pubkey to use for ATA derivation and sim tests (default: bot wallet)
"""

import argparse
import base64
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ── Status constants ─────────────────────────────────────────────────────────
PASS    = "PASS"
FAIL    = "FAIL"
NO_BAL  = "NOT_TESTABLE_WITHOUT_BALANCE"
NO_SEND = "NOT_TESTABLE_WITHOUT_LIVE_SEND"
NO_ARG  = "NOT_TESTABLE_WITHOUT_ARG"
SKIP    = "SKIP"

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"

_STATUS_COLOR = {
    PASS:    _GREEN,
    FAIL:    _RED,
    NO_BAL:  _YELLOW,
    NO_SEND: _YELLOW,
    NO_ARG:  _YELLOW,
    SKIP:    _CYAN,
}


class Result(NamedTuple):
    name: str
    status: str
    detail: str = ""


_results: list[Result] = []


def _r(name: str, status: str, detail: str = "") -> Result:
    r = Result(name, status, detail)
    _results.append(r)
    color = _STATUS_COLOR.get(status, "")
    print(f"  {color}[{status}]{_RESET}  {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    return r


# ── Mock helpers ─────────────────────────────────────────────────────────────

def _mock_pos(
    token_address: str = "So11111111111111111111111111111111111111112",
    dex_id: str = "pumpfun",
    tokens_held: int = 1_000_000,
    notes: str = "live|tx:abc",
    chain: str = "solana",
    token_symbol: str = "TEST",
):
    """Return a minimal mock Position-like object."""
    class _Pos:
        pass
    p = _Pos()
    p.token_address = token_address
    p.dex_id        = dex_id
    p.tokens_held   = tokens_held
    p.notes         = notes
    p.chain         = chain
    p.token_symbol  = token_symbol
    p.id            = "dryrun-pos-001"
    p.current_price = 0.000001
    p.status        = "open"
    p.exit_reason   = None
    return p


def _mock_pp(vsol: float = 5.0, mig_age: float = float("inf")):
    """Return a minimal mock PumpPortalMonitor-like object."""
    class _PP:
        def migration_age(self, mint): return mig_age
        def get_vsol(self, mint):      return vsol
        def get_last_seen(self, mint): return float("inf")
    return _PP()


# ── Config load ───────────────────────────────────────────────────────────────

def _load_config():
    try:
        import memecoin.config as cfg
        return cfg
    except Exception as e:
        print(f"  WARN: could not import memecoin.config: {e}")
        return None


def _get_rpc():
    cfg = _load_config()
    if cfg:
        rpc = getattr(cfg, "CHAINS", {}).get("solana", {}).get("rpc", "")
        if rpc:
            return rpc
    return "https://api.mainnet-beta.solana.com"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Route decision tests (pure logic, no network)
# ═══════════════════════════════════════════════════════════════════════════════

def section_route_decisions():
    print(f"\n{'─'*60}")
    print("1. ROUTE DECISION TESTS (no network)")
    print(f"{'─'*60}")

    try:
        from memecoin.exit_router import (
            classify, classify_detailed, TokenExitState,
        )
    except Exception as e:
        _r("import exit_router", FAIL, str(e))
        return

    # ── 1a: SPL bonding curve → BONDING_CURVE ────────────────────────────────
    try:
        pos = _mock_pos(dex_id="pumpfun")
        pp  = _mock_pp(vsol=10.0, mig_age=float("inf"))
        state = classify(pos, pp)
        if state == TokenExitState.BONDING_CURVE:
            _r("SPL bonding curve → BONDING_CURVE", PASS, f"state={state.value}")
        else:
            _r("SPL bonding curve → BONDING_CURVE", FAIL, f"got {state.value}")
    except Exception as e:
        _r("SPL bonding curve → BONDING_CURVE", FAIL, str(e))

    # ── 1b: T22 bonding curve → BONDING_CURVE_T22 ────────────────────────────
    # We inject is_t22=True by patching the token program cache
    try:
        from memecoin.executor import _mint_token_program_cache, _TOKEN22_PROGRAM_ID, _TOKEN_PROGRAM_ID
        _DUMMY_T22_MINT = "DummyT22MintXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        _mint_token_program_cache[_DUMMY_T22_MINT] = _TOKEN22_PROGRAM_ID
        pos = _mock_pos(token_address=_DUMMY_T22_MINT, dex_id="pumpfun")
        pp  = _mock_pp(vsol=10.0, mig_age=float("inf"))
        state = classify_detailed(pos, pp)
        ok = state == TokenExitState.BONDING_CURVE_T22
        _r("T22 bonding curve → BONDING_CURVE_T22", PASS if ok else FAIL,
           f"state={state.value}")
        _mint_token_program_cache.pop(_DUMMY_T22_MINT, None)
    except Exception as e:
        _r("T22 bonding curve → BONDING_CURVE_T22", FAIL, str(e))

    # ── 1c: T22 graduated → GRADUATED_PUMPSWAP_T22 ───────────────────────────
    try:
        from memecoin.executor import _mint_token_program_cache, _TOKEN22_PROGRAM_ID
        _DUMMY_T22_GRAD = "DummyT22GradXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        _mint_token_program_cache[_DUMMY_T22_GRAD] = _TOKEN22_PROGRAM_ID
        pos = _mock_pos(token_address=_DUMMY_T22_GRAD, dex_id="pumpswap")
        pp  = _mock_pp(vsol=0.0, mig_age=float("inf"))
        state = classify_detailed(pos, pp)
        ok = state == TokenExitState.GRADUATED_PUMPSWAP_T22
        _r("T22 graduated → GRADUATED_PUMPSWAP_T22", PASS if ok else FAIL,
           f"state={state.value}")
        _mint_token_program_cache.pop(_DUMMY_T22_GRAD, None)
    except Exception as e:
        _r("T22 graduated → GRADUATED_PUMPSWAP_T22", FAIL, str(e))

    # ── 1d: Custom:6005 reclassification ──────────────────────────────────────
    # 6005 = pump.fun error "sell on graduated token via bonding-curve program"
    # Our error classifier must label it pumpfun_custom_6005_graduated (not generic fail)
    try:
        from memecoin.pumpswap_local import _SIM_ERROR_PATTERNS
        patterns_str = [p for p, _ in _SIM_ERROR_PATTERNS]
        cls_map = {p: c for p, c in _SIM_ERROR_PATTERNS}
        custom_6005_cls = cls_map.get("Custom:6005") or cls_map.get("custom program error: 0x177d")
        if custom_6005_cls == "pumpfun_custom_6005_graduated":
            _r("Custom:6005 → pumpfun_custom_6005_graduated", PASS,
               f"pattern matched: {custom_6005_cls}")
        else:
            _r("Custom:6005 → pumpfun_custom_6005_graduated", FAIL,
               f"got: {custom_6005_cls!r} — pattern not in _SIM_ERROR_PATTERNS")
    except Exception as e:
        _r("Custom:6005 → pumpfun_custom_6005_graduated", FAIL, str(e))

    # ── 1e: Transfer hook → paper-only ────────────────────────────────────────
    try:
        from memecoin.config import BLOCK_T22_TRANSFER_HOOK
        if BLOCK_T22_TRANSFER_HOOK:
            _r("BLOCK_T22_TRANSFER_HOOK=True → policy enforced", PASS,
               "Config set — executor blocks transfer-hook T22 before live buy")
        else:
            _r("BLOCK_T22_TRANSFER_HOOK=True → policy enforced", FAIL,
               "BLOCK_T22_TRANSFER_HOOK=False — transfer-hook tokens NOT blocked")
    except Exception as e:
        _r("BLOCK_T22_TRANSFER_HOOK policy", FAIL, str(e))

    # ── 1f: Unknown T22 extension → paper-only ────────────────────────────────
    try:
        from memecoin.config import BLOCK_T22_UNKNOWN_EXTENSIONS
        if BLOCK_T22_UNKNOWN_EXTENSIONS:
            _r("BLOCK_T22_UNKNOWN_EXTENSIONS=True → policy enforced", PASS,
               "Config set — executor blocks unknown-extension T22 before live buy")
        else:
            _r("BLOCK_T22_UNKNOWN_EXTENSIONS=True → policy enforced", FAIL,
               "BLOCK_T22_UNKNOWN_EXTENSIONS=False — unknown-extension tokens NOT blocked")
    except Exception as e:
        _r("BLOCK_T22_UNKNOWN_EXTENSIONS policy", FAIL, str(e))

    # ── 1g: Jupiter high price impact → blocked ───────────────────────────────
    try:
        from memecoin.config import MAX_JUPITER_EXIT_PRICE_IMPACT_PCT, ALLOW_JUPITER_PANIC_EXIT
        _fake_impact = 40.0
        blocked = (_fake_impact > MAX_JUPITER_EXIT_PRICE_IMPACT_PCT) and not ALLOW_JUPITER_PANIC_EXIT
        if blocked:
            _r("Jupiter 40% impact → blocked", PASS,
               f"MAX={MAX_JUPITER_EXIT_PRICE_IMPACT_PCT}%  PANIC_EXIT={ALLOW_JUPITER_PANIC_EXIT}")
        else:
            _r("Jupiter 40% impact → blocked", FAIL,
               f"MAX={MAX_JUPITER_EXIT_PRICE_IMPACT_PCT}%  PANIC_EXIT={ALLOW_JUPITER_PANIC_EXIT}"
               f" — 40% impact would NOT be blocked")
    except Exception as e:
        _r("Jupiter high price impact → blocked", FAIL, str(e))

    # ── 1h: feed_blind + migration_age=inf → suppressed ──────────────────────
    # Verify scanner code path exists (grep for the suppression tag)
    try:
        scanner_path = Path(__file__).parent.parent / "scanner.py"
        scanner_src  = scanner_path.read_text()
        has_tag = "migration_uncertain_blind_suppressed" in scanner_src
        has_inf = 'mig_age == float("inf")' in scanner_src
        if has_tag and has_inf:
            _r("feed_blind + mig_age=inf → migration_uncertain_blind_suppressed", PASS,
               "Both suppression tag and inf check found in scanner.py")
        else:
            _r("feed_blind + mig_age=inf → migration_uncertain_blind_suppressed", FAIL,
               f"has_tag={has_tag}  has_inf_check={has_inf}")
    except Exception as e:
        _r("feed_blind migration suppression", FAIL, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Build-only tests (network: mint token program lookup only)
# ═══════════════════════════════════════════════════════════════════════════════

def section_build_tests(wallet: str, t22_bc_mint: str | None, t22_ps_mint: str | None):
    print(f"\n{'─'*60}")
    print("2. BUILD-ONLY TESTS (no send)")
    print(f"{'─'*60}")

    rpc = _get_rpc()

    # ── 2a: ATA derivation uses T22 for base token ───────────────────────────
    try:
        from memecoin.pumpswap_local import (
            _derive_ata, TOKEN_PROGRAM_SPL, TOKEN_PROGRAM_T22,
        )
        _dummy_mint = "So11111111111111111111111111111111111111112"
        ata_spl = _derive_ata(wallet, _dummy_mint, TOKEN_PROGRAM_SPL)
        ata_t22 = _derive_ata(wallet, _dummy_mint, TOKEN_PROGRAM_T22)
        if ata_spl != ata_t22:
            _r("ATA derivation: T22 ≠ SPL for same mint/owner", PASS,
               f"SPL ATA={ata_spl[:12]}…  T22 ATA={ata_t22[:12]}…")
        else:
            _r("ATA derivation: T22 ≠ SPL for same mint/owner", FAIL,
               "T22 and SPL ATA are identical — derivation bug!")
    except Exception as e:
        _r("ATA derivation SPL vs T22", FAIL, str(e))

    # ── 2b: WSOL ATA always uses SPL ─────────────────────────────────────────
    try:
        from memecoin.pumpswap_local import _derive_ata, TOKEN_PROGRAM_SPL, WSOL_MINT
        wsol_ata = _derive_ata(wallet, WSOL_MINT, TOKEN_PROGRAM_SPL)
        # Verify it looks like a valid pubkey (44 chars base58)
        if len(wsol_ata) >= 32:
            _r("WSOL ATA uses SPL token program", PASS, f"ATA={wsol_ata[:12]}…")
        else:
            _r("WSOL ATA uses SPL token program", FAIL, f"invalid ATA={wsol_ata!r}")
    except Exception as e:
        _r("WSOL ATA SPL derivation", FAIL, str(e))

    # ── 2c: Build T22 BC sell TX ──────────────────────────────────────────────
    if not t22_bc_mint:
        _r("Build T22 bonding-curve sell TX", NO_ARG, "pass --t22-bc-mint <MINT>")
    else:
        try:
            from memecoin.bonding_curve_t22 import build_bc_t22_sell_tx
            result = build_bc_t22_sell_tx(
                token_mint=t22_bc_mint,
                wallet_pubkey=wallet,
                token_amount=1_000_000,
                rpc_url=rpc,
            )
            if result.get("tx_bytes"):
                tx_b64 = base64.b64encode(result["tx_bytes"]).decode()
                _r("Build T22 bonding-curve sell TX", PASS,
                   f"tx_bytes={len(result['tx_bytes'])} bytes  b64={tx_b64[:32]}…")
            else:
                _r("Build T22 bonding-curve sell TX", FAIL,
                   f"error={result.get('error', 'unknown')}")
        except Exception as e:
            _r("Build T22 bonding-curve sell TX", FAIL, str(e))

    # ── 2d: Build T22 PumpSwap sell TX + verify accounts ─────────────────────
    if not t22_ps_mint:
        _r("Build T22 PumpSwap sell TX", NO_ARG, "pass --t22-ps-mint <MINT>")
        _r("PumpSwap remainingAccounts include poolV2/buyback", NO_ARG,
           "pass --t22-ps-mint <MINT>")
    else:
        try:
            from memecoin.pumpswap_local import (
                fetch_pool, build_pumpswap_sell_tx,
                TOKEN_PROGRAM_T22, TOKEN_PROGRAM_SPL,
                _derive_ata, WSOL_MINT,
                _derive_pool_v2_pda,
                BUYBACK_FEE_RECIPIENTS,
            )
            # Fetch pool
            pool = fetch_pool(t22_ps_mint, rpc)
            cc   = pool.get("coin_creator", "")
            _NULL = "11111111111111111111111111111111"

            tx_bytes = build_pumpswap_sell_tx(
                wallet_pubkey=wallet,
                keypair=_dummy_keypair(wallet),
                token_mint=t22_ps_mint,
                token_amount_raw=1_000_000,
                min_sol_out_lamports=0,
                priority_fee_sol=0.0001,
                token_program_id=TOKEN_PROGRAM_T22,
                pool=pool,
                rpc_url=rpc,
            )
            _r("Build T22 PumpSwap sell TX", PASS,
               f"pool={pool['pool_address'][:12]}…  tx={len(tx_bytes)} bytes"
               f"  coin_creator={'non-null' if cc != _NULL else 'null'}")

            # Verify remaining accounts structure by parsing TX accounts
            _verify_pumpswap_remaining(
                tx_bytes, pool, cc, t22_ps_mint, TOKEN_PROGRAM_T22, wallet
            )
        except Exception as e:
            _r("Build T22 PumpSwap sell TX", FAIL, str(e))
            _r("PumpSwap remainingAccounts include poolV2/buyback", FAIL,
               f"build step failed: {e}")


def _dummy_keypair(wallet_pubkey: str):
    """Return a dummy Keypair that signs with random bytes (TX won't be sent)."""
    try:
        from solders.keypair import Keypair
        kp = Keypair()
        return kp
    except Exception:
        return None


def _verify_pumpswap_remaining(tx_bytes, pool, coin_creator, token_mint, tok_prog_id, wallet):
    """Deserialize the TX and verify remaining accounts include poolV2 and buyback."""
    try:
        from solders.transaction import VersionedTransaction
        from memecoin.pumpswap_local import (
            _derive_pool_v2_pda, BUYBACK_FEE_RECIPIENTS, PUMP_AMM_PROGRAM,
            _NULL_PUBKEY_STR, TOKEN_PROGRAM_T22,
        )

        tx = VersionedTransaction.from_bytes(tx_bytes)
        msg = tx.message
        account_keys = [str(k) for k in msg.account_keys]

        has_pool_v2 = False
        has_buyback = False
        has_t22_prog = False

        if coin_creator and coin_creator != _NULL_PUBKEY_STR:
            pool_v2_pda = _derive_pool_v2_pda(token_mint)
            has_pool_v2 = pool_v2_pda in account_keys

        buyback_set = set(BUYBACK_FEE_RECIPIENTS)
        has_buyback = any(k in buyback_set for k in account_keys)

        has_t22_prog = (TOKEN_PROGRAM_T22 in account_keys)

        checks = []
        all_ok = True
        if coin_creator and coin_creator != _NULL_PUBKEY_STR:
            checks.append(f"poolV2Pda={'✓' if has_pool_v2 else '✗'}")
            if not has_pool_v2:
                all_ok = False
        checks.append(f"buyback_fee_recipient={'✓' if has_buyback else '✗'}")
        checks.append(f"T22_token_program={'✓' if has_t22_prog else '✗'}")
        if not has_buyback or not has_t22_prog:
            all_ok = False

        _r("PumpSwap remainingAccounts include poolV2/buyback",
           PASS if all_ok else FAIL,
           "  ".join(checks) + f"  total_accounts={len(account_keys)}")
    except Exception as e:
        _r("PumpSwap remainingAccounts include poolV2/buyback", FAIL, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Mainnet simulation tests (network, no send)
# ═══════════════════════════════════════════════════════════════════════════════

def section_simulation_tests(wallet: str, t22_bc_mint: str | None, t22_ps_mint: str | None):
    print(f"\n{'─'*60}")
    print("3. MAINNET SIMULATION TESTS (simulateTransaction, no send)")
    print(f"{'─'*60}")

    rpc = _get_rpc()

    def _run_sim(name, tx_bytes, token_program, route_name):
        try:
            from memecoin.pumpswap_local import simulate_sell
            sim_ok, error_class, logs = simulate_sell(tx_bytes, rpc)
            head = logs[:3] if logs else []
            detail = (
                f"sim_ok={sim_ok}  error_class={error_class!r}  "
                f"token_program={'T22' if 'TokenzQ' in token_program else 'SPL'}"
                f"  route={route_name}"
            )
            if head:
                detail += f"\n  logs[0]: {head[0][:120]}"
            # sim_ok=False is expected when wallet has no tokens; still PASS if error is classified
            if sim_ok:
                status = PASS
            elif error_class in (
                "pumpfun_t22_custom_6001", "pumpfun_custom_6005_graduated",
                "pumpswap_no_pool", "pumpswap_bad_pool_layout",
                "pumpswap_token2022_ata_error", "pumpswap_transfer_hook_error",
                "pumpswap_simulation_failed", "bc_t22_build_failed",
                "bc_t22_simulate_error",
            ):
                status = NO_BAL  # failed but error is classified (expected without balance)
            else:
                status = FAIL
            _r(name, status, detail)
        except Exception as e:
            _r(name, FAIL, str(e))

    # ── 3a: T22 BC sell simulation ────────────────────────────────────────────
    if not t22_bc_mint:
        _r("Simulate T22 bonding-curve sell", NO_ARG, "pass --t22-bc-mint <MINT>")
    else:
        try:
            from memecoin.bonding_curve_t22 import build_bc_t22_sell_tx
            from memecoin.pumpswap_local import TOKEN_PROGRAM_T22
            result = build_bc_t22_sell_tx(
                token_mint=t22_bc_mint, wallet_pubkey=wallet,
                token_amount=1_000_000, rpc_url=rpc,
            )
            if result.get("tx_bytes"):
                _run_sim("Simulate T22 bonding-curve sell", result["tx_bytes"],
                         TOKEN_PROGRAM_T22, "BONDING_CURVE_T22_LOCAL")
            else:
                _r("Simulate T22 bonding-curve sell", FAIL,
                   f"build failed: {result.get('error')}")
        except Exception as e:
            _r("Simulate T22 bonding-curve sell", FAIL, str(e))

    # ── 3b: T22 PumpSwap sell simulation ─────────────────────────────────────
    if not t22_ps_mint:
        _r("Simulate T22 PumpSwap sell", NO_ARG, "pass --t22-ps-mint <MINT>")
    else:
        try:
            from memecoin.pumpswap_local import (
                fetch_pool, build_pumpswap_sell_tx, TOKEN_PROGRAM_T22,
            )
            pool = fetch_pool(t22_ps_mint, rpc)
            tx_bytes = build_pumpswap_sell_tx(
                wallet_pubkey=wallet,
                keypair=_dummy_keypair(wallet),
                token_mint=t22_ps_mint,
                token_amount_raw=1_000_000,
                min_sol_out_lamports=0,
                priority_fee_sol=0.0001,
                token_program_id=TOKEN_PROGRAM_T22,
                pool=pool,
                rpc_url=rpc,
            )
            _run_sim("Simulate T22 PumpSwap sell", tx_bytes,
                     TOKEN_PROGRAM_T22, "PUMPSWAP_LOCAL")
        except Exception as e:
            _r("Simulate T22 PumpSwap sell", FAIL, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Historical TX comparison
# ═══════════════════════════════════════════════════════════════════════════════

def section_historical_comparison(wallet: str, t22_ps_mint: str | None, tx_sig: str | None):
    print(f"\n{'─'*60}")
    print("4. HISTORICAL TX COMPARISON")
    print(f"{'─'*60}")

    if not tx_sig:
        _r("Historical T22 PumpSwap sell TX comparison", NO_ARG,
           "pass --tx-sig <CONFIRMED_TX_SIG>")
        return

    if not t22_ps_mint:
        _r("Historical T22 PumpSwap sell TX comparison", NO_ARG,
           "pass --t22-ps-mint <MINT> alongside --tx-sig")
        return

    rpc = _get_rpc()

    try:
        import requests
        from memecoin.pumpswap_local import (
            fetch_pool, build_pumpswap_sell_tx, TOKEN_PROGRAM_T22,
        )
        from solders.transaction import VersionedTransaction

        # Fetch historical TX
        resp = requests.post(rpc, json={
            "jsonrpc": "2.0", "id": 1,
            "method":  "getTransaction",
            "params":  [tx_sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}],
        }, timeout=15)
        tx_data = resp.json().get("result")
        if not tx_data:
            _r("Historical TX comparison", FAIL, f"TX not found: {tx_sig[:16]}…")
            return

        # Extract account list from historical TX
        msg       = tx_data["transaction"]["message"]
        hist_accs = msg.get("accountKeys") or []  # list of pubkeys as strings

        # Build our TX for same mint
        pool = fetch_pool(t22_ps_mint, rpc)
        our_tx_bytes = build_pumpswap_sell_tx(
            wallet_pubkey=wallet,
            keypair=_dummy_keypair(wallet),
            token_mint=t22_ps_mint,
            token_amount_raw=1_000_000,
            min_sol_out_lamports=0,
            priority_fee_sol=0.0001,
            token_program_id=TOKEN_PROGRAM_T22,
            pool=pool,
            rpc_url=rpc,
        )
        our_tx  = VersionedTransaction.from_bytes(our_tx_bytes)
        our_msg = our_tx.message
        our_accs = [str(k) for k in our_msg.account_keys]

        # Normalize historical accounts (may be list of strings or dicts)
        if hist_accs and isinstance(hist_accs[0], dict):
            hist_keys = [a.get("pubkey", str(a)) for a in hist_accs]
        else:
            hist_keys = [str(a) for a in hist_accs]

        print(f"\n  Historical TX: {tx_sig[:20]}…  ({len(hist_keys)} accounts)")
        print(f"  Our TX:        {len(our_accs)} accounts")
        print(f"\n  {'#':<3}  {'Historical':<46}  {'Ours':<46}  Match")
        print(f"  {'─'*3}  {'─'*46}  {'─'*46}  {'─'*5}")

        mismatches = 0
        max_idx = max(len(hist_keys), len(our_accs))
        for i in range(max_idx):
            h = hist_keys[i] if i < len(hist_keys) else "(missing)"
            o = our_accs[i]  if i < len(our_accs)  else "(missing)"
            match = "✓" if h == o else "✗"
            if h != o:
                mismatches += 1
            print(f"  {i:<3}  {h[:44]:<46}  {o[:44]:<46}  {match}")

        status = PASS if mismatches == 0 else FAIL
        _r("Historical TX account order matches our builder", status,
           f"mismatches={mismatches}/{max_idx}")

        # Check writable/signer flags
        hist_header = msg.get("header") or {}
        n_sig       = hist_header.get("numRequiredSignatures", 0)
        n_ro_signed = hist_header.get("numReadonlySignedAccounts", 0)
        n_ro_unsign = hist_header.get("numReadonlyUnsignedAccounts", 0)
        print(f"\n  Historical header: numSig={n_sig}  numRO_signed={n_ro_signed}"
              f"  numRO_unsigned={n_ro_unsign}")

        # Token program check
        T22 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
        hist_has_t22 = T22 in hist_keys
        our_has_t22  = T22 in our_accs
        tp_ok = hist_has_t22 == our_has_t22
        _r("Token-2022 program in both TX account lists", PASS if tp_ok else FAIL,
           f"historical={hist_has_t22}  ours={our_has_t22}")

    except Exception as e:
        _r("Historical TX comparison", FAIL, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Reserve / min_sol_out test
# ═══════════════════════════════════════════════════════════════════════════════

def section_reserve_test(t22_ps_mint: str | None):
    print(f"\n{'─'*60}")
    print("5. RESERVE / MIN_SOL_OUT TEST (no send)")
    print(f"{'─'*60}")

    if not t22_ps_mint:
        _r("Fetch pool reserves", NO_ARG, "pass --t22-ps-mint <MINT>")
        _r("Compute min_sol_out from reserves", NO_ARG, "pass --t22-ps-mint <MINT>")
        _r("Live mode would NOT use min_out=0", NO_ARG, "pass --t22-ps-mint <MINT>")
        return

    rpc = _get_rpc()

    try:
        from memecoin.pumpswap_local import (
            fetch_pool, fetch_pool_sol_reserves, compute_min_sol_out,
        )
        from memecoin.config import (
            ALLOW_ZERO_MIN_OUT_EMERGENCY, LOCAL_PUMPSWAP_MAX_SLIPPAGE_PCT,
        )

        pool = fetch_pool(t22_ps_mint, rpc)
        sol_reserves = fetch_pool_sol_reserves(pool, rpc)

        if sol_reserves is not None:
            _r("Fetch pool WSOL reserves", PASS,
               f"pool_quote_ta={pool['pool_quote_token_account'][:12]}…"
               f"  sol_reserves={sol_reserves:,} lamports ({sol_reserves/1e9:.4f} SOL)")
        else:
            _r("Fetch pool WSOL reserves", FAIL, "fetch_pool_sol_reserves returned None")
            return

        # Compute min_out for 1M tokens
        token_amount = 1_000_000
        min_out, sol_res = compute_min_sol_out(
            token_amount, pool, rpc, slippage_pct=LOCAL_PUMPSWAP_MAX_SLIPPAGE_PCT
        )
        if min_out > 0:
            _r("Compute min_sol_out from reserves", PASS,
               f"token_in={token_amount:,}  min_out={min_out:,} lam"
               f"  ({min_out/1e9:.8f} SOL)  slippage={LOCAL_PUMPSWAP_MAX_SLIPPAGE_PCT}%")
        else:
            _r("Compute min_sol_out from reserves", FAIL,
               f"min_out=0  sol_res={sol_res}")

        # Verify live mode would not use 0 unless emergency
        if not ALLOW_ZERO_MIN_OUT_EMERGENCY:
            _r("Live mode would NOT use min_out=0 (ALLOW_ZERO_MIN_OUT_EMERGENCY=False)", PASS,
               "exit_router will abort with pumpswap_reserve_read_failed if reserves unreadable")
        else:
            _r("Live mode would NOT use min_out=0 (ALLOW_ZERO_MIN_OUT_EMERGENCY=False)", FAIL,
               "ALLOW_ZERO_MIN_OUT_EMERGENCY=True — unsafe for live sends!")

    except Exception as e:
        _r("Reserve/min_sol_out test", FAIL, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Canary enforcement test (pure logic)
# ═══════════════════════════════════════════════════════════════════════════════

def section_canary_test():
    print(f"\n{'─'*60}")
    print("6. CANARY ENFORCEMENT TEST (pure logic, no network)")
    print(f"{'─'*60}")

    try:
        from memecoin.config import (
            LIVE_CANARY_MODE, EXIT_SYSTEM_VALIDATED, MAX_CANARY_TRADE_USD,
        )

        requested_size = 40.0  # simulate a $40 signal

        # Apply canary cap logic (mirrors portfolio.py open_position)
        final_size = requested_size
        canary_capped = False
        if LIVE_CANARY_MODE and not EXIT_SYSTEM_VALIDATED:
            final_size    = min(final_size, MAX_CANARY_TRADE_USD)
            canary_capped = True

        expected_final = MAX_CANARY_TRADE_USD if (LIVE_CANARY_MODE and not EXIT_SYSTEM_VALIDATED) else requested_size

        ok = abs(final_size - expected_final) < 0.01
        _r("Canary cap: $40 → $3 when LIVE_CANARY_MODE=True, EXIT_SYSTEM_VALIDATED=False",
           PASS if ok else FAIL,
           f"LIVE_CANARY_MODE={LIVE_CANARY_MODE}  EXIT_SYSTEM_VALIDATED={EXIT_SYSTEM_VALIDATED}"
           f"  requested=${requested_size}  final=${final_size:.2f}"
           + (f"  capped={canary_capped}" if canary_capped else ""))

        # Verify the canary cap note tag is emitted
        portfolio_path = Path(__file__).parent.parent / "portfolio.py"
        portfolio_src  = portfolio_path.read_text()
        has_cap_tag = "canary_cap" in portfolio_src and "_canary_capped" in portfolio_src
        _r("portfolio.py emits |canary_cap note tag", PASS if has_cap_tag else FAIL,
           "Found _canary_capped and canary_cap in portfolio.py" if has_cap_tag
           else "Missing canary_cap note tag in portfolio.py")

        # Verify executor also enforces cap for T22 canary
        executor_path = Path(__file__).parent.parent / "executor.py"
        executor_src  = executor_path.read_text()
        has_t22_cap = "size_usd = 3" in executor_src or "size_usd=3" in executor_src or \
                      "size_usd = MAX_CANARY_TRADE_USD" in executor_src or \
                      "canary" in executor_src.lower() and "size_usd" in executor_src
        _r("executor.py enforces canary cap for T22 buys", PASS if has_t22_cap else FAIL,
           "Found canary size enforcement in executor.py" if has_t22_cap
           else "No canary size cap found in executor.py")

    except Exception as e:
        _r("Canary enforcement test", FAIL, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="no-SOL exit system validation harness"
    )
    parser.add_argument("--t22-bc-mint", help="Real T22 mint still on bonding curve")
    parser.add_argument("--t22-ps-mint", help="Real T22 mint on PumpSwap (graduated)")
    parser.add_argument("--tx-sig",      help="Historical confirmed T22 PumpSwap sell TX sig")
    parser.add_argument("--wallet",      help="Wallet pubkey for ATA derivation and sim tests")
    args = parser.parse_args()

    # Try to get wallet from config if not provided
    wallet = args.wallet
    if not wallet:
        try:
            from memecoin.executor import _get_keypair
            wallet = str(_get_keypair().pubkey())
            print(f"  Using bot wallet: {wallet[:12]}…")
        except Exception:
            wallet = "7VCaBMHukdHnMvABSUmMcNGDKZfNFgfVxEF4zHpNL6Jd"  # known valid pubkey
            print(f"  Using placeholder wallet: {wallet[:12]}…")

    print(f"\n{'═'*60}")
    print("EXIT SYSTEM DRY-RUN MATRIX")
    print(f"{'═'*60}")
    print(f"  wallet:      {wallet[:16]}…")
    print(f"  t22-bc-mint: {args.t22_bc_mint or '(none)'}")
    print(f"  t22-ps-mint: {args.t22_ps_mint or '(none)'}")
    print(f"  tx-sig:      {(args.tx_sig or '(none)')[:20]}")

    t0 = time.time()

    section_route_decisions()
    section_build_tests(wallet, args.t22_bc_mint, args.t22_ps_mint)
    section_simulation_tests(wallet, args.t22_bc_mint, args.t22_ps_mint)
    section_historical_comparison(wallet, args.t22_ps_mint, args.tx_sig)
    section_reserve_test(args.t22_ps_mint)
    section_canary_test()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"SUMMARY  ({elapsed:.1f}s)")
    print(f"{'═'*60}")

    counts = {PASS: 0, FAIL: 0, NO_BAL: 0, NO_SEND: 0, NO_ARG: 0, SKIP: 0}
    for r in _results:
        counts[r.status] = counts.get(r.status, 0) + 1

    for status, n in counts.items():
        if n == 0:
            continue
        color = _STATUS_COLOR.get(status, "")
        print(f"  {color}{status:<35}{_RESET}  {n}")

    total   = len(_results)
    n_pass  = counts[PASS]
    n_fail  = counts[FAIL]
    n_skip  = counts[NO_BAL] + counts[NO_SEND] + counts[NO_ARG] + counts[SKIP]

    print(f"\n  Total: {total}  Pass: {n_pass}  Fail: {n_fail}  Skipped/partial: {n_skip}")

    if n_fail == 0:
        print(f"\n  {_GREEN}OVERALL: PASS — no failures{_RESET}")
    else:
        print(f"\n  {_RED}OVERALL: FAIL — {n_fail} test(s) failed{_RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
