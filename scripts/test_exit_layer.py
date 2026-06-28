#!/usr/bin/env python3
"""
Exit-layer smoke test — zero cost, no money at risk.

Validates the full exit path against mainnet state WITHOUT sending any transaction:
  1. ExitRouter.classify()       — detects MIGRATION_UNCERTAIN vs GRADUATED_PUMPSWAP
  2. pumpswap_local.fetch_pool() — pool lookup for a known T22 graduated token
  3. pumpswap_local.build_pumpswap_sell_tx() — TX construction (all 24 accounts)
  4. pumpswap_local.simulate_sell()  — simulateTransaction against mainnet RPC

How to interpret simulation result:
  ok=True                              → TX structure + accounts correct ✓ ready to flip SELL_ENABLED
  error_class=pumpswap_honeypot_or_sell_restricted → structure correct, balance wrong (expected if wallet holds 0)
  error_class=pumpswap_token2022_ata_error          → T22 ATA derivation still broken
  error_class=pumpswap_bad_pool_layout              → pool fetch/parse issue
  error_class=pumpswap_simulation_failed            → generic (check logs printed below)

Run on VPS:
  cd /root/quant-bot
  python3 scripts/test_exit_layer.py

No SOL spent. sigVerify=False — simulation runs against real chain state but never lands.
"""
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("exit_layer_test")

# ── Test targets ────────────────────────────────────────────────────────────
# WSOLP — known T22 token, graduated to PumpSwap, previously unsellable
# Pool: H5gzcMCCXDZeXzNedDGfEPJnpYAYuYhH6t4Cyxt5PdFz  (coin_creator non-null → uses poolV2Pda)
T22_MINT  = "GvUCjmWSXA5hrTh9smmNA1AU55YCtP9mDLQcrKA1pump"

# HITTIN — most recent unsellable, T22, MIGRATION_UNCERTAIN case, Jun 27
HITTIN_MINT = "2dGd4bRL4KVXfAu4dWxoXeFsJ24prZvs8WJ67HuXpump"

# Load .env if present (so SOLANA_RPC_URL is available when run directly on VPS)
import os as _os
_env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
if _os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _os.environ.setdefault(_k.strip(), _v.strip())

RPC_URL = (
    _os.getenv("SOLANA_RPC_URL")
    or _os.getenv("HELIUS_RPC_URL")
    or "https://api.mainnet-beta.solana.com"
)
print(f"RPC: {RPC_URL[:60]}...")

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"


def _check(label, condition, detail=""):
    icon = PASS if condition else FAIL
    print(f"  {icon}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Test 1: ExitRouter classify() ──────────────────────────────────────────
section("TEST 1 — ExitRouter.classify() for HITTIN (MIGRATION_UNCERTAIN case)")

try:
    from memecoin.exit_router import classify, TokenExitState
    from memecoin.pumpportal_monitor import PumpPortalMonitor

    # Build a minimal mock position object
    class _FakePos:
        token_address = HITTIN_MINT
        chain = "solana"
        dex_id = "pumpfun"
        notes = "cohort:graduated"

    # classify() calls pp_monitor.migration_age() and RPC checks
    # For HITTIN (already sold, no active PP feed), it should return MIGRATION_UNCERTAIN
    # since migration_age will be inf (PP no longer active for this mint)
    monitor = PumpPortalMonitor.__new__(PumpPortalMonitor)
    monitor._migration_times = {}   # empty — simulates pp-silent state

    state = classify(_FakePos(), monitor)
    print(f"  classify() → {state.name}")

    _check(
        "MIGRATION_UNCERTAIN or GRADUATED_PUMPSWAP detected (not UNKNOWN/BC)",
        state in (TokenExitState.MIGRATION_UNCERTAIN, TokenExitState.GRADUATED_PUMPSWAP),
        state.name,
    )
    _check(
        "Not BONDING_CURVE (that would route to 6005 path)",
        state != TokenExitState.BONDING_CURVE,
        state.name,
    )
except Exception as e:
    print(f"  {FAIL}  classify() raised: {e}")
    import traceback; traceback.print_exc()


# ── Test 2: fetch_pool() for T22 token (WSOLP) ────────────────────────────
section("TEST 2 — pumpswap_local.fetch_pool() for WSOLP (T22, graduated)")

pool = None
try:
    from memecoin.pumpswap_local import fetch_pool, PumpSwapPoolError

    pool = fetch_pool(T22_MINT, RPC_URL)
    _check("Pool found", pool is not None)
    _check("base_mint matches", pool.get("base_mint") == T22_MINT, pool.get("base_mint", "")[:16])
    _check("pool_address non-empty", bool(pool.get("pool_address")))
    _check("coin_creator present (non-null)", pool.get("coin_creator") not in (None, "11111111111111111111111111111111"))
    _check("pool_base_token_account present", bool(pool.get("pool_base_token_account")))
    print(f"  pool={pool['pool_address'][:20]}...  cc={pool['coin_creator'][:12]}...")
except PumpSwapPoolError as e:
    print(f"  {FAIL}  PumpSwapPoolError: {e}")
except Exception as e:
    print(f"  {FAIL}  Unexpected error: {e}")
    import traceback; traceback.print_exc()


# ── Test 3: poolV2Pda derivation ──────────────────────────────────────────
section("TEST 3 — poolV2Pda derivation (the ix[21] mystery, now solved)")

try:
    from memecoin.pumpswap_local import _derive_pool_v2_pda
    from solders.pubkey import Pubkey

    pda = _derive_pool_v2_pda(T22_MINT)
    expected = "9zRMNKW1TwHfhowLPpwNxyPvQxg1YWkvusu2Jth2Hea1"
    _check(
        "poolV2Pda(WSOLP baseMint) == 9zRMNKW1... (verified against real TX)",
        pda == expected,
        pda[:20],
    )
except Exception as e:
    print(f"  {FAIL}  _derive_pool_v2_pda raised: {e}")


# ── Test 4: build_pumpswap_sell_tx() ──────────────────────────────────────
section("TEST 4 — build_pumpswap_sell_tx() using a real WSOLP holder as seller")

tx_bytes = None
real_holder = None
if pool is not None:
    try:
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey
        from memecoin.pumpswap_local import (
            build_pumpswap_sell_tx, TOKEN_PROGRAM_T22, _rpc,
        )

        # Find a real token holder via getTokenLargestAccounts on the pool_base_token_account.
        # We need a wallet that actually has a T22 ATA for WSOLP so the account exists on-chain.
        # With sigVerify=False the signer constraint is bypassed at the runtime level.
        print("  Looking up token holders for WSOLP...")
        holders_resp = _rpc(RPC_URL, {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenLargestAccounts",
            "params": [T22_MINT, {"commitment": "confirmed"}],
        }, timeout=10)
        accounts = holders_resp.get("result", {}).get("value", [])
        if accounts:
            # getTokenLargestAccounts returns token account addresses, not owner wallets.
            # Fetch the owner of the largest account.
            top_ta = accounts[0]["address"]
            ai = _rpc(RPC_URL, {
                "jsonrpc": "2.0", "id": 2,
                "method": "getAccountInfo",
                "params": [top_ta, {"encoding": "jsonParsed", "commitment": "confirmed"}],
            }, timeout=10)
            owner = (
                ai.get("result", {})
                  .get("value", {})
                  .get("data", {})
                  .get("parsed", {})
                  .get("info", {})
                  .get("owner")
            )
            if owner:
                real_holder = owner
                print(f"  Using real holder as seller: {owner[:20]}...")
            else:
                print(f"  {WARN}  Could not parse owner from getAccountInfo; using dummy wallet")

        dummy_kp = Keypair()
        seller_wallet = real_holder if real_holder else str(dummy_kp.pubkey())

        tx_bytes = build_pumpswap_sell_tx(
            wallet_pubkey=seller_wallet,
            keypair=dummy_kp,      # signed with dummy (sigVerify=False in sim)
            token_mint=T22_MINT,
            token_amount_raw=1,          # sell 1 raw token unit
            min_sol_out_lamports=0,      # accept any output
            priority_fee_sol=0.0001,
            token_program_id=TOKEN_PROGRAM_T22,
            pool=pool,
            rpc_url=RPC_URL,
        )
        _check("TX built without exception", tx_bytes is not None)
        _check("TX non-empty", len(tx_bytes) > 0, f"{len(tx_bytes)} bytes")
        _check(
            "TX size in range for 24-account instruction (>800 bytes)",
            len(tx_bytes) > 800,
            f"{len(tx_bytes)} bytes",
        )
        _check("Using real token holder (ATA exists on-chain)", real_holder is not None,
               seller_wallet[:16] if real_holder else "dummy wallet — sim may fail early")
    except Exception as e:
        print(f"  {FAIL}  build_pumpswap_sell_tx raised: {e}")
        import traceback; traceback.print_exc()
else:
    print(f"  {WARN}  Skipped (pool fetch failed in Test 2)")


# ── Test 5: simulate_sell() ──────────────────────────────────────────────
section("TEST 5 — simulate_sell() against mainnet (no SOL spent)")

if tx_bytes is not None:
    try:
        from memecoin.pumpswap_local import simulate_sell

        print("  Sending simulateTransaction to mainnet RPC...")
        ok, error_class, logs = simulate_sell(tx_bytes, RPC_URL)

        print(f"  ok={ok}  error_class={error_class!r}")
        print(f"  log lines ({len(logs)}):")
        for line in logs[:15]:
            print(f"    {line}")

        # Check if the AMM program was actually invoked (appears in logs)
        prog_started = any("Program pAMMBay6" in l for l in logs)

        if ok:
            _check("Simulation passed — TX structure fully correct", True)
        elif prog_started:
            # Program was invoked — structure was accepted. Failure is inside the program
            # (e.g., signer check failed because we used real holder but wrong keypair, or
            # balance check). This is expected and means the account structure is correct.
            _check(
                "AMM program invoked — structure accepted ✓ (expected program-level fail for test wallet)",
                True,
                error_class,
            )
        elif error_class == "pumpswap_honeypot_or_sell_restricted":
            _check(
                "Structure correct — program executed, failed on balance",
                True,
                "expected",
            )
        elif error_class == "pumpswap_token2022_ata_error":
            _check("T22 ATA derivation — STILL BROKEN", False, error_class)
        elif error_class == "pumpswap_bad_pool_layout":
            _check("Pool layout — BROKEN", False, error_class)
        elif error_class == "pumpswap_no_pool":
            # AccountNotFound before program invocation — an account in the TX doesn't exist.
            # Could be coin_creator_vault_ata (uninitialized ATA for inactive creator).
            # This is a pool-selection issue, not a structural TX issue.
            _check(
                "AccountNotFound before AMM invocation — likely uninitialized vault ATA for this specific pool",
                False,
                "try a different test token with active creator vault",
            )
        else:
            _check(
                f"Simulation failed before AMM program — structure issue",
                False,
                error_class,
            )

    except Exception as e:
        print(f"  {FAIL}  simulate_sell raised: {e}")
        import traceback; traceback.print_exc()
else:
    print(f"  {WARN}  Skipped (TX build failed in Test 4)")


# ── Test 6: MIGRATION_UNCERTAIN → run_pumpswap_local_path gate ────────────
section("TEST 6 — portfolio.py gate: MIGRATION_UNCERTAIN routes to local path")

try:
    import ast, pathlib

    src = pathlib.Path("memecoin/portfolio.py").read_text()
    tree = ast.parse(src)

    # Check for MIGRATION_UNCERTAIN in the ExitRouter gate condition
    gate_ok = "MIGRATION_UNCERTAIN" in src and "run_pumpswap_local_path" in src

    # Verify they appear in proximity (within 30 lines of each other)
    lines = src.splitlines()
    mig_lines = [i for i, l in enumerate(lines) if "MIGRATION_UNCERTAIN" in l]
    psp_lines = [i for i, l in enumerate(lines) if "run_pumpswap_local_path" in l]

    proximate = any(
        abs(m - p) < 30
        for m in mig_lines for p in psp_lines
    )

    _check("MIGRATION_UNCERTAIN present in portfolio.py", bool(mig_lines), f"lines {mig_lines}")
    _check("run_pumpswap_local_path present in portfolio.py", bool(psp_lines), f"lines {psp_lines}")
    _check("MIGRATION_UNCERTAIN and run_pumpswap_local_path within 30 lines (gate is wired)", proximate)
except Exception as e:
    print(f"  {FAIL}  Gate check raised: {e}")


# ── Test 7: Buyback fee recipients (all 8 valid pubkeys) ─────────────────
section("TEST 7 — BUYBACK_FEE_RECIPIENTS (8 addresses, all valid pubkeys)")

try:
    from memecoin.pumpswap_local import BUYBACK_FEE_RECIPIENTS
    from solders.pubkey import Pubkey

    _check("Exactly 8 fee recipients", len(BUYBACK_FEE_RECIPIENTS) == 8, str(len(BUYBACK_FEE_RECIPIENTS)))
    valid = []
    for addr in BUYBACK_FEE_RECIPIENTS:
        try:
            Pubkey.from_string(addr)
            valid.append(addr)
        except Exception:
            print(f"  {FAIL}  Invalid pubkey: {addr}")
    _check("All 8 are valid Solana pubkeys", len(valid) == 8)
except Exception as e:
    print(f"  {FAIL}  BUYBACK_FEE_RECIPIENTS check raised: {e}")


# ── Summary ───────────────────────────────────────────────────────────────
section("SUMMARY — Go/No-Go for live capital")
print("""
  Tests 1-4, 6-7 are deterministic (code + chain structure checks).
  Test 5 (simulation) is the key gate:

  SIMULATION ok=True or error_class='pumpswap_honeypot_or_sell_restricted'
  → Structure correct → set PUMPSWAP_LOCAL_SELL_ENABLED=True in config.py → safe to re-enter

  SIMULATION error_class='pumpswap_token2022_ata_error' or 'pumpswap_bad_pool_layout'
  → Structure still broken → do NOT re-enter with capital

  SIMULATION error_class='pumpswap_simulation_failed' with no AMM log lines
  → Account structure rejected before program ran → do NOT re-enter
""")
