# PUMPFUN_COMPATIBILITY_REPORT

Z8 — pump.fun bonding-curve program upgrade compatibility gate

## Verification Metadata

| Field                   | Value                                                              |
|-------------------------|--------------------------------------------------------------------|
| `program_address`       | `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`                    |
| `programdata_address`   | `B5MvUwXdiW1NMM6QFFD3ssPKBujD4zMohncbM73Z2BQu`                    |
| `current_deploy_slot`   | `433,095,571`                                                      |
| `upgrade_authority`     | `7gZufwwAo17y5kg8FMyJy2phgpvv9RSdzWtdXiWHjFr8`                    |
| `executable_data_hash`  | `5d65238ffd3513fa5980c01ba1eb2da9cc050091ad9c46b75916ce4a09b02bbf` |
| `verification_timestamp`| `2026-07-15`                                                       |
| `rpc_source`            | `https://api.mainnet-beta.solana.com` (public mainnet-beta)        |
| `prior_deploy_slot`     | `422,112,651` (reported, from program_monitor.py alert)            |

## On-Chain Verification Steps

### Step 1 — Program Account

```
getAccountInfo("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P", encoding="base64")
```

Raw program account (36 bytes):
```
hex: 0200000095b2d964aa5312ecabb16271c9d6f02f37100ada8ad478e5832cbd0993bf0bba
```

- Bytes 0-3 discriminant: `0x00000002` = 2 (Program) ✓
- Bytes 4-35 (ProgramData pubkey): `95b2d964aa5312ecabb16271c9d6f02f37100ada8ad478e5832cbd0993bf0bba`
- **ProgramData address: `B5MvUwXdiW1NMM6QFFD3ssPKBujD4zMohncbM73Z2BQu`**
- Owner: `BPFLoaderUpgradeab1e11111111111111111111111` ✓ (upgradeable)

### Step 2 — ProgramData Account (deploy slot)

```
getAccountInfo("B5MvUwXdiW1NMM6QFFD3ssPKBujD4zMohncbM73Z2BQu", encoding="base64")
```

First 48 bytes:
```
hex: 030000009383d019 00000000 016348fb820231fbab73dc3083e1f3f149ff125b718820f32eb1a15bfefcd61b09...
```

- Bytes 0-3 discriminant: `0x00000003` = 3 (ProgramData) ✓
- Bytes 4-11 (slot u64 LE): `9383d01900000000` → **slot = 433,095,571** ✓ CONFIRMED
- Byte 12: `0x01` = Some (upgrade authority present)
- Bytes 13-44: `6348fb820231fbab73dc3083e1f3f149ff125b718820f32eb1a15bfefcd61b09`
  → **upgrade_authority = `7gZufwwAo17y5kg8FMyJy2phgpvv9RSdzWtdXiWHjFr8`**

### Step 3 — ELF Data Hash

- Total ProgramData account size: 10,485,760 bytes
- ELF data starts at byte 45: 10,485,715 bytes
- ELF magic (bytes 45-48): `7f454c46` ✓ valid ELF
- **SHA-256 of ELF executable: `5d65238ffd3513fa5980c01ba1eb2da9cc050091ad9c46b75916ce4a09b02bbf`**

## Post-Upgrade Transaction Analysis (Z8.3)

All transactions examined are from slot `433,122,735` — confirmed post-upgrade
(25,164 slots after deploy slot `433,095,571`).

### Successful post-upgrade transactions examined

| Signature | Instruction | Result |
|-----------|-------------|--------|
| `h3rVR17am61iuQH5XPsRuzUxH4sAnFQfdyzhxGn4aKBeE6NhVNHcbT7okWJQXwzsgv1NhKLV4ADviLp7QBogccq` | BUY | ✓ success |
| `5RS2U4qFxHBDsXhuEuJPUUyR3WkcRRPMEh4dTdtXnXpM4J4MMHopDTLpjzi7Y1azr3WTj97YVUkWeCARYuW1rFfq` | SELL_V2 | ✓ success |
| `4bCL5vpyCSJ6scxo3RmTueyTzwAdvhx2Wm7e3zRo1n5LPbQVqa5UCBTeSSABj2cr7iv44ZHH2JN69SAoFYrSm4Wt` | unknown_new | ✓ success |

### Buy instruction comparison

| Field | V1 (pre-upgrade) | V2 (post-upgrade) | Change |
|-------|------------------|-------------------|--------|
| discriminator | `66063d1201daebea` | `66063d1201daebea` | **UNCHANGED** |
| data length | 24 bytes | 24 bytes | **UNCHANGED** |
| account count | 12 | **18** | **CHANGED (+6)** |
| account[0] global_pda | `4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf` | same | **UNCHANGED** |
| account[1] fee_recipient | writable | writable | **UNCHANGED** |
| account[2] mint | readonly | readonly | **UNCHANGED** |
| account[3] bonding_curve_pda | writable | writable | **UNCHANGED** |
| account[4] assoc_bc_token_acct | writable | writable | **UNCHANGED** |
| account[5] assoc_user_token_acct | writable | writable | **UNCHANGED** |
| account[6] wallet | WR+SIGN | WR+SIGN | **UNCHANGED** |
| account[7] system_program | readonly | readonly | **UNCHANGED** |
| account[8] token_program | readonly | readonly | **UNCHANGED** |
| account[9] rent_sysvar | readonly | writable (new role) | **CHANGED** |
| account[10] event_authority | `Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1` | same | **UNCHANGED** |
| account[11] program | readonly | readonly | **UNCHANGED** |
| accounts[12-17] | *(absent)* | new (creator vault + buyback system) | **ADDED** |

### Sell instruction comparison

| Field | V1 (pre-upgrade) | V2 (post-upgrade) | Change |
|-------|------------------|-------------------|--------|
| discriminator | `33e685a4017f83ad` (`global:sell`) | `5df6823ce7e940b2` (`global:sell_v2`) | **CHANGED** |
| data length | 24 bytes | 24 bytes | **UNCHANGED** |
| account count | 12 | **26** | **CHANGED (+14)** |
| event_authority | `Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1` | present (position unknown in new layout) | *not confirmed* |
| program_id | `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P` | same | **UNCHANGED** |

The sell instruction was renamed from `global:sell` to `global:sell_v2`. The old
discriminator (`33e685a4017f83ad`) is no longer accepted by the V2 program.

## Bonding-Curve Account Layout Verification (Z8.4)

Active bonding-curve account examined: `B24xqGezdJvjb2Ur3rzZtEyfjzGtTpcpY7a78SAWwN66`
Owner: `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P` ✓

| Bytes | Field | V1 | V2 post-upgrade | Status |
|-------|-------|----|-----------------|--------|
| 0:8 | discriminator | `17b7f83760d8ac60` | `17b7f83760d8ac60` | **UNCHANGED** |
| 8:16 | virtual_token_reserves (u64 LE) | present | `1,063,494,656,015,142` | **UNCHANGED** |
| 16:24 | virtual_sol_reserves (u64 LE) | present | `3,107,652,233` | **UNCHANGED** |
| 24:32 | real_token_reserves (u64 LE) | present | `783,594,656,015,142` | **UNCHANGED** |
| 32:40 | real_sol_reserves (u64 LE) | present | `2` | **UNCHANGED** |
| 40:48 | token_total_supply (u64 LE) | present | `1,000,000,000,000,000` | **UNCHANGED** |
| **48** | **complete (bool)** | **present** | **`0x00`** | **UNCHANGED** |
| 49:81 | creator (pubkey) | *(absent in V1)* | `FCJuEHYBo3fp1z8FsbNJAWDiZmWNDdUAZWAsvqKaLZfc` | **NEW V2 field** |
| 81:89 | unknown u64 | *(absent in V1)* | `1` | **NEW V2 field** |
| 89:151 | padding | *(absent in V1)* | `0x00…` (62 bytes) | **NEW V2 field** |

**Total size: 151 bytes** (V1 was 49 bytes).
**Bytes 0-48 are structurally identical to V1.**
**Byte 48 = 0 = active (not completed)** — confirmed for an active curve.

### Semantic invariants verified

| Invariant | Value | Pass? |
|-----------|-------|-------|
| virtual_token_reserves > 0 | 1,063,494,656,015,142 | ✓ |
| virtual_sol_reserves > 0 | 3,107,652,233 | ✓ |
| real_token_reserves ≤ virtual_token_reserves | ✓ | ✓ |
| token_total_supply = 1,000,000,000,000,000 (standard) | ✓ | ✓ |
| complete = 0 for active curve | ✓ | ✓ |
| derived price finite and positive | $4.38×10⁻⁷/token | ✓ |

Note: price from a single account was not compared to an independent live feed.
This is one of the conditions for upgrading to VERIFIED_UNCHANGED.

## Simulation Result

No simulation was performed.  The local TX builder is disabled (UNVERIFIED_SAFE_DEGRADE).
Simulation must be performed against a controlled test asset before re-enabling
the local builder.

## Impact Assessment

| Path | Status |
|------|--------|
| Buys (SPL, local builder) | **DISABLED** — buy account count changed 12→18 |
| Buys (T22, PumpPortal) | ✓ SAFE — PumpPortal adapts server-side |
| Sells (SPL, local builder) | **DISABLED** — sell discriminator changed |
| Sells (T22, PumpPortal) | ✓ SAFE — PumpPortal adapts server-side |
| Sells (PumpSwap, pumpswap_local.py) | ✓ SAFE — independent program |
| Sells (Jupiter fallback) | ✓ SAFE — independent program |
| Graduation detection: account_missing (Z1) | ✓ SAFE — not layout dependent |
| Graduation detection: complete=True (BC parse) | **BLOCKED** — compat state = CHANGED |
| Graduation detection: DexScreener pumpswap | ✓ SAFE — independent signal |
| BC price feed (virtual reserves bytes 8-23) | ✓ SAFE — layout unchanged in V2 |
| Tripwire (deploy slot monitoring) | ✓ ACTIVE |

## Final Verdict

```
UNVERIFIED_SAFE_DEGRADE
```

**Rationale**: A real interface change was confirmed:
1. Sell discriminator changed from `global:sell` to `global:sell_v2`
2. Buy account count increased from 12 to 18
3. Sell account count increased from 12 to 26

The local pump.fun TX builder is disabled for both buy and sell.  PumpPortal
server-built transactions and the PumpSwap/Jupiter fallback paths remain fully
active and are not affected by this upgrade.

The bonding-curve binary layout (bytes 0-48) appears unchanged based on one
account sample, but `complete=True` graduation from BC parse is also blocked
pending full verification (account owner check + discriminator check + length
validation now required by `validate_bc_account()`).

`account_missing` graduation (Z1) is unaffected — it does not parse any data.

## Conditions to Upgrade Verdict to VERIFIED_UNCHANGED

The verdict may be upgraded to `VERIFIED_UNCHANGED` when ALL of the following are met:

1. A confirmed SPL-token buy transaction post-upgrade shows account count = 12 (unchanged)
   OR the new 18-account buy interface is fully documented and implemented locally.
2. A confirmed SPL-token sell transaction post-upgrade shows discriminator `33e685a4017f83ad`
   (unchanged) OR the new sell_v2 interface is fully implemented and tested locally.
3. The full V2 IDL is obtained from pump.fun or decoded from the on-chain IDL account.
4. A local TX simulation against a controlled test asset passes with no instruction error.
5. `mark_interface_verified(433_095_571, "5d65238f...")` is called after steps 1-4.

## Epoch Gate Status

Z8 is complete as UNVERIFIED_SAFE_DEGRADE.  The epoch may begin under these conditions:

1. ✓ Local pump.fun builds disabled (Z8.5)
2. ✓ BC layout graduation gated (Z8.5)
3. ✓ account_missing still works via Z1 (unaffected)
4. ✓ PumpSwap/Jupiter active for graduated positions
5. ✓ Deployment tripwire active (Z8.6)
6. ✓ All Z8 tests pass (Z8.8)
7. ✓ Z1-Z7 invariants unchanged
8. ✓ PUMPFUN_COMPATIBILITY_REPORT.md contains reproducible evidence (this document)
