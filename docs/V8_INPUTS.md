# V8 Design Inputs — Four Analysis Artifacts

Generated: 2026-07-29

---

## N4(a) — Forward-Validation Table (v8_pass vs v7_pass vs population)

**Window**: most recent 13 days of data (Jul 16–Jul 29 2026)
**Era filter**: CLEAN ERA ONLY (rows where any price_source_*/price_status_* is non-NULL)
**Source**: report.py run 2026-07-29

Clean era has n=7 rows total — RF1 deployed Jul 28, only 41h old. Full-dataset smart_money used as proxy.

| Cohort | era | n | priced | win% | >=+30% | >=+50% | >=+100% | SM_v1_hit% | notes |
|---|---|---|---|---|---|---|---|---|---|
| v8_pass (SM_hit=True) | full | 91 | 91 | 92.3% | n/a | n/a | n/a | 100% | clean-era n too small; full dataset proxy |
| v7_pass (screener) | full | 205 | 196 | 42.3% | n/a | 17.9% | 4.6% | n/a | |
| population | full | 18282 | 1093 | 39.9% | n/a | 16.3% | 3.8% | n/a | |
| clean era (ALL) | clean | 7 | 7 | 0% | 0% | 0% | 0% | n/a | RF1 only 41h old; outcomes not yet resolved |

**Note**: clean era will grow ~500 rows/day. Re-run in 7 days for valid forward-validation.

Smart-money v1 pinned: smart_money_hit=True cohort n=91, win=92% across full dataset.

---

## N4(b) — Era-Split Re-Read: Progress Buckets + v7 Counterfactual

**Source**: report.py run 2026-07-29 (18,282 rows, 38 days)

### Win Rates

- social_alert_bc ALL (preRF1): n=14,393, priced=287 (2%), win=43.6%, >50%=20.2%, max=+1118%
- social_alert_grad ALL (preRF1): n=807, priced=771 (96%), win=38.7%, >50%=14.4%, max=+1064%
- V7 traded: n=753, win=37.3% vs NOT V7 traded: n=17,529, win=41.0%

### Survivorship-Bias Delta (clean vs preRF1)

- preRF1 BC: 2% priced (DexScreener misses BC tokens) — win rate 43.6% is computed on the 2% that graduated and got priced
- clean era: n=7, all priced (0% win — outcomes not yet resolved at 1m/3m intervals)
- Bias note: the preRF1 43.6% BC win rate is a survivor bias artifact — only tokens that hit DexScreener (graduated) are in the priced set

### Progress_at_signal Buckets

0 rows with pp_vsol data (field not yet flowing from outcome_poller to Supabase)

### Missed Winners (screener-rejected >=+50%)

| Filter | missed | med peak | max peak | >=+100% |
|---|---|---|---|---|
| liq<8k | 81 | +131.4% | +1118.4% | 47 |
| bsr<0.55 | 39 | +92.5% | +655.9% | 18 |
| rug>500 | 15 | +86.0% | +957.3% | 6 |
| vol>50k | 7 | +70.2% | +444.1% | 2 |
| vol<2k | 1 | +558.3% | +558.3% | 1 |

Single-filter removable: 71/143 (50%)

---

## N4(c) — path_stats Full Output

**Source**: path_stats run 2026-07-29

1,200 path files on disk, ALL header-only (no tick data).

**Root cause**: PC1 path collector opens a file per token but PP WebSocket does not deliver tick data for social-alert tokens (already graduated or dead before subscription fires). Path data accumulation requires either:
1. PP subscribeNewToken ticks (only works for tokens still on BC at signal time — rare)
2. PC2 Helius backfill (not yet run)

All cells: INSUFFICIENT (n=0, need ≥100 per cell). Will populate after PC2 backfill.

### Paths Sanity Line

| field | value |
|---|---|
| File count | 1,207 (1,187 .csv + 20 .csv.gz) |
| Date range | 2026-07-28 → 2026-07-29 (PC1 started with RF1 deployment) |
| Winner-path count | 0 with tick data (all header-only) |
| Backfill set | NOT YET RUN — run `python -m research.backfill_paths --dry-run` first |

---

## N4(d) — replay_exits

**Source**: replay_exits run 2026-07-29

0 results — all 1,201 paths too short or empty (same root cause as N4c).

### Three Comparison Configs (spec defined, data pending)

| Spec | hard_stop | trail_tiers | time_stop | notes |
|---|---|---|---|---|
| A (v7 current) | -35% | [+30%/−25%, +100%/−25%, +300%/−15%] | 90min | current production config |
| B (early-TP-heavy) | default | [+20%/−20%, +60%/−20%] | 45min | tighter exit |
| C (wide-stop/small-size) | -50% | default | 120min | size=0.5x |

Results: PENDING PC2 backfill. All three specs ready in replay_exits.py.
