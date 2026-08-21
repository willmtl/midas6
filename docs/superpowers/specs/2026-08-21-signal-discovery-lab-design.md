# Signal Discovery Lab — Design

**Date:** 2026-08-21
**Status:** Approved — building
**Author:** William + Claude

## Test matrix (v1) — broadened per user ("test more possibilities")

Each of the 381 signals is tested across the full cross-product below, all under the integrity harness:
- **Direction:** LONG (buy firing names) **and** SHORT (a significantly *negative* demeaned signal is a short/avoid strategy, not a discard). Reported as two rows per signal.
- **Selector strength:** ALL firing names **and** TOP-K by signal magnitude (tail-not-average — edge may live only in the strongest fires). K via quantile of the signal's continuous score where one exists.
- **Exits / holds:** fixed forward horizons (1/3/5/10/21/63/126d) **and** a compact set of the rule-based `studies.EXITS` (trailing-stop trail_5/trail_10, take-profit tp_10/tp_20, rsi_ob/rsi_x_dn) — a signal dead at a fixed hold can be alive with a trailing stop.
- **DEFERRED to v2 (gated):** signal **pairs / gating** (e.g., oversold × A/D-rising, the seq-winner shape). 381² is too large to sweep raw; v2 combines only the signals that cleared the v1 single-signal bar, keeping multiple-testing bounded.

## Problem

Every candidate signal in this project has historically been tested **as an overlay on the flagship**, which hides whether it has any standalone edge. Two standalone new-strategy attempts this session (G analyst-gap, D capitulation) died at the validation gate — but only after ad-hoc, one-off tests. We want a **systematic, standalone, all-timeframe bake-off of every signal**, integrity-clean, to surface a genuinely new strategy — and to do it once, reusably, rather than one hand-built validation at a time.

The critical constraint learned this session: **the harness must be integrity-clean or the leaderboard is garbage.** Naive tests produced +132% "bounces" (garbage-bar/penny outliers) and a +1,648% strategy that was purely a split-adjustment bug. The lab bakes in every guard: survivorship-aware, split-clean, market-demeaned, winsorized, honest monthly-panel t-stats, regime-split.

## Goal

Rank every one of the **381 signals** (`studies.SIGNALS`) as its **own standalone strategy** (buy all names firing it, own liquidity floor, no flagship gating) across **all timeframes**, producing a persisted leaderboard whose top entries are trustworthy candidates for a deeper strategy build.

## What "everything, all timeframes, individually" means here

- **Everything** = all 381 `studies.SIGNALS` (price/volume, RSI/momentum, gaps, vol-shock, A/D, oversold, fundamental dimensions, AND alt-data insider/activist/13G + market-relative signals — reusing `all_on_all_study`'s alt-data/market attach helpers).
- **Individually / standalone** = each signal is its own strategy: on each fire, buy the firing name; equal-weight the basket of names firing in each period; own $5M/day + >$5 floor; **no** sector-rotation / value / flagship gating.
- **All timeframes** = two arms:
  - **Daily→6mo arm:** forward horizons **1, 3, 5, 10, 21, 63, 126 trading days**.
  - **Intraday (H4) arm:** sub-daily horizons **1, 2, 3, 6 4h-bars** (~4h to ~3 days), scoped to the **562 names with 4h parquet** (`/app/.data/intraday/4h/`).

## Data reality (verified 2026-08-21)

- `Candle` is **daily-only** (18.1M rows, all `1d`).
- 4h intraday exists as parquet for **562 tickers only** (liquid large/mid-caps; survivors). **No delisted intraday data.** → the intraday arm is inherently **survivor-biased and large-cap-only**, and is labelled as such in every output. A full-universe intraday sweep would require a large EODHD 1h→4h backfill and still couldn't cover delisted names; **out of scope for v1**.
- Split cache (`splits_cache.json`) now covers ~5,251 names (refreshed from EODHD this session) — used for any as-traded/price-basis needs; returns use split-adjusted `Candle.close` (safe by construction).

## Integrity harness (non-negotiable, applied to every signal)

Generalizes `strategy_d_validate.py` (which correctly killed the D mirage):

1. **Universe:** analyst-covered ∪ delisted-with-candles, minus ETFs (survivorship-aware). Liquidity: 20d-avg dollar-vol ≥ $5M **and** price > $5 at the fire (kills penny junk + most garbage bars).
2. **Forward returns THROUGH delisting:** exit at last traded price (ffill, project-verified-neutral convention) so a falling knife into bankruptcy scores its real loss, not dropped.
3. **Split-clean:** returns on split-adjusted close.
4. **Market-demeaned:** each event's forward return minus SPY over the identical window.
5. **Winsorized** to [−100%, +150%] (robust to garbage bars / unadjusted seams).
6. **Edge-triggered** signals (fire on the crossing, per `studies.SIGNALS` definitions) — not every day in a state.
7. **Honest t-stat:** aggregate events to **calendar-monthly means, t over months** (N = months, not overlapping events); also report **median + win-rate** (robust) alongside mean.
8. **Regime split:** SPY above/below its 200d MA (bull/bear) reported separately — this session showed edges flip sign by regime.

## Ranking & output

- **Primary sort: absolute demeaned return** (user priority — maximize absolute return), per horizon.
- **Robustness filter:** min events (≥100 monthly-distinct), \|monthly-t\| shown; flag survivorship gap (all vs survivors-only) and regime dependence.
- A signal's row = its best horizon + the full horizon curve + regime split + integrity flags.
- **Persist** `BacktestResult(kind="signal_discovery")` (+ JSON): full leaderboard (daily arm + intraday arm, clearly separated). Optional frontend "Discovery" tab in a follow-up.
- Top standalone candidates (clean, robust, regime-aware) → each gets a dedicated deeper build (own spec), the way G/D did — but now starting from a signal that already cleared a standalone integrity bar.

## Architecture

**`signal_discovery.py`** (backend/, dir-mounted). Reuses:
- `studies.SIGNALS` (381 `fn(df)->bool entry series`), and `all_on_all_study`'s `_attach_altdata` + market-series injection for the alt-data/market signals.
- `strategy_d_validate.py`'s event-accumulation + integrity-stat machinery (refactored into shared helpers `forward_through_delisting`, `demean`, `winsorize`, `monthly_t`, `regime_split`).
- Daily candles from `Candle`; 4h from the parquet cache for the intraday arm.

**Compute:** MP over tickers (`--jobs 16`), event accumulation per (signal, horizon, regime), then aggregate. ~30–90 min for the daily arm (cf. `all_on_all` ~30 min); intraday arm is fast (562 names).

**CLI:** `--jobs N`, `--arm daily|intraday|both`, `--signals k1,k2` (subset for smoke-test), `--db`, `--min-events N`.

**Non-breaking:** additive; `studies.py`, `all_on_all_study.py`, `strategy_d_validate.py` imported read-only (the last refactored to expose its helpers without changing behavior).

## Risks & caveats
- **Multiple-testing:** 381 signals × 7 horizons × 2 regimes ≈ 5,300 tests → some will look significant by chance. Mitigations: honest monthly-t (not event-t), require robustness across *adjacent* horizons, and treat the leaderboard as a **candidate generator**, not proof — every top hit gets its own dedicated validation/build before deployment.
- **Intraday arm survivor-biased** (562 large-caps, no delisted) — labelled; not trusted at face value.
- **Alt-data signals** (insider/activist) only backtestable where the event data exists (Form 3/4/5 ~2020+); coverage reported, no silent truncation.
- **Compute/memory:** event accumulation is memory-light per batch; MP over ticker chunks avoids the Candle-hypertable DISTINCT trap (universe built from analyst-jsonl ∪ DelistedCompany, not a hypertable scan).

## Success criteria
1. `signal_discovery.py` produces a ranked, integrity-clean leaderboard for all 381 signals across both arms, persisted to `BacktestResult(kind="signal_discovery")`.
2. Re-running `strategy_d_validate`'s capitulation triggers *through* the lab reproduces the D verdict (harness parity — the lab must also find capitulation negative in bull).
3. At least the top-20 daily-arm candidates surfaced with full horizon curve + regime split + survivorship gap, ready for individual deeper builds.
