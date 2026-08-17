# H4 Indicator Sweep on C Candidates — Design

**Date:** 2026-08-17
**Status:** Approved (design), pending implementation plan
**Author:** William + Claude

## Problem

Inside daily-C candidate windows (cheapest-P/B value picks in top-momentum sectors), an H4 **mean-reversion** entry beats standalone H4 (`mr_rsi_os` +0.42%/3b, t=3.15; `mr_newlow60`, `mr_ndown` also significant). We tested only 14 hand-picked H4 signals. Question: does any **other** DB indicator — or a **sequenced combo** — beat `mr_rsi_os` as the H4 entry on C? User hypothesis: **gap-down → RSI-crossover** ("probably the one").

## Approach

**New `h4_c_indicators.py`, reusing everything** — imports `h4_on_signals_study.candidate_windows`/`backtest_ticker_masked`, `intraday_data.get_4h`, `h4_study` (exit ladder, agg, dedup), and `studies.SIGNALS` read-only. Additive: new `BacktestResult` kind, new tab view. Nothing existing is modified.

Two signal layers, both swept as the H4 entry inside C windows:

### 1. Singles — the full DB indicator library
Iterate `studies.SIGNALS` (~370, `key -> (name, fn)`). Each `fn(df)` returns a boolean entry series on a plain OHLCV frame. Signals needing extra args (the 14 SPY/QQQ `MARKET_SIGNAL_KEYS`) or fundamentals/seasonal inputs raise on a bare 4h df → **caught and skipped** (naturally filters to pure-OHLCV indicators). Skipped count is reported.

### 2. Combos — dip → confirmation sequences (the thesis)
A generic 4h sequence: **precondition P fired within the last K bars (K=5), then trigger T fires now.** Matrix (~12):
- **P** ∈ { gap-down ≥2%, RSI(14)<30, new-30-bar-low, ≥3 consecutive down bars }
- **T** ∈ { RSI(14) crosses above its SMA(14), A/D line rising, close > prior bar's high }
- Headline: **`gap_dn__rsi_x` = gap-down ≥2% then RSI-crosses-up within 5 bars.**

Combo entry series computed on the 4h frame; reused across C candidates.

## Backtest mechanic

For each C candidate 4h frame: compute each single/combo entry series, keep entries whose bar-date ∈ the C window (reuse the candidate mask), episode-dedup (`h4_study._episode_starts`, gap=3), enter at bar close, hold the 0–3-day exit ladder `[1,2,3,4,5,6,8,10]` bars. Boolean signals → **flat ladder only (no magnitude buckets).** Pool across candidates, aggregate odds (n, win%, avg, median, t) via `h4_study.agg_rows`.

## Ranking & significance guard

Rank singles and combos by **t-stat @3b**. Because ~380 signals are tested, apply a **multiple-comparisons guard**: flag as "real" only **t>3 AND n≥100**; report the Bonferroni threshold (α=0.05/N) in the payload. Consistent with the prior daily indicator-bakeoff finding (the zoo mostly loses) — this exists to avoid chasing noise.

## Benchmarks

- **Base rate** — unconditional 3b forward return of C candidates.
- **Champ** — `mr_rsi_os` @3b (the current best) shown as the bar to beat.
- Each single that overlaps the 14 H4 signals also carries its standalone-broad number (from `BacktestResult[h4_study]`) for context.

## Output & wiring

- `BacktestResult[h4_c_indicators]` + `.data/studies/h4_c_indicators.json`. Payload: `{singles: [ranked rows], combos: [ranked rows], champ, base_rate_3b, n_signals_tested, n_skipped, bonferroni_t, candidates}`.
- API `H4CIndicatorsView(_StudyResultView)` at `/api/h4-c-indicators`.
- Frontend: a "Full library on C" view under the existing **H4 on Signals** tab (toggle) — ranked tables for singles + combos, champ line, significance flag column.
- Mount new root file (create-file-first gotcha); run detached (env reaps long execs); B not involved so no `/dev/shm` DISTINCT hazard.

## Verification

- Combo builder unit test: on a synthetic frame with a hand-placed gap-down then RSI-cross, `gap_dn__rsi_x` fires on the expected bar and not otherwise.
- Singles skip test: a signal needing SPY/QQQ is skipped without aborting the sweep.
- Sanity: `mr_rsi_os`-equivalent single reproduces its ~+0.42%/3b on C (cross-check vs `h4_on_signals`).

## Caveats (payload + tab)

Daily-tuned indicator windows applied to 4h bars (documented); multiple comparisons (guard applied); C candidate universe = current-membership survivorship; gross of fees; small/mid-cap 4h liquidity thinner; only C candidates that have cached 4h data are included (counted).

## Out of scope (v1)

A/B/union library sweep; magnitude bucketing of DB singles; parameter re-tuning of indicators for 4h; live scanner.
