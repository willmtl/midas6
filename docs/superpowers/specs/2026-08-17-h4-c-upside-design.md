# Analyst Upside-Gap on H4-on-C Entry — Design

**Date:** 2026-08-17
**Status:** Approved (design), pending implementation plan

## Problem

The H4 mean-reversion entry works on daily-C value picks (`mr_rsi_os` +0.42%/3b t=3.15; `gap_dn__rsi_x` +1.04%/3b). Separately, the **analyst implied-upside gap** (target/price − 1) is monotonically predictive standalone (`analyst_ratings_study`: q4−q0 = +4.89pp/63d, n=148k). Never combined. Question: does the 0–3-day H4 dip-buy pay **more on deep-value (high analyst-upside) C names**?

## Approach

**New `h4_c_upside.py`, reusing everything** — `h4_on_signals_study.candidate_windows("C")`, `intraday_data.get_4h`, `h4_study` (SIGNALS/EXITS/`_episode_starts`/`agg_rows`), `h4_c_indicators.combo_entries` (for `gap_dn__rsi_x`), plus a PIT analyst-upside lookup from `.data/analyst_ratings.jsonl` (218k dated Benzinga rows, 925 tickers; 114/132 C names covered). Additive: new `BacktestResult` kind, new tab. Nothing existing modified.

## Components

### 1. PIT analyst-upside lookup
Load `.data/analyst_ratings.jsonl`; per ticker build a sorted list of `(date, price_target)` (drop null/≤0 targets). `upside_asof(tk, date, close)` = *most recent target on/before `date`* / `close` − 1, or `None` if no prior target. No look-ahead — uses only already-published targets.

### 2. Sweep with an upside dimension
For each C candidate 4h frame: run each H4 signal (the 14 `h4_study.SIGNALS` + the `gap_dn__rsi_x` combo). Mask entries to the C window, episode-dedup (gap=3). **For each surviving entry, compute `upside_asof(tk, bar_date, entry_close)`, bucket it, and pool the fixed exit-ladder return per *(signal × upside-bucket × exit)*.** Entries whose ticker has no target-as-of go in a counted `no_target` bucket.

**Upside buckets:** `[("<0%", -1e9, 0), ("0-25%", 0, 25), ("25-50%", 25, 50), ("50-100%", 50, 100), (">100%", 100, 1e9)]`.

### 3. Aggregation & test
Per signal × bucket: `h4_study.agg_rows` → n/win/avg/median/t at each exit. Report the **3b and 6b** rows per bucket, flag monotonicity (edge rising with upside). Benchmark = the un-split on-C number (from `BacktestResult[h4_on_signals]` C, or recomputed with `allowed`=all-upside). Small-sample guard: only treat a bucket cell as meaningful at n≥30; headline needs n≥100.

## Output & wiring
- `BacktestResult[h4_c_upside]` + `.data/studies/h4_c_upside.json`. Payload: `{signals: {key: {name, by_upside: {bucket: [ladder rows]}, bucket_order, all}}, bucket_order, coverage: {n_c_names, n_covered, n_no_target_entries}, note}`.
- API `H4CUpsideView(_StudyResultView)` at `/api/h4-c-upside`.
- Frontend sub-tab **Research → "H4 C-Upside"** (`/research/h4cups`): per-signal upside-bucket ladders (3b/6b), monotonic-in-upside highlighted, coverage line. Reuses `RsiLadderTable`.

## Verification
- `upside_asof` unit test: with a hand-built target series, returns the last-on-or-before target's upside, `None` before the first target, no future leak.
- Bucketing unit test: a known upside maps to the right bucket.
- Sanity: total entries across buckets (incl `no_target`) == the un-split C entry count for a given signal (mask conservation).

## Caveats (payload + tab)
114/132 C names covered (18 no-target, counted); dated-PIT (target known at entry); per-bucket samples get small — n<30 cells dim/hidden; gross of fees; C = current-membership survivorship; only cached-4h C names.

## Out of scope (v1)
Non-C selectors; estimate-revision or rating-action overlays (estimate momentum already refuted on value picks); live scanner; re-fetching missing C 4h.
