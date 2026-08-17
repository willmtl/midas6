# H4-on-Daily-Signals — Design

**Date:** 2026-08-16
**Status:** Approved (design), pending implementation plan
**Author:** William + Claude

## Problem

We have two proven-ish layers that currently live apart:
- **Daily selection systems (A/B/C)** — 1–2 month hold. C = div_2x sector flagship (cheapest-P/B in top-momentum sectors, +313%/DD−15.5%); B = capitulation (RSI<20 + A/D divergence, 64% win, t≈4.8); A = pure dip (rsi10<45, +349% but −43% DD). The daily uptrend gate *hurts* A.
- **H4 short-horizon engine** — 0–3 day hold. On the broad liquid universe the only edge that survived a base-rate check is **momentum continuation** (gap-up>4%, burst>7%), and it's largely daily-uptrend-conditional.

**Goal:** combine them — let a daily signal choose **which name** to trade, then use an **H4 entry** to time a short **0–3 day** trade on that name. Backtest first: does *daily-select → H4-entry* beat either layer alone? This is a research build; live scanner is a later follow-up.

## Handoff logic (decided)

Daily signal = **what** (candidate name); H4 signal = **when** (entry timing) + **the trade** (0–3 day hold). The H4 entry is the full engine swept on the *curated* candidate universe (the daily selection may change which H4 trigger wins vs the broad universe).

## Approach

**Chosen: a new `h4_on_signals_study.py` that reuses the H4 engine + intraday data layer and adds a candidate-window gate.** It imports `h4_study` (signals/exits/buckets/agg/backtest) and `intraday_data.get_4h`; it builds point-in-time candidate windows from A/B/C; it only counts an H4 entry whose 4h timestamp falls inside an active window for that name. Rejected: a from-scratch engine (duplicates H4 logic); live-first (premature — combo unproven).

**Non-breaking:** purely additive. `h4_study.py`, `rsi_4h_study.py`, `intraday_data.py`, and all A/B/C code (`backtest_lowpb`, `entry_signal_study`, `seq_fundamental_study`, `rotation_history_scan`, `studies.py`) are imported read-only, never modified. New `BacktestResult` kind, new URL, new tab.

## Components

### 1. Candidate windows (`h4_on_signals_study.py`, point-in-time per selector)

A candidate window is `(ticker, start_date, end_date)` during which the name is eligible for an H4 entry. Built PIT (only from data available at `start_date`):

- **C — div_2x flagship:** read saved `BacktestResult[rotation_history]`; each monthly pick → window = that holding month (`date` → next month `date`). No recompute.
- **A — pure dip (rsi10<45):** reuse the arm3_lowpb monthly reconstruction (loaders: `load_candles`, `load_financial_reports`, `_pit_monthly_panel`, `_available_at`, `price_basis.as_traded_close`, `sector_holdings`, `config`); each monthly value pick whose `rsi10<45` at month-end → window = that holding month.
- **B — capitulation:** run `SIGNALS["seq_rsi20_ad_rising_rsi"]` per name (reuse `_compute_trades`-style entry detection); each fire date → window = next **10 trading days**.
- **Union** = any of A/B/C active.

Selector is a CLI/config choice: `--selector C|A|B|union` (default: run all four, compared).

### 2. Data layer (staged fetch — "do both")

For every name that is ever a candidate, obtain its 4h frame via existing `intraday_data.get_4h(name)` (fetch+cache on demand). Two passes:
1. **Fast pass:** run immediately over only candidates that ALREADY have a cached 4h parquet — gives a first result in seconds.
2. **Full backfill:** fetch the missing candidate names' 1h→4h **detached** (setsid+nohup → log, the one long job), then re-run for the complete result.

Caveat: EODHD 1h depth varies; candidate names with `< MIN_BARS` 4h history are dropped and **counted** in the payload (no silent truncation).

### 3. Backtest mechanic

Sweep the full H4 engine on each candidate's 4h frame, but gate entries by the candidate mask:
- For ticker `tk`, compute each H4 signal's `(entry, mag)` via `h4_study.SIGNALS[k]["fn"]` (unchanged).
- Keep only entry bars whose timestamp date ∈ an active `(tk, window)` for the chosen selector.
- Episode-dedup the surviving entries (`h4_study._episode_starts`, gap=3), enter at bar close, hold the 0–3 day exit ladder `[1,2,3,4,5,6,8,10]` bars, bucket by magnitude — identical to `h4_study.backtest_ticker`, plus the candidate-mask filter.

Implementation: add a thin `backtest_ticker_masked(df, allowed_dates_set)` in the new script that mirrors `h4_study.backtest_ticker` but skips entries whose date ∉ `allowed_dates_set`. (Keeps `h4_study` untouched.)

### 4. Benchmarks (the point of the study)

For each *(selector × H4-signal × exit)* report odds (n, win%, avg win/loss, payoff, median, expectancy, t) + magnitude buckets, compared against:
- **Standalone H4** — the same H4 signal on the broad liquid-250 universe (from `BacktestResult[h4_study]`): does daily-selection *add* edge?
- **Daily-system own return** — the selector's own realized return on the same names/period (C: `port_ret`; A/B: the daily trade returns): does H4 timing *beat* buying at the daily signal and holding?
- **Base rate** — unconditional forward return of the candidate names over the same horizon.

### 5. Output & wiring

- `BacktestResult[h4_on_signals]` + `.data/studies/h4_on_signals.json`. Payload: per-selector → per-signal → {all, by_bucket, bucket_order, standalone_h4, daily_own, base_edge}, plus params (candidate counts, names fetched/dropped, windows).
- API `H4OnSignalsView(_StudyResultView)` at `/api/h4-on-signals`.
- Frontend sub-tab **Research → "H4 on Signals"** (`/research/h4-on-signals`), reusing `RsiLadderTable`/`RsiBucketTable`/`TailStrip`/`LastUpdatedChip`, with a selector toggle (A/B/C/union).
- Mount the new root file (create-file-first gotcha); run via `docker exec`.

## Data flow

```
A/B/C selectors ─► candidate windows (tk, start, end)  [PIT]
        │                         │
        │                get_4h(tk) fetch/cache (fast pass ∪ detached backfill)
        ▼                         ▼
   union of candidate names ─► 4h frames
                                  │
      h4_study.SIGNALS[k].fn ─► (entry, mag)  ── gate by candidate mask ──► entries
                                  │
        episode-dedup ─► exit ladder (0-3d bars) ─► magnitude buckets
                                  │
   aggregate odds + edge vs {standalone H4, daily-own, base rate}
                                  │
        BacktestResult[h4_on_signals] ─► /research/h4-on-signals tab
```

## Verification

- **Mask correctness:** with an all-True mask, `backtest_ticker_masked` reproduces `h4_study.backtest_ticker` exactly (unit test on a synthetic frame).
- **PIT / no look-ahead:** candidate `start_date` uses only data available then; H4 entries only on/after `start_date`.
- **Window counts:** payload reports #candidate windows, #names, #with-4h-data, #dropped per selector.
- **Fast pass vs full:** fast pass (cached-only) runs first and is a strict subset of the full backfill result.

## Caveats (in payload + tab)

- EODHD 1h depth varies; small-cap candidates may lack H4 history (counted, not hidden).
- Survivorship: A/B/C candidate universes are current-membership; documented.
- Gross of fees; H4 0–3 day frictions material — directional read.
- C/A candidates are small/mid-caps → thinner H4 liquidity than the liquid-250 broad benchmark; comparisons are directional.

## Out of scope (v1)

Live scanner; 8h/12h; stops/targets (fixed-bar exits only); intraday news/earnings triggers; optimizing the B window length (fixed 10 trading days).
