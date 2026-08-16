# H4 Short-Horizon Studies Engine — Design

**Date:** 2026-08-16
**Status:** Approved (design), pending implementation plan
**Author:** William + Claude

## Problem

The existing research and live stack is built for a **1–2 month hold** (sector rotation → cheapest-P/B value pick, monthly re-pick). We want a **second, independent trading style**: signals on the **4-hour (H4) timeframe** held for a **short horizon of 0–3 days**.

This is a *research* build. Goal: discover which H4 signal families produce real, tradeable 0–3 day edge — before committing to any live scanner. Live wiring is an explicit follow-up on the winners only.

## Prior art in the repo (reuse, don't rebuild)

- **`rsi_4h_study.py`** already proves the intraday pipeline: paginated EODHD **1h** fetch → resample to **4h** → parquet cache at `.data/intraday/4h/<ticker>.parquet` (idempotent), then backtests the RSI(14)×SMA(14) crossover across a **bar-based exit ladder**, **bucketed by how oversold** the cross was, with a **daily-candle benchmark** alongside, saved to `BacktestResult(kind="rsi_4h_backtest")`.
- **`studies.py`** — SIGNALS × EXITS sweep pattern, `_episode_starts`, `_tstat_from_returns`.
- **`sector_studies_mp.py`** — MP-over-signal-groups parallelism pattern.
- **`vol_shock_study.py`** — vol-normalized daily shock (down-days bounce) — the H4 mean-reversion analogue.
- **`burst_scan.py`** — 2-bar burst — the H4 momentum-continuation analogue.

The new engine **factors out** the fetch/cache/universe logic that currently lives inside `rsi_4h_study.py` into a shared module, then adds a broad multi-family sweep on top.

## Design principles (from project memory — hard rules)

- **Tail-not-average:** never report a signal by its blended mean; bucket every trade by *magnitude* (RSI depth, shock z, gap size) so the tail is visible.
- **Test signals individually:** each signal is a standalone selector with a dollar-volume floor (satisfied here by the liquid universe), not only as an overlay.
- **Always save backtests:** every run persists to `BacktestResult` (+ JSON); no throwaway prints.
- **Return priority:** rank/surface by absolute return; do not trade return for lower drawdown.

## Approach

**Chosen: a new standalone H4 engine mirroring `studies.py`**, plus a shared intraday-data module.
Rejected alternatives: extending `rsi_4h_study.py` in place (turns a single-purpose file into a tangle); one script per family (fragments the standardized exit-ladder / bucketing / DB-save logic).

## Components

### 1. `intraday_data.py` (new, repo root — shared)

Single owner of intraday data + universe selection. Extracted from `rsi_4h_study.py` (which is then refactored to import it, preserving its current numbers as a cross-check).

Responsibilities:
- `fetch_1h(sym, years)` — paginated EODHD 1h (moved verbatim from `rsi_4h_study`).
- `resample_ohlc(df, hours)` — 1h → 4h OHLCV.
- `get_4h(ticker, years, allow_fetch)` — cache-or-fetch, parquet at `.data/intraday/4h/<ticker>.parquet`, `MIN_BARS` guard.
- `liquid_universe(n=250, months=12)` — top-`n` stocks by **median daily dollar-volume** from DB daily `Candle`, restricted to `Candle ∩ Fundamental − ETFs`. Returns the fetch/backtest ticker list.
  - Uses a bounded scan / `max_parallel_workers_per_gather=0` per the Postgres `/dev/shm` caveat when aggregating over the Candle hypertable.

### 2. `h4_study.py` (new, repo root — the engine)

Mirrors `studies.py`: a registry of signals × an exit ladder, backtested over each cached 4h frame, bucketed, aggregated, saved.

**Exit ladder (0–3 day focus).** 4h RTH ≈ 1.6 bars/day. Ladder in bars: `[1, 2, 3, 4, 5, 6, 8, 10]` (≈ ½ day → ~6 days), each with a day-approx label, **plus one signal-native exit per family** (e.g. RSI-cross-back-below for the RSI signals).

**Signal families (all five).** Each signal is a function over the 4h OHLCV series returning `(entry_bool_array, magnitude_value_array)`; the magnitude drives bucketing.

1. **Mean-reversion (snap-back)**
   - `mr_rsi_os` — RSI(14) crosses up from oversold; bucket by RSI level at cross.
   - `mr_volshock_dn` — vol-normalized down-shock `z = ret / trailing_vol` (H4 `vol_shock`); bucket by |z|.
   - `mr_gap_dn` — large intraday gap-down bar; bucket by gap size.
   - `mr_bb_low` — close below lower Bollinger band / `%B < 0`; bucket by `%B`.
   - `mr_newlow` — new N-bar low (N ∈ {30, 60}); bucket by depth below prior low.
   - `mr_ndown` — K consecutive down bars; bucket by K.
2. **Momentum / breakout continuation**
   - `mo_burst` — 2-bar burst continuation (H4 `burst_scan`); bucket by burst magnitude.
   - `mo_break_hi` — new N-bar-high breakout; bucket by breakout distance.
   - `mo_rsi_ob` — RSI>SMA momentum continuation; bucket by RSI level.
   - `mo_gap_up` — gap-up continuation; bucket by gap size.
3. **Event-driven (price-based only in v1)**
   - `ev_open_gap` — session-open gap (first 4h bar of the day) up/down reaction; bucket by gap size.
   - *Deferred to v2:* news/earnings-tagged H4 (NewsItem not backfilled pre-2025).
4. **Trend / structure**
   - `st_ad_div` — A/D divergence via `ta.accdist` (ADL) rising while price falls — the strict-divergence edge, **not** slope/CMF/OBV.
   - `st_ma_pull` — pullback to a rising MA; bucket by distance to MA.
5. **Cross-timeframe conditioning**
   - Not new signals: every MR/MO signal is **additionally split by the stock's DAILY trend** (up/down, from DB daily candles), reported like the existing `by_spy_trend` regime splits. Shows whether an H4 edge only works with/against the daily trend.

**Bucketing & stats.** Reuse `rsi_4h_study`'s `by_bucket` machinery. Per signal × exit report: `n`, `avg%`, **`median%`**, `win%`, `t_stat` (`_tstat_from_returns`), with the **same signal on DAILY candles** as a side-by-side benchmark. Entries episode-deduped (`_episode_starts`); entry at signal-bar close; no fees.

**Parallelism & CLI.** MP over signals like `sector_studies_mp.py`. CLI: `--limit N`, `--no-fetch`, `--families a,b,c`, `--jobs J`, `--years Y` (default 5).

**Persistence.** `BacktestResult(kind="h4_study")` with the full payload + a per-family JSON under `.data/studies/`. Payload records the actual data span, universe size, n-with-data, params, and the documented caveats.

### 3. Frontend — `/h4` tab

New tab in `frontend/src/App.js` rendering the results table: rows per signal (grouped by family), the exit-ladder columns, magnitude buckets on expand, the daily benchmark column, and a last-updated chip. Reuse `usePagedList` + `ScrollSentinel` + the existing table/`TailStrip` components. Build with `docker compose up -d --build --no-deps frontend`.

### 4. Docker / ops

- Both new root `.py` files mounted individually in `docker-compose` — **create the file first, `rmdir backend/<x>.py` if a stub dir appears, then `up --force-recreate`** (the mount gotcha).
- Run: `MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_study.py ...`.
- Heavy Candle aggregation for `liquid_universe` respects the Postgres `/dev/shm` limit.

## Data flow

```
liquid_universe(250)  ──►  fetch_1h (EODHD)  ──►  resample 4h  ──►  parquet cache
                                                                        │
                              per ticker 4h frame ◄────────────────────┘
                                     │
             for each signal family: entry_bool + magnitude
                                     │
        episode-dedup ─► exit ladder (bars) + signal-native exit
                                     │
            bucket by magnitude ; split by daily trend
                                     │
     aggregate (n, avg, median, win, t) + daily benchmark
                                     │
        BacktestResult(kind="h4_study") + JSON  ──►  /h4 tab
```

## Verification

- **Cross-check:** after refactor, `rsi_4h_study.py` must reproduce its prior RSI(14) numbers (same fetch/cache path). The engine's own `mr_rsi_os` should match `rsi_4h_study` on the RSI-oversold bucket.
- **Individual-signal sanity:** each family run standalone with `--families` produces sensible `n` and non-degenerate buckets.
- **DB round-trip:** `BacktestResult(kind="h4_study")` exists and the `/h4` tab renders it.

## Caveats (documented in payload + tab)

- **EODHD 1h history depth** varies by ticker; report the actual span, don't assume 5y.
- **Survivorship:** the liquid universe is *current* top-250 → survivorship bias; flag it (consistent with prior discipline).
- **NewsItem not backfilled pre-2025** → event-driven family is price-based only in v1.
- **No fees / entry-at-close** — same convention as `rsi_4h_study`; real short-horizon trading has meaningful frictions, so treat absolute H4 returns as gross.

## Out of scope (v1)

8h/12h timeframes; live H4 scanner; stops/targets/trailing exits; news/earnings-tagged event signals; cross-timeframe as a full combinatorial sweep (it's a reporting split here).
