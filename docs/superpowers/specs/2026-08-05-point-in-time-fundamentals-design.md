# Point-in-Time Fundamentals — Design Spec

**Date:** 2026-08-05
**Status:** Implemented (data source swapped to SEC EDGAR mid-build — see addendum)
**Author:** (pairing session)

> **ADDENDUM (2026-08-05, during implementation): data source swapped yfinance → SEC EDGAR.**
> The approved plan backfilled quarterly financials from yfinance with
> `avail_date = period_end + 45d`. In verification this proved fatally shallow: yfinance's
> `quarterly_*` endpoints return only ~5 recent quarters, so PIT metrics were non-NA for
> only ~6% of a 5-year backtest (NVDA: 75/1254 bars). We swapped the `FinancialReport`
> backfill to the **SEC EDGAR `companyfacts` XBRL API** (free, ~19 years of filings,
> 2007→2026), which also gives the **true `filed` date** — so `avail_date` is now the
> actual public-disclosure date, not a +45-day approximation. Coverage jumped to 100% of
> bars for covered tickers; 683/1035 universe tickers have EDGAR data (the rest are
> ETFs/foreign 20-F filers with no us-gaap XBRL → fall back to price-derived + snapshot
> dims). Dividends still come from yfinance (deep history there is fine). PE/EPS are
> computed from `market_cap / net_income_ttm` and `net_income_ttm / shares` (NOT summed
> per-share EPS) to stay split-agnostic (NVDA 10:1). Everything downstream of
> `FinancialReport` (model, DIMENSIONS, PIT engine, UI) is unchanged by the swap.
> `REPORT_LAG_DAYS = 45` is retained only as a fallback for any source without a filing date.

## Problem

Fundamental bucketing today joins each historical trade to **today's** `Fundamental`
snapshot. A stock that is micro-cap / cheap-PE / negative-EPS *now* may not have been at
the time the trade fired. Combined with survivorship bias, the resulting buckets are
**directional hints, not tradeable signal** ("this signal likes beaten-down deep-value
names") — you cannot act on "+88% for micro-caps" because that classification uses
future information.

## Goal

Bucket every trade by the fundamentals **as they were knowable on the trade's entry
date** — genuine point-in-time (PIT). This removes the fundamental lookahead bias and
makes the amplification buckets actionable. (Survivorship bias in the *universe* is a
separate, out-of-scope problem.)

## Scope decisions (already made)

- **Tier A + Tier B, free, via yfinance.** No paid vendor.
- **`REPORT_LAG_DAYS = 45`** (configurable) — a quarter's numbers are treated as public
  `period_end + 45 days`, approximating filing lag so we never use figures before they
  were available.
- **Tier C is kept as snapshot-only, not dropped.** Free point-in-time is impossible for
  these (needs analyst estimates or historical ownership/short-interest data yfinance
  does not provide), so they bucket trades using **today's `Fundamental` snapshot** and
  are **clearly labeled "snapshot (lookahead)"** in the app. Upgrading them to true PIT
  is future work (SEC EDGAR + FINRA free path, or a paid vendor — see "Future work").

### Dimension model: a `pit` flag per dimension

Every `DIMENSIONS` entry gains a `pit: bool` flag. `pit=True` → value comes from the
per-trade point-in-time row (below). `pit=False` → value comes from the current
`Fundamental` snapshot (as today, carrying the lookahead caveat). The app badges each
dimension accordingly so PIT and snapshot buckets are never confused.

### Dimensions — 23 total (17 point-in-time + 6 snapshot-only)

**Point-in-time (17):**

**Price-derived (Tier A) — 5**
1. Market cap = `Close × shares_outstanding(as-of)`
2. Beta = rolling regression of returns vs SPY (from candles)
3. 52-week range position = `(Close − 52w low) / (52w high − 52w low)` (from candles)
4. Trailing avg volume (from candles)
5. Dividend yield = TTM dividends / Close

**Report-derived (Tier B) — 12**
6. PE (trailing) = `Close / TTM EPS`
7. PS = `market_cap / TTM revenue`
8. PB = `market_cap / total_equity`
9. PEG = `PE / earnings_growth`
10. EPS (TTM) sign — positive vs negative
11. Revenue growth (YoY, TTM vs prior TTM)
12. Earnings growth (YoY, TTM vs prior TTM)
13. Profit margin = `TTM net_income / TTM revenue`
14. Operating margin = `TTM operating_income / TTM revenue`
15. Debt-to-equity = `total_debt / total_equity`
16. Current ratio = `current_assets / current_liabilities`
17. Free cash flow sign — positive vs negative

**Snapshot-only (6)** — kept, labeled "snapshot (lookahead)", using today's `Fundamental`:
18. Forward PE
19. Forward EPS (sign)
20. Short % float
21. Institution %
22. Insider %
23. Float (current)

## Architecture

**Principle: store raw PIT inputs, derive metrics in the worker.** Matches the existing
"load raw → compute in worker" pattern (`_prepare_indicators`). Avoids a giant
per-ticker-per-day metrics table; metrics are computed once per ticker inside the sweep
where the DataFrame already lives, then each trade does an `asof(entry_date)` lookup.

### 1. Data model — `FinancialReport` (new)

One row per ticker per fiscal quarter (~1,149 tickers × ~20 quarters ≈ 23K rows).

```
ticker              CharField(db_index)
period_end          DateField            # fiscal quarter end
avail_date          DateField(db_index)  # period_end + REPORT_LAG_DAYS; the PIT key
revenue             BigIntegerField null
net_income          BigIntegerField null
eps_diluted         FloatField null
operating_income    BigIntegerField null
total_equity        BigIntegerField null
total_debt          BigIntegerField null
current_assets      BigIntegerField null
current_liabilities BigIntegerField null
free_cash_flow      BigIntegerField null
shares_outstanding  BigIntegerField null
unique_together = (ticker, period_end)
```

Dividends: a small `DividendHistory` (ticker, ex_date, amount) OR reuse an existing
dividend source if present in `.data/`. Decided at implementation time; TTM yield needs
only a per-ticker date→amount series.

Shares outstanding: from `yf.Ticker(t).get_shares_full(start, end)` (irregular time
series). Stored per report row (nearest as-of) and/or a light series; forward-filled for
daily market cap. Report-row granularity is sufficient for bucket boundaries.

Existing `Fundamental` (snapshot) model is **left intact** — still used for the live
Firing Now per-stock display and anywhere current values are wanted.

### 2. Backfill job — `fetch_financial_history.py` (new, root, mounted)

Parallelized with the established spawn pattern (`ProcessPoolExecutor`, `django.setup()`
per worker, `connections.close_all()` before spawn, chunked). Per ticker pulls:
`quarterly_income_stmt`, `quarterly_balance_sheet`, `quarterly_cashflow`,
`get_shares_full()`, `dividends`; normalizes yfinance's shifting row labels
(a small alias map: e.g. `Total Revenue`, `Net Income`, `Diluted EPS`,
`Total Stockholder Equity`/`Stockholders Equity`, `Total Debt`, `Current Assets`,
`Current Liabilities`, `Operating Income`, `Free Cash Flow`); computes
`avail_date = period_end + 45d`; upserts `FinancialReport`. yfinance provides ~4–5 years
of quarterly history, covering the 5y candle window. Flags: `--jobs`, `--limit`,
`--tickers`.

### 3. Sweep refactor

New helper `_prepare_fundamentals_pit(ticker, price_df, reports_df, dividends)` →
a **date-indexed metrics DataFrame** (index = trading days), columns = the 17 metric
values:

- Report-derived metrics: build TTM rolling sums (revenue, net_income, operating_income,
  fcf) and point-in-time balance-sheet values from `reports_df`, indexed by `avail_date`,
  then **forward-filled** onto trading days (a report's values apply from its `avail_date`
  until the next report's `avail_date`).
- Price-derived metrics computed directly from `price_df` (+ shares, dividends).
- Combine (PE, PS, PB, PEG, margins) from the aligned frame.

Bucketing moves **inside the per-trade path**: for each trade `(ticker, entry_idx)`,
`row = metrics.asof(entry_date)`. For each `DIMENSIONS` entry: if `pit=True` bucket
`row[field]` (point-in-time); if `pit=False` bucket the ticker's current
`Fundamental` snapshot value (same as the old behavior, now clearly the exception).

The **`DIMENSIONS` pattern is preserved** — `(name, field, bucket_fn, order)` tuples and
bucket functions are unchanged in shape; only the *source of values* changes from a
static snapshot dict to the per-trade PIT row. New dimensions add new tuples + bucket
functions. Downstream consumers (`StockStudy.by_dimension` JSON, the "Best fundamental
bucket" column, Firing Now) are unaffected in shape and pick up the new dimensions
automatically.

### 4. Coverage / NA handling

Trades before a ticker's first `avail_date` (pre-IPO window, or tickers with no yfinance
history) yield `None` for report-derived metrics → bucketed as **`NA (no PIT data)`**,
rendered as such, never silently mis-bucketed. Price-derived metrics still resolve
wherever candles exist. Per-dimension NA counts are surfaced so thin coverage is visible.

### 5. Recurring refresh

`fetch_financial_history` wired into the data-refresh flow (Celery beat), scheduled
**before** the nightly `stock-studies-daily` sweep so PIT inputs are fresh. Quarterly
data changes rarely, so a daily pull is cheap (mostly no-op upserts); frequency tunable.

## Testing

- Unit: `avail_date` lag; TTM rollups; each metric formula on a hand-built fixture;
  bucket-boundary edges; NA path when no report precedes a trade.
- PIT-correctness test: a trade dated *before* a report's `avail_date` must NOT see that
  report's numbers (the core anti-lookahead guarantee).
- Parity/sanity: latest PIT row per ticker ≈ current `Fundamental` snapshot (within
  restatement/lag tolerance) to catch normalization bugs.
- Integration: small `--limit` sweep produces populated `by_dimension` with all 23
  dimensions (17 PIT + 6 snapshot, correctly flagged) and non-trivial NA accounting.

## Out of scope

- Universe survivorship bias (delisted tickers).
- **True point-in-time** for the 6 snapshot-only dimensions — they ship as snapshot
  (lookahead-labeled) now; upgrading them to PIT is Future work.

## Future work

- **Tier C, free:** SEC EDGAR `companyfacts`/`frames` (financials with true filing
  dates — could *replace* the `+45d` approximation), EDGAR 13F (institutional), Forms
  3/4/5 (insider), FINRA short interest (free). More engineering, no cost.
- **Tier C, paid:** EODHD (cheapest broad), FMP (analyst history), Sharadar (best PIT
  correctness). Would slot in as additional `FinancialReport`-style sources + dimensions.

## Rollout

1. Model + migration (`FinancialReport`, optional `DividendHistory`).
2. Backfill job; run full backfill.
3. Sweep refactor (`_prepare_fundamentals_pit`) + new dimension tuples/bucket-fns.
4. Re-run all-on-all sweep → repopulate `StockStudy.by_dimension` (23 dims: 17 PIT +
   6 snapshot-only).
5. Wire recurring backfill into beat.
6. Verify app: Best-bucket column + row-expand show 23 dimensions, PIT vs snapshot
   badged, with NA accounting.
