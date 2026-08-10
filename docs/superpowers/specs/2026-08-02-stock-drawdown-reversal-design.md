# "30% off ATH + RSI reversal" Stock Strategy — Design (Core + Backtest)

**Date:** 2026-08-02
**Scope (this spec):** the shared drawdown indicator + entry signal, and a per-stock
**backtest study** that validates the idea. Live screen and daily alert are a deferred
follow-up (separate spec) — build only after the backtest looks good.

## Goal

Test a "buy the dip on quality names" strategy: enter a stock when it is **≥30% below its
all-time high** *and* momentum is turning up (RSI crosses above its SMA). The reversal
gate fires while the stock is still ≥30% down, so we catch the turn rather than a falling
knife. Validate historically across the sector-holding universe before building any UI.

## Universe & ATH

- **Universe:** the 858 distinct US holding tickers across all sector ETFs
  (`sector_holdings.get_holdings` per sector, deduped; `.`-containing foreign tickers
  dropped). Each ticker is assigned to ONE sector — the first sector whose holdings list
  contains it — so a trade is never double-counted across sectors.
- **ATH:** highest close in the available 5y daily data (`Close.cummax()`), an expanding
  running high. This is a 5-year high, not a true lifetime high (accepted limitation;
  no new data fetch). Data comes from `data_fetcher.fetch_tickers(universe)` (incremental,
  uses existing `.data/5y` / DB).

## Core (shared unit)

### Indicator — `indicators.py`
```python
def drawdown_from_high(df):
    """Fractional drop from the running all-time-high close (<= 0; -0.30 = 30% below ATH)."""
    close = df["Close"]
    return close / close.cummax() - 1.0
```
Returns a Series aligned to `df.index`. `-0.30` means 30% below the running high.

### Signal — `studies.py`
- Constant: `DD_THRESHOLD = 0.30` (tunable), placed with the other studies constants.
- Function:
```python
def sig_dd30_rsi_reversal(df):
    """>=30% below running ATH AND RSI(10) crosses above its SMA(10) on that bar."""
    dd = indicators.drawdown_from_high(df)
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    return (dd <= -DD_THRESHOLD) & cross
```
- Registered in the `SIGNALS` dict as `"dd30_rsi_reversal": ("30% off ATH + RSI Reversal",
  sig_dd30_rsi_reversal)`. Registration keeps the definition canonical and reusable, and
  means the main sector studies engine also gains this signal (+70 sector combos on the
  next `run_all()` — harmless bonus; noted, not the focus here).
- `studies.py` does **not** currently import `indicators` (confirmed) — add `import
  indicators` at the top alongside `import config` / `import data_fetcher`.

## Backtest study — new file `stock_drawdown_study.py`

A dedicated per-stock runner (the existing `stock_studies.py` / `stock_studies_v2.py` are
ETF-signal-driven — they don't score stocks individually — so they are not reused).

**Algorithm:**
1. Build the deduped universe + `ticker -> sector` map (as above).
2. `stock_data = data_fetcher.fetch_tickers(universe)`.
3. For each stock with ≥60 bars: compute `sig_dd30_rsi_reversal(sdf)` **once**, collect
   entry dates, cache `Close` array + drawdown-at-entry. Skip stocks with no entries.
4. For each `exit_key, exit_fn` in `EXITS` (all 70): for each stock, for each entry index,
   `exit_idx = exit_fn(sdf, idx)`; skip if `None`/`<= idx`; compute
   `ret = (close[exit_idx] - close[idx]) / close[idx] * 100` and `hold = exit_idx - idx`.
5. Aggregate per exit: `total_trades`, `avg_return`, `win_rate`, `avg_hold`,
   `sector_count`, plus `best_sectors` / `worst_sectors` (same shape as the studies engine:
   list of `{sector, trades, wins, losses, win_rate, avg_return, ...}`) and a
   `top_stocks` list (best individual tickers by avg_return, min-trades guard).
6. Save all 70 exit-results to `.data/studies/stock_drawdown.json` (mirrors the in-memory
   JSON pattern of `run_macd_studies.py`; DB persistence is out of scope here).

**Entrypoint:** `run_all()` under `if __name__ == "__main__":` prints a ranked summary
(top exits by avg_return, overall trade count, best sectors/stocks).

## Testing / verification

Standalone script `test_stock_drawdown.py` (`python test_stock_drawdown.py`), matching the
project's no-pytest convention:

- `drawdown_from_high`: on a synthetic series that peaks then falls 35%, assert the last
  value ≈ -0.35 and that it is 0 at a new high.
- `sig_dd30_rsi_reversal`: build a stock that rises, drops ≥35%, then turns up so RSI
  crosses its SMA — assert the signal is True on the reversal bar and False while still
  falling and False when <30% down.
- Threshold boundary: a stock exactly 25% down with an RSI cross does NOT fire; 30% down
  does.
- Study smoke test: run `stock_drawdown_study` for one exit (`"1w"`) over a 3-ticker
  subset of real data; assert it returns a result dict with `total_trades >= 0` and, when
  trades exist, a numeric `avg_return` and non-empty `best_sectors`.

Integration: run the full `run_all()` once; confirm `.data/studies/stock_drawdown.json`
is written with 70 exit-results and a non-zero total trade count.

## Out of scope (this spec)

- Live screen (current qualifying stocks) — deferred follow-up.
- Daily alert on newly-qualifying stocks — deferred follow-up.
- Backend endpoint / frontend dashboard tab — deferred follow-up.
- DB persistence of the stock study (JSON only for now).
- True lifetime ATH via max-period data (5y high is accepted).

## Notes / environment

- Not a git repository (`git: false`): plan uses verification steps, not commits.
- Adding the signal to `SIGNALS` adds +70 sector-ETF combos to the main studies engine on
  the next `run_all()`; acceptable and independent of the stock study.
