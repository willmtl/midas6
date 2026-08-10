# SPY + QQQ RSI-Cross, Correlation & Beta Market Signals — Design

**Date:** 2026-08-02
**Scope:** Add broad-market RSI-cross signals (daily + weekly), plus sector
correlation/beta signals and report columns, to the studies engine — then run a full
data / studies / report update.

## Goal

Add new entry signals to the studies engine (`studies.py`) driven by the market indices
SPY and QQQ:

1. **Market timing** — fire when SPY and/or QQQ cross their RSI above its SMA (daily and
   weekly variants; weekly treated as the stronger signal).
2. **Amplifier gating** — combine the market cross with the sector's correlation/beta to
   the indices, to pick sectors that amplify a confirmed market turn.
3. **Standalone** correlation/beta crossing signals.
4. **Report context** — attach per-sector correlation/beta to both indices.

Because the market-timing signals key off the indices (not the per-sector `df`), their
entry dates are market-wide; each study then measures each sector's forward returns from
those dates against the existing exit conditions.

## Constants (in `studies.py`, tunable)

- `CB_WINDOW = 60` — rolling window (trading days) for correlation & beta. Longer than
  the 10-bar RSI because corr/beta need more samples to be stable.
- `HI_CORR = 0.7`, `HI_BETA = 1.0` — "high" thresholds.
- RSI cross uses window=10 (engine convention), SMA window=10.
- `BOTH_WINDOW_D = 3` (daily bars), `BOTH_WINDOW_W = 2` (weekly bars) — how recently
  SPY and QQQ must *each* have crossed for the "both" signal to fire.
- Weekly resample: `W-FRI` (matches the FRESH composite's weekly Sortino gate).

## Helpers to add in `studies.py`

- `_rsi_cross_series(close) -> pd.Series[bool]` — daily RSI(10) crosses above SMA(10):
  `(rsi > sma) & (rsi.shift(1) <= sma.shift(1))`, indexed like `close`.
- `_rsi_cross_series_weekly(close) -> pd.Series[bool]` — resample `close` to `W-FRI`,
  compute the same cross on weekly bars, then map each weekly cross to a **single daily
  entry** on the last trading day of the crossing week (so studies enter on a real daily
  bar, once per weekly cross). All-False if too few weekly bars.
- `_rolling_corr(close, ref_close, w=CB_WINDOW) -> pd.Series` — rolling Pearson
  correlation of daily returns.
- `_rolling_beta_series(close, ref_close, w=CB_WINDOW) -> pd.Series` — cov/var of returns
  (mirrors `indicators.rolling_beta`).
- `_min_corr_spyqqq(close, spy_close, qqq_close) -> pd.Series` — `min` of corr-to-SPY and
  corr-to-QQQ, elementwise.
- `_min_beta_spyqqq(close, spy_close, qqq_close) -> pd.Series` — `min` of beta-to-SPY and
  beta-to-QQQ, elementwise.

Every signal function returns a boolean Series reindexed to `df.index`; if a required
market series is missing it returns an all-False Series (graceful degradation).

## Signals (14 new)

RSI cross = RSI(10) above SMA(10). "min-corr" / "min-β" = min against **both** SPY and
QQQ. All reference SPY/QQQ and are injected in `run_study` (see below).

| Group | Key | Fires when |
|---|---|---|
| A. Market timing (daily) | `spy_rsi_x` | SPY daily RSI crosses above SMA |
| | `qqq_rsi_x` | QQQ daily RSI crosses above SMA |
| | `spy_qqq_rsi_x_both` | SPY and QQQ each crossed up within last `BOTH_WINDOW_D` bars |
| A. Market timing (weekly) | `spy_rsi_x_wk` | SPY weekly RSI cross (mapped to daily entry) |
| | `qqq_rsi_x_wk` | QQQ weekly RSI cross |
| | `spy_qqq_rsi_x_both_wk` | SPY and QQQ each weekly-crossed within last `BOTH_WINDOW_W` weeks |
| B. Standalone corr/beta | `corr_spyqqq_x_high` | sector min-corr crosses above `HI_CORR` |
| | `beta_spyqqq_x_high` | sector min-β crosses above `HI_BETA` |
| C. Gated (daily) | `spy_qqq_rsi_x_hibeta` | `spy_qqq_rsi_x_both` and min-β > `HI_BETA` |
| | `spy_qqq_rsi_x_hicorr` | `spy_qqq_rsi_x_both` and min-corr > `HI_CORR` |
| | `spy_qqq_rsi_x_hibeta_hicorr` | `spy_qqq_rsi_x_both` and both |
| C. Gated (weekly) | `spy_qqq_rsi_x_hibeta_wk` | `spy_qqq_rsi_x_both_wk` and min-β > `HI_BETA` |
| | `spy_qqq_rsi_x_hicorr_wk` | `spy_qqq_rsi_x_both_wk` and min-corr > `HI_CORR` |
| | `spy_qqq_rsi_x_hibeta_hicorr_wk` | `spy_qqq_rsi_x_both_wk` and both |

Study count: 310 → **324** signals; +14 × 70 exits = **~980 new study combos**.

**Note:** QQQ is itself in the sector universe, so for the QQQ sector row corr/β to QQQ
≈ 1.0 (self-reference). Harmless; left as-is.

## `run_study` injection

Define `MARKET_SIGNAL_KEYS = {...}` (the 14 keys above). In `run_study` (near line 2462),
extend the existing special-case branch:

```python
if sig_key == "rsi_x_pos_updn":
    spy_close = all_data[config.BENCHMARK]["Close"] if all_data.get(config.BENCHMARK) is not None else None
    signals = sig_fn(df, spy_close=spy_close).fillna(False)
elif sig_key in MARKET_SIGNAL_KEYS:
    spy_df = all_data.get(config.BENCHMARK)
    qqq_df = all_data.get("QQQ")
    spy_close = spy_df["Close"] if spy_df is not None else None
    qqq_close = qqq_df["Close"] if qqq_df is not None else None
    signals = sig_fn(df, spy_close=spy_close, qqq_close=qqq_close).fillna(False)
else:
    signals = sig_fn(df).fillna(False)
```

No `config.py` change: QQQ is already fetched (`"Nasdaq 100": "QQQ"` in `SECTOR_ETFS`).

## Report context (Group 4 — "just report")

Add per-sector correlation/beta vs both indices to the trend scan:

- New helper `rolling_correlation(asset_returns, bench_returns, window)` in
  `indicators.py` (mirrors existing `rolling_beta`).
- In `trend_analyzer.analyze()`: compute and add `corr_spy`, `corr_qqq`, `beta_qqq`
  (beta-vs-SPY already exists as `beta`). Requires QQQ df — fetch from `all_data["QQQ"]`.
- In `report.py`: add columns for `corr_spy`, `corr_qqq`, `beta_qqq`.

## "Update everything" run sequence

Order so new signals compute against fresh data in one pass:

1. **Refresh data** — automatic: `data_fetcher.fetch_all()` does incremental updates
   (last_date → today) and persists to `.data/5y/`. Triggered by the report/studies runner.
2. **DB import** — run `import_json` so the Postgres candle hypertable (frontend/Django
   source) reflects refreshed JSON. If the DB is unreachable here, note and skip.
3. **Re-run studies** — `run_all()`. Its DB skip logic computes only the ~980 new signal
   combos; existing studies stay as-is (one incremental day doesn't materially change
   5-year aggregates; a full recompute is a separate deliberate action).
4. **Re-run trend report** — `report.py` (also refreshes data via step 1, and now shows
   the corr/beta columns).

## Testing / verification

The project has no pytest setup; it uses standalone runnable scripts. Add
`test_market_signals.py` (run with `python test_market_signals.py`) that:

- Builds small synthetic `close`/`spy_close`/`qqq_close` series with a known RSI cross bar
  and asserts each daily `sig_*` is True on the cross bar, False elsewhere.
- Asserts `_rsi_cross_series_weekly` marks exactly one daily bar (the week's last trading
  day) per weekly cross.
- Asserts the "both" signal requires both indices within the window, and the gated signals
  require the amplifier condition (min-corr / min-β) on top.
- Asserts `_rolling_corr` / `_rolling_beta_series` return sane values (corr in [-1, 1]).

Integration: run one study per new key against real data; confirm trades produced and
saved. Confirm `generate_studies()` count rises by ~980 and `run_all()` labels them "new
to compute."

## Out of scope

- Full recompute of existing studies on fresh data (separate deliberate action).
- Weekly correlation/beta (corr/beta stay on the 60-day daily window).
- Frontend/dashboard surfacing (backend `compute_scan` unchanged).
- Any change to the live `trend_analyzer` BULLISH/FRESH signal *rules* (we only add
  report columns, not new gating).

## Notes / environment

- Not a git repository (`git: false`), so the plan uses verification steps instead of
  commits.
