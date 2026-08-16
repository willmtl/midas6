#!/usr/bin/env python3
"""AS-TRADED price basis for POINT-IN-TIME valuation (finding #2). Candles are stored dividend/split-ADJUSTED
(auto_adjust=True), so a future split back-adjusts the historical price DOWN and makes a name that later split
10:1/50:1 look ~1/N as cheap on P/B at past dates — a LOOK-AHEAD that pulls future-splitters into the 'value'
book and inflated the backtest by ~+178pp. This recovers the AS-TRADED (split-as-known-at-t) price for building
P/B / market-cap in backtests: as_traded = adj_close * product(split ratios STRICTLY AFTER t). Returns must keep
using the adjusted close (correct for total return); only the valuation LEVEL/ranking uses as-traded.

NOTE: for a LIVE pick at today's date there are no future splits, so as_traded == adj_close — the live scanner
is already correct and needs no change; this matters only for historical (backtest / time-machine) P/B.

Splits are cached at /app/.data/splits_cache.json (populated by pb_split_fix_study.py; missing ticker -> no
correction). Refresh with refresh_splits(tickers)."""
import os, json, time
import numpy as np, pandas as pd

SPLIT_CACHE = "/app/.data/splits_cache.json"


def load_splits():
    try:
        return json.load(open(SPLIT_CACHE))
    except Exception:
        return {}


def refresh_splits(tickers):
    """Fetch (yfinance) & cache split history for any ticker not already cached. Returns the full cache."""
    import yfinance as yf
    cache = load_splits()
    todo = [t for t in tickers if t not in cache]
    if todo:
        print(f"price_basis: fetching splits for {len(todo)} tickers...", flush=True)
        for i, t in enumerate(todo):
            try:
                sp = yf.Ticker(t).splits
                cache[t] = {str(pd.Timestamp(d).tz_localize(None).date()): float(r) for d, r in sp.items()} if len(sp) else {}
            except Exception:
                cache[t] = {}
            if (i + 1) % 100 == 0:
                json.dump(cache, open(SPLIT_CACHE, "w")); time.sleep(0.1)
        json.dump(cache, open(SPLIT_CACHE, "w"))
    return cache


def _future_split_factor(split_map, midx):
    """product of split ratios STRICTLY AFTER each month-end (undoes future back-adjustment)."""
    if not split_map:
        return pd.Series(1.0, index=midx)
    items = sorted((pd.Timestamp(d), r) for d, r in split_map.items())
    out = []
    for m in midx:
        f = 1.0
        for d, r in items:
            if d > m:
                f *= r
        out.append(f)
    return pd.Series(out, index=midx)


def as_traded_close(monthly_adj_close, splits=None):
    """monthly_adj_close: DataFrame [month-end index x tickers] of ADJUSTED month-end Close.
    Returns the as-traded close (adj * future-split-factor). Tickers absent from the splits cache are unchanged.
    Use ONLY for P/B / market-cap ranking; keep the adjusted close for returns."""
    if splits is None:
        splits = load_splits()
    midx = monthly_adj_close.index
    sf = pd.DataFrame({t: _future_split_factor(splits.get(t, {}), midx) for t in monthly_adj_close.columns},
                      index=midx).reindex(index=midx, columns=monthly_adj_close.columns).fillna(1.0)
    return monthly_adj_close * sf
