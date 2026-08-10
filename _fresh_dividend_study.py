"""
Event study: when a SECTOR's FRESH composite completes ("fresh full rotation"),
buy an equal-weight basket of that sector's dividend stocks. Measure forward returns
and compare vs the sector ETF and vs SPY.
"""
import warnings; warnings.filterwarnings("ignore")
import json, os
import numpy as np, pandas as pd
import ta
import config, data_store
from indicators import (daily_returns, rolling_sortino,
                        RSI_PERIOD, RSI_SMA_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

HOLD = [5, 10, 20, 30, 60, 90]
WIN_D, WIN_W = 14, 8
SW = config.SORTINO_WINDOW

GICS_TO_ETF = {
    "Technology": "XLK", "Communication Services": "XLC", "Healthcare": "XLV",
    "Real Estate": "XLRE", "Consumer Defensive": "XLP", "Financial Services": "XLF",
    "Industrials": "XLI", "Utilities": "XLU", "Energy": "XLE",
    "Basic Materials": "XLB", "Consumer Cyclical": "XLY",
}

full = json.load(open(".data/fundamentals/all_fundamentals.json"))["tickers"]
# dividend-stock basket per GICS sector (payers with candle data)
basket = {g: [] for g in GICS_TO_ETF}
for tk, v in full.items():
    if not isinstance(v, dict):
        continue
    s = v.get("sector")
    if s not in GICS_TO_ETF:
        continue
    y = v.get("dividend_yield") or v.get("trailing_div_yield") or v.get("forward_div_yield")
    if isinstance(y, (int, float)) and y > 0 and os.path.exists(f".data/5y/{tk}.json"):
        basket[s].append(tk)

# preload candles
CACHE = {}
def load(tk):
    if tk not in CACHE:
        CACHE[tk] = data_store.load_ticker(tk, "5y")
    return CACHE[tk]

spy = load("SPY")


def _cross(x, sma):
    return (x > sma) & (x.shift(1) <= sma.shift(1))


def fresh_events(df, mode="D"):
    """Dates where the fresh composite COMPLETES: the later of the two crossovers
    lands today while the other crossover is within `window` and Sortino>0 & MACD green
    hold today (matches trend_analyzer fresh_days==0). Deduped within `window` bars."""
    if mode == "W":
        agg = {c: getattr(df[c].resample("W-FRI"), how)()
               for c, how in (("Open", "first"), ("High", "max"), ("Low", "min"),
                              ("Close", "last"), ("Volume", "sum")) if c in df.columns}
        d = pd.DataFrame(agg).dropna(subset=["Close"])
        if len(df.index) and df.index[-1].dayofweek != 4:
            d = d.iloc[:-1]
        window, sortino_weekly = WIN_W, False
    else:
        d, window, sortino_weekly = df, WIN_D, True
    if len(d) < 60:
        return []

    rsi = ta.momentum.rsi(d["Close"], window=RSI_PERIOD)
    rsi_cross = _cross(rsi, rsi.rolling(RSI_SMA_PERIOD).mean())
    rsi_recent = rsi_cross.rolling(window, min_periods=1).max().fillna(0).astype(bool)

    ret = daily_returns(d)
    sort = rolling_sortino(ret, SW).reindex(d.index).ffill().fillna(0)
    rs = ta.momentum.rsi(sort, window=RSI_PERIOD)
    rs_cross = _cross(rs, rs.rolling(RSI_SMA_PERIOD).mean())
    rs_recent = rs_cross.rolling(window, min_periods=1).max().fillna(0).astype(bool)

    if sortino_weekly:
        wk = d["Close"].resample("W-FRI").last().dropna()
        wkret = np.log(wk / wk.shift(1)).dropna()
        wksort = rolling_sortino(wkret, SW)
        sort_pos = (wksort > 0).reindex(d.index, method="ffill").fillna(False)
    else:
        sort_pos = (rolling_sortino(ret, SW).reindex(d.index) > 0).fillna(False)

    # completion = the later crossover lands today, other already within window, Sortino>0
    # (MACD removed from the rule)
    completion = ((rsi_cross & rs_recent) | (rs_cross & rsi_recent)) & sort_pos
    dates = list(d.index[completion.values])
    # dedup: drop events within `window` bars of a kept event
    kept = []
    for t in dates:
        if not kept or (d.index.get_loc(t) - d.index.get_loc(kept[-1])) > window:
            kept.append(t)
    return kept


def fwd_ret(df, t, N):
    if df is None or df.empty:
        return None
    pos = df.index.get_indexer([t], method="ffill")[0]
    if pos == -1 or pos + N >= len(df):
        return None
    c0, c1 = df["Close"].iloc[pos], df["Close"].iloc[pos + N]
    if not (c0 > 0):
        return None
    return c1 / c0 - 1


def basket_ret(tickers, t, N):
    rs = [fwd_ret(load(tk), t, N) for tk in tickers]
    rs = [r for r in rs if r is not None]
    return (np.mean(rs), len(rs)) if rs else (None, 0)


def run(mode):
    # rows: per (sector,event,N) -> basket, etf, spy returns
    per_sector = {g: {N: {"basket": [], "etf": [], "spy": []} for N in HOLD} for g in GICS_TO_ETF}
    pooled = {N: {"basket": [], "etf": [], "spy": []} for N in HOLD}
    event_count = {}
    for gics, etf in GICS_TO_ETF.items():
        edf = load(etf)
        if edf is None:
            continue
        events = fresh_events(edf, mode)
        event_count[gics] = len(events)
        for t in events:
            for N in HOLD:
                b, n = basket_ret(basket[gics], t, N)
                e = fwd_ret(edf, t, N)
                s = fwd_ret(spy, t, N)
                if b is None or e is None:
                    continue
                per_sector[gics][N]["basket"].append(b)
                per_sector[gics][N]["etf"].append(e)
                per_sector[gics][N]["spy"].append(s)
                pooled[N]["basket"].append(b)
                pooled[N]["etf"].append(e)
                pooled[N]["spy"].append(s)
    return per_sector, pooled, event_count


def stat(lst):
    if not lst:
        return None
    a = np.array(lst)
    return {"n": len(a), "avg": float(a.mean() * 100), "med": float(np.median(a) * 100),
            "hit": float((a > 0).mean() * 100)}


results = {}
for mode, label in (("D", "DAILY fresh"), ("W", "WEEKLY fresh")):
    per_sector, pooled, ec = run(mode)
    results[mode] = {"pooled": {N: {k: stat(v[k]) for k in v} for N, v in pooled.items()},
                     "events": ec,
                     "per_sector": {g: {N: {k: stat(v[k]) for k in v} for N, v in d.items()}
                                    for g, d in per_sector.items()}}

    print(f"\n{'='*78}\n{label} trigger — buy sector dividend basket on 'fresh full rotation'\n{'='*78}")
    total_ev = sum(ec.values())
    print(f"Total fresh events across 11 sectors: {total_ev}  |  windows D={WIN_D}td W={WIN_W}wk\n")
    print(f"{'Hold':>5} | {'BASKET avg%':>11} {'hit%':>5} {'n':>4} | {'ETF avg%':>9} {'hit%':>5} | {'SPY avg%':>9} | {'edge vs ETF':>11} {'edge vs SPY':>11}")
    print("-" * 92)
    for N in HOLD:
        b, e, s = results[mode]["pooled"][N]["basket"], results[mode]["pooled"][N]["etf"], results[mode]["pooled"][N]["spy"]
        if not b:
            continue
        print(f"{N:>5} | {b['avg']:>10.2f}% {b['hit']:>5.0f} {b['n']:>4} | {e['avg']:>8.2f}% {e['hit']:>5.0f} | "
              f"{s['avg']:>8.2f}% | {b['avg']-e['avg']:>+10.2f}% {b['avg']-s['avg']:>+10.2f}%")

    # per-sector at 20d
    print(f"\n  Per-sector basket return @ 20-day hold ({label}):")
    print(f"  {'Sector':24} {'events':>7} {'avg%':>8} {'hit%':>6} {'ETF avg%':>9} {'edge':>7}")
    rowsps = []
    for g in GICS_TO_ETF:
        st = results[mode]["per_sector"][g][20]
        if st["basket"]:
            rowsps.append((g, st["basket"], st["etf"]))
    for g, bs, es in sorted(rowsps, key=lambda r: r[1]["avg"], reverse=True):
        print(f"  {g:24} {bs['n']:>7} {bs['avg']:>7.2f}% {bs['hit']:>5.0f}% {es['avg']:>8.2f}% {bs['avg']-es['avg']:>+6.2f}%")

json.dump(results, open(".data/dividends/fresh_dividend_study.json", "w"), indent=2, default=str)
print("\nSaved -> .data/dividends/fresh_dividend_study.json")
