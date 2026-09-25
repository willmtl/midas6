#!/usr/bin/env python
"""NEW CONCEPT (user 2026-09-22): RSI-of-(relative-Sortino-vs-SPY).

For each selected stock:
  1. relative Sortino(14): rolling Sortino of the stock's EXCESS return over SPY (SPY is the reference / MAR),
        excess_t   = stock_ret_t - spy_ret_t
        sortino_t  = mean(excess over 14d) / downside_dev(excess over 14d)      (downside = min(excess,0), rms)
     -> a rolling, downside-only measure of how well the stock is beating SPY.
  2. RSI(14) on that Sortino series  (Wilder; ffill().fillna(0) before RSI, matching indicators.compute_rsi_of_sortino).

This is the Sortino analogue of the price-RS concept: instead of RSI on the stock/SPY PRICE ratio, it's RSI on the
stock's rolling risk-adjusted OUTperformance vs SPY. Builds the panel + sanity diagnostics. (backtest comes next.)

Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sortino_rsi_study.py
Env: WIN (Sortino window, default 14), RSI_N (default 14), PANELS (default /app/.data/panels.pkl), BENCH (default SPY)
"""
import os, json, pickle
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from seq_fundamental_study import load_candles

WIN = int(os.environ.get("WIN", 14))
RSI_N = int(os.environ.get("RSI_N", 14))
PANELS = os.environ.get("PANELS", "/app/.data/panels.pkl")
BENCH = os.environ.get("BENCH", "SPY")


def wilder_rsi(x, n):
    """Wilder RSI on a Series or DataFrame (column-wise)."""
    d = x.diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return (100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan)))


def rel_sortino_vs_bench(ret, bench_ret, window):
    """Rolling Sortino of (stock - benchmark) excess returns. ret is a DataFrame (cols=tickers), bench_ret a Series.

    Matches indicators.rolling_sortino's shape (mean/downside-rms, no annualization) but with MAR = benchmark
    instead of the risk-free rate. Vectorized across all tickers.
    """
    excess = ret.sub(bench_ret, axis=0)                       # stock excess over SPY, per day
    mean_ex = excess.rolling(window).mean()
    neg = excess.clip(upper=0.0)
    dd = np.sqrt((neg ** 2).rolling(window).mean())           # downside deviation over the window
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino = mean_ex / dd.replace(0.0, np.nan)
    # low/zero-downside sentinel (mirror indicators.rolling_sortino): flat window -> NaN, else finite ±9.99
    low = dd < 1e-10
    sortino = sortino.mask(low & (mean_ex.abs() >= 1e-10), np.sign(mean_ex) * 9.99)
    sortino = sortino.mask(low & (mean_ex.abs() < 1e-10), np.nan)
    return sortino


def main():
    b = pickle.load(open(PANELS, "rb"))
    universe = list(b["common"])
    cd = load_candles(universe + [BENCH])
    scl = pd.DataFrame({t: cd[t]["Close"] for t in universe if t in cd and "Close" in cd[t]}).sort_index()
    scl = scl[scl > 0]
    if BENCH not in cd:
        raise SystemExit("no benchmark candles")
    spy = cd[BENCH]["Close"].reindex(scl.index)
    ret = scl.pct_change()
    spy_ret = spy.pct_change()

    sortino = rel_sortino_vs_bench(ret, spy_ret, WIN)                     # relative Sortino(WIN) vs SPY
    rsi_sort = wilder_rsi(sortino.ffill().fillna(0.0), RSI_N)             # RSI(RSI_N) of that Sortino
    print(f"built panels: rel-Sortino{WIN} + RSI{RSI_N}-of-Sortino  shape={rsi_sort.shape}  "
          f"names_with_data={int(rsi_sort.notna().any().sum())}", flush=True)

    # ---- sanity diagnostics ----
    last = rsi_sort.iloc[-1].dropna()
    print(f"\nlatest RSI-of-rel-Sortino across {len(last)} names: "
          f"min {last.min():.1f}  p25 {last.quantile(.25):.1f}  median {last.median():.1f}  "
          f"p75 {last.quantile(.75):.1f}  max {last.max():.1f}", flush=True)
    print(f"  oversold (<30): {int((last < 30).sum())}   overbought (>70): {int((last > 70).sum())}", flush=True)

    # worked example so the math is inspectable
    ex = next((t for t in ("AAPL", "MSFT", "JPM", "XOM") if t in scl.columns), scl.columns[0])
    tail = pd.DataFrame({
        "stock_ret%": (ret[ex] * 100).round(2),
        "spy_ret%": (spy_ret * 100).round(2),
        "excess%": ((ret[ex] - spy_ret) * 100).round(2),
        f"relSortino{WIN}": sortino[ex].round(3),
        f"RSI{RSI_N}ofSortino": rsi_sort[ex].round(1),
    }).dropna().tail(8)
    print(f"\nworked example [{ex}] (last 8 rows):", flush=True)
    print(tail.to_string(), flush=True)

    # persist the panels for the next step (signal/backtest)
    with open("/app/.data/studies/sortino_rsi_panel.pkl", "wb") as fh:
        pickle.dump(dict(rel_sortino=sortino, rsi_of_sortino=rsi_sort, win=WIN, rsi_n=RSI_N, bench=BENCH), fh, protocol=4)
    print("\nsaved /app/.data/studies/sortino_rsi_panel.pkl", flush=True)


if __name__ == "__main__":
    main()
