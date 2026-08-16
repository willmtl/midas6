#!/usr/bin/env python3
"""Short-term oversold-reversal entry on ABSOLUTE single-stock price — the mirror of the RS-bar test.

On the RS bar, oversold crosses were a falling knife (relative strength is trend-persistent). Here we run
the SAME timing on ABSOLUTE stock price across the full stock universe and measure forward ABSOLUTE return
at short holds. If oversold crosses pay here (and DEEPER oversold pays MORE = the tail), that confirms the
reversal edge is an absolute single-stock phenomenon and pins the entry.

Events (RSI(10) on Close): cross up through 20 / 30 / 35 / 50.
Also: bucket the cross-up-through-30 events by the MIN RSI reached in the prior 5 bars (depth = the tail).
Diagnostic (forward windows), no fees; stock-universe survivorship applies.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/stock_rsi_timing.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd, ta
import config
from seq_fundamental_study import load_candles

HORIZONS = [1, 3, 5, 10, 21]
DEPTH_BUCKETS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 50)]


def _t(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if len(a) < 3 or a.std(ddof=1) == 0:
        return None
    return round(float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), 1)


def _m(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return (round(float(a.mean()) * 100, 2), round(float((a > 0).mean()) * 100, 1), len(a)) if len(a) else (None, None, 0)


def main():
    from core.models import Candle, Fundamental
    etfs = set(config.SECTOR_ETFS.values()) | {"SPY", "QQQ"}
    cand = set(Candle.objects.values_list("ticker", flat=True).distinct())
    fund = set(Fundamental.objects.values_list("ticker", flat=True).distinct())
    universe = sorted((cand & fund) - etfs)
    print(f"universe: {len(universe)} stocks (Candle ∩ Fundamental − ETFs)", flush=True)
    data = load_candles(universe)

    events = ["RSI x20 (deep)", "RSI x30", "RSI x35", "RSI x50"]
    thr = {"RSI x20 (deep)": 20, "RSI x30": 30, "RSI x35": 35, "RSI x50": 50}
    fwd = {ev: {h: [] for h in HORIZONS} for ev in events}
    depth = {b: {h: [] for h in (5, 10)} for b in DEPTH_BUCKETS}   # x30 events bucketed by prior-min RSI
    n = 0

    for tkr in universe:
        df = data.get(tkr)
        if df is None or len(df) < 120:
            continue
        c = df["Close"]
        rsi = ta.momentum.rsi(c, window=10)
        if rsi.notna().sum() < 60:
            continue
        n += 1
        fret = {h: c.shift(-h) / c - 1 for h in HORIZONS}
        for ev in events:
            m = ((rsi > thr[ev]) & (rsi.shift(1) <= thr[ev])).fillna(False)
            for h in HORIZONS:
                fwd[ev][h] += list(fret[h][m].dropna().values)
        # depth buckets on the x30 cross: how deep did RSI get in the last 5 bars?
        m30 = ((rsi > 30) & (rsi.shift(1) <= 30)).fillna(False)
        prior_min = rsi.rolling(5).min()
        for idx in np.where(m30.values)[0]:
            pm = prior_min.iloc[idx]
            if not np.isfinite(pm):
                continue
            for lo, hi in DEPTH_BUCKETS:
                if lo <= pm < hi:
                    for h in (5, 10):
                        v = fret[h].iloc[idx]
                        if np.isfinite(v):
                            depth[(lo, hi)][h].append(v)
                    break

    print(f"\n=== ABSOLUTE single-stock oversold entry on {n} stocks — forward ABSOLUTE return ===", flush=True)
    print(f"{'event':16} | " + "  ".join(f"+{h}d" for h in HORIZONS) + "   (mean% / %pos / t)", flush=True)
    for ev in events:
        cells = []
        for h in HORIZONS:
            mean, pos, cnt = _m(fwd[ev][h])
            cells.append(f"{mean:>+5}/{pos:>4}/{str(_t(fwd[ev][h])):>4}")
        print(f"{ev:16} | " + "  ".join(cells), flush=True)

    print(f"\n=== THE TAIL: x30 crosses bucketed by MIN RSI in prior 5 bars (deeper = bigger bounce?) ===", flush=True)
    print(f"{'prior-min RSI':14} | {'+5d mean/%pos/t/n':>26} | {'+10d mean/%pos/t/n':>26}", flush=True)
    for b in DEPTH_BUCKETS:
        m5, p5, n5 = _m(depth[b][5]); m10, p10, n10 = _m(depth[b][10])
        print(f"{f'{b[0]}-{b[1]}':14} | {f'{m5:>+5}/{p5:>4}/{str(_t(depth[b][5])):>4}/{n5:>6}':>26} | "
              f"{f'{m10:>+5}/{p10:>4}/{str(_t(depth[b][10])):>4}/{n10:>6}':>26}", flush=True)


if __name__ == "__main__":
    main()
