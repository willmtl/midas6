#!/usr/bin/env python3
"""Candlestick patterns + short-horizon TA on the 4h clock (the A/B horizon). For each pattern/signal, measure
the forward 1/2/3/6-bar return, subtract the BASE RATE (unconditional fwd return of any bar), split by 4h
trend (close vs 50-bar SMA), and report n / avg% / win% / t + edge-over-base. Refuted on C (monthly hold) —
this is the RIGHT clock (0-3 bar). Universe = cached-4h liquid names (reuses get_4h).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/candle_ta_h4_study.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import ta
import h4_on_signals_study as S
from intraday_data import get_4h

HORIZONS = [1, 2, 3, 6]


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def patterns(df):
    """Boolean pattern series on OHLC (index-aligned)."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - c.combine(o, max)
    lower = c.combine(o, min) - l
    po, pc = o.shift(1), c.shift(1)
    pbody = (pc - po)
    P = {}
    P["doji"] = (body <= 0.1 * rng)
    P["hammer"] = (lower >= 2 * body) & (upper <= body) & (body > 0)              # bullish reversal
    P["shooting_star"] = (upper >= 2 * body) & (lower <= body) & (body > 0)       # bearish reversal
    P["bull_engulf"] = (pc < po) & (c > o) & (c >= po) & (o <= pc)                # green engulfs prior red
    P["bear_engulf"] = (pc > po) & (c < o) & (c <= po) & (o >= pc)               # red engulfs prior green
    P["marubozu_up"] = (c > o) & (body >= 0.9 * rng)                             # strong-body continuation
    P["marubozu_dn"] = (c < o) & (body >= 0.9 * rng)
    # short-horizon TA (failed as monthly-value gates; tested here on the right clock)
    stoch = ta.momentum.stoch(h, l, c, window=14, smooth_window=3)
    P["stoch_x_up"] = (stoch > 20) & (stoch.shift(1) <= 20)                      # %K crossing out of oversold
    wr = ta.momentum.williams_r(h, l, c, lbp=14)
    P["williams_os"] = (wr <= -80)                                              # Williams %R deep oversold
    cci = ta.trend.cci(h, l, c, window=20)
    P["cci_os"] = (cci <= -100) & (cci.shift(1) > -100)                          # CCI entering oversold
    return P


def main():
    uni = S._stock_universe()
    frames = {}
    for tk in uni:
        df = get_4h(tk, 5, False)                       # cached only (fast)
        if df is not None and len(df) >= 120:
            frames[tk] = df
    print(f"4h cached for {len(frames)} names\n", flush=True)

    pooled = {}          # pat -> horizon -> {'all':[], 'up':[], 'dn':[]}
    base = {hz: [] for hz in HORIZONS}
    pat_names = None
    for tk, df in frames.items():
        c = df["Close"].values
        n = len(c)
        sma50 = df["Close"].rolling(50).mean().values
        fwd = {hz: np.full(n, np.nan) for hz in HORIZONS}
        for hz in HORIZONS:
            fwd[hz][:n - hz] = (c[hz:] - c[:n - hz]) / c[:n - hz] * 100
            base[hz].extend(fwd[hz][np.isfinite(fwd[hz])])
        P = patterns(df)
        if pat_names is None:
            pat_names = list(P); pooled = {p: {hz: {"all": [], "up": [], "dn": []} for hz in HORIZONS} for p in pat_names}
        for p, sig in P.items():
            s = sig.fillna(False).values
            idx = np.flatnonzero(s)
            for hz in HORIZONS:
                for i in idx:
                    r = fwd[hz][i]
                    if not np.isfinite(r):
                        continue
                    up = np.isfinite(sma50[i]) and c[i] > sma50[i]
                    pooled[p][hz]["all"].append(r)
                    pooled[p][hz]["up" if up else "dn"].append(r)

    base_avg = {hz: float(np.mean(base[hz])) for hz in HORIZONS}
    print("BASE RATE (unconditional fwd return, %):  " +
          "  ".join(f"{hz}b {base_avg[hz]:+.3f}" for hz in HORIZONS) + "\n", flush=True)
    print(f"{'pattern':14}{'n@3b':>7}{'avg%':>8}{'edge%':>8}{'win%':>7}{'t':>7}   {'UPtrend edge/t':>16}   {'DNtrend edge/t':>16}", flush=True)
    rows = []
    for p in pat_names:
        a = np.asarray(pooled[p][3]["all"], float); a = a[np.isfinite(a)]
        if len(a) < 30:
            continue
        edge = a.mean() - base_avg[3]
        up = np.asarray(pooled[p][3]["up"], float); up = up[np.isfinite(up)]
        dn = np.asarray(pooled[p][3]["dn"], float); dn = dn[np.isfinite(dn)]
        ue = (up.mean() - base_avg[3]) if len(up) >= 20 else None
        de = (dn.mean() - base_avg[3]) if len(dn) >= 20 else None
        rows.append((p, len(a), a.mean(), edge, (a > 0).mean() * 100, _t(a), ue, _t(up), de, _t(dn)))
    rows.sort(key=lambda r: -(r[3]))
    for p, n, av, ed, w, t, ue, ut, de, dt in rows:
        us = f"{ue:+.2f}/{ut}" if ue is not None else "—"
        ds = f"{de:+.2f}/{dt}" if de is not None else "—"
        print(f"{p:14}{n:>7}{av:>+8.2f}{ed:>+8.2f}{w:>7.0f}{str(t):>7}   {us:>16}   {ds:>16}", flush=True)


if __name__ == "__main__":
    main()
