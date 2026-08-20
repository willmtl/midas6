#!/usr/bin/env python3
"""ENGULFING candles, proper deep-dive on the 4h clock (not the naive average that came back -0.07% before).
Per tail-not-average: bucket by MAGNITUDE (vol-normalized body size), by CONTEXT (bullish-engulf after a decline
= reversal setup vs in an uptrend = continuation), base-rate-subtracted, with t-stats. Also tests BEARISH engulfing
as a forward-DOWN / short / veto signal (we've never found a tradeable short — this checks a pattern directly).
Universe = cached-4h liquid names, also split high-vol vs low-vol (vol is the short-horizon driver).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/engulf_study.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import ta
import h4_on_signals_study as S
from intraday_data import get_4h

H3, H6 = 3, 6


def _stat(a, base):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return None
    sd = a.std(ddof=1)
    t = a.mean() / (sd / np.sqrt(len(a))) if sd > 0 else 0
    return (len(a), a.mean(), a.mean() - base, (a > 0).mean() * 100, t)


def _line(lbl, s):
    if s is None:
        return f"    {lbl:26} (n<20)"
    n, av, ed, w, t = s
    return f"    {lbl:26} n={n:>6}  avg {av:+.2f}%  edge {ed:+.2f}%  win {w:.0f}%  t={t:+.2f}"


def main():
    uni = S._stock_universe()
    # per-pattern pooled records: dict side -> list of (ret3, ret6, mag, up, decl, hivol)
    rec = {"bull": [], "bear": []}
    base3, base6 = [], []
    nfr = 0
    for tk in uni:
        df = get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        nfr += 1
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        cv = c.values; n = len(cv)
        rng = (h - l)
        body = (c - o)
        po, pc = o.shift(1), c.shift(1)
        atr = ta.volatility.average_true_range(h, l, c, window=14)
        sma50 = c.rolling(50).mean()
        annv = c.pct_change().rolling(20).std() * (252 ** 0.5)
        prior3 = c.shift(1) / c.shift(4) - 1.0                                   # 3-bar move INTO the pattern bar
        bull = (pc < po) & (c > o) & (c >= po) & (o <= pc)                       # green body engulfs prior red
        bear = (pc > po) & (c < o) & (c <= po) & (o >= pc)                       # red body engulfs prior green
        f3 = np.full(n, np.nan); f6 = np.full(n, np.nan)
        f3[:n - H3] = (cv[H3:] - cv[:n - H3]) / cv[:n - H3] * 100
        f6[:n - H6] = (cv[H6:] - cv[:n - H6]) / cv[:n - H6] * 100
        base3.extend(f3[np.isfinite(f3)]); base6.extend(f6[np.isfinite(f6)])
        mag = (body.abs() / atr).values                                         # vol-normalized body size (tail metric)
        up = (c > sma50).values
        p3 = prior3.values
        for side, sig in (("bull", bull.fillna(False).values), ("bear", bear.fillna(False).values)):
            for i in np.flatnonzero(sig):
                if not (np.isfinite(f3[i]) and np.isfinite(mag[i]) and np.isfinite(p3[i]) and np.isfinite(annv.values[i])):
                    continue
                rec[side].append((f3[i], f6[i], mag[i], bool(up[i]), bool(p3[i] < 0), bool(annv.values[i] >= 0.50)))
    b3 = float(np.mean(base3)); b6 = float(np.mean(base6))
    print(f"4h cached for {nfr} names; base rate 3b {b3:+.3f}%  6b {b6:+.3f}%", flush=True)
    print(f"bull engulf n={len(rec['bull'])}  bear engulf n={len(rec['bear'])}\n", flush=True)

    for side in ("bull", "bear"):
        arr = np.array(rec[side], float)
        if not len(arr):
            continue
        r3, r6, mag, upf, decl, hiv = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3] > 0, arr[:, 4] > 0, arr[:, 5] > 0
        q75 = np.nanpercentile(mag, 75)
        strong = mag >= q75
        note = "(forward UP = bullish works)" if side == "bull" else "(forward DOWN/negative = bearish works, short/veto)"
        print(f"===== {side.upper()} ENGULFING {note}   strong = body/ATR >= {q75:.2f} (top quartile) =====", flush=True)
        print("  -- @3b (edge vs base) --", flush=True)
        print(_line("ALL", _stat(r3, b3)), flush=True)
        print(_line("STRONG (top-quartile mag)", _stat(r3[strong], b3)), flush=True)
        print(_line("in UPtrend", _stat(r3[upf], b3)), flush=True)
        print(_line("in DOWNtrend", _stat(r3[~upf], b3)), flush=True)
        print(_line("after DECLINE (reversal)", _stat(r3[decl], b3)), flush=True)
        print(_line("after ADVANCE (continuation)", _stat(r3[~decl], b3)), flush=True)
        print(_line("STRONG & UPtrend", _stat(r3[strong & upf], b3)), flush=True)
        print(_line("STRONG & after-DECLINE", _stat(r3[strong & decl], b3)), flush=True)
        print(_line("STRONG & high-vol name", _stat(r3[strong & hiv], b3)), flush=True)
        print("  -- @6b --", flush=True)
        print(_line("ALL", _stat(r6, b6)), flush=True)
        print(_line("STRONG (top-quartile mag)", _stat(r6[strong], b6)), flush=True)
        print(_line("STRONG & UPtrend", _stat(r6[strong & upf], b6)), flush=True)
        print(_line("STRONG & after-DECLINE", _stat(r6[strong & decl], b6)), flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
