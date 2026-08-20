#!/usr/bin/env python3
"""MAJOR (N-bar) engulfing on the 4h clock: a candle whose BODY engulfs the prior N candles' bodies (N=2,3) —
rare + large by construction (the magnitude tail, unlike the ~64k-fire 1-bar version that was noise). Hypothesis:
a 2-3 bar bull engulf is a big bullish THRUST -> should CONTINUE like a burst (positive edge), not mean-revert.
Bull tested forward-UP; bear tested forward-DOWN (short/veto). Base-rate-adjusted, magnitude/trend/context splits,
horizons 3/6/8 bars. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/engulf_major.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import ta
import h4_on_signals_study as S
from intraday_data import get_4h

HZ = [3, 6, 8]


def _stat(a, base):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return None
    sd = a.std(ddof=1)
    t = a.mean() / (sd / np.sqrt(len(a))) if sd > 0 else 0
    return (len(a), a.mean(), a.mean() - base, (a > 0).mean() * 100, t)


def _line(lbl, s):
    if s is None:
        return f"    {lbl:28} (n<20)"
    n, av, ed, w, t = s
    return f"    {lbl:28} n={n:>5}  avg {av:+.2f}%  edge {ed:+.2f}%  win {w:.0f}%  t={t:+.2f}"


def main():
    uni = S._stock_universe()
    # (N, side) -> list of (r3, r6, r8, mag, up, decl, hivol)
    rec = {(N, s): [] for N in (2, 3) for s in ("bull", "bear")}
    base = {hz: [] for hz in HZ}
    nfr = 0
    for tk in uni:
        df = get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        nfr += 1
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        cv = c.values; n = len(cv)
        bodyhi = pd.concat([o, c], axis=1).max(axis=1)
        bodylo = pd.concat([o, c], axis=1).min(axis=1)
        atr = ta.volatility.average_true_range(h, l, c, window=14)
        sma50 = c.rolling(50).mean()
        annv = c.pct_change().rolling(20).std() * (252 ** 0.5)
        body = (c - o).abs()
        f = {}
        for hz in HZ:
            arr = np.full(n, np.nan); arr[:n - hz] = (cv[hz:] - cv[:n - hz]) / cv[:n - hz] * 100
            f[hz] = arr; base[hz].extend(arr[np.isfinite(arr)])
        magv = (body / atr).values; upv = (c > sma50).values; annvv = annv.values
        for N in (2, 3):
            prior_hi = bodyhi.shift(1).rolling(N).max()      # highest body-top over prior N bars
            prior_lo = bodylo.shift(1).rolling(N).min()      # lowest body-bottom over prior N bars
            decl = (c.shift(1) < c.shift(N + 1)).values      # net decline INTO the pattern (prior N bars down)
            adv = (c.shift(1) > c.shift(N + 1)).values
            bull = ((c > o) & (o <= prior_lo) & (c >= prior_hi)).fillna(False).values
            bear = ((c < o) & (o >= prior_hi) & (c <= prior_lo)).fillna(False).values
            for side, sig, ctx in (("bull", bull, decl), ("bear", bear, adv)):
                for i in np.flatnonzero(sig):
                    if not (np.isfinite(f[3][i]) and np.isfinite(magv[i]) and np.isfinite(annvv[i])):
                        continue
                    rec[(N, side)].append((f[3][i], f[6][i], f[8][i], magv[i],
                                           bool(upv[i]), bool(ctx[i]), bool(annvv[i] >= 0.50)))
    b = {hz: float(np.mean(base[hz])) for hz in HZ}
    print(f"4h cached for {nfr} names; base rate  " + "  ".join(f"{hz}b {b[hz]:+.3f}%" for hz in HZ) + "\n", flush=True)

    for N in (2, 3):
        for side in ("bull", "bear"):
            arr = np.array(rec[(N, side)], float)
            print(f"{'='*90}", flush=True)
            note = "forward UP = bullish continuation works" if side == "bull" else "forward DOWN/negative = bearish works (short/veto)"
            if not len(arr):
                print(f"== {N}-BAR {side.upper()} ENGULFING: no fires", flush=True); continue
            r3, r6, r8, mag, upf, ctx, hiv = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                                              arr[:, 4] > 0, arr[:, 5] > 0, arr[:, 6] > 0)
            q75 = np.nanpercentile(mag, 75)
            strong = mag >= q75
            print(f"== {N}-BAR {side.upper()} ENGULFING  (n={len(arr)}, {note})   ctx = {'after-decline' if side=='bull' else 'after-advance'}", flush=True)
            for hz, rr in ((3, r3), (6, r6), (8, r8)):
                print(f"  -- @{hz}b --", flush=True)
                print(_line("ALL", _stat(rr, b[hz])), flush=True)
                print(_line("STRONG (top-quartile mag)", _stat(rr[strong], b[hz])), flush=True)
                print(_line("in UPtrend", _stat(rr[upf], b[hz])), flush=True)
                print(_line("in DOWNtrend", _stat(rr[~upf], b[hz])), flush=True)
                print(_line("context (reversal/contin.)", _stat(rr[ctx], b[hz])), flush=True)
                print(_line("STRONG & UPtrend", _stat(rr[strong & upf], b[hz])), flush=True)
                print(_line("STRONG & high-vol name", _stat(rr[strong & hiv], b[hz])), flush=True)
            print("", flush=True)


if __name__ == "__main__":
    main()
