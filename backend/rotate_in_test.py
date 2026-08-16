#!/usr/bin/env python3
"""Is the dashboard's ROTATE IN the best sector-rotation signal? Backtest it directly.

ROTATE IN = BULLISH + RSI(10) freshly crossed above its SMA(10) in the last 3 days (RSI<50 & Omega>1 at the
cross). BULLISH = RSI(10)>SMA(10) AND Omega(10)>1. Reproduce both on each sector ETF's ABSOLUTE price, sample
the state at each month-end, and measure forward RELATIVE return vs SPY (does that sector then beat SPY?).
Compare ROTATE-IN vs BULLISH-only vs all-sectors base rate.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rotate_in_test.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import config
from backtest_lowpb import BENCH, CRYPTO
from seq_fundamental_study import load_candles
import ta

HZ = [1, 3, 6]


def hit(vals):
    a = np.asarray(vals, float); a = a[~np.isnan(a)]
    if not len(a):
        return (None, None, 0)
    return (round(float((a > 0).mean()) * 100, 1), round(float(a.mean()) * 100, 2), len(a))


def main():
    etfs = [e for e in config.SECTOR_ETFS.values() if e and e not in CRYPTO]
    data = load_candles(etfs + [BENCH])
    spy_m = data[BENCH]["Close"].resample("ME").last()
    midx = spy_m.index

    groups = ["ROTATE IN", "BULLISH (all)", "BULLISH-only (no cross)", "all (base rate)"]
    fwd = {g: {h: [] for h in HZ} for g in groups}

    for e in etfs:
        df = data.get(e)
        if df is None or len(df) < 260:
            continue
        c = df["Close"]
        rsi = ta.momentum.rsi(c, window=10)
        rsi_sma = rsi.rolling(10).mean()
        above = rsi > rsi_sma
        cross = above & (~above.shift(1).fillna(False)) & (rsi < 50)
        fresh3 = cross.rolling(3).max().fillna(0).astype(bool)
        ret = c.pct_change()
        pos = ret.clip(lower=0).rolling(10).sum()
        neg = (-ret.clip(upper=0)).rolling(10).sum()
        omega = pos / neg.replace(0, np.nan)
        bullish = (above & (omega > 1)).fillna(False)
        rotate_in = (bullish & fresh3).fillna(False)

        em = c.resample("ME").last().reindex(midx)
        bull_m = bullish.reindex(midx, method="ffill").fillna(False)
        rot_m = rotate_in.reindex(midx, method="ffill").fillna(False)
        for i in range(12, len(midx)):
            for h in HZ:
                if i + h >= len(midx):
                    continue
                rel = (em.iloc[i + h] / em.iloc[i] - 1) - (spy_m.iloc[i + h] / spy_m.iloc[i] - 1)
                if not np.isfinite(rel):
                    continue
                fwd["all (base rate)"][h].append(rel)
                if rot_m.iloc[i]:
                    fwd["ROTATE IN"][h].append(rel)
                if bull_m.iloc[i]:
                    fwd["BULLISH (all)"][h].append(rel)
                    if not rot_m.iloc[i]:
                        fwd["BULLISH-only (no cross)"][h].append(rel)

    print(f"\n=== IS 'ROTATE IN' THE BEST SECTOR SIGNAL? forward RELATIVE return vs SPY ({len(midx)} months) ===", flush=True)
    print(f"{'group':26} | " + "  ".join(f"+{h}mo (hit% / mean%)" for h in HZ), flush=True)
    for g in groups:
        cells = []
        for h in HZ:
            hr, mn, nn = hit(fwd[g][h])
            cells.append(f"{hr}%/{mn:>+5}% (n{nn})")
        print(f"{g:26} | " + "  ".join(cells), flush=True)
    print("\n(if ROTATE IN doesn't beat BULLISH-only or the base rate, the fresh-cross EVENT adds nothing)", flush=True)


if __name__ == "__main__":
    main()
