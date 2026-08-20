#!/usr/bin/env python3
"""Strategy-A equity curve: A = 4h momentum burst (>=5% 2-bar) on a high-vol liquid name, hold H bars, run as
an equal-weight K-slot capacity-capped book (gross 1x, no leverage). The per-trade edge is thin and A fires a
firehose (~132/wk), so the strategy return hinges on capacity + fees — reported gross and net at 5/10/20 bps/side.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/a_return.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import h4_on_signals_study as S
from intraday_data import get_4h

H = 5                      # hold bars (4h) — momentum grows to ~8b; 5 is the balanced pick
BURST, VOL_LO, VOL_HI, DVOL = 0.05, 0.50, 3.00, 5e6


def collect():
    trades = []
    names = S._stock_universe()
    got = 0
    for tk in names:
        df = get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        got += 1
        c = df["Close"].values; idx = df.index
        vol = (pd.Series(c).pct_change().rolling(20).std() * (252 ** 0.5)).values
        dv = (df["Close"] * df["Volume"]).rolling(20).mean().values
        for i in range(2, len(c) - H):
            b = c[i] / c[i - 2] - 1.0
            if b >= BURST and VOL_LO <= vol[i] <= VOL_HI and dv[i] >= DVOL and c[i] > 0:
                trades.append((idx[i].value, idx[i + H].value, c[i + H] / c[i] - 1.0, b))
    trades.sort()
    return trades, got


def sim(trades, K, fee):
    """K equal-weight slots, gross 1x. Take a trade when a slot is free (else skip = capacity cap)."""
    busy = [0] * K                      # busy_until timestamp per slot
    eq = [1.0] * K
    taken = 0
    for ent, exit_, ret, _mag in trades:
        for k in range(K):
            if busy[k] <= ent:
                eq[k] *= (1.0 + ret - 2 * fee)
                busy[k] = exit_
                taken += 1
                break
    book = float(np.mean(eq))
    return book, taken


def main():
    trades, got = collect()
    span_days = (trades[-1][0] - trades[0][0]) / 1e9 / 86400 if trades else 1
    yrs = span_days / 365.25
    print(f"A trades collected: {len(trades)} over {got} names, ~{yrs:.1f}y (H={H} bars)\n", flush=True)
    for K in (10, 20, 40):
        print(f"  --- K={K} concurrent slots (each {100/K:.0f}% of book) ---", flush=True)
        for lab, fee in (("gross", 0.0), ("5bps", 5e-4), ("10bps", 1e-3), ("20bps", 2e-3)):
            book, taken = sim(trades, K, fee)
            tot = (book - 1) * 100
            cagr = ((book) ** (1 / yrs) - 1) * 100 if yrs > 0 and book > 0 else float("nan")
            print(f"    {lab:6}: total {tot:>10.0f}%   CAGR {cagr:>7.1f}%   trades taken {taken}", flush=True)


if __name__ == "__main__":
    main()
