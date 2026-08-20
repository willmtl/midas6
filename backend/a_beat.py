#!/usr/bin/env python3
"""Beat-tests for FLAGSHIP A (gap-up momentum, high-vol liquid, wide diversified book). Baseline = K=40, hold5,
FIFO, no gate. Variants: magnitude-ranked selection, uptrend gate (close>SMA50), longer hold (8b), wider K(60).
Reports net @10/20bps. Beats baseline?  Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/a_beat.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import h4_on_signals_study as S
from intraday_data import get_4h

GAP, VOL_LO, VOL_HI, DVOL = 0.05, 0.50, 3.00, 5e6


def collect():
    """Every vol+liquidity-qualified 4h bar with its burst/gap magnitudes, uptrend flag, and forward 5/8/12-bar
    returns — so the sim can filter by any trigger and hold without re-fetching."""
    trades = []
    for tk in S._stock_universe():
        df = get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        c = df["Close"].values; idx = df.index.astype("int64").values
        vol = (pd.Series(c).pct_change().rolling(20).std() * (252 ** 0.5)).values
        dv = (df["Close"] * df["Volume"]).rolling(20).mean().values
        sma = pd.Series(c).rolling(50).mean().values
        for i in range(2, len(c) - 12):
            if not (VOL_LO <= vol[i] <= VOL_HI and dv[i] >= DVOL and c[i] > 0):
                continue
            g1 = c[i] / c[i - 1] - 1.0; b2 = c[i] / c[i - 2] - 1.0
            up = bool(np.isfinite(sma[i]) and c[i] > sma[i])
            trades.append((idx[i], g1, b2, up,
                           idx[i + 5], c[i + 5] / c[i] - 1.0,
                           idx[i + 8], c[i + 8] / c[i] - 1.0,
                           idx[i + 12], c[i + 12] / c[i] - 1.0))
    return trades


HOLDS = {5: (4, 5), 8: (6, 7), 12: (8, 9)}   # hold -> (exit_ts_col, ret_col)


def sim(trades, trig, K=40, fee=1e-3, hold=5, rank=False, trend_only=False):
    exc, rc = HOLDS[hold]
    ts = [t for t in trades if trig(t) and (not trend_only or t[3])]
    ts.sort(key=lambda t: (t[0], -t[2]) if rank else (t[0],))
    busy = [0] * K; eq = [1.0] * K; taken = 0
    for t in ts:
        ent, ex, ret = t[0], t[exc], t[rc]
        for k in range(K):
            if busy[k] <= ent:
                eq[k] *= (1.0 + ret - 2 * fee); busy[k] = ex; taken += 1
                break
    return float(np.mean(eq)), taken


def main():
    trades = collect()
    ents = [t[0] for t in trades]
    yrs = (max(ents) - min(ents)) / 1e9 / 86400 / 365.25
    print(f"qualified bars {len(trades)}, {yrs:.1f}y\n", flush=True)
    T_gap5 = lambda t: t[1] >= 0.05
    T_b2_5 = lambda t: t[2] >= 0.05
    T_b2_8 = lambda t: t[2] >= 0.08
    variants = [
        ("burst2>=5% hold5 (a_return)", T_b2_5, dict(hold=5)),
        ("burst2>=5% hold8",            T_b2_5, dict(hold=8)),
        ("burst2>=5% hold12",           T_b2_5, dict(hold=12)),
        ("burst2>=8% hold8",            T_b2_8, dict(hold=8)),
        ("gap1>=5% hold5",              T_gap5, dict(hold=5)),
        ("gap1>=5% hold8",              T_gap5, dict(hold=8)),
        ("burst2>=5% hold8 +rank",      T_b2_5, dict(hold=8, rank=True)),
        ("burst2>=5% hold8 +uptrend",   T_b2_5, dict(hold=8, trend_only=True)),
    ]
    print(f"  {'variant':30}{'gross%':>11}{'net10%':>11}{'net20%':>11}{'CAGRn20':>9}{'taken':>8}", flush=True)
    for lab, trig, kw in variants:
        g, tk = sim(trades, trig, fee=0.0, **kw)
        n10, _ = sim(trades, trig, fee=1e-3, **kw)
        n20, _ = sim(trades, trig, fee=2e-3, **kw)
        cg = ((n20) ** (1 / yrs) - 1) * 100 if n20 > 0 else float("nan")
        print(f"  {lab:30}{(g-1)*100:>11.0f}{(n10-1)*100:>11.0f}{(n20-1)*100:>11.0f}{cg:>9.0f}{tk:>8}", flush=True)


if __name__ == "__main__":
    main()
