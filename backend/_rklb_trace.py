#!/usr/bin/env python
"""Single-ticker Pine trace for RKLB: reproduce the exact BUY/SELL round-trips the user's chart shows.
Runs BOTH the full Pine (ulcer gate ON, =their chart) and the no-ulcer tuned scanner, with real prices."""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S
from seq_fundamental_study import load_candles

TK = os.environ.get("TK", "RKLB")
cd = load_candles([TK])
d = cd[TK]
close = pd.DataFrame({TK: d["Close"]}).sort_index(); close = close[close > 0]
days = close.index
op = d["Open"].reindex(days); hp = d["High"].reindex(days); lp = d["Low"].reindex(days)
ohlc4 = (op + hp + lp + close[TK]) / 4.0
ohlc4 = pd.DataFrame({TK: ohlc4})
ind = S.compute_indicators(close, ohlc4)
close_np = close[TK].to_numpy(); valid = close[TK].notna().to_numpy()
srsi = ind["srsi"][TK].to_numpy(); srsi_ema = ind["srsi_ema"][TK].to_numpy()
nrsi = ind["nrsi"][TK].to_numpy(); nrsi_ema = ind["nrsi_ema"][TK].to_numpy()
dd = ind["dd_score"][TK].to_numpy(); ulcer = ind["ulcer"][TK].to_numpy()
warm = max(S.SORT_WIN + S.SORT_SMOOTH + S.SORT_RSI + S.SORT_EMA, S.DD_DD, 200)


def trace(ulcer_min, label):
    S.ULCER_MIN = ulcer_min
    trips = S.run_state_machine(0, close_np, srsi, srsi_ema, nrsi, nrsi_ema, dd, ulcer, valid, warm)
    print(f"\n=== {label} (ULCER_MIN={ulcer_min}) — {len(trips)} trades ===")
    print(f"  {'entry':>10} {'exit':>10} {'reason':>8} {'hold':>4} {'entry$':>9} {'exit$':>9} {'ret%':>8}  dd/ulcer/srsi@entry")
    tot = 1.0
    for (ei, xi, reason, bac) in trips:
        ep = close_np[ei]
        if xi > 0:
            xp = close_np[xi]; ret = xp / ep - 1.0; xd = str(days[xi].date()); hold = xi - ei
        else:
            xp = close_np[valid.nonzero()[0][-1]]; ret = xp / ep - 1.0; xd = "OPEN"; hold = int(valid.nonzero()[0][-1] - ei)
        tot *= (1 + ret)
        print(f"  {str(days[ei].date()):>10} {xd:>10} {reason:>8} {hold:>4} {ep:>9.2f} {xp:>9.2f} {ret*100:>+7.1f}%  "
              f"dd={dd[ei]:+.2f} ulc={ulcer[ei]:.1f} srsi={srsi[ei]:.0f}>{srsi_ema[ei]:.0f} nrsi={nrsi[ei]:.0f}")
    print(f"  compounded (all, sequential): {(tot-1)*100:+.0f}%")
    # last 12 months only
    cut = days[-1] - pd.Timedelta(days=365)
    last = [(ei, xi, reason) for (ei, xi, reason, bac) in trips if days[ei] >= cut]
    ly = 1.0
    for (ei, xi, reason) in last:
        xp = close_np[xi] if xi > 0 else close_np[valid.nonzero()[0][-1]]
        ly *= (1 + (xp / close_np[ei] - 1.0))
    print(f"  LAST 12 MONTHS: {len(last)} trades, compounded {(ly-1)*100:+.0f}%")


print(f"{TK}: {len(days)} bars, {days[0].date()} -> {days[-1].date()}, last close ${close_np[valid.nonzero()[0][-1]]:.2f}")
trace(10.0, "FULL PINE — your chart")
trace(0.0, "NO-ULCER — tuned scanner base")
