#!/usr/bin/env python3
"""Win/loss profile of FLAGSHIP-B (capitulation w15 -> gap-up @3b): avg winner, avg loser, worst loss, win/loss
ratio, distribution. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_loss.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS
from intraday_data import get_4h

_n, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
daily = load_candles(S._stock_universe())
allowed = {}
for tk, df in daily.items():
    if len(df) < 60:
        continue
    sig = seq_fn(df).fillna(False); idx = df.index
    dv = (df["Close"] * df["Volume"]).rolling(20).mean()
    s = set()
    for i in np.flatnonzero(sig.values):
        if dv.iloc[i] < 5e6:
            continue
        for j in range(i, min(i + 15, len(idx))):
            s.add(idx[j].date())
    if s:
        allowed[tk] = s

r = []
for tk, dates in allowed.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    entry, _m = H.SIGNALS["mo_gap_up"]["fn"](df)
    c = df["Close"].values; ts = df.index
    for i in range(len(c) - 3):
        if entry[i] and ts[i].date() in dates:
            r.append((c[i + 3] - c[i]) / c[i] * 100)

a = np.array(r, float)
w = a[a > 0]; l = a[a <= 0]
print(f"B gap-up@3b: n={len(a)}  avg {a.mean():+.2f}%  median {np.median(a):+.2f}%", flush=True)
print(f"  WINS  {len(w)} ({len(w)/len(a)*100:.0f}%)  avg win  {w.mean():+.2f}%  best {a.max():+.2f}%", flush=True)
print(f"  LOSSES {len(l)} ({len(l)/len(a)*100:.0f}%) avg loss {l.mean():+.2f}%  worst {a.min():+.2f}%", flush=True)
print(f"  win/loss size ratio {abs(w.mean()/l.mean()):.2f}  |  expectancy {a.mean():+.2f}% per trade", flush=True)
print(f"  loss percentiles: 25th {np.percentile(l,75):+.2f}  median {np.median(l):+.2f}  10th(worst) {np.percentile(l,10):+.2f}  5th {np.percentile(l,5):+.2f}", flush=True)
