#!/usr/bin/env python3
"""WHY does buying at month-end close beat next-morning open? Decompose the +overnight gap into:
  (1) generic small-cap OVERNIGHT premium — do these names gap up EVERY night, not just month-end?
  (2) broad TURN-OF-MONTH — does SPY also gap up on the same month-end nights?
  (3) name-specific EVENT bump at the month-end boundary — month-end gap >> the name's own typical night.

For each buy trade (ticker, bd=month-end): me_gap = open(next day)/close(bd)-1.
Compare vs the SAME name's mean overnight over its full series, and vs SPY's month-end-night gap.
"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import load_candles

d = json.load(open("/app/.data/studies/flagship_history.json"))
trades, tickers = [], set()
for m in d["months"]:
    bd, sd = m.get("date"), m.get("ndate")
    if not sd:
        continue
    for pk in m.get("picks") or []:
        tk, ret = pk.get("ticker"), pk.get("ret")
        if tk and ret is not None:
            trades.append({"tk": tk, "bd": bd}); tickers.add(tk)

from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
cand, tks = {}, sorted(tickers | {"SPY"})
for i in range(0, len(tks), 40):
    cand.update(load_candles(tks[i:i + 40]))
px = {}
for tk, df in cand.items():
    if df is None or df.empty:
        continue
    s = df[["Open", "Close"]].copy()
    s.index = pd.to_datetime(s.index).normalize()
    px[tk] = s[~s.index.duplicated(keep="last")].sort_index()

def overnight_series(tk):
    """open[t+1]/close[t]-1 across the whole series."""
    s = px.get(tk)
    if s is None or len(s) < 3:
        return None
    return (s["Open"].shift(-1) / s["Close"] - 1).dropna()

def me_gap(tk, bd):
    s = px.get(tk)
    if s is None:
        return None
    bd = pd.Timestamp(bd).normalize()
    if bd not in s.index:
        return None
    fut = s.index[s.index > bd]
    if len(fut) == 0:
        return None
    c = s.at[bd, "Close"]; o = s.at[fut[0], "Open"]
    if not (pd.notna(c) and pd.notna(o) and c > 0 and o > 0):
        return None
    return float(o / c - 1)

# precompute per-name full-series mean overnight (the "typical night" baseline)
name_base = {tk: (ov.mean() if (ov := overnight_series(tk)) is not None and len(ov) > 20 else None) for tk in tickers}

me, base_matched, spy_me, excess = [], [], [], []
for t in trades:
    g = me_gap(t["tk"], t["bd"])
    if g is None:
        continue
    me.append(g)
    b = name_base.get(t["tk"])
    if b is not None:
        base_matched.append(b); excess.append(g - b)
    sg = me_gap("SPY", t["bd"])
    if sg is not None:
        spy_me.append(sg)

def summ(x):
    x = np.array(x)
    t = x.mean() / (x.std() / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else 0.0
    return f"mean={x.mean()*1e4:+7.1f} bps  t={t:+.2f}  n={len(x)}"

print("=== WHY: month-end overnight gap decomposition (buy names) ===", flush=True)
print(f"(1) month-end overnight gap, held names   : {summ(me)}", flush=True)
print(f"(2) same names' TYPICAL night (full series): {summ(base_matched)}   <- generic overnight premium", flush=True)
print(f"(3) name EXCESS at month-end (1)-(2)       : {summ(excess)}   <- event/turn-of-month bump above their own norm", flush=True)
print(f"(4) SPY overnight gap on same month-end nts: {summ(spy_me)}   <- broad turn-of-month", flush=True)

# full-market baseline for SPY: typical SPY night
spy_ov = overnight_series("SPY")
if spy_ov is not None:
    print(f"(5) SPY TYPICAL night (full series)        : mean={spy_ov.mean()*1e4:+7.1f} bps   (n={len(spy_ov)})", flush=True)
