#!/usr/bin/env python3
"""Market-cap profile of the FLAGSHIP-B (capitulation w15 -> gap-up) trade names. Are they really big-cap?
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_mktcap.py"""
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
from core.models import Fundamental

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

# count gap-up trades per name
trades = {}
for tk, dates in allowed.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    entry, _m = H.SIGNALS["mo_gap_up"]["fn"](df)
    ts = df.index
    n = sum(1 for i in range(len(ts) - 3) if entry[i] and ts[i].date() in dates)
    if n:
        trades[tk] = n

mc = dict(Fundamental.objects.filter(ticker__in=list(trades)).values_list("ticker", "market_cap"))
buckets = [("mega  >$50B", 50e9, 9e15), ("large $10-50B", 10e9, 50e9), ("mid   $2-10B", 2e9, 10e9),
           ("small $0.3-2B", 3e8, 2e9), ("micro <$0.3B", 0, 3e8)]
by_names = {b[0]: 0 for b in buckets}; by_trades = {b[0]: 0 for b in buckets}
caps = []; nocap = 0
for tk, n in trades.items():
    m = mc.get(tk)
    if not m:
        nocap += 1; continue
    caps.append(m)
    for lab, lo, hi in buckets:
        if lo <= m < hi:
            by_names[lab] += 1; by_trades[lab] += n; break

tot_n = sum(by_names.values()) or 1; tot_t = sum(by_trades.values()) or 1
print(f"B gap-up trade names: {len(trades)} ({nocap} no mktcap)  |  median mktcap ${np.median(caps)/1e9:.1f}B\n", flush=True)
print(f"  {'bucket':16}{'names':>7}{'names%':>8}{'trades':>8}{'trades%':>9}", flush=True)
for lab, _lo, _hi in buckets:
    print(f"  {lab:16}{by_names[lab]:>7}{by_names[lab]/tot_n*100:>7.0f}%{by_trades[lab]:>8}{by_trades[lab]/tot_t*100:>8.0f}%", flush=True)
