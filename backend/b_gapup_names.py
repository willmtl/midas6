#!/usr/bin/env python3
"""Which stocks drive the B(capitulation, w15) x mo_gap_up cell (+1.96%/68%/t5.31)? Rebuild the seq w15
windows, run the masked H4 backtest per name, and record each ticker's mo_gap_up 3-bar returns.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_gapup_names.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_on_signals_study as S
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS
from intraday_data import get_4h

_name, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
DVOL, WIN = 5e6, 15

uni = S._stock_universe()
daily = load_candles(uni)
# build seq w15 windows
allowed = {}
for tk, df in daily.items():
    if len(df) < 60:
        continue
    sig = seq_fn(df).fillna(False); idx = df.index
    dvol = (df["Close"] * df["Volume"]).rolling(20).mean()
    s = set()
    for i, v in enumerate(sig.values):
        if not v or dvol.iloc[i] < DVOL:
            continue
        for j in range(i, min(i + WIN, len(idx))):
            s.add(idx[j].date())
    if s:
        allowed[tk] = s

rows = []
allr = []
for tk, dates in allowed.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    res = S.backtest_ticker_masked(df, dates)
    r = [x for x in res["mo_gap_up"]["flat"].get("3b", []) if np.isfinite(x)]
    if r:
        a = np.asarray(r, float)
        rows.append((tk, len(r), a.mean(), (a > 0).mean() * 100))
        allr.extend(r)

rows.sort(key=lambda x: -x[1])
A = np.asarray(allr, float)
print(f"TOTAL: {len(A)} mo_gap_up entries across {len(rows)} names | avg {A.mean():+.2f}% | win {(A>0).mean()*100:.0f}%\n", flush=True)
print(f"  {'ticker':8}{'n':>5}{'avg%':>9}{'win%':>7}", flush=True)
for tk, n, av, w in rows:
    print(f"  {tk:8}{n:>5}{av:>+9.2f}{w:>7.0f}", flush=True)
