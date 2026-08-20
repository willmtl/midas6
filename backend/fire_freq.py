#!/usr/bin/env python3
"""Trigger firing frequency (per week, universe-wide) for the A/B short-horizon selectors, over the 4h-tradeable
era (2021-09+). A = gap-up>=5% in a high-vol liquid name; B = capitulation seq (+$5M floor). Counts distinct
name-day trigger events, not window-days.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fire_freq.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import h4_on_signals_study as S
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS

_name, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
ERA = pd.Timestamp("2021-09-01")          # 4h-tradeable era

daily = load_candles(S._stock_universe())
a_fires = b_fires = 0
a_names = set(); b_names = set()
maxd = ERA
for tk, df in daily.items():
    if len(df) < 60:
        continue
    df = df[df.index >= ERA]
    if len(df) < 5:
        continue
    maxd = max(maxd, df.index.max())
    c = df["Close"]
    vol = c.pct_change().rolling(20).std() * (252 ** 0.5)
    dvol = (c * df["Volume"]).rolling(20).mean()
    gap = c / c.shift(1) - 1.0
    a = int(((gap >= 0.05) & vol.between(0.5, 3.0) & (dvol >= 5e6)).sum())
    b = int(((seq_fn(df).fillna(False)) & (dvol >= 5e6)).sum())
    a_fires += a; b_fires += b
    if a: a_names.add(tk)
    if b: b_names.add(tk)

weeks = max(1.0, (maxd - ERA).days / 7.0)
print(f"span {ERA.date()} -> {maxd.date()}  = {weeks:.0f} weeks\n", flush=True)
print(f"A (gap-up momentum): {a_fires} fires over {len(a_names)} names  =>  {a_fires/weeks:.1f} / week  "
      f"({a_fires/weeks/5:.1f} / trading day)", flush=True)
print(f"B (capitulation seq): {b_fires} fires over {len(b_names)} names  =>  {b_fires/weeks:.1f} / week  "
      f"({b_fires/weeks*4.3:.0f} / month)", flush=True)
