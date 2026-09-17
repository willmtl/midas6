#!/usr/bin/env python3
"""What would the best money-flow cell (CMF monthly bullish-divergence, top-200 mega-caps) actually BUY right now?
Shows the last few months' fired names. Divergence = price at a 20-month low while CMF(20) is higher than 10
months ago, on currently-listed top-200-by-mktcap names, $5M/day + >$5. Run:
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/moneyflow_live_picks.py"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, Fundamental, Sector

DVOL_FLOOR = 5e6; PRICE_FLOOR = 5.0; TOPN = 200
etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
recent = set(Candle.objects.filter(interval="1d", date__gte="2026-09-01").values_list("ticker", flat=True).distinct())
rows = [r for r in Fundamental.objects.filter(ticker__in=recent, market_cap__gt=0).exclude(ticker__in=etfs).values("ticker", "market_cap") if "." not in r["ticker"]]
rows.sort(key=lambda r: r["market_cap"], reverse=True)
uni = [r["ticker"] for r in rows[:TOPN]]
mcap = {r["ticker"]: r["market_cap"] for r in rows[:TOPN]}
print(f"universe: top {len(uni)} currently-listed by mktcap", flush=True)

from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
fires = {}   # month -> list of (ticker, dict)
close_m_all = {}
for i in range(0, len(uni), 100):
    q = Candle.objects.filter(ticker__in=uni[i:i + 100], interval="1d", date__gte="2010-01-01").values_list("ticker", "date", "high", "low", "close", "volume")
    df = pd.DataFrame(list(q), columns=["ticker", "date", "high", "low", "close", "volume"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"])
    for c in ("high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date")
        h = s["high"].resample("ME").max(); l = s["low"].resample("ME").min()
        c = s["close"].resample("ME").last(); v = s["volume"].resample("ME").sum()
        if c.notna().sum() < 40:
            continue
        rng = (h - l).replace(0, np.nan); mfm = ((c - l) - (h - c)) / rng; mfv = (mfm * v).fillna(0.0)
        cmf = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        dvol = (s["close"] * s["volume"]).rolling(20).mean().resample("ME").last()
        price_low = c <= c.rolling(20, min_periods=15).min()
        fired = price_low & (cmf > cmf.shift(10)) & (dvol >= DVOL_FLOOR) & (c > PRICE_FLOOR)
        for d in c.index[fired.fillna(False)]:
            fires.setdefault(d, []).append((tk, float(c[d]), float(cmf[d]), float(mcap.get(tk, 0))))

months = sorted(fires)[-4:]
print(f"\n=== CMF monthly bullish-divergence fires — last {len(months)} months (top-{TOPN} mega-caps) ===", flush=True)
for d in months:
    lst = sorted(fires[d], key=lambda x: -x[3])
    print(f"\n{d.date()}  ({len(lst)} names):", flush=True)
    for tk, px, cmf, mc in lst:
        print(f"   {tk:6} ${px:8.2f}  CMF {cmf:+.3f}  mktcap ${mc/1e9:.0f}B", flush=True)
if not months:
    print("no fires found", flush=True)
