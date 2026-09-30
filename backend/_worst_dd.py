#!/usr/bin/env python
"""Intra-trade max drawdown (MAE) for the current tuned book: how far each position went underwater
between entry and exit. Reads the entries CSV + candles; ranks the worst."""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import pandas as pd, numpy as np
from seq_fundamental_study import load_candles

t = pd.read_csv("/app/.data/studies/sortino_dd_entries.csv")
t = t[t.tuned == True].copy()
print(f"tuned book: {len(t)} trades, {t.ticker.nunique()} names")
cd = load_candles(list(t.ticker.unique()))
rows = []
for _, r in t.iterrows():
    d = cd.get(r.ticker)
    if d is None or "Close" not in d:
        continue
    cl = d["Close"].dropna(); lo = d.get("Low", cl).reindex(cl.index).fillna(cl)
    pos = cl.index.searchsorted(pd.Timestamp(r.date))
    if pos >= len(cl):
        continue
    epos = min(pos + int(r.hold_days), len(cl) - 1)
    seg_c = cl.iloc[pos:epos + 1]
    if len(seg_c) < 2:
        continue
    ep = float(seg_c.iloc[0]); seg_l = lo.reindex(seg_c.index).fillna(seg_c)
    trough = float(seg_l.min()); mdd = trough / ep - 1.0
    tdate = seg_l.idxmin()
    rows.append(dict(ticker=r.ticker, sector=r.sector, entry=r.date, exit=str(seg_c.index[-1].date()),
                     maxdd=mdd * 100, trough=str(tdate.date()), final=r.rt_ret,
                     recovered=(r.rt_ret > 0)))
df = pd.DataFrame(rows).sort_values("maxdd")
print(f"\n=== 20 trades with the DEEPEST intra-trade drawdown ===")
print(f"{'ticker':>7} {'entry':>11} {'trough':>11} {'maxDD':>7} {'final':>8} {'sector':>12}")
for _, r in df.head(20).iterrows():
    flag = "recovered" if r.recovered else "LOSS"
    print(f"{r.ticker:>7} {r.entry:>11} {r.trough:>11} {r.maxdd:>6.0f}% {r.final:>+7.0f}% {r.sector:>12}  {flag}")
print(f"\nmedian intra-trade maxDD across the book: {df.maxdd.median():.0f}%")
print(f"trades that went >30% underwater: {(df.maxdd<=-30).sum()}/{len(df)} "
      f"({(df.maxdd<=-30).mean()*100:.0f}%); of those, recovered to a win: "
      f"{df[df.maxdd<=-30].recovered.mean()*100:.0f}%")
# worst by stock
print(f"\n=== worst intra-trade drawdown BY STOCK ===")
g = df.groupby("ticker").maxdd.min().sort_values().head(12)
for tk, v in g.items():
    print(f"  {tk:>7} {v:>6.0f}%")
