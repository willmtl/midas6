#!/usr/bin/env python3
"""Do held names that CRASHED on the buy day (month-end) bounce back OVERNIGHT?
Condition the month-end overnight gap (open_next/close_bd - 1) on how the name traded ON the buy day.
Two definitions of "crashed on the 31":
  day_ret   = close(bd)/close(prev trading day) - 1   (down big on the day, close-to-close)
  intra_ret = close(bd)/open(bd) - 1                   (sold off during the session)
Bucket both; report mean overnight + t per bucket. Real flagship picks only (flagship_history.json).
"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import load_candles

d = json.load(open("/app/.data/studies/flagship_history.json"))
trades, tickers = [], set()
for m in d["months"]:
    if not m.get("ndate"):
        continue
    for pk in m.get("picks") or []:
        if pk.get("ticker") and pk.get("ret") is not None:
            trades.append({"tk": pk["ticker"], "bd": m["date"]}); tickers.add(pk["ticker"])

from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
cand, tks = {}, sorted(tickers)
for i in range(0, len(tks), 40):
    cand.update(load_candles(tks[i:i + 40]))
px = {}
for tk, df in cand.items():
    if df is None or df.empty:
        continue
    s = df[["Open", "Close"]].copy()
    s.index = pd.to_datetime(s.index).normalize()
    px[tk] = s[~s.index.duplicated(keep="last")].sort_index()

rows = []
for t in trades:
    s = px.get(t["tk"])
    if s is None:
        continue
    bd = pd.Timestamp(t["bd"]).normalize()
    if bd not in s.index:
        continue
    pos = s.index.get_loc(bd)
    if isinstance(pos, slice) or pos < 1 or pos >= len(s) - 1:
        continue
    c_bd, o_bd = s["Close"].iloc[pos], s["Open"].iloc[pos]
    c_prev = s["Close"].iloc[pos - 1]
    o_next = s["Open"].iloc[pos + 1]
    if not all(pd.notna(x) and x > 0 for x in (c_bd, o_bd, c_prev, o_next)):
        continue
    rows.append({"day_ret": c_bd / c_prev - 1, "intra_ret": c_bd / o_bd - 1,
                 "overnight": o_next / c_bd - 1})

df = pd.DataFrame(rows)
print(f"n usable buy-trades: {len(df)}", flush=True)

def summ(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return f"n={len(x)}"
    t = x.mean() / (x.std() / math.sqrt(len(x))) if x.std() > 0 else 0.0
    return f"mean_overnight={x.mean()*1e4:+7.1f} bps  t={t:+.2f}  n={len(x):>3d}  win%={100*(x>0).mean():.0f}"

def bucket_report(col, label, edges, names):
    print(f"\n--- conditioned on {label} ({col}) ---", flush=True)
    cats = pd.cut(df[col], bins=edges, labels=names)
    for nm in names:
        sub = df.loc[cats == nm, "overnight"]
        print(f"  {nm:16s} {summ(sub)}", flush=True)
    # correlation: more negative day -> bigger overnight bounce?
    c = df[[col, "overnight"]].corr().iloc[0, 1]
    print(f"  corr({col}, overnight) = {c:+.3f}  (negative => crashed names bounce MORE overnight)", flush=True)

bucket_report("day_ret", "how the stock did ON the 31 (close-to-close)",
              [-1, -0.05, -0.03, -0.01, 0.01, 1], ["crash<-5%", "-5..-3%", "-3..-1%", "flat±1%", "up>+1%"])
bucket_report("intra_ret", "intraday move on the 31 (open-to-close)",
              [-1, -0.05, -0.03, -0.01, 0.01, 1], ["sold-off<-5%", "-5..-3%", "-3..-1%", "flat±1%", "up>+1%"])

# headline: crashed (day_ret < -3%) vs the rest
crashed = df.loc[df["day_ret"] < -0.03, "overnight"]
rest = df.loc[df["day_ret"] >= -0.03, "overnight"]
print(f"\nCRASHED on 31 (day<-3%):  {summ(crashed)}", flush=True)
print(f"NOT crashed (day>=-3%) :  {summ(rest)}", flush=True)
