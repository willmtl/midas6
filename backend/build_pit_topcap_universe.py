#!/usr/bin/env python3
"""Build a POINT-IN-TIME top-market-cap universe (user: pick by PAST mktcap, not present) to de-survivorship the
SMC study. PIT mktcap = as-traded close x PIT shares (FinancialReport.shares_outstanding ffilled by avail_date).
At each month take the top-N; the UNION over 2016-2026 = every name that was EVER a top-N mega-cap (incl. fallen
angels that shrank). Reports union size, split into still-listed (fetchable from TV) vs delisted (NOT fetchable),
and the fallen-angels missing from a current-top-120 list. Writes the fetchable union to /app/.data/tv_symbols_pit.txt.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/build_pit_topcap_universe.py [N]"""
import os, sys, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
import price_basis
from core.models import Candle, Sector, DelistedCompany, FinancialReport, Fundamental

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
covered = set()
for line in open("/app/.data/analyst_ratings.jsonl", encoding="utf-8"):
    if line.strip():
        try:
            tk = json.loads(line).get("ticker")
            if tk and "." not in tk and tk not in etfs:
                covered.add(tk)
        except Exception:
            pass
delisted = set(DelistedCompany.objects.exclude(delisted_date=None).values_list("ticker", flat=True))
universe = sorted((covered | delisted) - etfs)
print(f"broad universe {len(universe)} ({len(delisted)} delisted)", flush=True)

from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")

spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2015-06-01").values_list("date", "close")), columns=["date", "close"])
spy["date"] = pd.to_datetime(spy["date"]); spy_m = spy.set_index("date")["close"].astype(float).resample("ME").last()
midx = spy_m.index[spy_m.index >= "2016-01-01"]
midx_ts = np.array([d.value for d in midx], dtype="int64")

close_m = {}
for i in range(0, len(universe), 200):
    q = Candle.objects.filter(ticker__in=universe[i:i + 200], interval="1d", date__gte="2015-06-01").values_list("ticker", "date", "close")
    df = pd.DataFrame(list(q), columns=["ticker", "date", "close"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        close_m[tk] = g.sort_values("date").set_index("date")["close"].resample("ME").last().reindex(midx)
close_m = pd.DataFrame(close_m)
cols = list(close_m.columns)
print(f"names with monthly closes {len(cols)}", flush=True)

frs = defaultdict(list)
for tk, ad, sh in FinancialReport.objects.filter(ticker__in=cols, shares_outstanding__gt=0).values_list("ticker", "avail_date", "shares_outstanding"):
    if ad and sh:
        frs[tk].append((pd.Timestamp(ad).value, float(sh)))
shares_m = {}
for tk in cols:
    pts = sorted(frs.get(tk, []))
    if not pts:
        continue
    di = np.array([t for t, _ in pts]); sv = np.array([v for _, v in pts]); o = np.full(len(midx), np.nan)
    for j, dd in enumerate(midx_ts):
        b = np.searchsorted(di, dd, side="right")
        if b > 0:
            o[j] = sv[b - 1]
    shares_m[tk] = pd.Series(o, index=midx)
shares_m = pd.DataFrame(shares_m)
mcap_pit = price_basis.as_traded_close(close_m[shares_m.columns]) * shares_m
print(f"names with PIT mktcap {mcap_pit.shape[1]}", flush=True)

# union of monthly top-N  (+ save the per-month membership so the SMC backtests can gate PIT, no look-ahead)
union = set()
membership_months = defaultdict(int)
membership = {}
for d in midx:
    row = mcap_pit.loc[d].dropna()
    if len(row):
        top = list(row.nlargest(N).index)
        membership[str(d.date())] = top
        for tk in top:
            union.add(tk); membership_months[tk] += 1
union = sorted(union)
open(f"/app/.data/pit_top{N}_membership.json", "w").write(json.dumps(membership))
print(f"saved monthly PIT-top-{N} membership -> /app/.data/pit_top{N}_membership.json", flush=True)

recent = set(Candle.objects.filter(interval="1d", date__gte="2026-09-01").values_list("ticker", flat=True).distinct())
listed = [t for t in union if t in recent]
gone = [t for t in union if t not in recent]
# current top-N (present mktcap) for the fallen-angel diff
curmc = {r["ticker"]: r["market_cap"] for r in Fundamental.objects.filter(ticker__in=recent, market_cap__gt=0).exclude(ticker__in=etfs).values("ticker", "market_cap") if "." not in r["ticker"]}
cur_top = set(sorted(curmc, key=lambda t: -curmc[t])[:120])
fallen = [t for t in listed if t not in cur_top]

print(f"\n=== PIT top-{N} UNION 2016-2026 ===", flush=True)
print(f"union size: {len(union)}", flush=True)
print(f"  still listed (fetchable from TV): {len(listed)}", flush=True)
print(f"  delisted / gone (NOT fetchable):  {len(gone)}", flush=True)
print(f"  fallen angels (listed, were top-{N} but NOT in current top-120): {len(fallen)}", flush=True)
print(f"\nFALLEN ANGELS we'd be MISSING with a current-top-120 list ({len(fallen)}):", flush=True)
for i in range(0, len(fallen), 8):
    print("  " + "  ".join(fallen[i:i + 8]), flush=True)
print(f"\nDELISTED former top-{N} (can't get SMC from TV; the true survivorship hole) ({len(gone)}):", flush=True)
for i in range(0, len(gone), 8):
    print("  " + "  ".join(gone[i:i + 8]), flush=True)

open("/app/.data/tv_symbols_pit.txt", "w").write("\n".join(listed) + "\n")
print(f"\nwrote {len(listed)} fetchable PIT-top-{N} tickers -> /app/.data/tv_symbols_pit.txt", flush=True)
