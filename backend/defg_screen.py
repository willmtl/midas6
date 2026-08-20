#!/usr/bin/env python3
"""NEW SHORT-TERM ALPHA — first-pass screens for D/E/F/G from NON-PRICE information sources (the pivot: stop
mining price patterns). All base-rate-subtracted, PIT (no look-ahead), $5M liquidity floor, tail-bucketed.
  D = OVERNIGHT drift: close->open vs open->close (equity premium accrues overnight?), conditioned.
  E = PEAD: forward drift after an earnings surprise, by surprise sign+magnitude (enter at reaction-day CLOSE).
  F = INSIDER cluster buys: forward drift after clustered insider buying (buy_count/value), enter at filed close.
  G = ETF FLOW: forward ETF return after big creation/redemption (flow_pct), continuation vs reversal.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/defg_screen.py"""
import os, bisect, warnings, datetime as dt
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
from seq_fundamental_study import load_candles
import h4_on_signals_study as S
from core.models import EarningsEvent, InsiderBuy, ETFFlow, Sector

DVOL_FLOOR = 5e6


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def _s(a, base=0.0):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a):
        return "  (none)"
    a = np.clip(a, -50, 50)                       # kill data-artifact moonshots for a clean screen
    return f"n={len(a):>6}  avg {a.mean():+.3f}%  edge {a.mean()-base:+.3f}%  med {np.median(a):+.3f}%  win {(a>0).mean()*100:.0f}%  t={_t(a)}"


# ---- load daily candles once ----
etf_set = set(Sector.objects.values_list("etf", flat=True))
flow_etfs = set(ETFFlow.objects.values_list("ticker", flat=True).distinct())
stocks = S._stock_universe()
print("loading candles...", flush=True)
daily = load_candles(sorted(set(stocks) | flow_etfs | etf_set | {"SPY"}))
print(f"loaded {len(daily)} names\n", flush=True)

# per-ticker precompute
PX = {}     # tk -> dict(close, open, dates[list date], dvol, annv, fwd{h})
FWD_H = [1, 3, 5, 10, 20, 40]
base_pool = {h: [] for h in FWD_H}
for tk, df in daily.items():
    if len(df) < 60:
        continue
    c = df["Close"].values.astype(float); o = df["Open"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(c)
    dvol = (df["Close"] * df["Volume"]).rolling(20).mean().values     # proper trailing 20d $volume (real floor)
    annv = (df["Close"].pct_change().rolling(20).std() * (252 ** 0.5)).values
    liq = dvol >= DVOL_FLOOR
    fwd = {}
    for h in FWD_H:
        f = np.full(n, np.nan); f[:n - h] = (c[h:] - c[:n - h]) / c[:n - h] * 100
        f = np.clip(f, -50, 50)                                       # data-artifact guard
        fwd[h] = f
        m = np.isfinite(f) & liq                                      # base rate on LIQUID bars only (match signals)
        base_pool[h].extend(f[m])
    PX[tk] = {"c": c, "o": o, "dates": [d.date() for d in df.index], "dvol": dvol, "annv": annv, "fwd": fwd}
base = {h: float(np.mean(base_pool[h])) for h in FWD_H}
basemed = {h: float(np.median(base_pool[h])) for h in FWD_H}
print("BASE RATE mean   (liquid): " + "  ".join(f"{h}d {base[h]:+.3f}" for h in FWD_H), flush=True)
print("BASE RATE median (liquid): " + "  ".join(f"{h}d {basemed[h]:+.3f}" for h in FWD_H) + "\n", flush=True)


def bar_at(tk, d, on_or_after=True):
    p = PX.get(tk)
    if not p:
        return None
    i = bisect.bisect_left(p["dates"], d)
    return i if (on_or_after and 0 <= i < len(p["dates"])) else None


# ================= D — OVERNIGHT vs INTRADAY =================
print("=" * 70, "\n== D — OVERNIGHT (close->open) vs INTRADAY (open->close)", flush=True)
on_all, in_all, on_hiv, in_hiv, on_afterup, on_afterdn = [], [], [], [], [], []
for tk, p in PX.items():
    c, o, dvol, annv = p["c"], p["o"], p["dvol"], p["annv"]
    n = len(c)
    for i in range(1, n):
        if dvol[i] < DVOL_FLOOR or o[i] <= 0 or c[i - 1] <= 0:
            continue
        onr = (o[i] - c[i - 1]) / c[i - 1] * 100      # overnight
        inr = (c[i] - o[i]) / o[i] * 100              # intraday
        on_all.append(onr); in_all.append(inr)
        if np.isfinite(annv[i]) and annv[i] >= 0.50:
            on_hiv.append(onr); in_hiv.append(inr)
        prev_in = (c[i - 1] - o[i - 1]) / o[i - 1] if o[i - 1] > 0 else 0
        (on_afterup if prev_in > 0 else on_afterdn).append(onr)
print(f"  overnight  ALL         {_s(on_all)}", flush=True)
print(f"  intraday   ALL         {_s(in_all)}", flush=True)
print(f"  overnight  high-vol    {_s(on_hiv)}", flush=True)
print(f"  intraday   high-vol    {_s(in_hiv)}", flush=True)
print(f"  overnight  after up-day {_s(on_afterup)}", flush=True)
print(f"  overnight  after dn-day {_s(on_afterdn)}", flush=True)


# ================= E — PEAD =================
print("\n" + "=" * 70, "\n== E — PEAD (forward drift from reaction-day close, by surprise)", flush=True)
ev = {}
for tk, rd, su, gs, ba in EarningsEvent.objects.values_list("ticker", "report_date", "eps_surprise_pct", "grounded_score", "before_after"):
    ev.setdefault(tk, []).append((rd, su, gs, ba))
pools = {("pos", h): [] for h in (5, 10, 20)}
pools.update({("neg", h): [] for h in (5, 10, 20)})
mag = {q: {h: [] for h in (5, 10, 20)} for q in range(5)}
sur_all = []
for tk, evs in ev.items():
    p = PX.get(tk)
    if not p:
        continue
    for rd, su, gs, ba in evs:
        if su is None:
            continue
        i = bar_at(tk, rd)
        if i is None:
            continue
        i = i + 1 if (ba == "AfterMarket") else i     # AMC reacts next session
        if i >= len(p["c"]) or p["dvol"][i] < DVOL_FLOOR:
            continue
        sur_all.append(su)
        side = "pos" if su > 0 else "neg"
        for h in (5, 10, 20):
            r = p["fwd"][h][i]
            if np.isfinite(r):
                pools[(side, h)].append(r)
qs = np.nanpercentile(np.abs(sur_all), [20, 40, 60, 80]) if sur_all else [0, 0, 0, 0]
for tk, evs in ev.items():
    p = PX.get(tk)
    if not p:
        continue
    for rd, su, gs, ba in evs:
        if su is None or su <= 0:
            continue                                   # positive-surprise magnitude tail
        i = bar_at(tk, rd)
        if i is None:
            continue
        i = i + 1 if (ba == "AfterMarket") else i
        if i >= len(p["c"]) or p["dvol"][i] < DVOL_FLOOR:
            continue
        q = int(np.searchsorted(qs, su, side="right"))
        for h in (5, 10, 20):
            r = p["fwd"][h][i]
            if np.isfinite(r):
                mag[min(q, 4)][h].append(r)
for h in (5, 10, 20):
    print(f"  POS surprise  @{h}d  {_s(pools[('pos', h)], base[h])}", flush=True)
    print(f"  NEG surprise  @{h}d  {_s(pools[('neg', h)], base[h])}", flush=True)
print("  positive-surprise magnitude tail @10d (Q0 small -> Q4 biggest beat):", flush=True)
for q in range(5):
    print(f"     Q{q}  {_s(mag[q][10], base[10])}", flush=True)


# ================= F — INSIDER CLUSTER BUYS =================
print("\n" + "=" * 70, "\n== F — INSIDER cluster buys (forward drift from filed-day close)", flush=True)
ins = {}
for tk, fd, bv, sv, bc in InsiderBuy.objects.values_list("ticker", "filed_date", "buy_value", "sell_value", "buy_count"):
    ins.setdefault(tk, []).append((fd, bv or 0, sv or 0, bc or 0))
net_all = {h: [] for h in (10, 20, 40)}
clus = {h: [] for h in (10, 20, 40)}       # buy_count>=3 AND net buyer
big = {h: [] for h in (10, 20, 40)}        # top-quartile buy_value
bvs = [x[1] for v in ins.values() for x in v if x[1] > 0]
bv_q75 = np.nanpercentile(bvs, 75) if bvs else 1e18
for tk, evs in ins.items():
    p = PX.get(tk)
    if not p:
        continue
    for fd, bv, sv, bc in evs:
        i = bar_at(tk, fd)
        if i is None or i >= len(p["c"]) or p["dvol"][i] < DVOL_FLOOR:
            continue
        net_buy = bv > sv
        for h in (10, 20, 40):
            r = p["fwd"][h][i]
            if not np.isfinite(r):
                continue
            if net_buy:
                net_all[h].append(r)
            if bc >= 3 and net_buy:
                clus[h].append(r)
            if bv >= bv_q75:
                big[h].append(r)
for h in (10, 20, 40):
    print(f"  net-buyer      @{h}d  {_s(net_all[h], base[h])}", flush=True)
    print(f"  cluster(>=3)   @{h}d  {_s(clus[h], base[h])}", flush=True)
    print(f"  big-$ (top-Q)  @{h}d  {_s(big[h], base[h])}", flush=True)


# ================= G — ETF FLOW =================
print("\n" + "=" * 70, "\n== G — ETF FLOW (forward ETF return by flow_pct quintile)", flush=True)
fl = {}
for tk, d, fp in ETFFlow.objects.values_list("ticker", "date", "flow_pct"):
    if fp is not None:
        fl.setdefault(tk, []).append((d, fp))
recs = []
for tk, rows in fl.items():
    p = PX.get(tk)
    if not p:
        continue
    for d, fp in rows:
        i = bar_at(tk, d)
        if i is None or i >= len(p["c"]):
            continue
        recs.append((fp, p["fwd"][3][i], p["fwd"][5][i], p["fwd"][10][i]))
if recs:
    arr = np.array(recs, float)
    fq = np.nanpercentile(arr[:, 0], [20, 40, 60, 80])
    qi = np.searchsorted(fq, arr[:, 0], side="right")
    lab = ["Q1 outflow", "Q2", "Q3", "Q4", "Q5 inflow"]
    for hcol, h in ((1, 3), (2, 5), (3, 10)):
        print(f"  -- fwd {h}d ETF return by flow_pct --", flush=True)
        for q in range(5):
            print(f"     {lab[q]:12} {_s(arr[qi == q, hcol], base[h])}", flush=True)
