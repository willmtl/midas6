#!/usr/bin/env python3
"""ISOLATED analyst price-target-gap signal test — variable isolation, none of the flagship machinery.

Strips away P/B, tl_rsi entry, div4x conviction, sector rotation. Measures ONE thing: does the gap between
price and analyst target predict next-month returns, cross-sectionally? Each month, rank every analyst-
covered US stock by implied upside = median(trailing-180d target)/close - 1, bucket into quintiles, and
look at the NEXT month's return. Q1 = smallest gap (price at/above target), Q5 = biggest gap (furthest
BELOW target). Reported BOTH raw and market-DEMEANED (subtract that month's universe mean) so we isolate
the signal from market beta. Tail check (tail-not-average): also the top/bottom DECILE and the >50%-gap
extreme. Long/short Q5-Q1 with a Welch t-stat.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/analyst_gap_study.py"""
import os, json, datetime as dt
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector

etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
tgts = defaultdict(list)
for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    tk, pt, d = r.get("ticker"), r.get("price_target"), r.get("date")
    if tk and pt and d and "." not in tk and tk not in etfs:   # US-listed only (target/price ccy must match)
        tgts[tk].append((pd.Timestamp(d).value, float(pt)))
uni = sorted(tgts)
print(f"analyst-covered US non-ETF universe: {len(uni)}", flush=True)

# month-end close panel
rows = list(Candle.objects.filter(ticker__in=uni, date__gte="2015-12-01").values_list("ticker", "date", "close"))
df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
df = df[(df["close"] > 1.0)]                                    # drop sub-$1 penny noise (light hygiene only)
mclose = df.pivot_table(index="date", columns="ticker", values="close").resample("ME").last()
midx = mclose.index
print(f"months {midx[0].date()}..{midx[-1].date()}  tickers {mclose.shape[1]}", flush=True)

# implied-upside panel: median target within trailing 180d, / month-end close - 1
midx_i = np.array([t.value for t in midx], dtype="int64")
stale = 180 * 86400 * 10**9
ups = pd.DataFrame(np.nan, index=midx, columns=mclose.columns)
for tk in mclose.columns:
    pts = tgts.get(tk)
    if not pts:
        continue
    arr = np.array(sorted(pts)); di, tv = arr[:, 0], arr[:, 1]
    col = np.full(len(midx), np.nan)
    for j, d in enumerate(midx_i):
        a = np.searchsorted(di, d - stale, side="right"); b = np.searchsorted(di, d, side="right")
        if b > a:
            col[j] = np.median(tv[a:b])
    ups[tk] = col / mclose[tk].values - 1.0

fwd = mclose.shift(-1) / mclose - 1.0                           # next-month return

# cross-sectional quintile/decile buckets, raw + market-demeaned
QN = 5
qraw = [[] for _ in range(QN)]; qdem = [[] for _ in range(QN)]
dec_lo, dec_hi = [], []                                        # bottom/top decile (demeaned)
big_gap, big_gap_dem = [], []                                  # >50% implied upside extreme (demeaned)
mo_q1, mo_q5, mo_ls = [], [], []                               # PER-MONTH demeaned means (honest t-stat basis)
nmonths = 0
for d in midx[:-1]:
    u = ups.loc[d].dropna()
    f = fwd.loc[d]
    valid = [t for t in u.index if pd.notna(f.get(t)) and np.isfinite(f[t])]
    if len(valid) < 25:
        continue
    nmonths += 1
    u = u[valid].sort_values()                                 # ascending: Q1=smallest gap ... Q5=biggest
    fv = f[valid]
    mkt = fv.mean()
    order = list(u.index)
    _qmeans = []
    for qi, members in enumerate(np.array_split(np.array(order), QN)):
        r = fv[list(members)].values
        qraw[qi].extend(r); qdem[qi].extend(r - mkt)
        _qmeans.append((r - mkt).mean())
    mo_q1.append(_qmeans[0]); mo_q5.append(_qmeans[-1]); mo_ls.append(_qmeans[-1] - _qmeans[0])
    deciles = np.array_split(np.array(order), 10)
    dec_lo.extend((fv[list(deciles[0])] - mkt).values); dec_hi.extend((fv[list(deciles[-1])] - mkt).values)
    ext = [t for t in order if u[t] > 0.50]                    # >50% implied upside
    if ext:
        big_gap.extend(fv[ext].values); big_gap_dem.extend((fv[ext] - mkt).values)


def stat(a):
    a = np.array(a, float); a = a[np.isfinite(a)]
    if len(a) < 2:
        return None
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
    return dict(n=len(a), mean=a.mean() * 100, med=np.median(a) * 100, win=(a > 0).mean() * 100, t=t)


print(f"\nmonths used: {nmonths}   (Q1 = price AT/ABOVE target, Q5 = price FURTHEST BELOW target)\n")
print(f"{'bucket':16}{'N':>7}{'mean%':>9}{'median%':>9}{'win%':>7}{'t(vs0)':>9}   (RAW / then market-DEMEANED)")
for qi in range(QN):
    sr, sd = stat(qraw[qi]), stat(qdem[qi])
    print(f"  Q{qi+1} {'(most below)' if qi==QN-1 else '(at/above)' if qi==0 else '':11}"
          f"{sr['n']:>7}{sr['mean']:>9.2f}{sr['med']:>9.2f}{sr['win']:>7.1f}{sr['t']:>9.2f}", flush=True)
    print(f"       demeaned {'':4}{sd['n']:>7}{sd['mean']:>9.2f}{sd['med']:>9.2f}{sd['win']:>7.1f}{sd['t']:>9.2f}", flush=True)

print(f"tail deciles (demeaned):  bottom-10% {stat(dec_lo)['mean']:+.2f}%  vs  top-10% {stat(dec_hi)['mean']:+.2f}%")
sg = stat(big_gap_dem)
print(f">50% implied-upside extreme (demeaned): n={sg['n']}  mean {sg['mean']:+.2f}%  win {sg['win']:.1f}%")

# HONEST significance: t-stat on the PER-MONTH demeaned means (N=months, not pooled stock-months)
def mt(x):
    x = np.array(x, float); x = x[np.isfinite(x)]
    return x.mean() * 100, x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)
m1, t1, _ = mt(mo_q1); m5, t5, _ = mt(mo_q5); mls, tls, nm = mt(mo_ls)
print(f"\n=== HONEST monthly-panel t-stats (N={nm} months, not pooled stock-months) ===")
print(f"  Q1 (at/above target) demeaned:  {m1:+.2f}%/mo   t = {t1:+.2f}")
print(f"  Q5 (furthest below)  demeaned:  {m5:+.2f}%/mo   t = {t5:+.2f}")
print(f"  Q5 - Q1 long/short:             {mls:+.2f}%/mo   t = {tls:+.2f}   (>2 = significant; autocorr not adjusted)")

# ── REDUNDANCY CHECK: where do the FLAGSHIP's actual picks fall on the implied-upside distribution? ──
try:
    J = json.load(open("/app/.data/studies/flagship_history.json"))
    pct = []
    for m in J["months"]:
        d = pd.Timestamp(m["date"])
        if d not in ups.index:
            # snap to nearest month-end index
            cand = ups.index[ups.index <= d]
            if len(cand) == 0: continue
            d = cand[-1]
        row = ups.loc[d].dropna()
        if len(row) < 25: continue
        for p in m["picks"]:
            tk = p["ticker"]
            if tk in row.index:
                pct.append((row < row[tk]).mean() * 100)   # percentile of the pick's upside within covered universe
    pct = np.array(pct)
    print(f"\n=== REDUNDANCY CHECK: flagship picks' implied-upside percentile (0=at/above target, 100=furthest below) ===")
    print(f"  picks with an upside value: {len(pct)}   MEAN percentile {pct.mean():.0f}   median {np.median(pct):.0f}")
    print(f"  share in top quintile (Q5, most-below-target): {(pct>=80).mean()*100:.0f}%   in bottom quintile (Q1): {(pct<20).mean()*100:.0f}%")
except Exception as e:
    print(f"redundancy check skipped: {e}")
