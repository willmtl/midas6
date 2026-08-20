#!/usr/bin/env python3
"""FLAGSHIP-B beat: do analyst-upside or IV-skew amplify the capitulation->gap-up bounce (both lifted the C
dip-buy)? Join each B(capitulation w15) x mo_gap_up entry to PIT analyst implied-upside (.data/analyst_ratings
.jsonl, last target on/before the entry) and OptionSnapshot iv_skew (as-of), bucket the 3-bar return.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/b_amplify.py"""
import os, json, bisect, warnings, datetime as dt
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS
from intraday_data import get_4h
from core.models import OptionSnapshot

_n, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 8:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


# B(capitulation w15) windows
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

# PIT analyst upside: ticker -> sorted [(date, upside)]
ups = {}
try:
    for line in open("/app/.data/analyst_ratings.jsonl"):
        r = json.loads(line)
        t, d, pt = r.get("ticker"), r.get("date"), r.get("adjusted_price_target") or r.get("price_target")
        if t and d and pt:
            ups.setdefault(t, []).append((d, float(pt)))
    for t in ups:
        ups[t].sort()
except Exception as e:
    print("analyst load err", e, flush=True)

# iv_skew series per ticker
skew = {}
for r in OptionSnapshot.objects.filter(iv_skew__isnull=False).values_list("ticker", "date", "iv_skew"):
    skew.setdefault(r[0], []).append((r[1], float(r[2])))
for t in skew:
    skew[t].sort()


def asof(ser, d):
    ds = [x[0] for x in ser]
    i = bisect.bisect_right(ds, d) - 1
    return ser[i][1] if i >= 0 else None


# collect mo_gap_up entries in B windows with amp features + 3b return
rows = []
for tk, dates in allowed.items():
    df = get_4h(tk, 5, False)
    if df is None or len(df) < 120:
        continue
    entry, mag = H.SIGNALS["mo_gap_up"]["fn"](df)
    c = df["Close"].values; ts = df.index
    for i in range(len(c) - 3):
        if not entry[i] or ts[i].date() not in dates:
            continue
        r3 = (c[i + 3] - c[i]) / c[i] * 100
        d = ts[i].date().isoformat()
        px = c[i]
        up = None
        if tk in ups:
            tgt = asof(ups[tk], d)
            if tgt and px > 0:
                up = tgt / px - 1.0
        sk = asof(skew[tk], ts[i].date()) if tk in skew else None
        rows.append((r3, up, sk))

allr = np.array([r[0] for r in rows], float)
print(f"B x mo_gap_up entries: {len(rows)} | overall 3b avg {allr.mean():+.2f}% win {(allr>0).mean()*100:.0f}% t{_t(allr)}\n", flush=True)

print("== by ANALYST UPSIDE bucket ==", flush=True)
buckets = [("<0", -9, 0), ("0-25%", 0, .25), ("25-50%", .25, .5), ("50-100%", .5, 1.0), (">100%", 1.0, 9)]
cov = [r for r in rows if r[1] is not None]
print(f"  (coverage {len(cov)}/{len(rows)})", flush=True)
for lab, lo, hi in buckets:
    a = np.array([r[0] for r in cov if lo <= r[1] < hi], float)
    if len(a) >= 5:
        print(f"  {lab:9} n{len(a):>4}  {a.mean():>+7.2f}%  win {(a>0).mean()*100:>3.0f}%  t{_t(a)}", flush=True)

print("\n== by IV-SKEW bucket (fear-priced = higher) ==", flush=True)
sc = [r for r in rows if r[2] is not None]
print(f"  (coverage {len(sc)}/{len(rows)})", flush=True)
if sc:
    sv = sorted(r[2] for r in sc)
    qs = [sv[int(len(sv) * q)] for q in (.2, .4, .6, .8)]
    labs = ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
    edges = [-9e9] + qs + [9e9]
    for k in range(5):
        a = np.array([r[0] for r in sc if edges[k] <= r[2] < edges[k + 1]], float)
        if len(a) >= 5:
            print(f"  {labs[k]:9} n{len(a):>4}  {a.mean():>+7.2f}%  win {(a>0).mean()*100:>3.0f}%  t{_t(a)}", flush=True)
