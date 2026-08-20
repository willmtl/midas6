#!/usr/bin/env python3
"""AUDIT the flagship DOC/enrich layer for display-metric bugs of the ENIA class (metric misrepresents the
correct underlying backtest). The engine P&L (basket_ret = Σ(ret·w)/Σw) is trusted; we check whether the
DERIVED/DISPLAYED aggregates faithfully represent it. Read-only. Run:
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/audit_flagship_doc.py"""
import os, json, numpy as np
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from pathlib import Path
from collections import defaultdict

J = json.load(open("/app/.data/studies/flagship_history.json"))
months = J["months"]
perf = J["perf"]
print(f"config={J.get('config')}  months={len(months)}  engine perf.total={perf.get('total')}%  dd={perf.get('dd')}%\n")

# --- key presence ---
mk = months[len(months)//2]
print("A) month keys present:", sorted(mk.keys()))
has_ts = all("top_sectors" in m for m in months)
print("   every month has top_sectors:", has_ts, " basket_ret:", all("basket_ret" in m for m in months))

# --- distinct weights (should be {1.0, 4.0} for size_mode=conv / CONV=4.0) ---
ws = defaultdict(int)
for m in months:
    for p in m["picks"]:
        if p.get("ret") is not None:
            ws[round(p["weight"], 3)] += 1
print("\nB) distinct pick weights -> count:", dict(sorted(ws.items())))

# --- weight vs conviction-flag consistency ---
mism = []
for m in months:
    for p in m["picks"]:
        if p.get("ret") is None:
            continue
        conv_by_weight = p["weight"] > 1.0
        if conv_by_weight != bool(p["conviction"]):
            mism.append((m["date"], p["ticker"], p["weight"], p["conviction"]))
print(f"\nC) weight>1 vs conviction-flag MISMATCHES: {len(mism)}")
for x in mism[:12]:
    print("   ", x)

# --- verify basket_ret == Σ(ret·w)/Σw for each month (engine normalization) ---
maxerr = 0.0; worst = None
for m in months:
    ps = [p for p in m["picks"] if p.get("ret") is not None]
    wsum = sum(p["weight"] for p in ps)
    if wsum <= 0:
        continue
    recomputed = sum(p["ret"] * p["weight"] for p in ps) / wsum
    err = abs(recomputed - m["basket_ret"])
    if err > maxerr:
        maxerr = err; worst = (m["date"], round(recomputed, 5), round(m["basket_ret"], 5))
print(f"\nD) basket_ret vs Σ(ret·w)/Σw  max abs err = {maxerr:.2e}  worst {worst}")
print("   (near-0 = engine monthly return is correctly weight-NORMALIZED; the doc curve/total are sound)")

# --- enrich-derived compounded total vs engine perf.total ---
eqf = 1.0
for m in months:
    eqf *= (1 + m["basket_ret"])
print(f"\nE) compounded Π(1+basket_ret)-1 = {(eqf-1)*100:,.0f}%   vs engine perf.total {perf.get('total')}%")

# --- CONTRIB: raw (buggy) vs normalized (true attribution) ---
raw = defaultdict(float); norm = defaultdict(float); nheld = defaultdict(int)
for m in months:
    ps = [p for p in m["picks"] if p.get("ret") is not None]
    wsum = sum(p["weight"] for p in ps) or 1.0
    for p in ps:
        raw[p["ticker"]] += p["ret"] * p["weight"]                 # current doc formula (un-normalized)
        norm[p["ticker"]] += p["ret"] * p["weight"] / wsum         # true share-of-book contribution
        nheld[p["ticker"]] += 1
sum_norm = sum(norm.values()); sum_basket = sum(m["basket_ret"] for m in months)
print(f"\nF) Σ normalized-contrib over all names = {sum_norm*100:+.1f}%   vs Σ monthly basket_ret = {sum_basket*100:+.1f}%  (should match)")

def top(d, k=12):
    return sorted(d.items(), key=lambda x: -x[1])[:k]
print("\n   TOP-12 by RAW contrib (current leaderboard sort):")
for t, v in top(raw):
    print(f"     {t:6} raw {v*100:+7.1f}%   norm {norm[t]*100:+6.1f}%   held {nheld[t]}")
print("\n   TOP-12 by NORMALIZED contrib (true attribution):")
for t, v in top(norm):
    print(f"     {t:6} norm {v*100:+6.1f}%   raw {raw[t]*100:+7.1f}%   held {nheld[t]}")

# rank churn: how many of the raw top-20 are NOT in the normalized top-20
rt = {t for t, _ in top(raw, 20)}; nt = {t for t, _ in top(norm, 20)}
print(f"\n   leaderboard churn: {len(rt-nt)}/20 names in RAW top-20 fall out of NORMALIZED top-20 -> {sorted(rt-nt)}")

# ENIA specifics
print(f"\nG) ENIA: raw contrib {raw.get('ENIA',0)*100:+.1f}%   normalized {norm.get('ENIA',0)*100:+.1f}%   held {nheld.get('ENIA',0)}")
