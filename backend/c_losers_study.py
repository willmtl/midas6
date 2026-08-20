#!/usr/bin/env python3
"""C WORST PICKS — mirror of c_runners_study. Rank names by ACTUAL portfolio damage (normalized contribution =
Σ ret·w/Σw, the metric we just fixed), not single-month return, and characterize the true losers EX-ANTE. Key
question: the worst DRAWDOWN months split into (a) temporary dips that later ran (SM/HL/VSAT/LRN) and (b)
permanent value traps that stayed dead / went bankrupt (BBBY/SAVE/ACB/SNDL). Is there a feature known AT PICK
that separates them — so we could skip the traps WITHOUT cutting the winners tail? Also: did div4x CONVICTION
ever 4×-amplify a loser? Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/c_losers_study.py"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import load_candles

J = json.load(open("/app/.data/studies/flagship_history.json"))
months = J["months"]

# ---- per-name aggregates: normalized contribution + ex-ante feature medians + conviction damage ----
norm = defaultdict(float); convdrag = defaultdict(float); nconv = defaultdict(int)
feat = defaultdict(lambda: {"roe": [], "pb": [], "pe": [], "rev_g": [], "mc": [], "de": [], "gpa": [],
                            "ni": [], "rev": [], "rets": [], "sectors": set(), "n": 0, "delisted": False,
                            "worst_m": 0.0})
for m in months:
    ps = [p for p in m["picks"] if p.get("ret") is not None]
    wsum = sum(p["weight"] for p in ps) or 1.0
    for p in ps:
        tk = p["ticker"]; c = p["ret"] * p["weight"] / wsum
        norm[tk] += c
        if p["weight"] > 1.0:                       # conviction (4×) hold
            nconv[tk] += 1
            convdrag[tk] += c
        f = feat[tk]; f["n"] += 1; f["sectors"].add(p["sector"]); f["delisted"] |= bool(p["delisted"])
        f["rets"].append(p["ret"]); f["worst_m"] = min(f["worst_m"], p["ret"])
        for k, col in (("roe", "roe"), ("pb", "pb"), ("pe", "pe"), ("rev_g", "rev_g"), ("mktcap_usd", "mc"),
                       ("de", "de"), ("gpa", "gpa"), ("ni", "ni"), ("revenue", "rev")):
            if p.get(k) is not None:
                f[col].append(p[k])

def med(a):
    a = [x for x in a if x is not None and np.isfinite(x)]; return float(np.median(a)) if a else None

# ---- held-to-today (buy first hold forever) to classify recovered vs permanent ----
tickers = sorted(feat.keys())
cands = load_candles(tickers)
first_date = {}
for m in months:
    for p in m["picks"]:
        if p.get("ret") is not None and p["ticker"] not in first_date:
            first_date[p["ticker"]] = m["date"]
def hold_today(tk):
    df = cands.get(tk)
    if df is None or not len(df): return None
    c = df["Close"]; p0 = c.asof(pd.Timestamp(first_date[tk])); p1 = c.iloc[-1]
    return float(p1/p0 - 1) if pd.notna(p0) and pd.notna(p1) and p0 > 0 else None

rows = []
for tk in tickers:
    f = feat[tk]
    rows.append(dict(tk=tk, contrib=norm[tk], n=f["n"], nconv=nconv[tk], convdrag=convdrag[tk],
                     roe=med(f["roe"]), pb=med(f["pb"]), pe=med(f["pe"]), rev_g=med(f["rev_g"]),
                     mc=med(f["mc"]), de=med(f["de"]), gpa=med(f["gpa"]), ni=med(f["ni"]), rev=med(f["rev"]),
                     worst_m=f["worst_m"], delisted=f["delisted"],
                     sector=sorted(f["sectors"])[0] if f["sectors"] else "", ht=hold_today(tk)))
df = pd.DataFrame(rows)

print(f"names={len(df)}  Σcontrib={df.contrib.sum()*100:+.0f}%  (winners {(df.contrib>0).sum()} / losers {(df.contrib<0).sum()})\n")

print("=== (1) WORST 18 PICKS by ACTUAL portfolio damage (normalized contribution) ===")
print(f"  {'tk':6}{'contrib':>9}{'hld':>4}{'cv':>3}{'ROE':>6}{'P/B':>5}{'P/E':>7}{'D/E':>6}{'GP/A':>6}"
      f"{'revg':>6}{'rev$M':>8}{'NI$M':>8}{'mc$B':>6}{'worstM':>7}{'held→tdy':>9}  sector / flag")
w = df.sort_values("contrib").head(18)
for _, r in w.iterrows():
    dl = " †DELISTED" if r.delisted else ""
    print(f"  {r.tk:6}{r.contrib*100:>+8.1f}%{r.n:>4}{r.nconv:>3}{(r.roe or 0)*100:>5.0f}%{r.pb or 0:>5.1f}"
          f"{r.pe if r.pe is not None else 0:>7.1f}{r.de if r.de is not None else 0:>6.2f}"
          f"{r.gpa if r.gpa is not None else 0:>6.2f}{(r.rev_g or 0)*100:>5.0f}%"
          f"{(r.rev or 0)/1e6:>8.0f}{(r.ni or 0)/1e6:>8.0f}{(r.mc or 0)/1e9:>6.1f}"
          f"{r.worst_m*100:>+6.0f}%{(r.ht if r.ht is not None else float('nan'))*100:>+8.0f}%  {r.sector}{dl}")

# ---- (2) classify losers: permanent (held-to-today deeply neg / delisted) vs recovered ----
los = df[df.contrib < 0].copy()
los["permanent"] = (los["ht"] < -0.5) | (los["delisted"] & (los["ht"] < 0))
print(f"\n=== (2) among {len(los)} losing names: PERMANENT value-destroyers vs later-RECOVERED ===")
for lab, sub in (("PERMANENT (held→today < -50% or delisted-negative)", los[los.permanent]),
                 ("RECOVERED / temporary (held→today >= -50%)", los[~los.permanent])):
    if not len(sub): continue
    print(f"  {lab}: n={len(sub)}  total drag {sub.contrib.sum()*100:+.0f}%")
    print(f"     median at pick:  ROE {med(sub.roe)*100 if med(sub.roe) is not None else float('nan'):+.0f}%   "
          f"P/B {med(sub.pb):.2f}   P/E {med([x for x in sub.pe if x is not None]) if any(sub.pe.notna()) else float('nan'):.1f}   "
          f"rev_g {med(sub.rev_g)*100 if med(sub.rev_g) is not None else float('nan'):+.0f}%   "
          f"loss-making(ROE<0) {100*(sub.roe<0).mean():.0f}%")

# ---- (3) does loss-making-at-pick separate permanent losers from the winners? (it's the obvious filter) ----
print("\n=== (3) would a PROFITABILITY filter (skip ROE<0 at pick) help? outcome by pick-time ROE sign ===")
df["roe_sign"] = np.where(df.roe.fillna(0) < 0, "loss-making (ROE<0)", "profitable (ROE>=0)")
for lab, sub in df.groupby("roe_sign"):
    print(f"  {lab:22} n={len(sub):>4}  Σcontrib {sub.contrib.sum()*100:>+6.0f}%  "
          f"mean {sub.contrib.mean()*100:>+5.1f}%  median held→today {med(sub.ht)*100 if med(sub.ht) is not None else float('nan'):>+5.0f}%  "
          f"%permanent-dead(ht<-50) {100*(sub.ht<-0.5).mean():.0f}%")

# ---- (4) conviction amplification: did 4× ever make a loser worse? ----
print("\n=== (4) CONVICTION (4×) on LOSERS — did div4x amplify damage? ===")
cd = df[df.nconv > 0].copy()
print(f"  names ever conviction-weighted: {len(cd)}   their net contrib {cd.contrib.sum()*100:+.0f}%")
print(f"  conviction-weighted DRAG (contrib booked on 4× holds only): {df.convdrag.sum()*100:+.0f}%")
worstconv = cd.sort_values("convdrag").head(8)
print("  worst conviction picks (drag on 4× holds):")
for _, r in worstconv.iterrows():
    print(f"     {r.tk:6} convdrag {r.convdrag*100:+.1f}%  ({r.nconv} conv holds)  ROE {(r.roe or 0)*100:+.0f}%  held→today {(r.ht or 0)*100:+.0f}%  {r.sector}")
