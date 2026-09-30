#!/usr/bin/env python
"""Partial-effect test: do low-ROE / cheap-P/E / thin-vol sort the flagship tail BEYOND market cap,
or is it all just the cap axis? Split picks into mcap terciles, then within each, sort by the candidate.
If the candidate's within-cap trend survives -> independent lever. If it flattens -> just mcap in disguise."""
import json, pandas as pd, numpy as np

d = json.load(open("/app/.data/studies/flagship_history.json"))
rows = [dict(x) for m in d["months"] for x in m.get("picks", []) if x.get("ret") is not None]
t = pd.DataFrame(rows)
t["ret"] = t["ret"] * 100.0
t["mktcap_b"] = t["mktcap_usd"] / 1e9
t["dvol_m"] = t["dvol_usd"] / 1e6
t = t[t.mktcap_b.notna()].copy()
t["capT"] = pd.qcut(t.mktcap_b.rank(method="first"), 3, labels=["small", "mid", "large"])
print(f"{len(t)} picks | overall mean {t.ret.mean():+.2f}%  big(>=20%) {(t.ret>=20).mean()*100:.1f}%\n")

def within(col, invert_label, n=2):
    print(f"### {col}  (does it sort WITHIN each mcap tercile?)")
    for cap in ["small", "mid", "large"]:
        g = t[(t.capT == cap) & t[col].notna()].copy()
        if len(g) < n * 12 or g[col].nunique() < n:
            print(f"  {cap:<6} n={len(g):<4} (too thin)")
            continue
        g["b"] = pd.qcut(g[col].rank(method="first"), n, labels=False)
        lo, hi = g[g.b == 0], g[g.b == n - 1]
        d_mean = hi.ret.mean() - lo.ret.mean(); d_big = (hi.ret >= 20).mean() - (lo.ret >= 20).mean()
        print(f"  {cap:<6} n={len(g):<4} {invert_label}: mean {lo.ret.mean():+6.1f}% -> {hi.ret.mean():+6.1f}%  "
              f"(Δ{d_mean:+5.1f})   big {(lo.ret>=20).mean()*100:4.1f}% -> {(hi.ret>=20).mean()*100:4.1f}% (Δ{d_big*100:+5.1f})")
    print()

# For each candidate: bucket 0 = "cheap/small/junky" end (the hypothesized tail side)
within("roe", "loROE->hiROE")
within("pe",  "loPE->hiPE")
within("dvol_m", "thin->thick")
within("mktcap_b", "smaller->bigger (within-tercile residual)")

# Direct 2x2: small-cap x low-ROE — is the flagship tail concentrated in the small&unprofitable corner?
print("### 2x2: market-cap x ROE (big-month rate, ret>=20%)")
sm = t.mktcap_b <= t.mktcap_b.median()
lo = t.roe <= t.roe.median()
for cap_lbl, cap_m in [("small", sm), ("big", ~sm)]:
    line = f"  {cap_lbl:<6}"
    for roe_lbl, roe_m in [("loROE", lo), ("hiROE", ~lo)]:
        g = t[cap_m & roe_m & t.roe.notna()]
        line += f"  {roe_lbl}: n={len(g):<3} mean{g.ret.mean():+6.1f}% big{(g.ret>=20).mean()*100:5.1f}%"
    print(line)
