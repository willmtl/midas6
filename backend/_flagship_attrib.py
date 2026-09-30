#!/usr/bin/env python
"""Same indicator-attribution as the Sortino scanner, on the DEPLOYED flagship trace.
Flagship = monthly value-rotation, so each pick's 'ret' is the ONE-MONTH forward return.
For each entry-time factor, bucket picks into quartiles -> mean/median monthly ret, win%,
and big-month rate (ret>=20%). Monotonic climb = the factor sorts return."""
import json, pandas as pd, numpy as np

d = json.load(open("/app/.data/studies/flagship_history.json"))
rows = [dict(x) for m in d["months"] for x in m.get("picks", []) if x.get("ret") is not None]
t = pd.DataFrame(rows)
t["ret"] = t["ret"] * 100.0                       # -> percent
t["mktcap_b"] = t["mktcap_usd"] / 1e9
t["dvol_m"] = t["dvol_usd"] / 1e6
print(f"flagship deployed: total {d['perf'].get('total')}%  |  {len(t)} realized picks, "
      f"{t.ticker.nunique()} names, {len(d['months'])} months")
print(f"monthly ret: mean {t.ret.mean():+.2f}%  median {t.ret.median():+.2f}%  "
      f"win {(t.ret>0).mean()*100:.0f}%  big(>=20%) {(t.ret>=20).mean()*100:.1f}%\n")

def buckets(df, col, n=4):
    dd = df[df[col].notna()].copy()
    if len(dd) < n * 15 or dd[col].nunique() < n:
        return None
    try:
        dd["q"] = pd.qcut(dd[col].rank(method="first"), n, labels=False)
    except Exception:
        return None
    out = []
    for q in range(n):
        g = dd[dd.q == q]
        out.append((q, len(g), g[col].mean(), g.ret.mean(), g.ret.median(),
                    (g.ret > 0).mean() * 100, (g.ret >= 20).mean() * 100))
    return out

def trend(vals):
    q = np.arange(len(vals)); rt = np.array([v[3] for v in vals]); big = np.array([v[6] for v in vals])
    c = lambda a: 0.0 if a.std() == 0 else np.corrcoef(q, a)[0, 1]
    return c(rt), c(big)

CANDS = [
    ("pb",        "P/B (the deployed selector)"),
    ("pe",        "P/E ttm"),
    ("roe",       "ROE ttm"),
    ("de",        "Debt / equity"),
    ("gpa",       "Gross profits / assets"),
    ("rev_g",     "Revenue growth"),
    ("mktcap_b",  "Market cap ($B)"),
    ("dvol_m",    "Dollar volume ($M/day)"),
    ("weight",    "Conviction weight (A/D div sizing)"),
]

print(f"{'':<36}{'n':>5}{'metric':>10}{'mean%':>8}{'med%':>7}{'win%':>6}{'big%':>6}")
print("-" * 80)
scored = []
for col, label in CANDS:
    b = buckets(t, col)
    if b is None:
        print(f"{label:<36}  (insufficient / non-numeric)")
        continue
    c_rt, c_big = trend(b)
    print(f"{label}   [ret-trend r={c_rt:+.2f}  big-trend r={c_big:+.2f}]")
    for q, n, mv, mean, med, win, big in b:
        bar = "#" * int(max(0, big) / 2)
        print(f"   Q{q+1:<33}{n:>5}{mv:>10.2f}{mean:>+8.1f}{med:>+7.1f}{win:>6.0f}{big:>6.1f}  {bar}")
    scored.append((label, c_rt, c_big, abs(c_big)))
    print()

# conviction as a binary split
if "conviction" in t.columns:
    for val, lab in [(True, "conviction=TRUE (A/D divergence)"), (False, "conviction=FALSE")]:
        g = t[t.conviction == val]
        if len(g):
            print(f"{lab:<36}{len(g):>5}{'':>10}{g.ret.mean():>+8.1f}{g.ret.median():>+7.1f}"
                  f"{(g.ret>0).mean()*100:>6.0f}{(g.ret>=20).mean()*100:>6.1f}")
print()
print("=" * 80)
print("RANKED by |effect| on big months (ret>=20%):")
for label, c_rt, c_big, a in sorted(scored, key=lambda x: -x[3]):
    tag = "STRONG" if a > 0.8 else ("moderate" if a > 0.5 else "weak/none")
    dirn = "high end" if c_big > 0 else "low end"
    print(f"  |big|={a:.2f}  (tail at {dirn:>8})  ret-trend r={c_rt:+.2f}  {tag:<10} {label}")
