#!/usr/bin/env python
"""Which entry-time indicators sort the tail like Ulcer does?
For each candidate metric available AT ENTRY, bucket all signals into quintiles and
report round-trip mean/median, win%, and big-trade rate (rt>=50%). A monotonic climb =
the indicator is a real selector; flat = it doesn't separate winners from losers."""
import pandas as pd, numpy as np

t = pd.read_csv("/app/.data/studies/sortino_dd_entries.csv")
t = t[t.rt_ret.notna()].copy()
print(f"signals with a completed round-trip: {len(t)}  ({t.ticker.nunique()} names)\n")

def buckets(df, col, n=5, invert=False):
    d = df[df[col].notna()].copy()
    if len(d) < n * 20:
        return None
    try:
        d["q"] = pd.qcut(d[col].rank(method="first"), n, labels=False)
    except Exception:
        return None
    rows = []
    for q in range(n):
        g = d[d.q == q]
        rows.append((q, len(g), g[col].mean(), g.rt_ret.mean(), g.rt_ret.median(),
                     (g.rt_ret > 0).mean() * 100, (g.rt_ret >= 50).mean() * 100))
    return rows

CANDS = [
    ("ulcer",         "Ulcer index (baseline)"),
    ("dd_score",      "DD downside score"),
    ("mcap_b",        "Market cap ($B)"),
    ("srsi",          "Sortino-RSI value at entry"),
    ("srsi_spread",   "Sortino-RSI minus its EMA"),
    ("nrsi",          "OHLC4-RSI value at entry"),
    ("nrsi_spread",   "OHLC4-RSI minus its EMA"),
    ("bars_after_cross", "Bars since Sortino-RSI cross"),
    ("dist_hi",       "Distance below 52w high (%)"),
    ("mom",           "Trailing momentum (own return)"),
    ("dvol_m",        "Dollar volume ($M/day)"),
]

def monotonic(vals):
    # Spearman-ish: correlation of quintile index vs metric (mean rt & big-rate)
    q = np.arange(len(vals))
    rt = np.array([v[3] for v in vals]); big = np.array([v[6] for v in vals])
    def corr(a):
        if a.std() == 0: return 0.0
        return np.corrcoef(q, a)[0, 1]
    return corr(rt), corr(big)

print(f"{'':<34}{'n':>6}{'metric':>10}{'mean%':>8}{'med%':>7}{'win%':>7}{'big%':>7}")
print("-" * 82)
scored = []
for col, label in CANDS:
    b = buckets(t, col)
    if b is None:
        print(f"{label:<34}  (insufficient / non-numeric)")
        continue
    c_rt, c_big = monotonic(b)
    print(f"{label}   [rt-trend r={c_rt:+.2f}  big-trend r={c_big:+.2f}]")
    for q, n, mv, mean, med, win, big in b:
        bar = "#" * int(max(0, big) / 1.5)
        print(f"   Q{q+1:<31}{n:>6}{mv:>10.2f}{mean:>+8.1f}{med:>+7.1f}{win:>7.0f}{big:>7.1f}  {bar}")
    scored.append((label, c_rt, c_big))
    print()

print("=" * 82)
print("RANKED by how strongly the top quintile concentrates BIG trades (rt>=50%):")
for label, c_rt, c_big in sorted(scored, key=lambda x: -x[2]):
    tag = "STRONG selector" if c_big > 0.8 else ("weak" if c_big > 0.4 else "no effect")
    print(f"  big-trend r={c_big:+.2f}  rt-trend r={c_rt:+.2f}  {tag:<16} {label}")
