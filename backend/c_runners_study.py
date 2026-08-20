#!/usr/bin/env python3
"""C 'ONES THAT GOT AWAY' — characterize the picks that ran AFTER we rotated out, using features known AT EXIT.
Blanket hold-longer is REFUTED (churn is the edge); this asks the DIFFERENT question: of the names C sold, which
ones became big post-exit runners, and what do they share EX-ANTE (at the sell month) so we could selectively
HOLD or RE-ENTER just those. Guards against hindsight bias by only comparing exit-time-observable features.

Pipeline: load flagship_history.json (adaptive=C) -> reconstruct maximal consecutive-month HOLD EPISODES per
ticker -> for each episode record exit-month ex-ante features + in-hold return -> from daily candles compute
trailing-6mo momentum AT exit and forward returns AFTER exit (+3/6/12mo) + max 12mo run-up ('if we kept holding')
-> (1) size the problem (distribution + concentration of missed upside), (2) split runners vs rest and compare
exit-time features, (3) bucket forward return by exit-momentum quintile (key hypothesis: winners were still
rising at exit = we sold for SECTOR reasons, not stock weakness). Run:
MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/c_runners_study.py"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from seq_fundamental_study import load_candles

TD3, TD6, TD12 = 63, 126, 252   # trading-day forward windows
J = json.loads(Path("/app/.data/studies/flagship_history.json").read_text())
months = J["months"]
mdates = [m["date"] for m in months]                       # ordered month-end strings
midx = {d: i for i, d in enumerate(mdates)}

# ---- reconstruct hold episodes (maximal consecutive-month runs per ticker) ----
# held[ticker] = ordered list of (month_index, pick_record) for months the name was a real forward pick
held = {}
for i, m in enumerate(months):
    for p in m["picks"]:
        if p.get("ticker") and p.get("ret") is not None:
            held.setdefault(p["ticker"], []).append((i, p))

episodes = []   # dict per episode
for tk, seq in held.items():
    seq.sort(key=lambda x: x[0])
    run = [seq[0]]
    for prev, cur in zip(seq, seq[1:]):
        if cur[0] == prev[0] + 1:
            run.append(cur)
        else:
            episodes.append((tk, run)); run = [cur]
    episodes.append((tk, run))

tickers = sorted({tk for tk, _ in episodes})
print(f"episodes={len(episodes)} across {len(tickers)} tickers; months {mdates[0]}..{mdates[-1]}", flush=True)

daily = load_candles(tickers)
close = {t: d["Close"].sort_index() for t, d in daily.items() if d is not None and len(d)}

def px_at(tk, dstr):
    s = close.get(tk)
    if s is None or not len(s):
        return None, None
    ts = pd.Timestamp(dstr)
    s2 = s[s.index <= ts]
    if not len(s2):
        return None, None
    return float(s2.iloc[-1]), s.index.get_loc(s2.index[-1])

def fwd(tk, pos, nd):
    s = close.get(tk)
    if s is None or pos is None or pos + 1 >= len(s):
        return None
    j = min(pos + nd, len(s) - 1)                          # cap at last available (delisted -> last price)
    p0 = float(s.iloc[pos]); p1 = float(s.iloc[j])
    return (p1 / p0 - 1) if p0 > 0 else None

def runup(tk, pos, nd):
    s = close.get(tk)
    if s is None or pos is None or pos + 1 >= len(s):
        return None
    win = s.iloc[pos + 1: min(pos + 1 + nd, len(s))]
    p0 = float(s.iloc[pos])
    return (float(win.max()) / p0 - 1) if len(win) and p0 > 0 else None

rows = []
for tk, run in episodes:
    exit_i, exit_p = run[-1]
    exit_date = mdates[exit_i]
    if exit_i >= len(mdates) - 2:                          # too close to data end for forward measurement
        continue
    p_exit, pos = px_at(tk, exit_date)
    if pos is None:
        continue
    inhold = float(np.prod([1 + r[1]["ret"] for r in run]) - 1)     # compounded in-hold return
    mom6 = fwd(tk, pos - TD6, TD6) if pos - TD6 >= 0 else None      # trailing 6mo momentum AT exit
    rows.append(dict(
        ticker=tk, sector=exit_p.get("sector"), exit=exit_date, nmo=len(run),
        pb=exit_p.get("pb"), roe=exit_p.get("roe"), rev_g=exit_p.get("rev_g"),
        de=exit_p.get("de"), mktcap=exit_p.get("mktcap_usd"), conv=bool(exit_p.get("conviction")),
        inhold=inhold, mom6_exit=mom6,
        f3=fwd(tk, pos, TD3), f6=fwd(tk, pos, TD6), f12=fwd(tk, pos, TD12),
        run12=runup(tk, pos, TD12)))
df = pd.DataFrame(rows)
print(f"measurable episodes (with forward data): {len(df)}\n", flush=True)

def pct(a, q):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return np.percentile(a, q) if len(a) else float("nan")

# ---- (1) size the problem ----
print("=== (1) POST-EXIT forward returns of C's picks (did we sell too early?) ===", flush=True)
for col, lab in (("f6", "+6mo post-exit"), ("f12", "+12mo post-exit"), ("run12", "MAX 12mo run-up ('if held')")):
    a = df[col].dropna()
    print(f"  {lab:30} n={len(a):>4}  median {a.median()*100:+6.1f}%  mean {a.mean()*100:+6.1f}%  "
          f"p90 {pct(a,90)*100:+6.0f}%  %>50% {(a>0.5).mean()*100:4.0f}  %>100% {(a>1.0).mean()*100:4.0f}", flush=True)

# concentration of missed upside (positive post-exit f12 only)
g = df.dropna(subset=["f12"]).copy(); g["gain"] = g["f12"].clip(lower=0)
tot = g["gain"].sum()
gs = g.sort_values("gain", ascending=False)
for k in (5, 10, 20):
    print(f"  top {k:>2} episodes = {gs['gain'].head(k).sum()/tot*100:4.0f}% of all positive post-exit upside", flush=True)
print("  biggest 'ones that got away' (post-exit +12mo):", flush=True)
for _, r in gs.head(12).iterrows():
    print(f"     {r['ticker']:6} {r['sector']:18} exit {r['exit']}  f12 {r['f12']*100:+6.0f}%  "
          f"run12 {r['run12']*100:+6.0f}%  mom6@exit {(r['mom6_exit'] or 0)*100:+5.0f}%  inhold {r['inhold']*100:+5.0f}%  conv={r['conv']}", flush=True)

# ---- (2) runners vs rest, features known AT EXIT ----
thr = df["f12"].dropna().quantile(0.80)
df["runner"] = df["f12"] >= thr
print(f"\n=== (2) RUNNER (top-quintile post-exit +12mo, >= {thr*100:.0f}%) vs REST — EXIT-TIME features ===", flush=True)
print(f"  {'feature':16}{'RUNNERS':>14}{'REST':>14}", flush=True)
for col in ("mom6_exit", "inhold", "pb", "roe", "rev_g", "de", "mktcap"):
    rmed = df.loc[df.runner, col].median(); omed = df.loc[~df.runner & df.f12.notna(), col].median()
    sc = 1e-9 if col == "mktcap" else 1
    unit = "$B" if col == "mktcap" else ("%" if col in ("mom6_exit", "inhold", "roe", "rev_g") else "")
    m = 100 if col in ("mom6_exit", "inhold", "roe", "rev_g") else sc
    print(f"  {col:16}{rmed*m:>12.2f}{unit:>2}{omed*m:>12.2f}{unit:>2}", flush=True)
print(f"  conviction rate  runners {df.loc[df.runner,'conv'].mean()*100:.0f}%   rest {df.loc[~df.runner & df.f12.notna(),'conv'].mean()*100:.0f}%", flush=True)

# ---- (3) the key hypothesis: still-rising-at-exit -> post-exit run? bucket f12 by exit momentum quintile ----
print("\n=== (3) POST-EXIT +12mo by TRAILING-6mo-MOMENTUM-AT-EXIT quintile (were runners still rising when sold?) ===", flush=True)
d3 = df.dropna(subset=["mom6_exit", "f12"]).copy()
d3["q"] = pd.qcut(d3["mom6_exit"], 5, labels=["Q1 weakest", "Q2", "Q3", "Q4", "Q5 strongest"])
for q, sub in d3.groupby("q"):
    print(f"  {q:14} n={len(sub):>4}  mom6@exit med {sub['mom6_exit'].median()*100:+6.0f}%  "
          f"-> post-exit f12 median {sub['f12'].median()*100:+6.1f}%  mean {sub['f12'].mean()*100:+6.1f}%  "
          f"runner-rate {sub['runner'].mean()*100:4.0f}%", flush=True)

# also split by in-hold return sign (did we sell winners or losers?)
print("\n  by IN-HOLD return (did the stock rise or fall while we held it?):", flush=True)
for lab, mask in (("in-hold UP (>0)", df.inhold > 0), ("in-hold DOWN (<=0)", df.inhold <= 0)):
    sub = df[mask].dropna(subset=["f12"])
    print(f"     {lab:18} n={len(sub):>4}  post-exit f12 median {sub['f12'].median()*100:+6.1f}%  runner-rate {sub['runner'].mean()*100:4.0f}%", flush=True)

Path("/app/.data/studies/c_runners.json").write_text(df.to_json(orient="records"))
print("\nwrote /app/.data/studies/c_runners.json", flush=True)
