"""
LOCAL-LABEL re-validation of the strong-bullish-news FADE edge — the cutover gate.

Identical methodology to news_drift_robust.py (the Haiku-label version) but reads the LOCAL qwen
labels (local_rating / local_impact) instead of llm_rating / llm_impact. Purpose: before repointing
the live consumers (playbook fade flag, news-horizon scanner) from Haiku to local, confirm the SAME
edge is present in local labels at the (much larger) local sample.

Baseline to beat (Haiku, news_drift_robust.py): BULL imp>=2 STRONG fades ~-5% to -6% @63d and is
SIZE-conditioned — mid/small negative, large >=10B ~flat (coin-flip). We want the local labels to
reproduce: STRONG negative, both time halves same sign, mid/small << large.

Run:  docker compose exec -T backend python -u news_drift_robust_local.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import build_universe, load_candles
from core.models import NewsItem, Fundamental

DRIFT = 63; BETA_WIN = 60; MIN_PRICE = 3.0; STRONG = 6.0

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"]); spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
    mcap = {f["ticker"]: (f["market_cap"] or 0) for f in Fundamental.objects.values("ticker", "market_cap")}
    news_by_tk = defaultdict(list)
    for r in NewsItem.objects.filter(local_rating__isnull=False).values("ticker", "dt", "local_rating", "local_impact"):
        news_by_tk[r["ticker"]].append(r)

    ev = []
    for tk, df in cds.items():
        if tk == "SPY" or df is None or len(df) < BETA_WIN + DRIFT + 5:
            continue
        items = news_by_tk.get(tk)
        if not items:
            continue
        idx = df.index; close = df["Close"].values; n = len(close)
        mkt = spy["Close"].reindex(idx).ffill().values
        r = df["Close"].pct_change()
        both = pd.concat([r.rename("s"), spy_ret.rename("m")], axis=1).reindex(idx)
        beta = (both["s"].rolling(BETA_WIN).cov(both["m"]) / both["m"].rolling(BETA_WIN).var()).values
        day = defaultdict(lambda: {"net": 0, "maximp": 0})
        for it in items:
            d = pd.Timestamp(it["dt"]).tz_localize(None).normalize()
            pos = int(idx.searchsorted(d))
            if pos <= 0 or pos >= n:
                continue
            a = day[pos]; a["net"] += (it["local_rating"] or 0); a["maximp"] = max(a["maximp"], it["local_impact"] or 0)
        for d0, a in day.items():
            if a["net"] <= 0 or a["maximp"] < 2:      # BULLISH, impact>=2 only
                continue
            rfrom, rto = d0 - 1, d0 + 1
            if rfrom < BETA_WIN or rto + DRIFT >= n or close[rto] < MIN_PRICE or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            react = ((close[rto] / close[rfrom] - 1.0) - bc * (mkt[rto] / mkt[rfrom] - 1.0)) * 100
            dm = mkt[rto + DRIFT] / mkt[rto] - 1.0
            fwd = ((close[rto + DRIFT] / close[rto] - 1.0) - bc * dm) * 100
            if not (np.isfinite(react) and np.isfinite(fwd)):
                continue
            ev.append(dict(imp=int(a["maximp"]), react_abs=abs(react),
                           orient=fwd, date=idx[rto], mcap=mcap.get(tk, 0)))
    print("bullish impact>=2 events (LOCAL labels):", len(ev), flush=True)
    if len(ev) < 100:
        print("too few — let the moved-first classification pass fill more first"); return

    dates = sorted(e["date"] for e in ev); mid = dates[len(dates) // 2]

    def stat(g, label):
        if len(g) < 20:
            print("  %-26s n=%-4d (thin)" % (label, len(g))); return
        v = np.array([e["orient"] for e in g])
        print("  %-26s n=%-4d  +63d med=%+5.1f%%  win=%2d%%  mean=%+5.1f%%"
              % (label, len(g), np.median(v), round((v > 0).mean() * 100), v.mean()))

    def report(name, sub):
        print(f"\n=== {name}  (n={len(sub)}) — oriented drift, NEGATIVE = fade/reversal ===")
        stat(sub, "ALL")
        print(" by TIME half:")
        stat([e for e in sub if e["date"] <= mid],  "  1st half")
        stat([e for e in sub if e["date"] >  mid],  "  2nd half")
        print(" by MCAP:")
        stat([e for e in sub if 0 < e["mcap"] < 2e9],        "  micro/small <2B")
        stat([e for e in sub if 2e9 <= e["mcap"] < 10e9],    "  mid 2-10B")
        stat([e for e in sub if e["mcap"] >= 10e9],          "  large >=10B")

    report("BULL imp>=2 (monotonic fade)", ev)
    report("BULL imp>=2 STRONG (|day1|>=6%)", [e for e in ev if e["react_abs"] >= STRONG])
    report("BULL imp==3 (really matters)", [e for e in ev if e["imp"] == 3])
    print(f"\ntime split at {mid.date()}  (window {dates[0].date()} .. {dates[-1].date()})")

if __name__ == "__main__":
    main()
