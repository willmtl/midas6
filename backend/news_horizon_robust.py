"""
Robustness gate for the horizon-conditioned news signals — same discipline as news_drift_robust.py
(the bull-pop fade). For each (signed news TYPE, its own horizon) we split the ORIENTED abnormal
drift by TIME-half and by MARKET-CAP to rule out a single-regime / size artifact.

Focus signals:
  contract   @month  (+1.7% candidate UNDER-reaction — buy/hold)
  product    @month  (-3.7% candidate OVER-reaction — fade)
  guidance_up@3mo    (-7.0% candidate OVER-reaction — fade)
  earnings_beat@3mo  (-1.7% fade, high-n reference)

oriented drift = dir * abn (β-adj). + = drifts WITH the news, − = reverses.

Run:  docker compose exec -T backend python -u news_horizon_robust.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import build_universe, load_candles
from core.models import NewsItem, Fundamental

HZ = {"day": 1, "week": 5, "month": 21, "3mo": 63}
BETA_WIN = 60; MIN_PRICE = 3.0
SIGNALS = [("contract", "month"), ("product", "month"), ("guidance_up", "3mo"), ("earnings_beat", "3mo")]

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"]); spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
    mcap = {f["ticker"]: (f["market_cap"] or 0) for f in Fundamental.objects.values("ticker", "market_cap")}
    news_by_tk = defaultdict(list)
    for r in NewsItem.objects.filter(llm_rating__isnull=False, llm_impact__gte=2).values(
            "ticker", "dt", "llm_rating", "llm_impact", "llm_cat"):
        news_by_tk[r["ticker"]].append(r)

    ev = defaultdict(list)     # cat -> list of events
    maxd = max(HZ.values())
    for tk, df in cds.items():
        if tk == "SPY" or df is None or len(df) < BETA_WIN + maxd + 5:
            continue
        items = news_by_tk.get(tk)
        if not items:
            continue
        idx = df.index; close = df["Close"].values; n = len(close)
        mkt = spy["Close"].reindex(idx).ffill().values
        r = df["Close"].pct_change()
        both = pd.concat([r.rename("s"), spy_ret.rename("m")], axis=1).reindex(idx)
        beta = (both["s"].rolling(BETA_WIN).cov(both["m"]) / both["m"].rolling(BETA_WIN).var()).values
        # per (day,cat): dominant item
        day = defaultdict(lambda: defaultdict(lambda: {"net": 0, "imp": 0}))
        for it in items:
            d = pd.Timestamp(it["dt"]).tz_localize(None).normalize()
            pos = int(idx.searchsorted(d))
            if pos <= 0 or pos >= n:
                continue
            a = day[pos][it["llm_cat"]]; a["net"] += (it["llm_rating"] or 0); a["imp"] = max(a["imp"], it["llm_impact"] or 0)
        for d0, cats in day.items():
            rfrom, rto = d0 - 1, d0 + 1
            if rfrom < BETA_WIN or rto + maxd >= n or close[rto] < MIN_PRICE or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            for cat, a in cats.items():
                if a["net"] == 0 or a["imp"] < 2:
                    continue
                dr = int(np.sign(a["net"]))
                fwd = {}
                for name, k in HZ.items():
                    dm = mkt[rto + k] / mkt[rto] - 1.0
                    fwd[name] = ((close[rto + k] / close[rto] - 1.0) - bc * dm) * 100 * dr
                if all(np.isfinite(v) for v in fwd.values()):
                    ev[cat].append(dict(fwd=fwd, date=idx[rto], mcap=mcap.get(tk, 0)))

    all_dates = sorted(e["date"] for lst in ev.values() for e in lst)
    mid = all_dates[len(all_dates) // 2]

    def stat(g, label, hz):
        if len(g) < 15:
            print("    %-20s n=%-4d (thin)" % (label, len(g))); return
        v = np.array([e["fwd"][hz] for e in g])
        print("    %-20s n=%-4d  med=%+5.1f%%  win=%2d%%" % (label, len(g), np.median(v), round((v > 0).mean()*100)))

    for cat, hz in SIGNALS:
        g = ev.get(cat, [])
        if len(g) < 15:
            print(f"\n{cat} @{hz}: n={len(g)} (thin, skip)"); continue
        v = np.array([e["fwd"][hz] for e in g]); direction = "UNDER-react (drift with)" if np.median(v) > 0 else "OVER-react (fade)"
        print(f"\n=== {cat} @{hz}  n={len(g)}  ALL med={np.median(v):+.1f}%/{round((v>0).mean()*100)}%win  [{direction}] ===")
        print("  by TIME half:")
        stat([e for e in g if e["date"] <= mid], "1st half", hz)
        stat([e for e in g if e["date"] >  mid], "2nd half", hz)
        print("  by MCAP:")
        stat([e for e in g if 0 < e["mcap"] < 2e9],     "micro/small <2B", hz)
        stat([e for e in g if 2e9 <= e["mcap"] < 10e9], "mid 2-10B", hz)
        stat([e for e in g if e["mcap"] >= 10e9],       "large >=10B", hz)
    print(f"\ntime split at {mid.date()}  (window {all_dates[0].date()} .. {all_dates[-1].date()})")

if __name__ == "__main__":
    main()
