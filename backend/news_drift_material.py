"""
Material-news drift — the CORRECTED design.

'Matters' is judged EX-ANTE by the LLM impact rating (known at publication, no lookahead) —
NOT by the same-day move (that would pre-select already-priced news and discard the very
under-reaction cases we're hunting). The same-day reaction is used only as a FEATURE.

Horizons: 7 / 30 / 90 days (≈ 5 / 21 / 63 trading days) — 90d is where "room to digest" shows up.

Analyses:
  (A) DIRECTION x IMPACT: raw abnormal drift by LLM direction and impact (does high-impact news drift,
      and does bullish vs bearish separate?).
  (B) UNDER-REACTION within impact>=2: split by day-1 reaction magnitude — MUTED (barely moved) vs
      STRONG (already moved) — and measure ORIENTED drift. Hypothesis: high-impact + MUTED day-1 ->
      forward drift in the news direction (room left); STRONG -> reverts.

Run:  docker compose exec -T backend python -u news_drift_material.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from core.models import NewsItem

DRIFTS = [5, 21, 63]        # ≈ 7 / 30 / 90 calendar days
BETA_WIN = 60; MIN_PRICE = 3.0
MUTED = 3.0; STRONG = 6.0   # |day-1 abnormal reaction| % thresholds

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"]); spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
    news_by_tk = defaultdict(list)
    for r in NewsItem.objects.filter(llm_rating__isnull=False).values("ticker", "dt", "llm_rating", "llm_impact"):
        news_by_tk[r["ticker"]].append(r)
    print("classified items:", sum(len(v) for v in news_by_tk.values()), "tickers:", len(news_by_tk), flush=True)

    events = []
    for tk, df in cds.items():
        if tk == "SPY" or df is None or len(df) < BETA_WIN + max(DRIFTS) + 5:
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
            a = day[pos]; a["net"] += (it["llm_rating"] or 0); a["maximp"] = max(a["maximp"], it["llm_impact"] or 0)
        for d0, a in day.items():
            if a["net"] == 0:
                continue
            rfrom, rto = d0 - 1, d0 + 1
            if rfrom < BETA_WIN or rto + max(DRIFTS) >= n or close[rto] < MIN_PRICE or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            dr = int(np.sign(a["net"]))
            b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            react = ((close[rto] / close[rfrom] - 1.0) - bc * (mkt[rto] / mkt[rfrom] - 1.0)) * 100  # signed abn day-1
            fwd = {}; ok = True
            for k in DRIFTS:
                dm = mkt[rto + k] / mkt[rto] - 1.0
                v = ((close[rto + k] / close[rto] - 1.0) - bc * dm) * 100
                if not np.isfinite(v):
                    ok = False; break
                fwd[k] = v
            if not ok or not np.isfinite(react):
                continue
            events.append(dict(dir=dr, imp=int(a["maximp"]), react_abs=abs(react), fwd=fwd))
    print("usable classified events:", len(events), flush=True)
    if len(events) < 100:
        print("too few"); return

    def raw(g, label):   # raw (unoriented) abnormal drift
        if len(g) < 25:
            print("  %-28s n=%d (thin)" % (label, len(g))); return
        cells = [ "+%dd abn=%+5.1f%% up=%2d%%" % (k, np.median([e["fwd"][k] for e in g]),
                  round((np.array([e["fwd"][k] for e in g]) > 0).mean()*100)) for k in DRIFTS ]
        print("  %-28s n=%-5d %s" % (label, len(g), " | ".join(cells)))
    def orient(g, label):  # oriented drift = dir * abn
        if len(g) < 25:
            print("  %-28s n=%d (thin)" % (label, len(g))); return
        cells = [ "+%dd med=%+5.1f%% win=%2d%%" % (k, np.median([e["fwd"][k]*e["dir"] for e in g]),
                  round((np.array([e["fwd"][k]*e["dir"] for e in g]) > 0).mean()*100)) for k in DRIFTS ]
        print("  %-28s n=%-5d %s" % (label, len(g), " | ".join(cells)))

    print("\n(A) DIRECTION x IMPACT — raw abnormal drift (7/30/90d)")
    for imp in (1, 2, 3):
        raw([e for e in events if e["dir"] > 0 and e["imp"] == imp], f"BULLISH impact={imp}")
    for imp in (1, 2, 3):
        raw([e for e in events if e["dir"] < 0 and e["imp"] == imp], f"BEARISH impact={imp}")

    hi = [e for e in events if e["imp"] >= 2]
    print(f"\n(B) UNDER-REACTION within impact>=2 (n={len(hi)}) — ORIENTED drift by day-1 reaction size")
    print("    hypothesis: MUTED day-1 (barely moved) -> drifts in news direction over 30/90d")
    for nm, g in [("MUTED (|day1|<%.0f%%)" % MUTED,  [e for e in hi if e["react_abs"] < MUTED]),
                  ("MID",                             [e for e in hi if MUTED <= e["react_abs"] < STRONG]),
                  ("STRONG (|day1|>=%.0f%%)" % STRONG,[e for e in hi if e["react_abs"] >= STRONG])]:
        orient(g, nm)
    print("\n    same split, BULLISH only:")
    for nm, g in [("bull MUTED", [e for e in hi if e["dir"] > 0 and e["react_abs"] < MUTED]),
                  ("bull STRONG",[e for e in hi if e["dir"] > 0 and e["react_abs"] >= STRONG])]:
        orient(g, nm)
    print("    same split, BEARISH only:")
    for nm, g in [("bear MUTED", [e for e in hi if e["dir"] < 0 and e["react_abs"] < MUTED]),
                  ("bear STRONG",[e for e in hi if e["dir"] < 0 and e["react_abs"] >= STRONG])]:
        orient(g, nm)
    print("\nimpact distribution:", dict(sorted(Counter(e["imp"] for e in events).items())))

if __name__ == "__main__":
    main()
