"""
News drift study using the LLM RATING as orientation (news_classifier.py output),
replacing EODHD's degenerate sentiment. Only uses classified NewsItem rows.

Per (ticker, trading-day) with classified news:
  net    = sum(llm_rating) that day ; dir = sign(net) ; primary cat = cat of the strongest item.
  reaction R = oriented ABNORMAL return over [d-1 -> d+1] ; forward drift = oriented abn at +10/+21/+42.
  expected E = expanding PIT median R in (cat x mcap x beta) ; residual = R - E.

Two things to check that the EODHD version couldn't:
  (1) DIRECTION: do LLM-bearish days underperform and LLM-bullish outperform (RAW abnormal, not oriented)?
  (2) RESIDUAL: does under-reaction (resid<0) predict more forward drift?

Run:  docker compose exec -T backend python -u news_drift_llm.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from core.models import NewsItem

DRIFTS = [10, 21, 42]
BETA_WIN = 60
MIN_PRICE = 3.0
MIN_PRIOR = 15

def mcap_bucket(m):
    if m is None or not np.isfinite(m): return "unk"
    if m < 500e6:  return "micro"
    if m < 2e9:    return "small"
    if m < 10e9:   return "mid"
    return "large"

def beta_bucket(b):
    if b is None or not np.isfinite(b): return "unk"
    if b < 0.8:  return "lo"
    if b <= 1.2: return "mid"
    return "hi"

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"])
    spy = cds.get("SPY")
    if spy is None or not len(spy):
        print("no SPY"); return
    spy_ret = spy["Close"].pct_change()
    print("loading financial reports ...", flush=True)
    reports = load_financial_reports(tks)

    print("loading CLASSIFIED news ...", flush=True)
    news_by_tk = defaultdict(list)
    for r in NewsItem.objects.filter(llm_rating__isnull=False).values(
            "ticker", "dt", "llm_rating", "llm_impact", "llm_cat"):
        news_by_tk[r["ticker"]].append(r)
    n_class = sum(len(v) for v in news_by_tk.values())
    print("classified items:", n_class, "tickers:", len(news_by_tk), flush=True)
    if n_class < 500:
        print("too few classified — run news_classifier.classify_news first"); return

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
        rep = reports.get(tk)
        if rep is not None and len(rep):
            r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
            pdd = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()) if len(r2) else None
            sh = r2["shares_outstanding"].to_numpy(float) if len(r2) else None
        else:
            pdd, sh = None, None

        day = defaultdict(lambda: {"net": 0, "maximp": 0, "best": (0, "other")})
        for it in items:
            d = pd.Timestamp(it["dt"]).tz_localize(None).normalize()
            pos = int(idx.searchsorted(d))
            if pos <= 0 or pos >= n:
                continue
            rat = it["llm_rating"] or 0
            a = day[pos]
            a["net"] += rat
            a["maximp"] = max(a["maximp"], it["llm_impact"] or 0)
            if abs(rat) >= abs(a["best"][0]):
                a["best"] = (rat, it["llm_cat"] or "other")

        for d0, a in day.items():
            net = a["net"]
            if net == 0:
                continue
            rfrom, rto = d0 - 1, d0 + 1
            if rfrom < BETA_WIN or rto + max(DRIFTS) >= n:
                continue
            if close[rto] < MIN_PRICE or close[rfrom] <= 0 or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            dr = np.sign(net)
            b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            react_mkt = mkt[rto] / mkt[rfrom] - 1.0
            base = (close[rto] / close[rfrom] - 1.0) - bc * react_mkt   # unoriented abnormal reaction
            R = base * dr
            if not np.isfinite(R):
                continue
            m = (sh[max(0, int(pdd.searchsorted(idx[d0], "right")) - 1)] * close[rto]
                 if (pdd is not None and len(pdd)) else None)
            adrift = {}; ok = True
            for k in DRIFTS:
                dm = mkt[rto + k] / mkt[rto] - 1.0
                fd = (close[rto + k] / close[rto] - 1.0) - bc * dm       # RAW abnormal (unoriented)
                if not np.isfinite(fd):
                    ok = False; break
                adrift[k] = fd * 100
            if not ok:
                continue
            events.append(dict(tk=tk, react=idx[rto], R=R, dir=int(dr), imp=int(a["maximp"]),
                               cat=a["best"][1], mb=mcap_bucket(m), bb=beta_bucket(b), adrift=adrift))

    print("usable classified news events:", len(events), flush=True)
    if len(events) < 150:
        print("too few events"); return

    # ---- (1) DIRECTION CHECK: raw abnormal forward return by LLM direction & impact ----
    def rawrow(g, label):
        if len(g) < 30:
            print("  %-26s n=%d (thin)" % (label, len(g))); return
        cells = []
        for k in DRIFTS:
            arr = np.array([e["adrift"][k] for e in g])
            cells.append("+%dd abn=%+5.1f%% up=%2d%%" % (k, np.median(arr), round((arr > 0).mean()*100)))
        print("  %-26s n=%-5d %s" % (label, len(g), " | ".join(cells)))
    print("\n(RAW abnormal forward return — bullish should be >0, bearish <0 if the LLM read has signal)")
    print("\n=== DIRECTION ===")
    rawrow([e for e in events if e["dir"] > 0], "BULLISH")
    rawrow([e for e in events if e["dir"] < 0], "BEARISH")
    print("\n=== by IMPACT (major news should move more) ===")
    for imp in (1, 2, 3):
        rawrow([e for e in events if e["dir"] > 0 and e["imp"] == imp], f"bullish impact={imp}")
    for imp in (1, 2, 3):
        rawrow([e for e in events if e["dir"] < 0 and e["imp"] == imp], f"bearish impact={imp}")

    # ---- (2) RESIDUAL / under-reaction: oriented drift by reaction residual ----
    events.sort(key=lambda x: x["react"])
    running = defaultdict(list)
    for ev in events:
        E = None
        for key in ((ev["cat"], ev["mb"], ev["bb"]), (ev["cat"], ev["mb"]), (ev["cat"],), ("ALL",)):
            h = running[key]
            if len(h) >= MIN_PRIOR:
                E = float(np.median(h)); break
        ev["resid"] = (ev["R"] - E) if E is not None else None
        for key in ((ev["cat"], ev["mb"], ev["bb"]), (ev["cat"], ev["mb"]), (ev["cat"],), ("ALL",)):
            running[key].append(ev["R"])
    scored = [e for e in events if e["resid"] is not None]

    def residrow(sub, label):
        if len(sub) < 60:
            print("  %-20s n=%d (too few)" % (label, len(sub))); return
        resids = np.array([e["resid"] for e in sub]); lo, hi = np.quantile(resids, [1/3, 2/3])
        print("  %s (n=%d)" % (label, len(sub)))
        for nm, g in [("UNDER", [e for e in sub if e["resid"] <= lo]),
                      ("MID",   [e for e in sub if lo < e["resid"] < hi]),
                      ("OVER",  [e for e in sub if e["resid"] >= hi])]:
            if len(g) < 20:
                print("      %-6s n=%d (thin)" % (nm, len(g))); continue
            cells = []
            for k in DRIFTS:
                arr = np.array([e["adrift"][k] * e["dir"] for e in g])   # oriented drift
                cells.append("+%dd med=%+5.1f%% win=%2d%%" % (k, np.median(arr), round((arr > 0).mean()*100)))
            print("      %-6s n=%-5d %s" % (nm, len(g), " | ".join(cells)))
    print("\n(ORIENTED abnormal drift by reaction residual; UNDER = under-reacted = hypothesis: more drift)")
    print("\n=== RESIDUAL: ALL ===");     residrow(scored, "ALL")
    print("\n=== RESIDUAL: BULLISH ==="); residrow([e for e in scored if e["dir"] > 0], "bullish")
    print("\n=== RESIDUAL: BEARISH ==="); residrow([e for e in scored if e["dir"] < 0], "bearish")
    print("\ncat distribution:", dict(Counter(e["cat"] for e in events).most_common()))

if __name__ == "__main__":
    main()
