"""
Phase 2b — NEWS under-/over-reaction (drift) study, by news TYPE.

Same mechanic as the earnings drift study, over news events instead of earnings:
  - event   = a (ticker, trading-day) with news; signed sentiment ss = mean(pos - neg) that day
              (the stored `polarity` field is junk-skewed ~+1, so we use pos-neg), oriented dir=sign(ss).
  - category = mapped from EODHD tags (EARNINGS / GUIDANCE / ANALYST / M&A / DIV-BUYBACK / LEGAL / PRODUCT / OTHER).
  - reaction R = oriented ABNORMAL return over [d-1 -> d+1] (2-day window, robust to intraday timing).
  - expected E = expanding POINT-IN-TIME median oriented reaction in (category x mcap x beta) bucket.
  - residual   = R - E ; UNDER (<0) = under-reacted -> hypothesis: forward drift in sentiment direction.
  - forward drift = oriented abnormal return at +10/+21/+42 trading days.

CAVEAT: news history is ~13 months (recent single regime) -> this is EXPLORATORY / lower-powered
than the 5y earnings study. Horizons kept short to preserve sample.

Run:  docker compose exec -T backend python -u news_drift_study.py
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
MIN_ABS_SS = 0.05          # min |pos-neg| daily sentiment to count as a directional event
MIN_PRIOR = 20

CAT_RULES = [   # (category, set of tag-substrings) — first match wins, in priority order
    ("EARNINGS",   ("EARNINGS", "EPS", "QUARTERLY RESULT")),
    ("M&A",        ("MERGER", "ACQUISITION", "TAKEOVER", "BUYOUT", "DEAL")),
    ("GUIDANCE",   ("GUIDANCE", "OUTLOOK", "FORECAST")),
    ("ANALYST",    ("PRICE-TARGET", "PRICE TARGET", "ANALYST", "RATING", "CONSENSUS", "UPGRADE", "DOWNGRADE")),
    ("DIV-BUYBACK",("DIVIDEND", "BUYBACK", "SHARE-BUYBACK", "SHAREHOLDER")),
    ("LEGAL",      ("LAWSUIT", "LEGAL", "INVESTIGATION", "REGULAT", "SEC FILING", "FRAUD")),
    ("PRODUCT",    ("PRODUCT", "LAUNCH", "PARTNERSHIP", "CONTRACT")),
]

def categorize(tags):
    up = [str(t).upper() for t in (tags or [])]
    for cat, subs in CAT_RULES:
        for t in up:
            if any(sub in t for sub in subs):
                return cat
    return "OTHER"

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

    # news grouped by ticker -> list of (date, pos, neg, tags)
    print("loading news ...", flush=True)
    news_by_tk = defaultdict(list)
    for r in NewsItem.objects.filter(pos__isnull=False, neg__isnull=False).values(
            "ticker", "dt", "pos", "neg", "tags"):
        news_by_tk[r["ticker"]].append(r)
    print("tickers with news:", len(news_by_tk), flush=True)

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

        # aggregate news to (trading-day index) -> mean ss, tag counter
        day_agg = defaultdict(lambda: {"ss": [], "tags": Counter()})
        for it in items:
            d = pd.Timestamp(it["dt"]).tz_localize(None).normalize()
            pos = int(idx.searchsorted(d))
            if pos <= 0 or pos >= n:
                continue
            ss = (it["pos"] or 0) - (it["neg"] or 0)
            day_agg[pos]["ss"].append(ss)
            day_agg[pos]["tags"].update(str(t).upper() for t in (it["tags"] or []))

        for d0, agg in day_agg.items():
            ss = float(np.mean(agg["ss"]))
            if abs(ss) < MIN_ABS_SS:
                continue
            rfrom, rmid, rto = d0 - 1, d0, d0 + 1
            if rfrom < BETA_WIN or rto + max(DRIFTS) >= n:
                continue
            if close[rto] < MIN_PRICE or close[rfrom] <= 0 or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            dr = np.sign(ss)
            b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            react_mkt = mkt[rto] / mkt[rfrom] - 1.0
            R = ((close[rto] / close[rfrom] - 1.0) - bc * react_mkt) * dr
            if not np.isfinite(R):
                continue
            m = (sh[max(0, int(pdd.searchsorted(idx[d0], "right")) - 1)] * close[rto]
                 if (pdd is not None and len(pdd)) else None)
            cat = categorize(agg["tags"])
            drift = {}; ok = True
            for k in DRIFTS:
                dm = mkt[rto + k] / mkt[rto] - 1.0
                fd = ((close[rto + k] / close[rto] - 1.0) - bc * dm) * dr
                if not np.isfinite(fd):
                    ok = False; break
                drift[k] = fd * 100
            if not ok:
                continue
            events.append(dict(tk=tk, react=idx[rto], R=R, dir=int(dr), cat=cat,
                               mb=mcap_bucket(m), bb=beta_bucket(b), drift=drift))

    print("usable news events:", len(events), flush=True)
    if len(events) < 200:
        print("too few"); return

    # expanding PIT peer baseline of oriented reaction R
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
    print("events with a PIT baseline:", len(scored), flush=True)

    def report(sub, label):
        if len(sub) < 60:
            print("  %-26s n=%d (too few)" % (label, len(sub))); return
        resids = np.array([e["resid"] for e in sub]); lo, hi = np.quantile(resids, [1/3, 2/3])
        print("  %s (n=%d)" % (label, len(sub)))
        for nm, g in [("UNDER", [e for e in sub if e["resid"] <= lo]),
                      ("MID",   [e for e in sub if lo < e["resid"] < hi]),
                      ("OVER",  [e for e in sub if e["resid"] >= hi])]:
            if len(g) < 20:
                print("      %-6s n=%d (thin)" % (nm, len(g))); continue
            cells = []
            for k in DRIFTS:
                a = np.array([e["drift"][k] for e in g])
                cells.append("+%dd med=%+5.1f%% win=%2d%%" % (k, np.median(a), round((a > 0).mean()*100)))
            print("      %-6s n=%-5d %s" % (nm, len(g), " | ".join(cells)))

    print("\n(oriented abnormal drift by reaction residual; UNDER = under-reacted = hypothesis: more drift)")
    print("\n=== ALL news events ===")
    report(scored, "ALL")
    print("\n=== POSITIVE news (ss>0) ===")
    report([e for e in scored if e["dir"] > 0], "positive")
    print("\n=== NEGATIVE news (ss<0) ===")
    report([e for e in scored if e["dir"] < 0], "negative")
    print("\n=== by news CATEGORY ===")
    cats = Counter(e["cat"] for e in scored)
    for cat, cnt in cats.most_common():
        if cnt >= 120:
            report([e for e in scored if e["cat"] == cat], "cat=" + cat)

if __name__ == "__main__":
    main()
