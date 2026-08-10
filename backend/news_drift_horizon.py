"""
Horizon-conditioned news drift — evaluate each news at DAY / WEEK / MONTH / 3-MONTH, and read the
horizon that matters FOR THAT NEWS TYPE (user: "evaluated on the day, week, month and 3 months
depending on the news"). Uses the existing LLM category on each item — NO re-classification.

Idea: different news digests over different windows. M&A reprices same-day; analyst calls fade in a
week; earnings/guidance drift over 1-3 months. So we measure abnormal oriented drift at all four
horizons and flag each category's PRIMARY horizon, then ask: at its own horizon, does the news
CONTINUE (under-reaction, drift with the news) or REVERSE (over-reaction)?

Oriented drift = dir * abnormal fwd return (β-adj). +ve = moved WITH the news; -ve = reversed.

Run:  docker compose exec -T backend python -u news_drift_horizon.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from seq_fundamental_study import build_universe, load_candles
from core.models import NewsItem

HZ = {"day": 1, "week": 5, "month": 21, "3mo": 63}     # trading-day horizons
BETA_WIN = 60; MIN_PRICE = 3.0

# expected digestion horizon per SIGNED news TYPE (which horizon to READ for that category)
CAT_HORIZON = {
    "ma": "day",             # definitive deal terms reprice immediately
    "upgrade": "week",       # analyst calls move fast then fade
    "downgrade": "week",
    "capital": "week",       # offering/dilution/buyback digests in days
    "dividend": "week",
    "macro": "week",
    "product": "month",      # product/launch traction builds over weeks
    "contract": "month",
    "legal": "month",
    "other": "month",
    "earnings_beat": "3mo",  # PEAD — validated 1-3mo drift
    "earnings_miss": "3mo",
    "guidance_up": "3mo",    # forward guidance reprices over months
    "guidance_down": "3mo",
    "clinical": "3mo",       # biotech catalysts play out over months
    "mgmt": "3mo",           # leadership change = slow burn
}

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"]); spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
    news_by_tk = defaultdict(list)
    for r in NewsItem.objects.filter(llm_rating__isnull=False, llm_impact__gte=1).values(
            "ticker", "dt", "llm_rating", "llm_impact", "llm_cat"):
        news_by_tk[r["ticker"]].append(r)

    events = []
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
        day = defaultdict(lambda: {"net": 0, "maximp": 0, "cat": "other"})
        for it in items:
            d = pd.Timestamp(it["dt"]).tz_localize(None).normalize()
            pos = int(idx.searchsorted(d))
            if pos <= 0 or pos >= n:
                continue
            a = day[pos]; a["net"] += (it["llm_rating"] or 0)
            if (it["llm_impact"] or 0) >= a["maximp"]:      # dominant (highest-impact) item sets the category
                a["maximp"] = it["llm_impact"] or 0; a["cat"] = (it["llm_cat"] or "other")
        for d0, a in day.items():
            if a["net"] == 0:
                continue
            rfrom, rto = d0 - 1, d0 + 1
            if rfrom < BETA_WIN or rto + maxd >= n or close[rto] < MIN_PRICE or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            dr = int(np.sign(a["net"])); b = beta[rfrom]
            bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            fwd = {}; ok = True
            for name, k in HZ.items():
                dm = mkt[rto + k] / mkt[rto] - 1.0
                v = ((close[rto + k] / close[rto] - 1.0) - bc * dm) * 100
                if not np.isfinite(v):
                    ok = False; break
                fwd[name] = v * dr        # ORIENTED
            if ok:
                events.append(dict(imp=int(a["maximp"]), cat=a["cat"], fwd=fwd))
    print("usable classified events (impact>=1):", len(events), flush=True)
    if len(events) < 100:
        print("too few"); return

    def row(g, label, primary):
        cells = []
        for name in HZ:
            v = np.array([e["fwd"][name] for e in g])
            mark = "*" if name == primary else " "
            cells.append("%s%-4s %+5.1f%%/%2d%%" % (mark, name, np.median(v), round((v > 0).mean()*100)))
        print("  %-22s n=%-5d %s" % (label, len(g), " ".join(cells)))

    print("\nORIENTED drift by NEWS TYPE  (+ = continues WITH news, − = reverses).  * = type's own horizon")
    print("  " + " " * 30 + "  ".join("%-11s" % h for h in HZ))
    order = sorted(set(e["cat"] for e in events),
                   key=lambda c: -sum(1 for e in events if e["cat"] == c))
    for cat in order:
        g = [e for e in events if e["cat"] == cat and e["imp"] >= 2]   # ones that MATTER
        if len(g) >= 25:
            row(g, cat, CAT_HORIZON.get(cat, "month"))

    print("\nAT EACH TYPE'S OWN HORIZON (impact>=2) — continuation vs reversal:")
    for cat in order:
        g = [e for e in events if e["cat"] == cat and e["imp"] >= 2]
        if len(g) < 25:
            continue
        h = CAT_HORIZON.get(cat, "month"); v = np.array([e["fwd"][h] for e in g])
        med = np.median(v); verdict = "CONTINUES (under-reaction)" if med > 0.5 else "REVERSES (over-reaction)" if med < -0.5 else "flat"
        print("  %-10s @%-5s n=%-4d  med=%+5.1f%%  -> %s" % (cat, h, len(g), med, verdict))
    print("\ncategory counts (impact>=2):", dict(Counter(e["cat"] for e in events if e["imp"] >= 2).most_common()))

if __name__ == "__main__":
    main()
