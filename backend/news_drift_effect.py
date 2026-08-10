"""
Same-day-EFFECT-conditioned news drift.

The point of the day_effect split (compute_news_effect.py): only ~1 news item in ~7 actually moves
the stock on its reaction session; the rest is noise that drags every average toward zero. Here we
ask the question the split was built for (user: "analyze only the news that had an effect on the
day"): once the stock HAS reacted day-1, does it CONTINUE (drift, under-reaction — tradeable
momentum) or GIVE IT BACK (reversal, over-reaction — the fade edge)?

For each material (impact>=2) news bar we:
  - read the pre-computed same-day move  day_abn  and  day_effect,
  - orient forward abnormal returns by the day-1 move direction  dir = sign(day_abn):
        oriented_fwd(k) = dir * [ (stock t->t+k) - β·(spy t->t+k) ] * 100    (from the reaction close t)
    +ve = keeps moving the way it popped (continuation);  -ve = reverses (fade).
  - horizons day/week/month/3mo (1/5/21/63 bars).

We print EFFECT vs NO-EFFECT side by side (the control), then within EFFECT split UP-moves vs
DOWN-moves and by news TYPE at its own horizon. If the day_effect filter is doing its job, the
signal in the EFFECT column should be much sharper than the noise-diluted NO-EFFECT column.

Run:  docker compose exec -T backend python -u news_drift_effect.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from zoneinfo import ZoneInfo
from seq_fundamental_study import build_universe, load_candles
from core.models import NewsItem, Fundamental

HZ = {"day": 1, "week": 5, "month": 21, "3mo": 63}
BETA_WIN = 60; MIN_PRICE = 3.0
ET = ZoneInfo("America/New_York")
CAT_HORIZON = {
    "ma": "day", "upgrade": "week", "downgrade": "week", "capital": "week", "dividend": "week",
    "macro": "week", "product": "month", "contract": "month", "legal": "month", "other": "month",
    "earnings_beat": "3mo", "earnings_miss": "3mo", "guidance_up": "3mo", "guidance_down": "3mo",
    "clinical": "3mo", "mgmt": "3mo",
}


def reaction_pos(dt, idx, n):
    """Index of the session that first trades on the news (>=16:00 ET -> next day), snapped forward."""
    et = pd.Timestamp(dt).tz_convert(ET) if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt).tz_localize("UTC").tz_convert(ET)
    d = et.normalize().tz_localize(None)
    if et.hour >= 16:
        d = d + pd.Timedelta(days=1)
    pos = int(idx.searchsorted(d, side="left"))
    return pos if 0 < pos < n else -1


def cap_bucket(mc):
    if not mc or mc <= 0:
        return ""
    return "small" if mc < 2e9 else "mid" if mc < 10e9 else "large"


def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"]); spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
    mcap = {f["ticker"]: (f["market_cap"] or 0) for f in Fundamental.objects.values("ticker", "market_cap")}

    news_by_tk = defaultdict(list)
    for r in (NewsItem.objects.filter(llm_rating__isnull=False, llm_impact__gte=2, day_abn__isnull=False)
              .exclude(day_suspect=True)      # drop bad-candle / illiquid-OTC artifacts from BOTH groups
              .values("ticker", "dt", "llm_rating", "llm_impact", "llm_cat", "day_abn", "day_effect")):
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
        mc = mcap.get(tk, 0); cb = cap_bucket(mc)
        # collapse to the dominant (highest-impact) item per reaction session
        day = {}
        for it in items:
            t = reaction_pos(it["dt"], idx, n)
            if t < 0:
                continue
            a = day.setdefault(t, {"net": 0.0, "maximp": 0, "cat": "other", "abn": it["day_abn"], "eff": bool(it["day_effect"])})
            a["net"] += (it["llm_rating"] or 0)
            if (it["llm_impact"] or 0) >= a["maximp"]:
                a["maximp"] = it["llm_impact"] or 0; a["cat"] = (it["llm_cat"] or "other")
                a["abn"] = it["day_abn"]; a["eff"] = bool(it["day_effect"])
        for t, a in day.items():
            if a["net"] == 0 or a["abn"] is None:
                continue
            if t < BETA_WIN or t + maxd >= n or close[t] < MIN_PRICE or mkt[t] <= 0:
                continue
            dr = 1 if a["abn"] > 0 else -1          # orient by the DAY-1 MOVE, not the LLM sign
            bc = min(max(beta[t - 1] if np.isfinite(beta[t - 1]) else 1.0, 0.0), 3.0)
            fwd = {}; ok = True
            for name, k in HZ.items():
                dm = mkt[t + k] / mkt[t] - 1.0
                v = ((close[t + k] / close[t] - 1.0) - bc * dm) * 100
                if not np.isfinite(v):
                    ok = False; break
                fwd[name] = v * dr
            if ok:
                events.append(dict(imp=int(a["maximp"]), cat=a["cat"], eff=a["eff"],
                                   abn=float(a["abn"]), cb=cb, fwd=fwd))
    print("usable material events (impact>=2, day_abn computed):", len(events), flush=True)
    eff_n = sum(e["eff"] for e in events)
    print(f"  had same-day EFFECT: {eff_n} ({100*eff_n/max(len(events),1):.1f}%)  |  no-effect: {len(events)-eff_n}", flush=True)
    if len(events) < 100:
        print("too few"); return

    def row(g, label, primary=None):
        cells = []
        for name in HZ:
            v = np.array([e["fwd"][name] for e in g])
            mark = "*" if name == primary else " "
            cells.append("%s%-4s %+5.1f%%/%2d%%" % (mark, name, np.median(v), round((v > 0).mean()*100)))
        print("  %-26s n=%-5d %s" % (label, len(g), " ".join(cells)))

    hdr = "  " + " " * 34 + "  ".join("%-11s" % h for h in HZ)
    print("\nORIENTED forward drift  (+ = CONTINUES the day-1 move, − = REVERSES/fades).")
    print(hdr)
    row([e for e in events if e["eff"]], "EFFECT (moved day-1)")
    row([e for e in events if not e["eff"]], "NO-EFFECT (control)")

    print("\nWithin EFFECT — by day-1 direction:")
    print(hdr)
    row([e for e in events if e["eff"] and e["abn"] > 0], "UP-move (good-news pop)")
    row([e for e in events if e["eff"] and e["abn"] < 0], "DOWN-move (bad-news drop)")

    print("\nWithin EFFECT UP-moves — by market-cap bucket (the fade was size-conditioned):")
    print(hdr)
    for cb in ("small", "mid", "large"):
        g = [e for e in events if e["eff"] and e["abn"] > 0 and e["cb"] == cb]
        if len(g) >= 20:
            row(g, f"UP · {cb}")

    print("\nEFFECT news by TYPE at its own horizon (continuation vs reversal):")
    order = sorted(set(e["cat"] for e in events),
                   key=lambda c: -sum(1 for e in events if e["cat"] == c and e["eff"]))
    for cat in order:
        g = [e for e in events if e["cat"] == cat and e["eff"]]
        if len(g) < 20:
            continue
        h = CAT_HORIZON.get(cat, "month"); v = np.array([e["fwd"][h] for e in g])
        med = np.median(v)
        verdict = "CONTINUES" if med > 0.5 else "REVERSES/fade" if med < -0.5 else "flat"
        print("  %-12s @%-5s n=%-4d  med=%+5.1f%%  wr=%2d%%  -> %s"
              % (cat, h, len(g), med, round((v > 0).mean()*100), verdict))
    print("\nEFFECT category counts:", dict(Counter(e["cat"] for e in events if e["eff"]).most_common()))


if __name__ == "__main__":
    main()
