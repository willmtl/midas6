"""
News-horizon scanner — the live surface for the horizon-conditioned news work.

For every RECENT, material (impact>=2), LLM-classified news event whose type-specific horizon window
is still OPEN, join it to what our data says happens over THAT horizon for THAT news type, and take a
stance. Only the FADES survived robustness (news_horizon_robust.py + news_drift_robust.py):
  - earnings_beat  @3mo   -> FADE  (size-conditioned: mid/small ~-5%, large ~-1.4%)
  - product        @month -> FADE  (~-3.7%, all sizes)
  - strong bullish POP (dir>0, impact>=2, day-1 β-adj >= +6%) in mid/small cap -> FADE (bull-pop)
Everything else is WATCH — informational (its measured drift shown) but NOT robustness-validated.

Writes NewsHorizonSignal (cleared each run). Run:
  docker compose exec -T backend python -u news_horizon_scan.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from datetime import date, timedelta
from collections import defaultdict
from django.utils import timezone
from seq_fundamental_study import build_universe, load_candles
from core.models import NewsItem, Fundamental, NewsHorizonSignal

HZ_BARS = {"day": 1, "week": 5, "month": 21, "3mo": 63}
CAT_HORIZON = {
    "ma": "day", "upgrade": "week", "downgrade": "week", "capital": "week", "dividend": "week",
    "macro": "week", "product": "month", "contract": "month", "legal": "month", "other": "month",
    "earnings_beat": "3mo", "earnings_miss": "3mo", "guidance_up": "3mo", "guidance_down": "3mo",
    "clinical": "3mo", "mgmt": "3mo",
}
# robust FADE table: cat -> {cap_bucket: expected oriented drift %}. Validated by news_horizon_robust.py.
ROBUST_FADE = {
    "earnings_beat": {"small": -5.7, "mid": -5.3, "large": -1.4},
    "product":       {"small": -3.9, "mid": -3.8, "large": -3.5},
}
# informational (NOT robust) measured drift at own horizon, for the WATCH rows.
OBSERVED = {"earnings_miss": -0.9, "ma": -0.1, "upgrade": -0.5, "contract": +1.4, "legal": -0.7,
            "downgrade": -1.3, "clinical": -1.4, "guidance_up": -5.7, "guidance_down": +1.7,
            "mgmt": +0.3, "capital": +0.9, "dividend": 0.0, "macro": 0.0, "other": -0.8}
BETA_WIN = 60; MIN_PRICE = 3.0; POP = 6.0
SCAN_DAYS = 40   # look back far enough that month/3mo windows can still be open

def _signed_cat(cat, direction):
    """The horizon tables (CAT_HORIZON/ROBUST_FADE/OBSERVED) key on the Anthropic SIGNED taxonomy
    (earnings_beat/…). The local model emits the UNSIGNED cat_llm + a separate local_dir, so fold the
    two back into a signed label. dir>=0 -> bullish variant (ties to bullish, matching how the fade
    tables are indexed); dir<0 -> bearish variant."""
    c = (cat or "other").lower(); d = direction or 0
    if c == "earnings":    return "earnings_beat" if d >= 0 else "earnings_miss"
    if c == "guidance":    return "guidance_up" if d >= 0 else "guidance_down"
    if c == "analyst":     return "upgrade" if d >= 0 else "downgrade"
    if c in ("offering", "buyback"): return "capital"
    if c == "partnership": return "contract"   # no signed 'partnership'; nearest is contract (@month)
    if c == "insider":     return "other"       # not in CAT_HORIZON; treat as generic @month
    return c   # ma / product / contract / legal / clinical / mgmt / macro / dividend / other pass through

def cap_bucket(mc):
    if not mc or mc <= 0:
        return ""
    return "small" if mc < 2e9 else "mid" if mc < 10e9 else "large"

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"]); spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
    mcap = {f["ticker"]: (f["market_cap"] or 0) for f in Fundamental.objects.values("ticker", "market_cap")}
    try:
        from sector_holdings import get_sectors_for_ticker
    except Exception:
        get_sectors_for_ticker = lambda t: []

    since = timezone.now() - timedelta(days=SCAN_DAYS)
    news_by_tk = defaultdict(list)
    for r in (NewsItem.objects.filter(local_rating__isnull=False, local_impact__gte=2, dt__gte=since)
              .values("ticker", "dt", "local_rating", "local_impact", "cat_llm", "local_dir", "title")):
        news_by_tk[r["ticker"]].append(r)
    print("tickers with recent material news:", len(news_by_tk), flush=True)

    today = date.today(); now = timezone.now()
    rows = []
    for tk, items in news_by_tk.items():
        df = cds.get(tk)
        if df is None or len(df) < BETA_WIN + 5:
            continue
        idx = df.index; close = df["Close"].values; n = len(close)
        mkt = spy["Close"].reindex(idx).ffill().values
        r = df["Close"].pct_change()
        both = pd.concat([r.rename("s"), spy_ret.rename("m")], axis=1).reindex(idx)
        beta = (both["s"].rolling(BETA_WIN).cov(both["m"]) / both["m"].rolling(BETA_WIN).var()).values
        # dominant signed item per (news day, cat)
        day = defaultdict(lambda: defaultdict(lambda: {"net": 0, "imp": 0, "title": ""}))
        for it in items:
            d = pd.Timestamp(it["dt"]).tz_localize(None).normalize()
            pos = int(idx.searchsorted(d))
            if pos <= 0 or pos >= n:
                continue
            a = day[pos][_signed_cat(it["cat_llm"], it["local_dir"])]
            a["net"] += (it["local_rating"] or 0)
            if (it["local_impact"] or 0) >= a["imp"]:
                a["imp"] = it["local_impact"] or 0; a["title"] = it["title"] or ""
        mc = mcap.get(tk, 0); cb = cap_bucket(mc)
        best = {}   # keep one row per (tk,news_date,cat) — take the most actionable
        for pos, cats in day.items():
            days_since = (n - 1) - pos            # bars since the news bar (0 = latest bar)
            for cat, a in cats.items():
                if a["net"] == 0 or a["imp"] < 2:
                    continue
                hz = CAT_HORIZON.get(cat, "month"); hb = HZ_BARS[hz]
                days_left = hb - days_since
                if days_left <= 0 or days_since < 1:      # window closed, or no day-1 bar yet
                    continue
                dr = int(np.sign(a["net"]))
                # day-1 β-adj abnormal move (context / bull-pop trigger)
                rfrom, rto = pos - 1, pos + 1
                pop = None
                if rfrom >= BETA_WIN and rto < n and mkt[rfrom] > 0 and mkt[rto] > 0 and close[rto] >= MIN_PRICE:
                    b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
                    v = ((close[rto] / close[rfrom] - 1.0) - bc * (mkt[rto] / mkt[rfrom] - 1.0)) * 100
                    pop = float(v) if np.isfinite(v) else None
                # stance
                stance, robust, exp = "WATCH", False, OBSERVED.get(cat, 0.0)
                # a fade needs something still elevated to give back — if the "good news" already
                # cratered day-1 (pop strongly negative), the thesis is spent -> leave it WATCH.
                if cat in ROBUST_FADE and (pop is None or pop >= -2.0):
                    stance, robust, exp = "FADE", True, ROBUST_FADE[cat].get(cb, ROBUST_FADE[cat]["large"])
                elif dr > 0 and a["imp"] >= 2 and pop is not None and pop >= POP and 0 < mc < 10e9:
                    stance, robust = "FADE", True                      # bull-pop (news_drift_robust)
                    exp = -11.0 if cb == "small" else -6.0
                rec = dict(ticker=tk, news_date=idx[pos].date(), cat=cat, direction=dr, impact=int(a["imp"]),
                           horizon=hz, pop_pct=(round(pop, 1) if pop is not None else None), market_cap=mc,
                           cap_bucket=cb, exp_drift=round(exp, 1), stance=stance, robust=robust,
                           days_since=int(days_since), days_left=int(days_left),
                           last_close=round(float(close[-1]), 2), title=(a["title"] or "")[:240])
                key = (tk, rec["news_date"], cat)
                # prefer robust rows, then higher impact, then more days_left
                cur = best.get(key)
                if cur is None or (rec["robust"], rec["impact"], rec["days_left"]) > (cur["robust"], cur["impact"], cur["days_left"]):
                    best[key] = rec
        for rec in best.values():
            rec["sectors"] = get_sectors_for_ticker(tk) or []
            rows.append(rec)

    NewsHorizonSignal.objects.all().delete()
    NewsHorizonSignal.objects.bulk_create([NewsHorizonSignal(computed_at=now, **rec) for rec in rows], batch_size=500)
    fades = [r for r in rows if r["stance"] == "FADE"]
    print(f"wrote {len(rows)} news-horizon signals ({len(fades)} FADE / {len(rows)-len(fades)} WATCH)", flush=True)
    from collections import Counter
    print("FADE by cat:", dict(Counter(r["cat"] for r in fades).most_common()))
    print("FADE by horizon:", dict(Counter(r["horizon"] for r in fades).most_common()))
    return {"total": len(rows), "fade": len(fades)}

if __name__ == "__main__":
    main()
