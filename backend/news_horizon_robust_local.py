"""
LOCAL-LABEL re-validation of the TYPE-conditioned news FADE edges — the cutover gate for the
earnings_beat / product (and contract / guidance_up) horizon signals.

Identical methodology to news_horizon_robust.py (the retired Anthropic/Haiku-label version) but reads
the LOCAL qwen labels (local_rating / local_impact / cat_llm + local_dir) instead of
llm_rating / llm_impact / llm_cat. The live scanner (news_horizon_scan.py) already reads local labels,
so its ROBUST_FADE table must be re-derived on the SAME labels — a drift measured on the retired
Anthropic taxonomy is not guaranteed to hold on the local one. Only edges that reproduce here (STRONG
negative, BOTH time halves same sign, size-consistent) stay robust; the rest drop to WATCH.

oriented drift = dir * abn (beta-adj). + = drifts WITH the news, - = reverses (fade).

Run:  docker compose exec -T backend python -u news_horizon_robust_local.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import build_universe, load_candles
from core.models import NewsItem, Fundamental

HZ = {"day": 1, "week": 5, "month": 21, "3mo": 63}
BETA_WIN = 60; MIN_PRICE = 3.0
# The signals whose fade the live scanner treats as robust (+ the two references validated alongside).
SIGNALS = [("earnings_beat", "3mo"), ("product", "month"), ("contract", "month"),
           ("guidance_up", "3mo"), ("earnings_miss", "3mo")]


def _signed_cat(cat, direction):
    """Fold the unsigned local cat_llm + local_dir into the signed taxonomy the horizon tables key
    on. Copy of news_horizon_scan._signed_cat (kept local so this validator is standalone)."""
    c = (cat or "other").lower(); d = direction or 0
    if c == "earnings":    return "earnings_beat" if d >= 0 else "earnings_miss"
    if c == "guidance":    return "guidance_up" if d >= 0 else "guidance_down"
    if c == "analyst":     return "upgrade" if d >= 0 else "downgrade"
    if c in ("offering", "buyback"): return "capital"
    if c == "partnership": return "contract"
    if c == "insider":     return "other"
    return c


def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"]); spy = cds.get("SPY"); spy_ret = spy["Close"].pct_change()
    mcap = {f["ticker"]: (f["market_cap"] or 0) for f in Fundamental.objects.values("ticker", "market_cap")}
    news_by_tk = defaultdict(list)
    for r in NewsItem.objects.filter(local_rating__isnull=False, local_impact__gte=2).values(
            "ticker", "dt", "local_rating", "local_impact", "cat_llm", "local_dir"):
        news_by_tk[r["ticker"]].append(r)

    ev = defaultdict(list)     # signed cat -> list of events
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
        # per (day, signed cat): dominant item (net rating + max impact)
        day = defaultdict(lambda: defaultdict(lambda: {"net": 0, "imp": 0}))
        for it in items:
            scat = _signed_cat(it["cat_llm"], it["local_dir"])
            d = pd.Timestamp(it["dt"]).tz_localize(None).normalize()
            pos = int(idx.searchsorted(d))
            if pos <= 0 or pos >= n:
                continue
            a = day[pos][scat]
            a["net"] += (it["local_rating"] or 0); a["imp"] = max(a["imp"], it["local_impact"] or 0)
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
    if not all_dates:
        print("no local-labeled events — let the classification drain fill more first"); return
    mid = all_dates[len(all_dates) // 2]

    def stat(g, label, hz):
        if len(g) < 15:
            print("    %-20s n=%-4d (thin)" % (label, len(g))); return
        v = np.array([e["fwd"][hz] for e in g])
        print("    %-20s n=%-4d  med=%+5.1f%%  win=%2d%%  mean=%+5.1f%%"
              % (label, len(g), np.median(v), round((v > 0).mean() * 100), v.mean()))

    print("\n=== LOCAL-LABEL type-conditioned drift (oriented; NEGATIVE = fade) ===", flush=True)
    for cat, hz in SIGNALS:
        g = ev.get(cat, [])
        if len(g) < 15:
            print(f"\n{cat} @{hz}: n={len(g)} (thin, skip)"); continue
        v = np.array([e["fwd"][hz] for e in g])
        direction = "UNDER-react (drift with)" if np.median(v) > 0 else "OVER-react (fade)"
        print(f"\n=== {cat} @{hz}  n={len(g)}  ALL med={np.median(v):+.1f}%/{round((v>0).mean()*100)}%win/mean={v.mean():+.1f}%  [{direction}] ===")
        print("  by TIME half:")
        stat([e for e in g if e["date"] <= mid], "1st half", hz)
        stat([e for e in g if e["date"] >  mid], "2nd half", hz)
        print("  by MCAP (bucket keys used by ROBUST_FADE):")
        stat([e for e in g if 0 < e["mcap"] < 2e9],     "small <2B", hz)
        stat([e for e in g if 2e9 <= e["mcap"] < 10e9], "mid 2-10B", hz)
        stat([e for e in g if e["mcap"] >= 10e9],       "large >=10B", hz)
    print(f"\ntime split at {mid.date()}  (window {all_dates[0].date()} .. {all_dates[-1].date()})")


if __name__ == "__main__":
    main()
