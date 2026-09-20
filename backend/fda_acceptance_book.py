#!/usr/bin/env python3
"""FDA CATALYST — Phase 3 prep: EXPAND acceptance-date coverage beyond the 26 news-covered tickers by ESTIMATING the
acceptance/PDUFA anchor from the openFDA review clock: PRIORITY review => ~8mo before approval, STANDARD => ~12mo
(PDUFA = filing + 60d + 6/10mo; approval ~= PDUFA if on-time). Then:
  (1) VALIDATE the estimate vs the real news acceptance dates on the overlap subset (median error in days).
  (2) Re-run the acceptance-entry hold (enter est. acceptance, exit 5d before approval) at FULL coverage vs the
      news-only version — same edge on more events? Segmented by cap x kind, both halves, vs XBI.
Saves BacktestResult[fda_acceptance_book].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fda_acceptance_book.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from django.db.models import Q
from core.models import Candle, NewsItem, FinancialReport

APPROVALS = "/app/.data/fda_approvals.json"
TMAP = "/app/.data/fda_ticker_map.json"
SPLIT = pd.Timestamp("2021-01-01")
CLOCK = {"PRIORITY": 243, "STANDARD": 365}                 # est days from acceptance-anchor to approval
KW = ["pdufa", "accepts", "accepted for review", "priority review", "action date", "target action",
      "granted priority", "advisory committee", "adcom", "files nda", "files bla", "submits nda", "submits bla"]


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def cap_bucket(mc):
    if mc is None or mc <= 0:
        return "unknown"
    return "micro(<300M)" if mc < 3e8 else "small(300M-2B)" if mc < 2e9 else "mid(2-10B)" if mc < 1e10 else "large(>10B)"


def load_close(tickers):
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 200):
        rows = Candle.objects.filter(ticker__in=tickers[i:i + 200], interval="1d", date__gte="2013-06-01"
                                     ).values_list("ticker", "date", "close")
        df = pd.DataFrame(list(rows), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            out[tk] = g.sort_values("date").set_index("date")["close"]
    return out


def main():
    events = json.load(open(APPROVALS))
    tmap = json.load(open(TMAP))
    ev = [dict(e, ticker=tmap[e["sponsor_name"]]) for e in events
          if e["action_date"] >= "2015-01-01" and tmap.get(e["sponsor_name"])]
    tickers = {e["ticker"] for e in ev}
    close = load_close(tickers | {"XBI"}); xbi = close.get("XBI")

    # news acceptance dates (for validation)
    q = Q()
    for k in KW:
        q |= Q(title__icontains=k)
    nd = {}
    for tk, dt in NewsItem.objects.filter(q, ticker__in=list(tickers)).values_list("ticker", "dt"):
        nd.setdefault(tk, []).append(pd.Timestamp(dt).normalize().tz_localize(None))
    for tk in nd:
        nd[tk] = sorted(set(nd[tk]))

    # PIT shares for cap bucket
    shr = {}
    for tk, av, sh in FinancialReport.objects.filter(ticker__in=list(tickers)).exclude(shares_outstanding=None
            ).values_list("ticker", "avail_date", "shares_outstanding"):
        shr.setdefault(tk, []).append((pd.Timestamp(av), float(sh)))
    for tk in shr:
        shr[tk].sort()

    def pos(s, d):
        if s is None or len(s) == 0:
            return None
        p = s.index.searchsorted(d, side="right") - 1
        return p if 0 <= p < len(s) else None

    def ret_between(s, di, do):
        pi, po = pos(s, di), pos(s, do)
        if pi is None or po is None or po <= pi:
            return None
        a, b = float(s.iloc[pi]), float(s.iloc[po])
        return (b / a - 1.0) if a > 0 else None

    def px_before(s, ref, back):
        p = pos(s, ref)
        return (float(s.iloc[p - back]), s.index[p - back]) if (p is not None and p - back >= 0) else (None, None)

    def mktcap(tk, when):
        val = None
        for av, sh in shr.get(tk, []):
            if av <= when:
                val = sh
            else:
                break
        if val is None:
            return None
        p = pos(close.get(tk), when)
        return val * float(close[tk].iloc[p]) if p is not None else None

    def est_acc(e):
        d = pd.Timestamp(e["action_date"])
        return d - pd.Timedelta(days=CLOCK.get(e.get("review_priority"), 365))

    def news_acc(e):
        d = pd.Timestamp(e["action_date"])
        cand = [x for x in nd.get(e["ticker"], []) if d - pd.Timedelta(days=540) <= x <= d - pd.Timedelta(days=15)]
        return min(cand) if cand else None

    # (1) validate estimate vs news
    errs = []
    for e in ev:
        na = news_acc(e)
        if na is not None:
            errs.append((est_acc(e) - na).days)
    errs = pd.Series(errs)
    print(f"events {len(ev)} | review_priority present: {sum(1 for e in ev if e.get('review_priority') in CLOCK)}", flush=True)
    if len(errs):
        print(f"\n(1) VALIDATION est-acceptance vs NEWS-acceptance (n={len(errs)} overlap): "
              f"median err {errs.median():.0f}d | mean {errs.mean():.0f}d | |err|<90d: {(errs.abs()<90).mean()*100:.0f}%", flush=True)

    # (2) acceptance-hold at full coverage (estimated) vs news-only
    def book(anchor_fn, label):
        rows = []
        for e in ev:
            s = close.get(e["ticker"]); d = pd.Timestamp(e["action_date"])
            a = anchor_fn(e)
            if a is None:
                continue
            p = pos(s, a)
            if p is None or p + 1 >= len(s):
                continue
            din = s.index[p + 1]
            pout, dout = px_before(s, d, 5)
            if dout is None or dout <= din:
                continue
            r = ret_between(s, din, dout)
            if r is None:
                continue
            xr = ret_between(xbi, din, dout) or 0.0
            rows.append({"date": d, "r": r, "excess": r - xr, "hold_d": (dout - din).days,
                         "kind": e["kind"], "cap": cap_bucket(mktcap(e["ticker"], din))})
        v = pd.Series([x["r"] for x in rows]); ex = pd.Series([x["excess"] for x in rows])
        h1 = [x for x in rows if x["date"] < SPLIT]; h2 = [x for x in rows if x["date"] >= SPLIT]
        print(f"\n(2) {label}: n={len(rows)} raw {v.mean()*100:+.2f}% t={tstat(v):.2f} win {(v>0).mean()*100:.0f}% "
              f"| exXBI {ex.mean()*100:+.2f}% t={tstat(ex):.2f} | hold {np.mean([x['hold_d'] for x in rows]):.0f}d "
              f"| H1t {tstat([x['r'] for x in h1]):.2f} H2t {tstat([x['r'] for x in h2]):.2f}", flush=True)
        return rows, {"n": len(rows), "raw_pct": round(float(v.mean()*100), 2), "raw_t": round(tstat(v), 2),
                      "win_pct": round(float((v > 0).mean()*100), 1), "excess_pct": round(float(ex.mean()*100), 2),
                      "excess_t": round(tstat(ex), 2)}

    print("\n=== acceptance-entry hold: NEWS-only vs ESTIMATED (full coverage) ===", flush=True)
    _, news_stats = book(news_acc, "NEWS-only")
    est_rows, est_stats = book(est_acc, "ESTIMATED (all events)")

    # segmentation of the estimated book by cap x kind
    print("\nestimated book segmented (raw / excess vs XBI):", flush=True)
    seg = {}
    for kind in ("ORIG", "sNDA"):
        for cap in ("micro(<300M)", "small(300M-2B)", "mid(2-10B)", "large(>10B)", "unknown"):
            rs = [x for x in est_rows if x["kind"] == kind and x["cap"] == cap]
            if len(rs) >= 8:
                v = pd.Series([x["r"] for x in rs]); ex = pd.Series([x["excess"] for x in rs])
                seg[f"{kind}/{cap}"] = {"n": len(rs), "raw_pct": round(float(v.mean()*100), 2),
                                        "excess_pct": round(float(ex.mean()*100), 2), "excess_t": round(tstat(ex), 2)}
                print(f"  {kind:5} {cap:16} n={len(rs):4} raw {v.mean()*100:+6.2f}% exXBI {ex.mean()*100:+6.2f}% t {tstat(ex):5.2f}", flush=True)

    res = {"validation_median_err_days": float(errs.median()) if len(errs) else None,
           "validation_n": int(len(errs)), "news_only": news_stats, "estimated_full": est_stats, "segmented": seg}
    json.dump(res, open("/app/.data/studies/fda_acceptance_book.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="fda_acceptance_book", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[fda_acceptance_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
