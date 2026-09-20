#!/usr/bin/env python3
"""FDA CATALYST — Phase 2 PIT integrity check. Phase 1's run-up assumed the action date was knowable ~T_entry days
ahead. Prove it isn't look-ahead using truly point-in-time news: the "FDA accepts / PDUFA / priority review / action
date" ANNOUNCEMENT (when we would ACTUALLY learn the catalyst is coming). Two tests:
  A) PIT-FILTERED run-up: re-measure the 42->5 run-up ONLY on events whose PDUFA/acceptance was publicly announced
     >=42 trading days before the action date. If the drift survives on this provably-knowable subset -> not look-ahead.
  B) ACCEPTANCE-ENTRY hold (the honestly-tradeable version): enter the trading day AFTER the announcement, exit 5
     trading days before the action date. Report raw, excess vs XBI, hold length, win, t, both halves.
Saves BacktestResult[fda_pit_check].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fda_pit_check.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from django.db.models import Q
from core.models import Candle, NewsItem

APPROVALS = "/app/.data/fda_approvals.json"
TMAP = "/app/.data/fda_ticker_map.json"
SPLIT = pd.Timestamp("2021-01-01")
# pre-decision announcement keywords (catalyst becomes publicly KNOWN) — exclude the approval/decision itself
KW = ["pdufa", "accepts", "accepted for review", "priority review", "action date", "target action",
      "granted priority", "advisory committee", "adcom", "fast track", "breakthrough therapy",
      "files nda", "files bla", "submits nda", "submits bla", "accepted the"]


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


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


def news_dates(tickers):
    """per ticker: sorted list of dates of pre-decision FDA announcements (PIT catalyst-known dates)."""
    q = Q()
    for k in KW:
        q |= Q(title__icontains=k)
    m = {}
    rows = NewsItem.objects.filter(q, ticker__in=list(tickers)).values_list("ticker", "dt")
    for tk, dt in rows:
        m.setdefault(tk, []).append(pd.Timestamp(dt).normalize().tz_localize(None))
    for tk in m:
        m[tk] = sorted(set(m[tk]))
    return m


def main():
    events = json.load(open(APPROVALS))
    tmap = json.load(open(TMAP))
    ev = [dict(e, ticker=tmap[e["sponsor_name"]]) for e in events
          if e["action_date"] >= "2015-01-01" and tmap.get(e["sponsor_name"])]
    tickers = {e["ticker"] for e in ev}
    close = load_close(tickers | {"XBI"})
    nd = news_dates(tickers)
    xbi = close.get("XBI")
    n_news_tickers = len(nd)
    print(f"events {len(ev)} | tickers {len(tickers)} | tickers with any pre-decision FDA news: {n_news_tickers}", flush=True)

    def pos(s, d):
        if s is None or len(s) == 0:
            return None
        p = s.index.searchsorted(d, side="right") - 1
        return p if 0 <= p < len(s) else None

    def ret_between(s, d_in, d_out):
        pi, po = pos(s, d_in), pos(s, d_out)
        if pi is None or po is None or po <= pi:
            return None, None, None
        a, b = float(s.iloc[pi]), float(s.iloc[po])
        return (b / a - 1.0 if a > 0 else None), s.index[pi], s.index[po]

    def px_before(s, ref, back):
        p = pos(s, ref)
        if p is None or p - back < 0:
            return None, None
        return float(s.iloc[p - back]), s.index[p - back]

    def known_date(e):
        """earliest pre-decision announcement for this event, within [action-540d, action-15d]."""
        d = pd.Timestamp(e["action_date"])
        lst = nd.get(e["ticker"]) or []
        cand = [x for x in lst if d - pd.Timedelta(days=540) <= x <= d - pd.Timedelta(days=15)]
        return min(cand) if cand else None

    # ---- Test A: PIT-filtered 42->5 run-up (date provably known >=42 trading days ahead) ----
    full, pit = [], []
    for e in ev:
        s = close.get(e["ticker"]); d = pd.Timestamp(e["action_date"])
        p_in, din = px_before(s, d, 42); p_out, dout = px_before(s, d, 5)
        if not p_in or not p_out or p_in <= 0:
            continue
        r = p_out / p_in - 1.0
        xr = 0.0
        if xbi is not None and din is not None and dout is not None:
            xi, _, xo = None, None, None
            xrr, _, _ = ret_between(xbi, din, dout)
            xr = xrr if xrr is not None else 0.0
        rec = {"date": d, "r": r, "excess": r - xr, "kind": e["kind"]}
        full.append(rec)
        kd = known_date(e)
        if kd is not None and din is not None and kd <= din:      # announced on/before the entry day
            pit.append(rec)

    def summ(rows, key="r"):
        v = pd.Series([x[key] for x in rows]).dropna()
        if len(v) < 5:
            return {"n": int(len(v))}
        h1 = v[[r["date"] < SPLIT for r in rows][:len(v)]] if False else None
        return {"n": int(len(v)), "mean_pct": round(float(v.mean() * 100), 2), "win_pct": round(float((v > 0).mean() * 100), 1),
                "t": round(tstat(v), 2)}

    def halves(rows, key="r"):
        h1 = [x for x in rows if x["date"] < SPLIT]; h2 = [x for x in rows if x["date"] >= SPLIT]
        return summ(h1, key).get("t"), summ(h2, key).get("t")

    res = {"testA_full": summ(full, "r"), "testA_full_excess": summ(full, "excess"),
           "testA_pit": summ(pit, "r"), "testA_pit_excess": summ(pit, "excess")}
    ha = halves(full); hp = halves(pit)
    print("\n=== Test A: 42->5 run-up, full vs PIT-clean subset (PDUFA announced >=42d ahead) ===", flush=True)
    print(f"  FULL      n={res['testA_full'].get('n')} raw {res['testA_full'].get('mean_pct')}% t={res['testA_full'].get('t')} "
          f"| exXBI {res['testA_full_excess'].get('mean_pct')}% t={res['testA_full_excess'].get('t')} | halves t {ha}", flush=True)
    print(f"  PIT-CLEAN n={res['testA_pit'].get('n')} raw {res['testA_pit'].get('mean_pct')}% t={res['testA_pit'].get('t')} "
          f"| exXBI {res['testA_pit_excess'].get('mean_pct')}% t={res['testA_pit_excess'].get('t')} | halves t {hp}", flush=True)

    # ---- Test B: acceptance-entry hold (enter day after announcement, exit 5d before action) ----
    bt = []
    for e in ev:
        s = close.get(e["ticker"]); d = pd.Timestamp(e["action_date"])
        kd = known_date(e)
        if kd is None:
            continue
        p = pos(s, kd)
        if p is None or p + 1 >= len(s):
            continue
        d_in = s.index[p + 1]                                    # next trading day after announcement
        p_out, d_out = px_before(s, d, 5)
        if p_out is None or d_out is None or d_out <= d_in:
            continue
        r, _, _ = ret_between(s, d_in, d_out)
        if r is None:
            continue
        xr, _, _ = ret_between(xbi, d_in, d_out)
        bt.append({"date": d, "r": r, "excess": r - (xr or 0.0), "hold_d": (d_out - d_in).days, "kind": e["kind"]})
    hb = halves(bt)
    res["testB_accept_hold"] = summ(bt, "r"); res["testB_accept_hold_excess"] = summ(bt, "excess")
    res["testB_mean_hold_days"] = round(float(np.mean([x["hold_d"] for x in bt])), 0) if bt else None
    print("\n=== Test B: acceptance-entry hold (enter after announcement, exit 5d before decision) ===", flush=True)
    print(f"  n={res['testB_accept_hold'].get('n')} raw {res['testB_accept_hold'].get('mean_pct')}% "
          f"t={res['testB_accept_hold'].get('t')} win {res['testB_accept_hold'].get('win_pct')}% "
          f"| exXBI {res['testB_accept_hold_excess'].get('mean_pct')}% t={res['testB_accept_hold_excess'].get('t')} "
          f"| mean hold {res['testB_mean_hold_days']}d | halves t {hb}", flush=True)

    verdict = "PIT-CONFIRMED" if (res["testA_pit"].get("t") or 0) >= 2 and (res["testA_pit"].get("mean_pct") or 0) > 0 else "WEAK/INCONCLUSIVE"
    res["verdict"] = verdict
    print(f"\nVERDICT: run-up on the PIT-clean subset -> {verdict}", flush=True)

    json.dump(res, open("/app/.data/studies/fda_pit_check.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="fda_pit_check", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[fda_pit_check]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
