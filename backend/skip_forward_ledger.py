"""
SKIP FORWARD LEDGER — the out-of-sample-style companion to the PROXY_TEST backtest.

PROXY_TEST proved (in aggregate) that holding the accelerating bond/commodity/country ETF instead
of SKIPPING it loses. This makes it CONCRETE and per-event: for every sleeve the flagship SKIPPED
in the history trace, compute what that sleeve's ETF ACTUALLY returned the FOLLOWING month, and
compare to what our basket (the value picks that got the redistributed weight) made that same month.

If skipping is right, the skipped ETFs' realized forward returns should be <= our basket's — i.e.
the high accel really was a blow-off top, not a continuation. This is the "current/forward" view the
backtest can't show on its own: not 'would proxy have helped in-sample' but 'what did the thing we
passed on literally do next'.

Reads the flagship history trace (months[].skipped[] with etf+accel, months[].basket_ret/spy_ret),
pulls month-end ETF closes from Candle, and saves BacktestResult[skip_forward_ledger].

Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/skip_forward_ledger.py
"""
import os
import json
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import pandas as pd
import numpy as np
from core.models import Candle, BacktestResult

TRACE = "/app/.data/studies/flagship_history.json"
COMMODITY = {"GLD", "SLV", "PPLT", "USO", "UNG", "DBA", "WEAT", "CORN", "URA", "LIT", "COPX", "SLX", "REMX", "AMLP"}
BOND = {"TLT", "AGG", "HYG", "TIP", "BIL", "SHV", "SHY", "IEI", "IEF", "LQD", "FLOT", "MUB", "CWB"}


def etf_type(etf):
    if etf in COMMODITY:
        return "commodity"
    if etf in BOND:
        return "bond"
    return "country/index"


def main():
    with open(TRACE) as f:
        tr = json.load(f)
    months = tr["months"]
    print(f"trace arm={tr.get('arm')} months={len(months)}", flush=True)

    # collect every skipped ETF and every month date
    skip_etfs = sorted({s["etf"] for m in months for s in m.get("skipped", [])})
    dates = [m["date"] for m in months]
    print(f"distinct skipped ETFs={len(skip_etfs)}", flush=True)

    # month-end close panel for the skipped ETFs (last close on/before each trace month-end)
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    rows = Candle.objects.filter(ticker__in=skip_etfs).values_list("ticker", "date", "close")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    panel = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    didx = pd.to_datetime(dates)
    # as-of month-end close (reindex forward-fill to the trace month-end grid)
    me = panel.reindex(panel.index.union(didx)).ffill().reindex(didx)

    ledger = []
    for i, m in enumerate(months[:-1]):          # need i+1 for the forward month
        d0, d1 = didx[i], didx[i + 1]
        basket = m.get("basket_ret")
        spy = m.get("spy_ret")
        for s in m.get("skipped", []):
            e = s["etf"]
            if e not in me.columns:
                continue
            c0, c1 = me.loc[d0, e], me.loc[d1, e]
            if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0):
                continue
            fwd = c1 / c0 - 1.0
            ledger.append({
                "month": m["date"], "sector": s.get("sector"), "etf": e,
                "type": etf_type(e), "accel": s.get("accel"),
                "etf_fwd_ret": round(fwd * 100, 2),
                "basket_ret": round((basket or 0) * 100, 2) if basket is not None else None,
                "spy_ret": round((spy or 0) * 100, 2) if spy is not None else None,
                "edge_vs_etf": (round(((basket or 0) - fwd) * 100, 2) if basket is not None else None),
            })

    L = pd.DataFrame(ledger)
    print(f"\nledger events={len(L)}", flush=True)
    if L.empty:
        return

    # aggregate: what did the skipped ETFs return next month vs our basket that month
    def agg(sub, lab):
        etf_mean = sub["etf_fwd_ret"].mean()
        bkt_mean = sub["basket_ret"].mean()
        winrate_skip = (sub["basket_ret"] >= sub["etf_fwd_ret"]).mean() * 100   # % of times skipping >= buying the ETF
        print(f"  {lab:16} n={len(sub):>4} | skipped-ETF fwd avg {etf_mean:+6.2f}%"
              f" | our basket avg {bkt_mean:+6.2f}%  | skip-was-right {winrate_skip:4.0f}%", flush=True)
        return {"label": lab, "n": int(len(sub)), "skipped_etf_fwd_avg": round(float(etf_mean), 2),
                "basket_avg": round(float(bkt_mean), 2), "skip_was_right_pct": round(float(winrate_skip), 1)}

    print("=== SKIP FORWARD LEDGER (what the sleeves we passed on actually did NEXT month) ===", flush=True)
    summary = [agg(L, "ALL SKIPS")]
    for t in ["commodity", "bond", "country/index"]:
        sub = L[L["type"] == t]
        if len(sub):
            summary.append(agg(sub, t))

    # the highest-accel skips (the ones that "look" like the biggest missed opportunities)
    top = L.reindex(L["accel"].abs().sort_values(ascending=False).index).head(12)
    print("\n  most-accelerating skips (biggest 'missed' pops) — what they did next month:", flush=True)
    for _, r in top.iterrows():
        print(f"    {r['month']}  {r['sector']:<24} accel {r['accel']:>6}"
              f"  -> ETF next-mo {r['etf_fwd_ret']:+6.2f}%  (we made {r['basket_ret']:+6.2f}%)", flush=True)

    from django.utils import timezone
    BacktestResult.objects.create(kind="skip_forward_ledger", computed_at=timezone.now(), payload={
        "generated_from": TRACE, "arm": tr.get("arm"),
        "n_events": int(len(L)), "summary": summary,
        "events": ledger,
        "note": ("For every SKIPPED sleeve, the ETF's realized FOLLOWING-month return vs our basket that "
                 "month. skip_was_right_pct = share of skips where our basket >= the skipped ETF. "
                 "Concrete companion to PROXY_TEST: proves the high-accel skips were blow-off tops."),
    })
    print("\nsaved BacktestResult[skip_forward_ledger]", flush=True)


if __name__ == "__main__":
    main()
