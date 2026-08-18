#!/usr/bin/env python3
"""INTRA-MONTH CASH overlay: rebalance TWICE a month — hold the flagship through weeks 1/2/4/5 but sit in
CASH during the dead middle week (week 3), which the seasonality study found is the weakest (SPY slightly
negative, flagship only ~+0.05%/day vs +0.33%/day in week 1). Does dodging week 3 help?

Per flagship hold (buy month-end -> next month-end), decompose the daily path by WEEK-OF-MONTH; the overlay
return compounds every day EXCEPT week-3 days (0/cash there). Aggregate the weighted monthly basket and
compound vs the baseline (hold full month). Also a round-trip transaction-cost sensitivity (the mid-month
exit+reenter is extra turnover). Controls: skip week 2 / week 4 instead (should be worse — they're good weeks).
-> .data/studies/intramonth_cash.json + BacktestResult[intramonth_cash].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/intramonth_cash_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection

TRACE = Path("/app/.data/studies/flagship_history.json")
OUT = Path("/app/.data/studies/intramonth_cash.json")


def _weekmap(cal):
    df = pd.DataFrame({"d": cal})
    df["ym"] = df["d"].dt.to_period("M")
    df["tdom"] = df.groupby("ym").cumcount() + 1
    df["wk"] = ((df["tdom"] - 1) // 5 + 1).clip(upper=5)
    return {r.d: int(r.wk) for r in df.itertuples()}


def _compound(rows):
    r = np.asarray(rows, float)
    if len(r) == 0:
        return 0.0, 0.0
    tot = float(np.prod(1 + r) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    return round(tot, 1), round(sh, 2)


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, BacktestResult
    from django.utils import timezone

    spy = pd.Series({pd.Timestamp(d): float(c) for d, c in
                     Candle.objects.filter(ticker="SPY", interval="1d").order_by("date").values_list("date", "close")}).sort_index()
    wmap = _weekmap(spy.index)

    d = json.load(open(TRACE))
    holds = [(pd.Timestamp(m["date"]), pd.Timestamp(m["ndate"]), p["ticker"], p["weight"], p["ret"])
             for m in d["months"] for p in m["picks"] if p.get("ret") is not None]
    tickers = sorted({h[2] for h in holds})
    cand = {}
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 40:
            cand[tk] = pd.Series({pd.Timestamp(dt): float(c) for dt, c in q}).sort_index()

    # per-hold return skipping a given week (cash that week); n_switches = extra round-trips incurred that month
    def hold_ret(path_ret, skip_wk):
        acc = 1.0
        for dt, rv in path_ret.items():
            if wmap.get(dt) == skip_wk:
                continue                      # cash this day
            acc *= (1 + rv)
        return acc - 1

    rows_full, rows = {}, {"skip3": {}, "skip2": {}, "skip4": {}}
    for bd, nd, tk, w, mret in holds:
        s = cand.get(tk)
        if s is None:
            continue
        pr = s[(s.index > bd) & (s.index <= nd)].pct_change().dropna()
        if len(pr) < 3:
            continue
        key = str(bd.date())
        full = float(np.prod(1 + pr.values) - 1)
        rows_full.setdefault(key, []).append((w, full))
        for lab, wk in (("skip3", 3), ("skip2", 2), ("skip4", 4)):
            rows[lab].setdefault(key, []).append((w, hold_ret(pr, wk)))

    def basket(dct):
        out = []
        for k, lst in dct.items():
            ws = np.array([w for w, _ in lst]); rs = np.array([r for _, r in lst])
            out.append(float(np.average(rs, weights=ws)))
        return out

    base = basket(rows_full)
    bt, bsh = _compound(base)
    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "n_months": len(base),
               "baseline": {"total": bt, "sharpe": bsh}, "variants": {}}
    print(f"\n  {'policy':34}{'total':>10}{'Sharpe':>9}{'vs base':>10}", flush=True)
    print(f"  {'baseline: hold full month':34}{bt:>9.1f}%{bsh:>9}{0.0:>+9.1f}", flush=True)
    for lab, note in [("skip3", "CASH week 3 (dead middle)"), ("skip2", "CASH week 2 (control)"),
                      ("skip4", "CASH week 4 (control, a good week)")]:
        r = basket(rows[lab])
        t, sh = _compound(r)
        payload["variants"][lab] = {"total": t, "sharpe": sh, "vs_base": round(t - bt, 1)}
        print(f"  {note:34}{t:>9.1f}%{sh:>9}{t-bt:>+9.1f}", flush=True)
    # transaction-cost sensitivity on the winning idea (skip3): 2 extra round-trips/month at c bps each
    print(f"\n  skip-week-3 net of round-trip cost (mid-month exit+reenter = 2 legs/month):", flush=True)
    tc = {}
    for cbps in (0, 5, 10, 20):
        r = [x - 2 * cbps / 1e4 for x in basket(rows["skip3"])]   # 2 legs × cbps
        t, sh = _compound(r)
        tc[f"{cbps}bps"] = {"total": t, "sharpe": sh, "vs_base": round(t - bt, 1)}
        print(f"    @ {cbps:>2}bps/leg  total {t:>8.1f}%  Sharpe {sh}  vs base {t-bt:+.1f}", flush=True)
    payload["skip3_after_costs"] = tc

    s3 = payload["variants"]["skip3"]
    payload["verdict"] = (f"CASH-week-3 gross {s3['total']}% (Sh {s3['sharpe']}) vs baseline {bt}% (Sh {bsh}): "
                          f"{s3['vs_base']:+.1f}pp gross. " +
                          ("Sidestepping the dead week helps." if s3["vs_base"] > 0 else
                           "Week 3 is weak but still POSITIVE for the holds, so going to cash FORFEITS it -> costs return; only Sharpe may improve."))
    print("\nVERDICT:", payload["verdict"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="intramonth_cash",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[intramonth_cash]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_INTRAMONTH_CASH", flush=True)


if __name__ == "__main__":
    main()
