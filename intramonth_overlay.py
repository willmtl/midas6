#!/usr/bin/env python3
"""TURN-OF-MONTH exposure overlay on the flagship — trade the intramonth seasonality.

intramonth_seasonality_study showed the flagship's daily gains cluster at the TURN of the month (week 1
+0.334%/day t3.3, week 4 +0.267% t3.3) and go DEAD mid-month (week 3 +0.046% t0.6; SPY actually negative).
This turns that diagnostic into a tradeable EXPOSURE overlay: scale the book's exposure by week-of-month
(full at the turn, cash/half in the dead middle) and backtest total return / Sharpe / drawdown vs always-
invested. Modelled as a daily exposure scalar on the flagship's realized daily return (in practice an index
hedge, NOT churning the basket). -> .data/studies/intramonth_overlay.json + BacktestResult[intramonth_overlay].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/intramonth_overlay.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from intramonth_seasonality_study import _tdom_maps, _week

TRACE = Path("/app/.data/studies/flagship_history.json")
OUT = Path("/app/.data/studies/intramonth_overlay.json")


def _is_tom(tdom, tdte):
    """Turn-of-month = first 3 or last 3 trading days of the month."""
    return tdom <= 3 or tdte <= 2


# Exposure schemes: (tdom, tdte) -> exposure fraction. week 3 = tdom 11-15 (the dead middle).
SCHEMES = {
    "baseline (always 1x)":      lambda tdom, tdte: 1.0,
    "cash in week 3":            lambda tdom, tdte: 0.0 if _week(tdom) == 3 else 1.0,
    "half in week 3":            lambda tdom, tdte: 0.5 if _week(tdom) == 3 else 1.0,
    "turn-of-month only (1x/0)": lambda tdom, tdte: 1.0 if _is_tom(tdom, tdte) else 0.0,
    "ToM tilt (1.5x turn/0.5 mid)": lambda tdom, tdte: 1.5 if _is_tom(tdom, tdte) else 0.5,
}
ANN = 252


def _metrics(dates, ret):
    eq = np.cumprod(1 + ret)
    total = round((eq[-1] - 1) * 100, 1)
    sd = ret.std(ddof=1) if len(ret) > 1 else 0
    sharpe = round(float(ret.mean() / sd * np.sqrt(ANN)), 2) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    maxdd = round(float(((eq - peak) / peak).min() * 100), 1)
    return total, sharpe, maxdd


def main():
    from core.models import Candle, BacktestResult
    from django.utils import timezone
    from django.db import connection
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")

    spy = pd.Series({pd.Timestamp(d): float(c) for d, c in
                     Candle.objects.filter(ticker="SPY", interval="1d").order_by("date").values_list("date", "close")}).sort_index()
    tmap = _tdom_maps(spy.index)

    d = json.load(open(TRACE))
    holds = [(pd.Timestamp(m["date"]), pd.Timestamp(m["ndate"]), p["ticker"])
             for m in d["months"] for p in m["picks"] if p.get("ret") is not None]
    tickers = sorted({h[2] for h in holds})
    cand = {}
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 40:
            cand[tk] = pd.Series({pd.Timestamp(dt): float(c) for dt, c in q}).sort_index()
    # flagship realized daily return = equal-weight avg of held names' daily returns per date
    by_date = {}
    for bd, nd, tk in holds:
        s = cand.get(tk)
        if s is None:
            continue
        w = s[(s.index > bd) & (s.index <= nd)]
        for dt, rv in w.pct_change().dropna().items():
            by_date.setdefault(dt, []).append(float(rv))
    dates = sorted(dt for dt in by_date if dt in tmap)
    flag = np.array([float(np.mean(by_date[dt])) for dt in dates])

    rows = []
    base_total = None
    for name, fn in SCHEMES.items():
        expo = np.array([fn(*tmap[dt]) for dt in dates])
        ret = expo * flag
        total, sharpe, maxdd = _metrics(dates, ret)
        pct_deployed = round(float((expo > 0).mean() * 100), 1)
        if name.startswith("baseline"):
            base_total = total
        rows.append({"scheme": name, "total_return_pct": total, "sharpe": sharpe, "max_dd_pct": maxdd,
                     "pct_days_deployed": pct_deployed, "avg_exposure": round(float(expo.mean()), 2)})
    for r in rows:
        r["ret_vs_base_pp"] = round(r["total_return_pct"] - base_total, 1) if base_total is not None else None
    # SPY buy-hold over the same span, for context
    span = spy[(spy.index >= dates[0]) & (spy.index <= dates[-1])]
    spy_total = round((float(span.iloc[-1]) / float(span.iloc[0]) - 1) * 100, 1) if len(span) > 1 else None

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "n_days": len(dates), "span": [str(dates[0])[:10], str(dates[-1])[:10]],
               "spy_buyhold_pct": spy_total, "rows": rows,
               "note": ("Turn-of-month exposure overlay on the flagship's realized daily return (equal-weight "
                        "held-names proxy). Exposure scaled by week/turn-of-month; modelled as an index hedge, "
                        "not churning the basket. ⚠️ IN-SAMPLE calendar effect on the same ~5y — turn-of-month "
                        "is a documented anomaly (more defensible than a random cut) but this is not walk-forward "
                        "validated. Gross of fees / hedge cost.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="intramonth_overlay",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[intramonth_overlay]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== TURN-OF-MONTH OVERLAY ({len(dates)} days, SPY buy-hold {spy_total}%) ===", flush=True)
    print(f"  {'scheme':30}{'total%':>9}{'sharpe':>8}{'maxDD%':>8}{'deployed%':>10}{'Δret':>8}", flush=True)
    for r in sorted(rows, key=lambda x: -x["total_return_pct"]):
        print(f"  {r['scheme']:30}{r['total_return_pct']:>9}{r['sharpe']:>8}{r['max_dd_pct']:>8}"
              f"{r['pct_days_deployed']:>10}{r['ret_vs_base_pp']:>+8}", flush=True)


if __name__ == "__main__":
    main()
