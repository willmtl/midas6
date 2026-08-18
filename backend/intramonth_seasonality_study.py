#!/usr/bin/env python3
"""INTRA-MONTH SEASONALITY — is there a week/day of the month where gains/losses cluster? Tests the
turn-of-month effect on (A) the market (SPY) and (B) the flagship's actual holds. Buckets every daily
return by its TRADING-DAY-OF-MONTH (1 = first trading day, ...) and by TRADING-DAYS-TILL-MONTH-END, then
by WEEK (1-5) and turn-of-month (first-3 / last-3 / middle). Answers: which week makes the money.
-> .data/studies/intramonth_seasonality.json + BacktestResult[intramonth_seasonality].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/intramonth_seasonality_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection
from studies import _tstat_from_returns

TRACE = Path("/app/.data/studies/flagship_history.json")
OUT = Path("/app/.data/studies/intramonth_seasonality.json")


def _tdom_maps(cal):
    """cal = DatetimeIndex of trading days. Returns {date: (trading_day_of_month, trading_days_to_month_end)}."""
    df = pd.DataFrame({"d": cal})
    df["ym"] = df["d"].dt.to_period("M")
    df["tdom"] = df.groupby("ym").cumcount() + 1
    df["n"] = df.groupby("ym")["d"].transform("count")
    df["tdte"] = df["n"] - df["tdom"]          # 0 = last trading day of month
    return {r.d: (int(r.tdom), int(r.tdte)) for r in df.itertuples()}


def _week(tdom):
    return min((tdom - 1) // 5 + 1, 5)          # week 1 = tdom 1-5, etc.


def _bucket_report(series_by_date, tmap, label, out):
    """series_by_date: dict date->list of daily returns. Aggregates by week + turn-of-month."""
    wk = {w: [] for w in range(1, 6)}
    tom = {"first_3": [], "last_3": [], "middle": []}
    for dt, rets in series_by_date.items():
        if dt not in tmap:
            continue
        tdom, tdte = tmap[dt]
        wk[_week(tdom)].extend(rets)
        if tdom <= 3:
            tom["first_3"].extend(rets)
        elif tdte <= 2:
            tom["last_3"].extend(rets)
        else:
            tom["middle"].extend(rets)
    print(f"\n=== {label}: avg DAILY return by WEEK of month ===", flush=True)
    print(f"  {'week':10}{'n days':>9}{'avg/day':>10}{'t':>7}", flush=True)
    wko = {}
    for w in range(1, 6):
        r = np.asarray(wk[w], float)
        if len(r) < 20:
            continue
        t = _tstat_from_returns(list(r))
        wko[f"week{w}"] = {"n": len(r), "avg_bps": round(float(r.mean()) * 1e4, 1), "t": round(t, 2) if t else None}
        print(f"  week {w:<5}{len(r):>9}{r.mean()*100:>+9.3f}%{(round(t,2) if t else 0):>7}", flush=True)
    print(f"  -- turn-of-month --", flush=True)
    tomo = {}
    for k in ("first_3", "middle", "last_3"):
        r = np.asarray(tom[k], float)
        if len(r) < 20:
            continue
        t = _tstat_from_returns(list(r))
        tomo[k] = {"n": len(r), "avg_bps": round(float(r.mean()) * 1e4, 1), "t": round(t, 2) if t else None}
        print(f"  {k:<10}{len(r):>9}{r.mean()*100:>+9.3f}%{(round(t,2) if t else 0):>7}", flush=True)
    out[label] = {"by_week": wko, "turn_of_month": tomo}


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, BacktestResult
    from django.utils import timezone

    # trading calendar from SPY
    spy = pd.Series({pd.Timestamp(d): float(c) for d, c in
                     Candle.objects.filter(ticker="SPY", interval="1d").order_by("date").values_list("date", "close")}).sort_index()
    cal = spy.index
    tmap = _tdom_maps(cal)
    spy_ret = spy.pct_change().dropna()
    out = {"computed_at": pd.Timestamp.utcnow().isoformat()}

    # (A) SPY (the market)
    spy_by_date = {dt: [float(spy_ret[dt])] for dt in spy_ret.index}
    _bucket_report(spy_by_date, tmap, "SPY (market)", out)

    # (B) flagship holds: daily returns of each held name during its hold window
    d = json.load(open(TRACE))
    holds = [(pd.Timestamp(m["date"]), pd.Timestamp(m["ndate"]), p["ticker"])
             for m in d["months"] for p in m["picks"] if p.get("ret") is not None]
    tickers = sorted({h[2] for h in holds})
    cand = {}
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 40:
            cand[tk] = pd.Series({pd.Timestamp(dt): float(c) for dt, c in q}).sort_index()
    fh_by_date = {}
    for bd, nd, tk in holds:
        s = cand.get(tk)
        if s is None:
            continue
        w = s[(s.index > bd) & (s.index <= nd)]
        r = w.pct_change().dropna()
        for dt, rv in r.items():
            fh_by_date.setdefault(dt, []).append(float(rv))
    _bucket_report(fh_by_date, tmap, "FLAGSHIP holds", out)

    # verdict
    sm = out["SPY (market)"]["turn_of_month"]
    if sm:
        best = max(out["SPY (market)"]["by_week"].items(), key=lambda kv: kv[1]["avg_bps"])
        worst = min(out["SPY (market)"]["by_week"].items(), key=lambda kv: kv[1]["avg_bps"])
        out["verdict"] = (f"SPY: best week = {best[0]} ({best[1]['avg_bps']}bps/day), worst = {worst[0]} "
                          f"({worst[1]['avg_bps']}bps/day); turn-of-month first_3 {sm.get('first_3',{}).get('avg_bps')}bps "
                          f"vs middle {sm.get('middle',{}).get('avg_bps')}bps vs last_3 {sm.get('last_3',{}).get('avg_bps')}bps.")
        print("\nVERDICT:", out["verdict"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="intramonth_seasonality",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[intramonth_seasonality]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_INTRAMONTH", flush=True)


if __name__ == "__main__":
    main()
