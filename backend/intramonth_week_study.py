#!/usr/bin/env python3
"""FIRST-WEEK-PREDICTS-MONTH study (user hypothesis, 2026-08-19): returns concentrate at the turn of month
(first few / last few trading days). So if a held name FAILS to go up in the FIRST WEEK of the hold, does the
whole month tend to fail? If yes, we could exit at end of week 1 (or SHORT weeks 2-4) instead of riding a
loser to month-end.

DIAGNOSTIC on the flagship trace (no re-simulation). For each (month, held pick) we slice the name's DAILY
closes over the hold window [buy=date, sell=ndate] and split:
    week1_ret = close[+5 trading days] / close[buy] - 1     (the first week)
    rest_ret  = close[sell] / close[+5td] - 1                (weeks 2..end — what an early exit/short would act on)
    full_ret  = close[sell] / close[buy] - 1                 (the actual hold-month return)

We then ask: conditional on week1 FAILING (<= T), what is E[full_ret] and E[rest_ret]? If rest_ret is clearly
negative when week1 failed, an early-exit (go to cash) or a short-the-rest overlay adds value. We also sim three
overlays at basket level (mean over all picks):
    hold_all           = full_ret                                   (baseline flagship)
    exit_on_fail(T)     = week1_ret if week1<=T else full_ret        (cash after a failed first week)
    short_rest(T)       = week1_ret - rest_ret if week1<=T else full_ret   (short the name weeks 2..end)

Point-in-time: week1_ret is known at buy+5td, before the exit/short decision. No look-ahead.
Reads /app/.data/studies/flagship_history.json.  -> BacktestResult[intramonth_week] + prints.
Run: docker exec -w /app rotation-backend-1 python -u intramonth_week_study.py
"""
import os, sys, json, warnings
sys.path.insert(0, "/app")
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import pandas as pd
from pathlib import Path
from seq_fundamental_study import load_candles

TRACE = Path("/app/.data/studies/flagship_history.json")
WK = 5                                     # trading days = "first week"
THRs = [0.0, -0.02, -0.05]                 # first-week "failure" thresholds to test


def main():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")

    D = json.load(open(TRACE))
    months = D["months"]
    tickers = sorted({p["ticker"] for m in months for p in m.get("picks", []) if p.get("ticker")})
    print(f"trace: {len(months)} months, {len(tickers)} unique pick names, flagship total {D['perf']['total']}%",
          flush=True)
    cand = load_candles(tickers)

    recs = []          # (week1, rest, full) per held pick-month with enough daily data
    for m in months:
        d0 = pd.Timestamp(m["date"]); d1 = pd.Timestamp(m["ndate"]) if m.get("ndate") else None
        if d1 is None:
            continue
        for p in m.get("picks", []):
            t = p.get("ticker")
            df = cand.get(t) if t else None
            if df is None or "Close" not in df:
                continue
            dd = df["Close"].dropna()
            # SNAP to actual TRADING days: the trace date/ndate are CALENDAR month-end labels (resample ME),
            # whose value is the last trading day's close. Slicing raw daily by the calendar label desyncs when
            # month-end is a weekend/holiday (~1/3 of months) — it would start on the NEXT month's first day.
            left = dd.index[dd.index <= d0]            # buy = last trading day on/before the month-end label
            right = dd.index[dd.index <= d1]           # sell = last trading day on/before next month-end
            if len(left) == 0 or len(right) == 0:
                continue
            i0 = dd.index.get_loc(left[-1]); iN = dd.index.get_loc(right[-1])
            if iN - i0 < WK + 1 or i0 + WK > iN:       # need a full week inside the hold window
                continue
            p0 = float(dd.iloc[i0]); pw = float(dd.iloc[i0 + WK]); pn = float(dd.iloc[iN])
            if p0 <= 0 or pw <= 0:
                continue
            recs.append((pw / p0 - 1, pn / pw - 1, pn / p0 - 1))
    if not recs:
        print("no usable pick-months", flush=True); return
    w1 = np.array([r[0] for r in recs]); rest = np.array([r[1] for r in recs]); full = np.array([r[2] for r in recs])
    n = len(recs)

    print(f"\n=== FIRST-WEEK vs FULL-MONTH ({n} held pick-months; week = {WK} trading days) ===", flush=True)
    print(f"  corr(week1, full_month)   = {np.corrcoef(w1, full)[0,1]:+.3f}", flush=True)
    print(f"  corr(week1, rest_of_month)= {np.corrcoef(w1, rest)[0,1]:+.3f}   "
          f"(<0 => a bad first week is FOLLOWED by a good rest = mean-reversion, kills the 'short the rest' idea)",
          flush=True)
    print(f"  overall: week1 {w1.mean()*100:+.2f}%/mo | rest {rest.mean()*100:+.2f}% | full {full.mean()*100:+.2f}%",
          flush=True)

    print("\n  conditional on FIRST WEEK outcome:", flush=True)
    print(f"    {'first-week bucket':22}{'n':>5}{'E[full]':>10}{'E[rest]':>10}{'P(full<0)':>11}{'P(rest<0)':>11}", flush=True)
    def _bucket(mask, lab):
        if mask.sum() == 0:
            return
        print(f"    {lab:22}{int(mask.sum()):>5}{full[mask].mean()*100:>9.2f}%{rest[mask].mean()*100:>9.2f}%"
              f"{np.mean(full[mask]<0)*100:>10.0f}%{np.mean(rest[mask]<0)*100:>10.0f}%", flush=True)
    _bucket(w1 > 0, "week1 UP (>0)")
    _bucket(w1 <= 0, "week1 flat/down (<=0)")
    _bucket(w1 <= -0.02, "week1 down > 2%")
    _bucket(w1 <= -0.05, "week1 down > 5%")

    # ── overlay simulation (mean simple return across all pick-months; a proxy for the basket) ──
    print("\n  OVERLAY (mean pick return; baseline hold_all = full month):", flush=True)
    base = full.mean()
    print(f"    {'policy':30}{'mean ret':>10}{'vs hold':>10}", flush=True)
    print(f"    {'hold_all (flagship)':30}{base*100:>9.2f}%{'—':>10}", flush=True)
    out = {"n": n, "corr_week1_full": float(np.corrcoef(w1, full)[0,1]),
           "corr_week1_rest": float(np.corrcoef(w1, rest)[0,1]),
           "mean_week1": float(w1.mean()), "mean_rest": float(rest.mean()), "mean_full": float(full.mean()),
           "hold_all": float(base), "overlays": {}}
    for T in THRs:
        fail = w1 <= T
        exit_cash = np.where(fail, w1, full)                     # go to cash after a failed first week
        short_rest = np.where(fail, w1 - rest, full)             # short the name for weeks 2..end
        e_c, e_s = exit_cash.mean(), short_rest.mean()
        print(f"    exit-to-cash if wk1<={T*100:>+.0f}%{'':10}{e_c*100:>9.2f}%{(e_c-base)*100:>+9.2f}pp", flush=True)
        print(f"    short-rest   if wk1<={T*100:>+.0f}%{'':10}{e_s*100:>9.2f}%{(e_s-base)*100:>+9.2f}pp", flush=True)
        out["overlays"][str(T)] = {"n_fail": int(fail.sum()), "exit_cash": float(e_c), "short_rest": float(e_s),
                                   "exit_vs_hold_pp": float((e_c-base)*100), "short_vs_hold_pp": float((e_s-base)*100)}

    print("\n  VERDICT: 'short the rest' needs corr(week1,rest)>0 AND E[rest|wk1 fail]<0. If corr(week1,rest) is "
          "NEGATIVE, a bad first week REVERTS (rest is positive) -> exiting/shorting after week 1 is WRONG.", flush=True)

    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="intramonth_week", defaults={"payload": json.loads(json.dumps(out, default=str)),
                                              "computed_at": timezone.now()})
        print("\nSaved BacktestResult[intramonth_week]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
