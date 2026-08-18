#!/usr/bin/env python3
"""MULTI-DAY POP + TAKE-PROFIT-AT-100% event study (broad universe; follow-up to vol_pop_reversion which
found SINGLE-day pops in low-vol names DON'T revert — they drift up). Two new questions:

(A) MULTI-DAY POP: a run-up accumulated over 5 trading days (>= threshold) in a LOW-vol name — does the
    multi-day version revert where the single-day didn't? Forward f1/f5/f21 from the run's end. Gap-deduped
    (>=5d between events) so overlapping windows don't inflate the t-stat.
(B) TAKE-PROFIT at +100%: after a stock has DOUBLED over trailing 126d (6mo), does it revert (take-profit
    smart) or keep running (let it ride)? Buckets 50-100 / 100-200 / >200%. Forward f21/f63. Gap-deduped 21d.
-> .data/studies/vol_multiday_takeprofit.json + BacktestResult[vol_multiday_takeprofit].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/vol_multiday_takeprofit_study.py
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

OUT = Path("/app/.data/studies/vol_multiday_takeprofit.json")
POP5_THR = [0.15, 0.20, 0.30, 0.40]
RUNUP_BUCKETS = [("50-100%", 0.50, 1.0), ("100-200%", 1.0, 2.0), (">200%", 2.0, 99.0)]
VOL_BUCKETS = [("LOW <2%/d", 0.0, 0.02), ("MID 2-4%", 0.02, 0.04), ("HIGH >4%", 0.04, 9.9)]


def _dedup(idx_positions, gap):
    keep, last = [], -10 ** 9
    for i in idx_positions:
        if i - last >= gap:
            keep.append(i); last = i
    return keep


def _vb(v):
    for name, lo, hi in VOL_BUCKETS:
        if lo <= v < hi:
            return name
    return "HIGH >4%"


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, Fundamental, Sector, BacktestResult
    from django.utils import timezone

    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    tickers = sorted(set(Fundamental.objects.values_list("ticker", flat=True)) - etfs)
    print(f"universe: {len(tickers)}", flush=True)

    pop5, runup = [], []
    n_names = 0
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) < 200:
            continue
        s = pd.Series({pd.Timestamp(d): float(c) for d, c in q}).sort_index()
        s = s[s > 0]
        if len(s) < 200:
            continue
        n_names += 1
        arr = s.values
        tvol = s.pct_change().rolling(60).std().shift(1).values
        cum5 = np.concatenate([[np.nan] * 5, arr[5:] / arr[:-5] - 1])
        run126 = np.concatenate([[np.nan] * 126, arr[126:] / arr[:-126] - 1])
        n = len(arr)

        def fwd(i, h):
            return arr[i + h] / arr[i] - 1 if i + h < n else np.nan
        # (A) multi-day 5d pop, low events gap-deduped 5d
        cand = [i for i in range(126, n - 21) if np.isfinite(cum5[i]) and cum5[i] >= POP5_THR[0]
                and np.isfinite(tvol[i]) and tvol[i] > 0]
        for i in _dedup(cand, 5):
            pop5.append({"tk": tk, "pop5": float(cum5[i]), "vol": float(tvol[i]),
                         "f1": fwd(i, 1), "f5": fwd(i, 5), "f21": fwd(i, 21)})
        # (B) runup >=50% over 126d, gap-deduped 21d
        cand = [i for i in range(126, n - 63) if np.isfinite(run126[i]) and run126[i] >= 0.50]
        for i in _dedup(cand, 21):
            runup.append({"tk": tk, "runup": float(run126[i]),
                          "f21": fwd(i, 21), "f63": fwd(i, 63)})
    dfp = pd.DataFrame(pop5); dfr = pd.DataFrame(runup)
    print(f"names {n_names} | 5d-pop events {len(dfp)} | runup(>=50%) events {len(dfr)}", flush=True)
    out = {"computed_at": pd.Timestamp.utcnow().isoformat(), "n_names": n_names,
           "n_pop5": int(len(dfp)), "n_runup": int(len(dfr)), "multiday": {}, "takeprofit": {}}

    # (A) multi-day pop by vol bucket x threshold
    dfp["vb"] = dfp["vol"].apply(_vb)
    print("\n=== (A) MULTI-DAY (5-day) POP -> forward return (reversion => negative) ===", flush=True)
    for vb, lo, hi in VOL_BUCKETS:
        print(f"  --- trailing vol {vb} ---", flush=True)
        print(f"    {'5dpop>=':>8}{'n':>7}{'f1':>8}{'f5':>8}{'f21':>8}{'f5 t':>8}{'win%':>7}", flush=True)
        for thr in POP5_THR:
            g = dfp[(dfp["vb"] == vb) & (dfp["pop5"] >= thr)]
            g5 = g["f5"].dropna()
            if len(g5) < 10:
                continue
            t5 = _tstat_from_returns(list(g5))
            rr = {h: round(float(g[f"f{h}"].mean()) * 100, 2) for h in (1, 5, 21)}
            out["multiday"][f"{vb}|{int(thr*100)}"] = {"n": int(len(g)), **rr, "f5_t": round(t5, 2) if t5 else None}
            print(f"    {int(thr*100):>7}%{len(g):>7}{rr[1]:>+7.2f}{rr[5]:>+7.2f}{rr[21]:>+7.2f}"
                  f"{(round(t5,2) if t5 else 0):>8}{100*(g5>0).mean():>6.0f}%", flush=True)
        print("", flush=True)

    # (B) take-profit: forward return after a big run-up
    print("=== (B) TAKE-PROFIT: forward return AFTER a trailing-6mo run-up (revert => take profit; continue => ride) ===", flush=True)
    print(f"  {'runup':>10}{'n':>7}{'f21':>9}{'f63':>9}{'f21 t':>8}{'f63 t':>8}{'f63 win%':>9}", flush=True)
    for lab, lo, hi in RUNUP_BUCKETS:
        g = dfr[(dfr["runup"] >= lo) & (dfr["runup"] < hi)]
        g21, g63 = g["f21"].dropna(), g["f63"].dropna()
        if len(g63) < 10:
            continue
        t21 = _tstat_from_returns(list(g21)); t63 = _tstat_from_returns(list(g63))
        out["takeprofit"][lab] = {"n": int(len(g)), "f21": round(float(g21.mean()) * 100, 2),
                                  "f63": round(float(g63.mean()) * 100, 2),
                                  "f21_t": round(t21, 2) if t21 else None, "f63_t": round(t63, 2) if t63 else None,
                                  "f63_win": round(float((g63 > 0).mean()) * 100, 1)}
        print(f"  {lab:>10}{len(g):>7}{g21.mean()*100:>+8.2f}{g63.mean()*100:>+8.2f}"
              f"{(round(t21,2) if t21 else 0):>8}{(round(t63,2) if t63 else 0):>8}{100*(g63>0).mean():>8.0f}%", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="vol_multiday_takeprofit",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[vol_multiday_takeprofit]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_MDTP", flush=True)


if __name__ == "__main__":
    main()
