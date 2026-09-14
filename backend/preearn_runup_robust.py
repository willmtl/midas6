#!/usr/bin/env python3
"""ROBUSTNESS pass for the pre-earnings RUN-UP stock book (after it passed the equity-curve gate). Three tests:
  (1) K SWEEP {3,5,7,10,15} trading days before the print (ADV>=100M, 5bps) — how the edge scales with the
      run-up window and the date-known-ahead risk.
  (2) SUBPERIODS (2015-19 / 2020-22 / 2023-26) + FULL + EX-2020 for headline K=10 — is the edge one era / one year?
  (3) VALUE/QUALITY OVERLAY on K=10 — split events by PIT profitability (TTM net_income>0 as-of entry) and by
      P/B (mktcap/total_equity, cheap = below pooled median). Does SELECTION add alpha over plain long-beta?
All net of cost, EW continuously-invested, survivorship-aware, vs buy-hold SPY same window.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_runup_robust.py"""
import os, sys, json, math, bisect
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

LIQ = 100e6
COST_HALF = (5 / 2) / 1e4
KS_SWEEP = [3, 5, 7, 10, 15]
HEADLINE_K = 10


def perf(dret):
    dret = dret.dropna()
    if len(dret) < 60:
        return None
    eq = (1 + dret.values).cumprod(); yrs = len(dret) / 252.0
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    sh = dret.mean() / dret.std() * math.sqrt(252) if dret.std() > 0 else 0.0
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return dict(total=round((eq[-1] - 1) * 100, 1), cagr=round(cagr, 1), sharpe=round(sh, 2), maxdd=round(dd, 1))


def book_series(contrib):
    days = sorted(contrib)
    if len(days) < 60:
        return None
    b = pd.Series([np.mean(contrib[d]) for d in days], index=pd.DatetimeIndex(days)).sort_index()
    full = pd.date_range(b.index[0], b.index[-1], freq="B")
    return b.reindex(full).fillna(0.0)


def main():
    universe, _ = _universe()
    from core.models import EarningsEvent, Candle
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    edates = defaultdict(list)
    for tk, rd in EarningsEvent.objects.values_list("ticker", "report_date"):
        if tk in have:
            edates[tk].append(pd.Timestamp(rd))
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_ret = spy["c"].astype(float).groupby(spy["d"]).last().sort_index().pct_change()

    sweep = {K: defaultdict(list) for K in KS_SWEEP}          # K-sweep contrib
    # headline K=10 events with attrs: list of (dates_slice, rets_slice, profitable, pb)
    ev_daily = []   # (list[date], list[ret], profitable_bool_or_None, pb_or_None)
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            dret = pd.Series(close, index=dates).pct_change().values; di = dates.values
            # PIT fundamentals arrays
            r = reps.get(tk); av = ni = te = sh = None
            if r is not None and len(r):
                rr = r.dropna(subset=["avail_date"]).sort_values("avail_date")
                av = pd.to_datetime(rr["avail_date"]).values
                ni = rr["net_income"].values if "net_income" in rr else None
                te = rr["total_equity"].values if "total_equity" in rr else None
                sh = rr["shares_outstanding"].values if "shares_outstanding" in rr else None
            for rd in edates.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t >= n:
                    continue
                # K sweep
                for K in KS_SWEEP:
                    a = t - K; b = t - 1
                    if a < 0 or b <= a or close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                        continue
                    if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ):
                        continue
                    for d in range(a + 1, b + 1):
                        rr_ = dret[d]
                        if not np.isfinite(rr_):
                            continue
                        if d == a + 1:
                            rr_ -= COST_HALF
                        if d == b:
                            rr_ -= COST_HALF
                        sweep[K][dates[d]].append(rr_)
                # headline K=10 with attrs
                a = t - HEADLINE_K; b = t - 1
                if a < 0 or b <= a or close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                    continue
                if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ):
                    continue
                entry = dates[a]
                prof = None; pb = None
                if av is not None and len(av):
                    j = bisect.bisect_right(av, np.datetime64(entry)) - 1
                    if j >= 0:
                        if ni is not None and j >= 3:
                            ttm = np.nansum(ni[j - 3:j + 1])
                            prof = bool(ttm > 0)
                        if te is not None and sh is not None and np.isfinite(te[j]) and te[j] > 0 and np.isfinite(sh[j]):
                            pb = (sh[j] * float(close[a])) / te[j]
                ds = []; rs = []
                for d in range(a + 1, b + 1):
                    rr_ = dret[d]
                    if not np.isfinite(rr_):
                        continue
                    if d == a + 1:
                        rr_ -= COST_HALF
                    if d == b:
                        rr_ -= COST_HALF
                    ds.append(dates[d]); rs.append(rr_)
                if ds:
                    ev_daily.append((ds, rs, prof, pb))
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    def bench(idx):
        return perf(spy_ret.reindex(idx).fillna(0.0))

    # ---- (1) K SWEEP ----
    print("\n=== (1) K SWEEP  (ADV>=100M, 5bps) ===", flush=True)
    print(f"  {'K':>4}{'total%':>10}{'CAGR%':>8}{'Sh':>6}{'maxDD%':>8}{'  vs SPY(CAGR)':>16}", flush=True)
    out = {"k_sweep": {}}
    for K in KS_SWEEP:
        bs = book_series(sweep[K])
        if bs is None:
            continue
        bp = perf(bs); sp = bench(bs.index)
        out["k_sweep"][K] = dict(book=bp, spy=sp)
        print(f"  {K:>4}{bp['total']:>10.1f}{bp['cagr']:>8.1f}{bp['sharpe']:>6.2f}{bp['maxdd']:>8.1f}"
              f"{sp['cagr']:>12.1f}", flush=True)

    # ---- build a book from a filtered subset of ev_daily ----
    def build(filt):
        c = defaultdict(list)
        nev = 0
        for ds, rs, prof, pb in ev_daily:
            if not filt(prof, pb):
                continue
            nev += 1
            for d, r in zip(ds, rs):
                c[d].append(r)
        return c, nev

    # pooled P/B median (positive pb only)
    pbs = [pb for _, _, _, pb in ev_daily if pb is not None and pb > 0]
    pbmed = float(np.median(pbs)) if pbs else None

    # ---- (2) SUBPERIODS on headline all-book ----
    call_c, _ = build(lambda p, pb: True)
    allbook = book_series(call_c)
    print("\n=== (2) SUBPERIODS — headline K=10 all-book vs SPY ===", flush=True)
    print(f"  {'period':14}{'bookCAGR%':>11}{'spyCAGR%':>10}{'bookSh':>8}{'excess':>8}", flush=True)
    out["subperiods"] = {}
    periods = {"2015-2019": ("2015-01-01", "2019-12-31"), "2020-2022": ("2020-01-01", "2022-12-31"),
               "2023-2026": ("2023-01-01", "2026-12-31"), "FULL": (None, None),
               "EX-2020": ("EX", "EX")}
    for label, (s, e) in periods.items():
        if label == "EX-2020":
            mask = allbook.index.year != 2020
            bb = allbook[mask]; sb = spy_ret.reindex(allbook.index).fillna(0.0)[mask]
        elif s is None:
            bb = allbook; sb = spy_ret.reindex(allbook.index).fillna(0.0)
        else:
            m = (allbook.index >= s) & (allbook.index <= e)
            bb = allbook[m]; sb = spy_ret.reindex(allbook.index).fillna(0.0)[m]
        bp = perf(bb); sp = perf(sb)
        if not bp or not sp:
            continue
        out["subperiods"][label] = dict(book=bp, spy=sp)
        print(f"  {label:14}{bp['cagr']:>11.1f}{sp['cagr']:>10.1f}{bp['sharpe']:>8.2f}{bp['cagr']-sp['cagr']:>8.1f}", flush=True)

    # ---- (3) VALUE/QUALITY OVERLAY ----
    print(f"\n=== (3) SELECTION OVERLAY on K=10 (pooled P/B median={pbmed:.2f}) vs SPY ===", flush=True)
    print(f"  {'filter':22}{'events':>8}{'CAGR%':>8}{'Sh':>6}{'maxDD%':>8}{'vsSPY':>8}", flush=True)
    filts = {
        "ALL": lambda p, pb: True,
        "profitable (TTM>0)": lambda p, pb: p is True,
        "unprofitable": lambda p, pb: p is False,
        "cheap (P/B<median)": lambda p, pb: pb is not None and 0 < pb < pbmed,
        "expensive (P/B>median)": lambda p, pb: pb is not None and pb >= pbmed,
        "profitable & cheap": lambda p, pb: p is True and pb is not None and 0 < pb < pbmed,
    }
    out["overlay"] = {}
    for label, f in filts.items():
        c, nev = build(f)
        bs = book_series(c)
        if bs is None:
            continue
        bp = perf(bs); sp = bench(bs.index)
        out["overlay"][label] = dict(events=nev, book=bp, spy=sp)
        print(f"  {label:22}{nev:>8}{bp['cagr']:>8.1f}{bp['sharpe']:>6.2f}{bp['maxdd']:>8.1f}{bp['cagr']-sp['cagr']:>8.1f}", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), liq=LIQ, cost_bps=5, ks=KS_SWEEP,
                   headline_k=HEADLINE_K, pb_median=pbmed, results=out,
                   caveat="Robustness for pre-earnings run-up stock book. K-sweep, subperiods+ex-2020, and PIT "
                   "value/quality overlay (TTM net_income>0; P/B=mktcap/total_equity vs pooled median). EW, net 5bps.")
    Path("/app/.data/studies/preearn_runup_robust.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_runup_robust",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_runup_robust]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
