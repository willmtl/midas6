#!/usr/bin/env python3
"""Does the PREVIOUS earnings reaction condition the pre-earnings RUN-UP? Split every run-up trade into the
next print by the SIGN of the stock's reaction to its PRIOR earnings print. Question: do stocks that POPPED
last quarter run up harder (momentum / good-news drift) than stocks that DROPPED (rebound / no drift)?

For each earnings event at trading index t (first bar >= report_date), find the PRIOR earnings event at
t_prev. Prior reaction = close[t_prev+1]/close[t_prev-1]-1 (2-day window straddling the print to absorb
BMO/AMC timing). Bucket = PREV_UP (reaction>0) / PREV_DOWN (reaction<0) / (skip if no prior print in-sample).
Run-up trade = buy close t-K, sell close t-1 (day before the print). Headline K=10 (+K=5), ADV>=100M, 5bps.

Two views: (A) per-trade stats (mean/median/win/excess-vs-SPY) per bucket; (B) EW continuously-invested book
per bucket (net-of-cost equity curve vs SPY) -- does the split change CAGR/Sharpe/DD? Persists
BacktestResult[preearn_runup_prevsign]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_runup_prevsign.py"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

KS = [5, 10]
LIQ = 100e6
COST_BPS = 5
COST_HALF = (COST_BPS / 2) / 1e4
PREV_WIN = 1          # reaction window: close[t_prev-PREV_WIN] -> close[t_prev+PREV_WIN]


def perf(dret):
    dret = dret.dropna()
    if len(dret) < 60:
        return None
    eq = (1 + dret.values).cumprod(); yrs = len(dret) / 252.0
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    sh = dret.mean() / dret.std() * math.sqrt(252) if dret.std() > 0 else 0.0
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return dict(total=round((eq[-1] - 1) * 100, 1), cagr=round(cagr, 1), sharpe=round(sh, 2),
                maxdd=round(dd, 1), inv=round((dret != 0).mean() * 100, 1))


def book_series(contrib):
    days = sorted(contrib)
    if len(days) < 60:
        return None
    b = pd.Series([np.mean(contrib[d]) for d in days], index=pd.DatetimeIndex(days)).sort_index()
    full = pd.date_range(b.index[0], b.index[-1], freq="B")
    return b.reindex(full).fillna(0.0)


def tstats(arr, earr):
    a = np.asarray(arr, float) * 100; e = np.asarray(earr, float) * 100
    a = a[np.isfinite(a)]
    if len(a) < 50:
        return None
    return dict(n=len(a), mean=round(float(a.mean()), 3), med=round(float(np.median(a)), 3),
                win=round(float((a > 0).mean() * 100), 1), p10=round(float(np.percentile(a, 10)), 2),
                p90=round(float(np.percentile(a, 90)), 2), excess=round(float(np.nanmean(e)), 3),
                t=round(float(a.mean() / (a.std() / math.sqrt(len(a)))), 2) if a.std() > 0 else 0.0)


def main():
    universe, _ = _universe()
    from core.models import EarningsEvent, Candle
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    edates = defaultdict(list)
    for tk, rd in EarningsEvent.objects.values_list("ticker", "report_date"):
        if tk in have:
            edates[tk].append(pd.Timestamp(rd))
    for tk in edates:
        edates[tk] = sorted(set(edates[tk]))
    print(f"names with earnings dates: {len(names)}", flush=True)

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()
    spy_ret = spy_c.pct_change()

    BUCKETS = ["PREV_UP", "PREV_DOWN", "ALL"]
    # per-trade
    tret = {(K, b): [] for K in KS for b in BUCKETS}
    texc = {(K, b): [] for K in KS for b in BUCKETS}
    # book contributions
    contrib = {(K, b): defaultdict(list) for K in KS for b in BUCKETS}
    n_noprior = 0; n_events = 0
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            dret = pd.Series(close, index=dates).pct_change().values; di = dates.values
            spy_al = spy_c.reindex(dates).ffill().values
            evs = edates.get(tk, [])
            # map each earnings date to its trading index t
            tpos = []
            for rd in evs:
                p = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                tpos.append(p if p < n else None)
            for i, rd in enumerate(evs):
                t = tpos[i]
                if t is None:
                    continue
                n_events += 1
                # prior earnings reaction
                tprev = tpos[i - 1] if i > 0 else None
                bucket = None
                if tprev is not None and 0 <= tprev - PREV_WIN and tprev + PREV_WIN < n:
                    pa, pb = close[tprev - PREV_WIN], close[tprev + PREV_WIN]
                    if np.isfinite(pa) and np.isfinite(pb) and pa > 0:
                        prev_reac = pb / pa - 1.0
                        bucket = "PREV_UP" if prev_reac > 0 else "PREV_DOWN"
                if bucket is None:
                    n_noprior += 1
                for K in KS:
                    a = t - K; b = t - 1
                    if a < 0 or b <= a or close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                        continue
                    if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ):
                        continue
                    pa, pb = float(close[a]), float(close[b])
                    if not np.isfinite(pb):
                        continue
                    tr = (pb - pa) / pa - 2 * COST_HALF
                    sa, sb = spy_al[a], spy_al[b]
                    sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                    targets = ["ALL"] + ([bucket] if bucket else [])
                    for bk in targets:
                        tret[(K, bk)].append(tr); texc[(K, bk)].append(tr - sr)
                        for d in range(a + 1, b + 1):
                            rr = dret[d]
                            if not np.isfinite(rr):
                                continue
                            if d == a + 1:
                                rr -= COST_HALF
                            if d == b:
                                rr -= COST_HALF
                            contrib[(K, bk)][dates[d]].append(rr)
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    print(f"\nevents={n_events}  no-prior-in-sample(first prints)={n_noprior}", flush=True)

    out = {"per_trade": {}, "book": {}}
    print("\n=== (A) PER-TRADE by prior-earnings-reaction sign (buy t-K, sell t-1; net {}bps) ===".format(COST_BPS), flush=True)
    print(f"  {'K':>3} {'bucket':10}{'n':>7}{'mean':>8}{'med':>8}{'win%':>7}{'exSPY':>8}{'t':>7}", flush=True)
    for K in KS:
        for bk in BUCKETS:
            s = tstats(tret[(K, bk)], texc[(K, bk)])
            if not s:
                continue
            out["per_trade"][f"K{K}|{bk}"] = s
            print(f"  {K:>3} {bk:10}{s['n']:>7}{s['mean']:>8.3f}{s['med']:>8.3f}{s['win']:>7.1f}"
                  f"{s['excess']:>8.3f}{s['t']:>7.2f}", flush=True)

    print("\n=== (B) EW CONTINUOUSLY-INVESTED BOOK by prior-reaction sign vs SPY ===", flush=True)
    print(f"  {'K':>3} {'bucket':10}{'total%':>10}{'CAGR%':>8}{'Sh':>6}{'maxDD%':>8}{'inv%':>6}{'vsSPY':>8}", flush=True)
    for K in KS:
        for bk in BUCKETS:
            bs = book_series(contrib[(K, bk)])
            if bs is None:
                continue
            bp = perf(bs); sp = perf(spy_ret.reindex(bs.index).fillna(0.0))
            out["book"][f"K{K}|{bk}"] = dict(book=bp, spy=sp)
            print(f"  {K:>3} {bk:10}{bp['total']:>10.1f}{bp['cagr']:>8.1f}{bp['sharpe']:>6.2f}"
                  f"{bp['maxdd']:>8.1f}{bp['inv']:>6.1f}{bp['cagr']-sp['cagr']:>8.1f}", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), liq=LIQ, cost_bps=COST_BPS, ks=KS,
                   prev_win=PREV_WIN, results=out, n_events=n_events, n_noprior=n_noprior,
                   caveat="Pre-earnings run-up (buy close t-K, sell close t-1) split by SIGN of prior earnings "
                   "reaction (close[t_prev-1]->close[t_prev+1]). PREV_UP/PREV_DOWN vs ALL. Per-trade + EW "
                   "continuously-invested book, net 5bps, ADV>=100M. First-in-sample prints have no prior -> ALL only.")
    Path("/app/.data/studies/preearn_runup_prevsign.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_runup_prevsign",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_runup_prevsign]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
