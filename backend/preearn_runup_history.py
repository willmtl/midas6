#!/usr/bin/env python3
"""Does a LONGER HISTORY of good earnings reactions sharpen the run-up? Extends preearn_runup_prevsign from
lag-1 to a lookback of the last N prints (N=1..10). For each event, feature = MEAN of the reactions to the last
N prior prints (each reaction = close[t_prev+1]/close[t_prev-1]-1). Bucket the run-up trade by sign of that mean
(POS history / NEG history). Question: as N grows (more confirmation of a good-reaction track record), does the
POS bucket's run-up edge get stronger, peak, or wash out vs just using the last print (N=1)?

Per (K, N): POS-bucket per-trade exSPY + win + book CAGR/Sharpe/vsSPY, NEG-bucket book vsSPY, and the continuous
corr(hist_mean, run-up excess). Requires N prior in-sample prints (sample shrinks as N grows -> n reported).
K in {5,10}, ADV>=100M, net 5bps. Persists BacktestResult[preearn_runup_history]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_runup_history.py"""
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
NS = list(range(1, 11))         # lookback = last N prior prints
LIQ = 100e6
COST_BPS = 5
COST_HALF = (COST_BPS / 2) / 1e4


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
    for tk in edates:
        edates[tk] = sorted(set(edates[tk]))
    print(f"names with earnings dates: {len(names)}", flush=True)

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()
    spy_ret = spy_c.pct_change()

    # per-trade rows per K: (hist_mean_by_N dict, runup_excess, runup_ret, dates_slice, rets_slice)
    # accumulate into: per (K,N,bucket) -> per-trade excess list, and book contrib
    texc = {(K, N, s): [] for K in KS for N in NS for s in ("POS", "NEG")}
    twin = {(K, N, s): [] for K in KS for N in NS for s in ("POS", "NEG")}
    contrib = {(K, N, s): defaultdict(list) for K in KS for N in NS for s in ("POS", "NEG")}
    corr_x = {(K, N): [] for K in KS for N in NS}   # (hist_mean, excess) pairs
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
            tpos = []
            reac = []
            for rd in evs:
                p = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                tpos.append(p if p < n else None)
                r = np.nan
                if p < n and 0 <= p - 1 and p + 1 < n and np.isfinite(close[p - 1]) and np.isfinite(close[p + 1]) and close[p - 1] > 0:
                    r = close[p + 1] / close[p - 1] - 1.0
                reac.append(r)
            for i, rd in enumerate(evs):
                t = tpos[i]
                if t is None:
                    continue
                # prior reactions (most recent first): reac[i-1], reac[i-2], ...
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
                    exc = tr - sr
                    # daily contribution slice (shared across N/buckets it lands in)
                    slice_d = []
                    for d in range(a + 1, b + 1):
                        rr = dret[d]
                        if not np.isfinite(rr):
                            continue
                        if d == a + 1:
                            rr -= COST_HALF
                        if d == b:
                            rr -= COST_HALF
                        slice_d.append((dates[d], rr))
                    for N in NS:
                        if i - N < 0:
                            continue
                        prior = [reac[i - k] for k in range(1, N + 1)]
                        prior = [x for x in prior if np.isfinite(x)]
                        if len(prior) < N:      # require full N history
                            continue
                        hm = float(np.mean(prior))
                        s = "POS" if hm > 0 else "NEG"
                        texc[(K, N, s)].append(exc); twin[(K, N, s)].append(1.0 if tr > 0 else 0.0)
                        corr_x[(K, N)].append((hm, exc))
                        for dd_, rr in slice_d:
                            contrib[(K, N, s)][dd_].append(rr)
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    out = {}
    for K in KS:
        print(f"\n=== K={K}  run-up bucketed by SIGN of mean of last N earnings reactions (ADV>=100M, net {COST_BPS}bps) ===", flush=True)
        print(f"  {'N':>3}{'POSn':>7}{'POSexSPY':>10}{'POSwin%':>9}{'POScagr':>9}{'POSsh':>7}{'POSvsSPY':>10}"
              f"{'NEGn':>7}{'NEGcagr':>9}{'NEGvsSPY':>10}{'corr':>8}", flush=True)
        out[f"K{K}"] = {}
        for N in NS:
            pe = np.array(texc[(K, N, "POS")], float); pw = np.array(twin[(K, N, "POS")], float)
            ne = np.array(texc[(K, N, "NEG")], float)
            if len(pe) < 100:
                continue
            pbs = book_series(contrib[(K, N, "POS")]); nbs = book_series(contrib[(K, N, "NEG")])
            pbp = perf(pbs) if pbs is not None else None
            psp = perf(spy_ret.reindex(pbs.index).fillna(0.0)) if pbs is not None else None
            nbp = perf(nbs) if nbs is not None else None
            nsp = perf(spy_ret.reindex(nbs.index).fillna(0.0)) if nbs is not None else None
            xy = np.array(corr_x[(K, N)], float)
            cc = float(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]) if len(xy) > 10 else float("nan")
            row = dict(POSn=len(pe), POSexSPY=round(float(pe.mean()) * 100, 3), POSwin=round(float(pw.mean()) * 100, 1),
                       POScagr=pbp["cagr"] if pbp else None, POSsh=pbp["sharpe"] if pbp else None,
                       POSvsSPY=round(pbp["cagr"] - psp["cagr"], 1) if (pbp and psp) else None,
                       NEGn=len(ne), NEGcagr=nbp["cagr"] if nbp else None,
                       NEGvsSPY=round(nbp["cagr"] - nsp["cagr"], 1) if (nbp and nsp) else None,
                       corr=round(cc, 4))
            out[f"K{K}"][N] = row
            print(f"  {N:>3}{row['POSn']:>7}{row['POSexSPY']:>10.3f}{row['POSwin']:>9.1f}"
                  f"{(row['POScagr'] or 0):>9.1f}{(row['POSsh'] or 0):>7.2f}{(row['POSvsSPY'] or 0):>10.1f}"
                  f"{row['NEGn']:>7}{(row['NEGcagr'] or 0):>9.1f}{(row['NEGvsSPY'] or 0):>10.1f}{cc:>8.4f}", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), liq=LIQ, cost_bps=COST_BPS, ks=KS, ns=NS,
                   results=out, caveat="Run-up (buy t-K sell t-1) bucketed by sign of MEAN of last N earnings "
                   "reactions (N=1..10; reaction=close[tp-1]->[tp+1]). Requires full N in-sample history. POS/NEG "
                   "per-trade exSPY + EW book vs SPY; corr(hist_mean, run-up excess). ADV>=100M.")
    Path("/app/.data/studies/preearn_runup_history.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_runup_history",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_runup_history]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
