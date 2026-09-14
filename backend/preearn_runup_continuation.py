#!/usr/bin/env python3
"""Does a BIG pre-earnings RUN-UP CONTINUE through the print? I.e. does a strong drift-up into earnings predict
a POSITIVE earnings-day reaction (momentum continuation), or does it fade/reverse (run-up already priced it)?

For each event at trading index t (first bar >= report_date):
  RUNUP  = close[t-1]/close[t-K] - 1     (the drift we bought, buy t-K sell t-1)
  PRINT  = close[t+1]/close[t-1] - 1     (through-the-print gap, t-1 close -> t+1 close, 2-day to absorb BMO/AMC)
Bucket events into QUINTILES by RUNUP magnitude (within each K). Report, per quintile: run-up mean, PRINT mean/
median/win%, PRINT excess-vs-SPY same 2-day window, and P(PRINT>0). Also the JOINT cell 'big run-up AND positive
print' = P(both) and its combined RUNUP+PRINT return. Answers: if we DON'T sell before the print on the big
run-ups, do we get paid or punished? K in {5,10}, ADV>=100M, net 5bps. Persists BacktestResult[preearn_runup_cont].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_runup_continuation.py"""
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
NQ = 5           # quintiles by run-up magnitude


def main():
    universe, _ = _universe()
    from core.models import EarningsEvent, Candle
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    edates = defaultdict(list)
    for tk, rd in EarningsEvent.objects.values_list("ticker", "report_date"):
        if tk in have:
            edates[tk].append(pd.Timestamp(rd))
    print(f"names with earnings dates: {len(names)}", flush=True)

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()

    # collect per event: (runup, print_ret, print_excess) for each K
    rows = {K: [] for K in KS}
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            di = dates.values; spy_al = spy_c.reindex(dates).ffill().values
            for rd in edates.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t + 1 >= n:            # need t+1 for the print window
                    continue
                for K in KS:
                    a = t - K
                    if a < 0 or close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                        continue
                    if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ):
                        continue
                    pa, pm1, pp1 = float(close[a]), float(close[t - 1]), float(close[t + 1])
                    if not (np.isfinite(pm1) and np.isfinite(pp1) and pm1 > 0):
                        continue
                    runup = pm1 / pa - 1.0
                    printr = pp1 / pm1 - 1.0
                    sa, sb = spy_al[t - 1], spy_al[t + 1]
                    sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                    rows[K].append((runup, printr, printr - sr))
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    cost = COST_BPS / 1e4
    out = {}
    for K in KS:
        arr = np.array(rows[K], float)
        if len(arr) < 200:
            continue
        ru, pr, pe = arr[:, 0], arr[:, 1], arr[:, 2]
        qedges = np.quantile(ru, np.linspace(0, 1, NQ + 1))
        qidx = np.clip(np.searchsorted(qedges, ru, side="right") - 1, 0, NQ - 1)
        print(f"\n=== K={K}  RUN-UP quintiles -> does the drift CONTINUE through the print? (ADV>=100M) ===", flush=True)
        print(f"  {'quint':7}{'n':>7}{'runup%':>9}{'PRINT%':>9}{'PRINTmed':>10}{'PRINTwin%':>11}"
              f"{'exSPY%':>9}{'thru%':>9}", flush=True)
        cells = {}
        for q in range(NQ):
            m = qidx == q
            nq = int(m.sum())
            if nq < 20:
                continue
            rmean = ru[m].mean() * 100
            pmean = pr[m].mean() * 100; pmed = float(np.median(pr[m])) * 100
            pwin = (pr[m] > 0).mean() * 100; pex = float(np.nanmean(pe[m])) * 100
            # through-print total: hold from t-K close to t+1 close = (1+runup)(1+print)-1, net one round trip
            thru = ((1 + ru[m]) * (1 + pr[m]) - 1 - cost).mean() * 100
            label = f"Q{q+1}" + ("(top)" if q == NQ - 1 else "(bot)" if q == 0 else "")
            cells[f"Q{q+1}"] = dict(n=nq, runup=round(rmean, 3), print_mean=round(pmean, 3),
                                    print_med=round(pmed, 3), print_win=round(pwin, 1),
                                    print_excess=round(pex, 3), through_total=round(thru, 3))
            print(f"  {label:7}{nq:>7}{rmean:>9.2f}{pmean:>9.3f}{pmed:>10.3f}{pwin:>11.1f}{pex:>9.3f}{thru:>9.3f}", flush=True)

        # joint: top-quintile run-up AND positive print
        top = qidx == NQ - 1
        p_pos_given_top = (pr[top] > 0).mean() * 100
        p_pos_all = (pr > 0).mean() * 100
        # correlation runup vs print
        corr = float(np.corrcoef(ru, pr)[0, 1])
        print(f"  P(print>0 | top-quintile run-up) = {p_pos_given_top:.1f}%   vs baseline P(print>0) = {p_pos_all:.1f}%", flush=True)
        print(f"  corr(runup, print) = {corr:+.4f}  (>0 = continuation, <0 = reversal/priced-in)", flush=True)
        out[f"K{K}"] = dict(quintiles=cells, p_pos_given_top=round(p_pos_given_top, 1),
                            p_pos_baseline=round(p_pos_all, 1), corr_runup_print=round(corr, 4), n=len(arr))

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), liq=LIQ, cost_bps=COST_BPS, ks=KS, nq=NQ,
                   results=out, caveat="Run-up (close t-K -> t-1) vs through-print return (close t-1 -> t+1), by "
                   "run-up quintile. corr(runup,print) and P(print>0|top-quintile) test continuation vs priced-in. "
                   "through_total = hold t-K close to t+1 close net one round trip. ADV>=100M.")
    Path("/app/.data/studies/preearn_runup_cont.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_runup_cont",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_runup_cont]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
