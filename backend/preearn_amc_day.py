#!/usr/bin/env python3
"""Is there an EXTRA pre-announcement session to capture on the ANNOUNCEMENT DAY for AfterMarket reports?
The run-up book sells at close[t-1] (t = first trading day >= report_date). For AfterMarket (AMC) prints the
announcement drops AFTER the close of day t -> day t is itself a full PRE-print session we currently skip.
For BeforeMarket (BMO) the reaction IS day t (announcement before the open), so day t is post-print.

Measure the day-t session return R_t = close[t]/close[t-1]-1, split by before_after:
  AMC  -> R_t is a PRE-print run-up day (candidate extra capture, sell close[t] instead of close[t-1])
  BMO  -> R_t is the reaction itself (NOT run-up; sanity check it's the gap)
Report mean/median/win/exSPY of R_t per timing. Then: does EXTENDING the AMC hold by one day (sell close[t]
not close[t-1]) improve the K=10 run-up book? Build BOTH books (AMC-extended vs baseline) and compare.
Also the AMC through-print R = close[t+1]/close[t]-1 (the real overnight print reaction) as a sanity check.
ADV>=100M, net 5bps. Persists BacktestResult[preearn_amc_day]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_amc_day.py"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

K = 10
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


def sstat(a, e):
    a = np.asarray(a, float) * 100; e = np.asarray(e, float) * 100
    a = a[np.isfinite(a)]
    if len(a) < 50:
        return None
    return dict(n=len(a), mean=round(float(a.mean()), 3), med=round(float(np.median(a)), 3),
                win=round(float((a > 0).mean() * 100), 1), excess=round(float(np.nanmean(e)), 3),
                t=round(float(a.mean() / (a.std() / math.sqrt(len(a)))), 2) if a.std() > 0 else 0.0)


def main():
    universe, _ = _universe()
    from core.models import EarningsEvent, Candle
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    # ticker -> list of (report_date, before_after)
    ev = defaultdict(list)
    for tk, rd, ba in EarningsEvent.objects.values_list("ticker", "report_date", "before_after"):
        if tk in have:
            ev[tk].append((pd.Timestamp(rd), ba or ""))
    for tk in ev:
        ev[tk] = sorted(set(ev[tk]))
    print(f"names with earnings dates: {len(names)}", flush=True)

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()
    spy_ret = spy_c.pct_change()

    # day-t session return R_t split by timing
    rt = defaultdict(list); rt_ex = defaultdict(list)
    # AMC print overnight reaction close[t]->close[t+1]
    amc_print = []; amc_print_ex = []
    # books: baseline (sell close[t-1] for all) vs amc-extended (AMC sell close[t])
    base = defaultdict(list); ext = defaultdict(list)
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
            for rd, ba in ev.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t >= n or t - 1 < 0:
                    continue
                a = t - K
                if a < 0 or close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                    continue
                if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ):
                    continue
                key = ba if ba in ("BeforeMarket", "AfterMarket") else "unknown"
                # day-t session return
                pm1, pt = float(close[t - 1]), float(close[t])
                if np.isfinite(pm1) and np.isfinite(pt) and pm1 > 0:
                    r = pt / pm1 - 1.0
                    sa, sb = spy_al[t - 1], spy_al[t]
                    sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                    rt[key].append(r); rt_ex[key].append(r - sr)
                # AMC overnight print reaction close[t]->close[t+1]
                if key == "AfterMarket" and t + 1 < n:
                    pp1 = float(close[t + 1])
                    if np.isfinite(pt) and np.isfinite(pp1) and pt > 0:
                        rr = pp1 / pt - 1.0
                        sa, sb = spy_al[t], spy_al[t + 1]
                        sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                        amc_print.append(rr); amc_print_ex.append(rr - sr)
                # books: baseline sells close[t-1]; extended sells close[t] IFF AfterMarket
                sell_end = t if key == "AfterMarket" else t - 1
                for bookc, end in ((base, t - 1), (ext, sell_end)):
                    for d in range(a + 1, end + 1):
                        v = dret[d]
                        if not np.isfinite(v):
                            continue
                        if d == a + 1:
                            v -= COST_HALF
                        if d == end:
                            v -= COST_HALF
                        bookc[dates[d]].append(v)
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    out = {}
    print(f"\n=== DAY-t SESSION return  R_t = close[t]/close[t-1]-1  by timing (ADV>=100M, K={K}) ===", flush=True)
    print(f"  {'timing':14}{'n':>7}{'mean%':>9}{'med%':>9}{'win%':>7}{'exSPY%':>9}{'t':>7}   interpretation", flush=True)
    interp = {"AfterMarket": "PRE-print run-up day (candidate extra capture)",
              "BeforeMarket": "reaction itself (the gap; NOT run-up)",
              "unknown": "timing unknown"}
    for key in ("AfterMarket", "BeforeMarket", "unknown"):
        s = sstat(rt.get(key, []), rt_ex.get(key, []))
        if not s:
            continue
        out[f"day_t_{key}"] = s
        print(f"  {key:14}{s['n']:>7}{s['mean']:>9.3f}{s['med']:>9.3f}{s['win']:>7.1f}{s['excess']:>9.3f}{s['t']:>7.2f}   {interp[key]}", flush=True)

    sp = sstat(amc_print, amc_print_ex)
    if sp:
        out["amc_overnight_print"] = sp
        print(f"\n  AMC overnight print close[t]->close[t+1]: n={sp['n']} mean {sp['mean']}% med {sp['med']}% "
              f"win {sp['win']}% exSPY {sp['excess']}%  (the real binary reaction)", flush=True)

    print(f"\n=== BOOK: baseline (sell close[t-1]) vs AMC-EXTENDED (AMC sells close[t]) — K={K} ADV>=100M net {COST_BPS}bps ===", flush=True)
    for label, c in (("baseline", base), ("amc_extended", ext)):
        bs = book_series(c)
        if bs is None:
            continue
        bp = perf(bs); spf = perf(spy_ret.reindex(bs.index).fillna(0.0))
        out[f"book_{label}"] = dict(book=bp, spy=spf)
        print(f"  {label:14} total {bp['total']:>8.1f}%  CAGR {bp['cagr']:>5.1f}%  Sh {bp['sharpe']:>4.2f}  "
              f"maxDD {bp['maxdd']:>6.1f}%  vsSPY {bp['cagr']-spf['cagr']:+.1f}pp", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), liq=LIQ, cost_bps=COST_BPS, k=K, results=out,
                   caveat="Day-t session return close[t-1]->close[t] split by before_after. AMC: day t is a PRE-"
                   "print session (candidate extra run-up capture); BMO: day t is the reaction. AMC-extended book "
                   "sells close[t] for AfterMarket names vs baseline close[t-1]. DAILY resolution (no intraday). ADV>=100M.")
    Path("/app/.data/studies/preearn_amc_day.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_amc_day",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_amc_day]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
