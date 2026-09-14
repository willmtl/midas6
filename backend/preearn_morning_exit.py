#!/usr/bin/env python3
"""INTRADAY test of the user's idea: for AfterMarket (AMC) earnings, don't sell at close[t-1] and don't hold the
full announcement day -- SELL IN THE MORNING of day t (capture the overnight gap + morning drift into the AMC
print, then exit before the afternoon unwinds, still before the after-close print).

Uses cached 4h parquets (.data/intraday/4h/<tk>.parquet, EODHD 1h->4h, ~2021-08+). Each RTH day has a 12:00-UTC
'morning' bar (open~US open, close~noon ET) and a 16:00-UTC bar (close = daily close). 'Morning exit' = Close of
the 12:00-UTC bar on day t. Buy daily close[t-K].

Decompose (AMC events, ADV>=100M, K=10, ~2022-2026):
  LEG overnight+morning : morning_close[t] / daily_close[t-1] - 1   (the candidate extra capture)
  LEG afternoon         : daily_close[t]   / morning_close[t] - 1   (does the morning drift fade pre-print?)
Run-up TOTALS (buy daily close[t-K], net one 5bps round trip):
  exit_prevclose : sell daily close[t-1]        (BASELINE run-up book)
  exit_morning   : sell morning_close[t]        (USER'S IDEA)
  exit_dayclose  : sell daily close[t]          (full announcement day)
All demeaned vs SPY over the identical window (SPY 4h parquet + daily). Per-trade mean/median/win/t + by-year.
Persists BacktestResult[preearn_morning_exit]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_morning_exit.py"""
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
COST = 5 / 1e4
PARQ = Path("/app/.data/intraday/4h")


def bar_map(tk):
    """date -> (morning_close [12:00-UTC bar], regclose [16:00-UTC bar]) from the cached 4h parquet.
    Both from the SAME parquet => split-adjustment-consistent (intraday legs are pure ratios, so the raw-vs-
    adjusted basis cancels as long as no split falls inside the ~1-day leg). None if no usable parquet."""
    p = PARQ / f"{tk}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    df = df.copy(); df["d"] = df.index.tz_convert(None).normalize(); df["h"] = df.index.hour
    morn = df[df["h"] == 12].groupby("d")["Close"].last()
    reg = df[df["h"] == 16].groupby("d")["Close"].last()
    out = {}
    for d in morn.index.union(reg.index):
        out[d] = (float(morn.get(d, np.nan)), float(reg.get(d, np.nan)))
    return out or None


def sstat(a, e=None):
    a = np.asarray(a, float) * 100
    a = a[np.isfinite(a)]
    if len(a) < 30:
        return None
    d = dict(n=len(a), mean=round(float(a.mean()), 3), med=round(float(np.median(a)), 3),
             win=round(float((a > 0).mean() * 100), 1),
             t=round(float(a.mean() / (a.std() / math.sqrt(len(a)))), 2) if a.std() > 0 else 0.0)
    if e is not None:
        e = np.asarray(e, float) * 100; e = e[np.isfinite(e)]
        d["excess"] = round(float(e.mean()), 3) if len(e) else None
    return d


def main():
    universe, _ = _universe()
    from core.models import EarningsEvent, Candle
    amc = defaultdict(list)
    for tk, rd, ba in EarningsEvent.objects.values_list("ticker", "report_date", "before_after"):
        if ba == "AfterMarket":
            amc[tk].append(pd.Timestamp(rd))
    names = [t for t in universe if t in amc and (PARQ / f"{t}.parquet").exists()]
    print(f"AMC names with 4h parquet ∩ universe: {len(names)}", flush=True)

    # SPY daily + SPY intraday legs (same within-parquet basis)
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()
    spy_bars = bar_map("SPY") or {}

    def spy_legs(dt_prev, dt_t):
        """SPY (on-leg, af-leg) using its own parquet bars; (nan,nan) if unavailable."""
        bp = spy_bars.get(dt_prev); bt = spy_bars.get(dt_t)
        if not bp or not bt:
            return (np.nan, np.nan)
        pm, pr = bp; tm, tr = bt
        on = tm / pr - 1.0 if (np.isfinite(pr) and pr > 0 and np.isfinite(tm)) else np.nan
        af = tr / tm - 1.0 if (np.isfinite(tm) and tm > 0 and np.isfinite(tr)) else np.nan
        return (on, af)

    CLIP = 0.60      # drop legs beyond +-60% (residual split/bad-print rows)
    legs = defaultdict(list); legs_ex = defaultdict(list)
    totals = defaultdict(list); totals_ex = defaultdict(list)
    byyear = defaultdict(lambda: defaultdict(list))   # exit -> year -> [total]
    dropped = 0; done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            bm = bar_map(tk)
            if not bm:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            di = dates.values
            spy_al = spy_c.reindex(dates).ffill().values
            for rd in amc.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t >= n or t - 1 < 0:
                    continue
                a = t - K
                if a < 0 or close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                    continue
                if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ):
                    continue
                dt_t = pd.Timestamp(dates[t]).normalize(); dt_p = pd.Timestamp(dates[t - 1]).normalize()
                bp = bm.get(dt_p); bt = bm.get(dt_t)
                if not bp or not bt:
                    continue
                prev_reg = bp[1]; morn_t = bt[0]; reg_t = bt[1]
                if not (np.isfinite(prev_reg) and prev_reg > 0 and np.isfinite(morn_t) and morn_t > 0
                        and np.isfinite(reg_t) and reg_t > 0):
                    continue
                pa, pm1, pt = float(close[a]), float(close[t - 1]), float(close[t])
                if not (np.isfinite(pm1) and np.isfinite(pt) and pa > 0 and pm1 > 0):
                    continue
                # INTRADAY LEGS from the same parquet (adjustment-invariant ratios)
                l_on = morn_t / prev_reg - 1.0             # overnight + morning (close[t-1] -> noon[t])
                l_af = reg_t / morn_t - 1.0                # afternoon (noon[t] -> close[t])
                if abs(l_on) > CLIP or abs(l_af) > CLIP:
                    dropped += 1
                    continue
                s_on, s_af = spy_legs(dt_p, dt_t)
                yr = pd.Timestamp(dates[t]).year

                legs["overnight_morning"].append(l_on); legs["afternoon"].append(l_af)
                if np.isfinite(s_on):
                    legs_ex["overnight_morning"].append(l_on - s_on)
                if np.isfinite(s_af):
                    legs_ex["afternoon"].append(l_af - s_af)

                # daily run-up to close[t-1] (adjusted), then COMPOSE intraday legs on top
                ru = pm1 / pa - 1.0                         # baseline run-up gross
                sru = (spy_al[t - 1] / spy_al[a] - 1.0) if (spy_al[a] > 0 and np.isfinite(spy_al[t - 1])) else 0.0
                sdc = (spy_al[t] / spy_al[a] - 1.0) if (spy_al[a] > 0 and np.isfinite(spy_al[t])) else 0.0
                defs = {
                    "exit_prevclose": (ru, sru),
                    "exit_morning": ((1 + ru) * (1 + l_on) - 1.0, sru + (s_on if np.isfinite(s_on) else 0.0)),
                    "exit_dayclose": ((1 + ru) * (1 + l_on) * (1 + l_af) - 1.0, sdc),
                }
                for name, (gr, spx) in defs.items():
                    r = gr - COST
                    totals[name].append(r); byyear[name][yr].append(r)
                    totals_ex[name].append(r - spx)
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)
    print(f"dropped legs (|leg|>{CLIP}, split/bad-print): {dropped}", flush=True)

    out = {}
    print(f"\n=== INTRADAY LEG decomposition (AMC, K={K}, ADV>=100M, %/trade) ===", flush=True)
    print(f"  {'leg':20}{'n':>7}{'mean':>9}{'med':>9}{'win%':>7}{'exSPY':>9}{'t':>7}", flush=True)
    for lg in ("overnight_morning", "afternoon"):
        s = sstat(legs[lg], legs_ex.get(lg))
        if s:
            out[f"leg_{lg}"] = s
            print(f"  {lg:20}{s['n']:>7}{s['mean']:>9.3f}{s['med']:>9.3f}{s['win']:>7.1f}"
                  f"{s.get('excess',0):>9.3f}{s['t']:>7.2f}", flush=True)

    print(f"\n=== RUN-UP TOTAL by EXIT rule (buy daily close[t-K], net {int(COST*1e4)}bps round trip) ===", flush=True)
    print(f"  {'exit':16}{'n':>7}{'mean':>9}{'med':>9}{'win%':>7}{'exSPY':>9}{'t':>7}", flush=True)
    for ex in ("exit_prevclose", "exit_morning", "exit_dayclose"):
        s = sstat(totals[ex], totals_ex.get(ex))
        if s:
            out[ex] = s
            print(f"  {ex:16}{s['n']:>7}{s['mean']:>9.3f}{s['med']:>9.3f}{s['win']:>7.1f}"
                  f"{s.get('excess',0):>9.3f}{s['t']:>7.2f}", flush=True)

    print(f"\n=== BY YEAR — mean %/trade per exit ===", flush=True)
    yrs = sorted({y for ex in byyear for y in byyear[ex]})
    print("  year    " + "".join(f"{ex.replace('exit_',''):>13}" for ex in ("exit_prevclose", "exit_morning", "exit_dayclose")), flush=True)
    yr_out = {}
    for y in yrs:
        row = {}
        cells = []
        for ex in ("exit_prevclose", "exit_morning", "exit_dayclose"):
            arr = np.asarray(byyear[ex].get(y, []), float) * 100
            if len(arr) >= 15:
                row[ex] = dict(n=len(arr), mean=round(float(arr.mean()), 3))
                cells.append(f"{arr.mean():>8.3f}(n{len(arr)})")
            else:
                cells.append(f"{'--':>13}")
        yr_out[int(y)] = row
        print(f"  {y}  " + "".join(f"{c:>13}" for c in cells), flush=True)
    out["by_year"] = yr_out

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), liq=LIQ, k=K, cost_bps=5, results=out,
                   caveat="Intraday morning-exit for AMC earnings run-up. Morning = 12:00-UTC 4h bar close (~noon "
                   "ET) on day t. Legs overnight+morning / afternoon; run-up totals under 3 exit rules, demeaned "
                   "vs SPY. 4h parquet coverage ~2021-08+ (SHORT, no pre-COVID; suggestive-only). ADV>=100M AMC.")
    Path("/app/.data/studies/preearn_morning_exit.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_morning_exit",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_morning_exit]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
