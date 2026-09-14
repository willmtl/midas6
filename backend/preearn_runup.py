#!/usr/bin/env python3
"""PRE-EARNINGS RUN-UP study (user: short-term, 'right before earnings we want a big pop, ideally big stocks').
Uses EarningsEvent.report_date (EODHD announcement date, PIT). Two short-term plays, SEGMENTED by cap bucket:
  RUNUP_K : buy close of day (t-K), sell close of day (t-1)  -- pure pre-announcement drift, NO binary risk
  THROUGH : buy close (t-1), sell close (t+1)                -- holds through the print, captures the gap pop
report_date mapped to first trading day >= report_date = t. Excludes future-dated events (t+1 must exist).
Metrics per (cap x play): n, mean, median, win%, p90, UP-TAIL mean (top decile -> 'big pop' magnitude),
mean SPY-excess over the identical window (strip beta). Net of COST_BPS round-trip. By-year on best large cell.
Survivorship: uses only bars that exist; delisted names contribute their real pre-earnings windows.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_runup.py"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, DVOL_FLOOR, PRICE_FLOOR

RUNUP_KS = [3, 5, 10]
COST_BPS = 10                     # large/mega-cap round-trip (tight spreads)
CAPS = ["ADV>500M", "ADV100-500M", "ADV20-100M", "ADV5-20M"]   # option-liquidity tiers ($ADV proxy)


def cap_bucket(dvol):
    """Segment by 20d avg DOLLAR volume = optionability proxy (tight spreads + OI track $volume)."""
    if dvol is None or not np.isfinite(dvol) or dvol <= 0:
        return "cap?"
    m = dvol / 1e6
    return "ADV>500M" if m >= 500 else "ADV100-500M" if m >= 100 else "ADV20-100M" if m >= 20 else "ADV5-20M"


def shares_ff(reps, tk, dates):
    r = reps.get(tk)
    if r is None or "shares_outstanding" not in r.columns:
        return None
    d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
    if d.empty:
        return None
    s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates).values


def main():
    universe, delisted = _universe()
    from core.models import EarningsEvent, Candle
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    print(f"universe {len(universe)} | with earnings dates {len(have)} | tradable ∩ {len(names)}", flush=True)

    # SPY series for same-window demeaning
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()

    # earnings dates per ticker
    ev_qs = EarningsEvent.objects.values_list("ticker", "report_date")
    edates = defaultdict(list)
    for tk, rd in ev_qs:
        edates[tk].append(pd.Timestamp(rd))

    # rec accumulators: (cap, play) -> list of net returns ; parallel list of spy-excess ; year rows for best cell
    ret = defaultdict(list); exc = defaultdict(list)
    byyear = defaultdict(lambda: defaultdict(list))   # (cap,play) -> year -> [net]
    cost = COST_BPS / 1e4
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            sh = shares_ff(reps, tk, dates)
            spy_al = spy_c.reindex(dates).ffill().values
            di = dates.values
            for rd in edates.get(tk, []):
                # t = first trading day >= report_date
                pos = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if pos >= n:                       # future-dated / beyond data
                    continue
                t = pos
                ep_ref = float(close[t]) if t < n else None
                if ep_ref is None or ep_ref <= PRICE_FLOOR or not (dvol20[t] >= DVOL_FLOOR):
                    continue
                cap = cap_bucket(float(dvol20[t]) if np.isfinite(dvol20[t]) else None)
                yr = pd.Timestamp(dates[t]).year
                # RUN-UP plays: buy close t-K, sell close t-1  (both strictly before the print)
                for K in RUNUP_KS:
                    a = t - K; b = t - 1
                    if a < 0 or b <= a:
                        continue
                    pa, pb = float(close[a]), float(close[b])
                    if pa <= PRICE_FLOOR or not np.isfinite(pa) or not np.isfinite(pb):
                        continue
                    r = (pb - pa) / pa - cost
                    sa, sb = spy_al[a], spy_al[b]
                    sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                    play = f"runup{K}"
                    ret[(cap, play)].append(r); exc[(cap, play)].append(r - sr)
                    byyear[(cap, play)][yr].append(r)
                # THROUGH: buy close t-1, sell close t+1  (holds through the print)
                if t - 1 >= 0 and t + 1 < n:
                    pa, pb = float(close[t - 1]), float(close[t + 1])
                    if pa > PRICE_FLOOR and np.isfinite(pa) and np.isfinite(pb):
                        r = (pb - pa) / pa - cost
                        sa, sb = spy_al[t - 1], spy_al[t + 1]
                        sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                        ret[(cap, "through")].append(r); exc[(cap, "through")].append(r - sr)
                        byyear[(cap, "through")][yr].append(r)
        done += len(ch)
        print(f"  scanned {done}/{len(names)}  cells {len(ret)}", flush=True)

    def stats(arr, earr):
        a = np.asarray(arr, float) * 100; e = np.asarray(earr, float) * 100
        a = a[np.isfinite(a)]
        if len(a) < 50:
            return None
        up = a[a >= np.percentile(a, 90)]
        return dict(n=len(a), mean=round(a.mean(), 3), med=round(float(np.median(a)), 3),
                    win=round((a > 0).mean() * 100, 1), p10=round(float(np.percentile(a, 10)), 2),
                    p90=round(float(np.percentile(a, 90)), 2),
                    uptail=round(float(up.mean()), 2), excess=round(float(np.nanmean(e)), 3),
                    std=round(float(a.std()), 3), sh=round(a.mean() / a.std(), 3) if a.std() > 0 else 0.0)

    plays = [f"runup{K}" for K in RUNUP_KS] + ["through"]
    out = {}
    print("\n=== PRE-EARNINGS short-term plays by CAP (net {}bps, %/trade) ===".format(COST_BPS), flush=True)
    print(f"  {'liq_tier':14}{'play':9}{'n':>7}{'mean':>8}{'med':>8}{'win%':>7}{'p10':>7}{'p90':>7}{'uptail':>8}{'exSPY':>8}", flush=True)
    for cap in CAPS + ["cap?"]:
        for play in plays:
            s = stats(ret.get((cap, play), []), exc.get((cap, play), []))
            if not s:
                continue
            out[f"{cap}|{play}"] = s
            print(f"  {cap:14}{play:9}{s['n']:>7}{s['mean']:>8.3f}{s['med']:>8.3f}{s['win']:>7.1f}"
                  f"{s['p10']:>7.2f}{s['p90']:>7.2f}{s['uptail']:>8.2f}{s['excess']:>8.3f}", flush=True)

    # by-year robustness: best mega/large cell by mean among run-ups
    cand = [(cap, play) for cap in ("ADV>500M", "ADV100-500M") for play in plays
            if (cap, play) in ret and len(ret[(cap, play)]) >= 200]
    cand.sort(key=lambda cp: -np.mean(ret[cp]))
    yr_out = {}
    if cand:
        best = cand[0]
        print(f"\n=== BY-YEAR robustness — best large cell: {best[0]} / {best[1]} (net %/trade) ===", flush=True)
        print(f"  {'year':>6}{'n':>7}{'mean':>9}{'win%':>8}", flush=True)
        for y in sorted(byyear[best]):
            a = np.asarray(byyear[best][y], float) * 100
            if len(a) < 20:
                continue
            yr_out[int(y)] = dict(n=len(a), mean=round(a.mean(), 3), win=round((a > 0).mean() * 100, 1))
            print(f"  {y:>6}{len(a):>7}{a.mean():>9.3f}{(a > 0).mean() * 100:>8.1f}", flush=True)
        out["_best_cell"] = f"{best[0]}|{best[1]}"; out["_best_by_year"] = yr_out

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), cost_bps=COST_BPS, runup_ks=RUNUP_KS,
                   cells=out, caveat="Pre-earnings run-up (buy t-K, sell t-1) + through-event (t-1..t+1) by cap "
                   "bucket. report_date -> first trading day >= it = t. Net of cost, demeaned vs SPY same window. "
                   "Future-dated events excluded. Not episode-deduped (each earnings date is its own event).")
    Path("/app/.data/studies/preearn_runup.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_runup",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_runup]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
