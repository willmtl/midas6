#!/usr/bin/env python3
"""DECISIVE GATE for the pre-earnings RUN-UP: build a real, tradable, daily EQUAL-WEIGHT book of run-up positions
in the UNDERLYING STOCK and measure the net-of-cost equity curve vs buy-hold SPY. The play: buy at close t-K
(K trading days before the earnings report_date), sell at close t-1 (day before the print). Overlapping positions
across the liquid universe are equal-weighted each day; in cash when no position is on. Round-trip cost charged on
entry+exit day. Survivorship-aware (uses bars that exist). Full earnings history (no options needed).
PIT NOTE: uses report_date at t-K, i.e. assumes the earnings date is known ~K days ahead — true for the
scheduled 'earnings announcement premium' (companies pre-announce dates weeks out); K=5 is safest, K=10 mild.
Variants: liquidity floor (ADV) x K x cost. By-year book-vs-SPY. Persists BacktestResult[preearn_runup_book]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_runup_book.py"""
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
LIQ_FLOORS = [100e6, 500e6]
COSTS = [5, 10]                 # bps round-trip; headline = 5 (big liquid names)


def perf(dret):
    dret = dret.dropna()
    if len(dret) < 60:
        return None
    eq = (1 + dret.values).cumprod(); yrs = len(dret) / 252.0
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    sh = dret.mean() / dret.std() * math.sqrt(252) if dret.std() > 0 else 0.0
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return dict(days=len(dret), start=str(dret.index[0].date()), end=str(dret.index[-1].date()),
                total=round((eq[-1] - 1) * 100, 1), cagr=round(cagr, 1), sharpe=round(sh, 2), maxdd=round(dd, 1))


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
    spy_ret = spy_c.pct_change()

    # accumulate daily contributions per (K, liq, cost): date -> list of cost-adjusted daily position returns
    contrib = {(K, L, C): defaultdict(list) for K in KS for L in LIQ_FLOORS for C in COSTS}
    n_events = {(K, L): 0 for K in KS for L in LIQ_FLOORS}
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            dret = pd.Series(close, index=dates).pct_change().values; di = dates.values
            for rd in edates.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t >= n:
                    continue
                for K in KS:
                    a = t - K; b = t - 1
                    if a < 0 or b <= a:
                        continue
                    if close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                        continue
                    dv = dvol20[a] if np.isfinite(dvol20[a]) else 0.0
                    for L in LIQ_FLOORS:
                        if dv < L:
                            continue
                        n_events[(K, L)] += 1
                        for C in COSTS:
                            half = (C / 2) / 1e4
                            for d in range(a + 1, b + 1):
                                r = dret[d]
                                if not np.isfinite(r):
                                    continue
                                if d == a + 1:
                                    r -= half
                                if d == b:
                                    r -= half
                                contrib[(K, L, C)][dates[d]].append(r)
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    out = {}
    print(f"\n=== PRE-EARNINGS RUN-UP STOCK BOOK — net-of-cost equity curve vs SPY ===", flush=True)
    print(f"  {'K':>3}{'liqFloor':>10}{'cost':>6}{'events':>8}{'inv%':>6}{'total%':>10}{'CAGR%':>8}"
          f"{'Sh':>6}{'maxDD%':>8}{'SPYtot%':>9}{'SPYcagr':>8}", flush=True)
    for K in KS:
        for L in LIQ_FLOORS:
            for C in COSTS:
                days = sorted(contrib[(K, L, C)])
                if len(days) < 60:
                    continue
                book = pd.Series([np.mean(contrib[(K, L, C)][d]) for d in days], index=pd.DatetimeIndex(days)).sort_index()
                full = pd.date_range(book.index[0], book.index[-1], freq="B")
                book = book.reindex(full).fillna(0.0)
                bp = perf(book)
                spy_al = spy_ret.reindex(book.index).fillna(0.0)
                sp = perf(spy_al)
                inv = (book != 0).mean() * 100
                key = f"K{K}|L{int(L/1e6)}M|C{C}"
                out[key] = dict(events=n_events[(K, L)], invested_pct=round(inv, 1), book=bp, spy=sp)
                print(f"  {K:>3}{int(L/1e6):>8}M{C:>6}{n_events[(K,L)]:>8}{inv:>6.0f}{bp['total']:>10.1f}"
                      f"{bp['cagr']:>8.1f}{bp['sharpe']:>6.2f}{bp['maxdd']:>8.1f}{sp['total']:>9.1f}{sp['cagr']:>8.1f}", flush=True)

    # headline by-year: K=10, L=100M, C=5
    hk = (10, 100e6, 5)
    days = sorted(contrib[hk])
    book = pd.Series([np.mean(contrib[hk][d]) for d in days], index=pd.DatetimeIndex(days)).sort_index()
    full = pd.date_range(book.index[0], book.index[-1], freq="B"); book = book.reindex(full).fillna(0.0)
    spy_al = spy_ret.reindex(book.index).fillna(0.0)
    byb = (1 + book).groupby(book.index.year).prod() - 1
    bys = (1 + spy_al).groupby(spy_al.index.year).prod() - 1
    print(f"\n=== BY YEAR — headline book K=10 ADV>=100M cost5bps vs SPY ===", flush=True)
    print(f"  {'year':>6}{'book%':>9}{'spy%':>9}{'excess':>9}", flush=True)
    yr = {}
    for y in byb.index:
        bv = byb[y] * 100; sv = bys.get(y, np.nan) * 100
        yr[int(y)] = dict(book=round(bv, 1), spy=round(sv, 1), excess=round(bv - sv, 1))
        print(f"  {y:>6}{bv:>9.1f}{sv:>9.1f}{bv-sv:>9.1f}", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), ks=KS, liq_floors=LIQ_FLOORS, costs=COSTS,
                   variants=out, headline_by_year=yr,
                   caveat="Daily EW book of pre-earnings run-up positions in the STOCK: buy close t-K, sell close "
                   "t-1; overlapping EW; round-trip cost on entry+exit; in cash when none on. Survivorship-aware. "
                   "PIT assumes earnings date known ~K days ahead (scheduled announcement premium; K=5 safest).")
    Path("/app/.data/studies/preearn_runup_book.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_runup_book",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_runup_book]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
