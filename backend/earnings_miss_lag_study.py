#!/usr/bin/env python3
"""EXIT X DAYS AFTER A MISS — earnings_cut_timing showed exact-day exit is the WORST (locks in the overreaction
bottom) and month-end is better (lets the drop bounce). So is there an optimal LAG? For every flagship pick that
hits an earnings MISS during its hold, trace the stock's path AFTER the print and find the exit lag N (trading
days after the report) that maximises the entry->exit return.

For each (pick, miss-in-hold) event:
  entry->print = close at print / close at pick date  (what we've earned up to the print)
  post(N)      = close at print+N / close at print     (the post-miss path: drop then bounce?)
  total(N)     = compounded entry->print->(print+N)     (what we'd book exiting N days after the print)
Average total(N) across events for N in {0(exact-day),1,2,3,5,7,10,15,21(~month-end)}. Peak N = best lag.
-> BacktestResult[earnings_miss_lag] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_miss_lag_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
MIN_DVOL = 5e6
LAGS = [0, 1, 2, 3, 5, 7, 10, 15, 21]
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "earnings_miss_lag.json"


def load_misses(tickers):
    from core.models import EarningsEvent
    out = {}
    for r in EarningsEvent.objects.filter(ticker__in=list(tickers)).values(
            "ticker", "report_date", "grounded_score", "eps_surprise_pct"):
        if r["report_date"] is None:
            continue
        gs, eps = r["grounded_score"], r["eps_surprise_pct"]
        is_miss = (gs is not None and gs < 0) or (gs is None and eps is not None and eps < -1.0)
        if is_miss:
            out.setdefault(r["ticker"], []).append(pd.Timestamp(r["report_date"]))
    for t in out:
        out[t].sort()
    return out


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol, DCL = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        DCL[t] = d["Close"]
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    miss = load_misses(common)
    print(f"months {len(midx)} | stocks {len(common)} | names with a miss {len(miss)}", flush=True)

    # collect miss-in-hold events across the flagship's picks
    events = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        for etf in top:
            name, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date)
                 and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if not g:
                continue
            pick = min(g, key=lambda h: pb.loc[date, h])
            md = next((rd for rd in miss.get(pick, []) if date < rd < ndate), None)
            if md is None:
                continue
            s = DCL.get(pick)
            if s is None or len(s) == 0:
                continue
            k = s.index.searchsorted(md, side="right")   # first close strictly after the report
            if k >= len(s):
                continue
            p_entry = s.asof(date); p_print = float(s.iloc[k - 1]) if k > 0 else s.asof(md)
            if not (np.isfinite(p_entry) and p_entry > 0):
                continue
            row = {"ticker": pick, "date": str(date.date()), "print": str(md.date())}
            ok = True
            for N in LAGS:
                j = k + N - 1 if N > 0 else k - 1     # N trading days after the print close (N=0 => print-day close)
                j = max(j, 0)
                if j >= len(s):
                    ok = False; break
                pN = float(s.iloc[j])
                row[f"total_{N}"] = (pN / p_entry - 1) if p_entry > 0 else None
                row[f"post_{N}"] = (pN / p_print - 1) if p_print > 0 else None
            if ok:
                events.append(row)

    def avg(key):
        v = [e[key] for e in events if e.get(key) is not None and np.isfinite(e[key])]
        return round(float(np.mean(v)) * 100, 2) if v else None

    total_curve = {str(N): avg(f"total_{N}") for N in LAGS}
    post_curve = {str(N): avg(f"post_{N}") for N in LAGS}
    best_N = max(LAGS, key=lambda N: (total_curve[str(N)] if total_curve[str(N)] is not None else -1e9))
    exact = total_curve["0"]; monthend = total_curve[str(LAGS[-1])]
    verdict = (
        f"{len(events)} miss-in-hold events. Entry->exit return by exit lag (days after the print): "
        + ", ".join(f"{N}d {total_curve[str(N)]}%" for N in LAGS) + f". Best lag = {best_N} days "
        f"({total_curve[str(best_N)]}%) vs exact-day {exact}% vs ~month-end {monthend}%. "
        + (f"Waiting ~{best_N} days after the miss beats both cutting on the day AND holding to month-end -> the "
           f"post-miss bounce peaks around day {best_N}, then fades." if best_N not in (0, LAGS[-1]) else
           ("Cutting on the exact day is best -> no bounce, just get out." if best_N == 0 else
            "Holding to ~month-end is best -> the bounce runs the whole month, no earlier peak."))
    )
    print(f"\n=== EXIT X DAYS AFTER A MISS ({len(events)} events) ===", flush=True)
    print("  lag(d):  " + "  ".join(f"{N:>3}" for N in LAGS), flush=True)
    print("  post %:  " + "  ".join(f"{post_curve[str(N)]!s:>5}" for N in LAGS) + "   (path from the print itself)", flush=True)
    print("  total%:  " + "  ".join(f"{total_curve[str(N)]!s:>5}" for N in LAGS) + "   (entry->exit booked)", flush=True)
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "lags_days": LAGS, "benchmark": BENCH,
                   "months": int(len(midx)), "pb_basis": "as-traded",
                   "miss_rule": "grounded_score<0 (else eps_surprise<-1%)"},
        "n_events": len(events), "total_return_by_lag_pct": total_curve, "post_miss_path_by_lag_pct": post_curve,
        "best_lag_days": best_N, "verdict": verdict,
        "caveat": "Single-name path from daily closes; exit at close of print-day+N (N=0 => print-day close). Does "
                  "NOT model opportunity cost of redeployed capital, and lags past month-end aren't realisable in "
                  "the monthly framework (shown for shape). grounded_score PIT. present-day-holdings survivorship.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="earnings_miss_lag", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                 "computed_at": timezone.now()})
        print("Saved BacktestResult[earnings_miss_lag]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
