#!/usr/bin/env python3
"""VALUE RANKING LAB — chase more return by changing WHICH value metric picks the stock.

C picks the cheapest positive-P/B name in each strong sector (passing guard+low_debt). Book value can
be distorted; maybe a cash-flow or earnings lens picks a better name. Hold selection FIXED (rotation +
guard + low_debt) and vary ONLY the value metric used to choose the one pick per sector:

  pb            cheapest P/B (baseline = current C)
  ev_ebit       cheapest EV/EBIT (enterprise value / operating income; positive EBIT only)
  fcf_yield     highest FCF / EV (cash-flow cheapness)
  earn_yield    highest net income / market cap (earnings cheapness)
  ps            cheapest P/S (market cap / revenue)
  composite     best average z-rank of (cheap P/B + cheap EV/EBIT + high FCF-yield)

All PIT from FinancialReport (operating_income / total_debt / cash / revenue / net_income / free_cash_
flow / total_equity / shares). Monthly, equal-weight, top-momentum sectors. Reports vs-SPY / t / Sharpe
/ drawdown per metric (no-fallback + fallback), ranks, finds the best return.
-> BacktestResult[value_ranking] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/value_ranking_lab.py  (--limit 300)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "value_ranking.json"
LOOKBACK, TOP_N = 6, 10
METRICS = ["pb", "ev_ebit", "fcf_yield", "earn_yield", "ps", "composite"]


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return {"total_return": 0, "vs_spy": 0, "sharpe": 0, "max_drawdown": 0, "t_stat": None, "periods": 0}
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(tot, 1), "vs_spy": round(tot - sp, 1), "sharpe": round(sh, 2),
            "max_drawdown": round(dd, 1), "t_stat": round(t, 2) if t is not None else None, "periods": n}


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    if limit:
        all_holds = all_holds[:limit]; hs = set(all_holds)
        sector_map = {e: (n, [h for h in hh if h in hs]) for e, (n, hh) in sector_map.items()}

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    opinc, cash, rev, fcf = P("operating_income"), P("cash_and_equivalents"), P("revenue"), P("free_cash_flow")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt, opinc, cash, rev, fcf = map(R, (shares, equity, ni, debt, opinc, cash, rev, fcf))
    mktcap = px * shares
    ev = mktcap + debt.fillna(0) - cash.fillna(0)
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0

    # metric panels — LOWER = better (negate the yields so argmin picks the best)
    ev_ebit = (ev / opinc.where(opinc > 0))
    fcf_yield = -(fcf / ev.where(ev > 0))
    earn_yield = -(ni / mktcap.where(mktcap != 0))
    ps = (mktcap / rev.where(rev > 0))

    def _z(p):
        return (p.sub(p.mean(axis=1), axis=0)).div(p.std(axis=1).replace(0, np.nan), axis=0)
    composite = _z(pb) + _z(ev_ebit) + _z(fcf_yield)          # all "lower=better" -> sum of z, pick min

    MP = {"pb": pb, "ev_ebit": ev_ebit, "fcf_yield": fcf_yield, "earn_yield": earn_yield,
          "ps": ps, "composite": composite}

    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def run(metric, fallback=False):
        mp = MP[metric]
        rets, spies = [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0
                         and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                use = [c for c in use if c in mp.columns and pd.notna(mp.loc[date, c])]
                pick = mp.loc[date, use].idxmin() if use else None
                if pick is not None:
                    r = _ret_delist(px[pick], date, ndate)
                    if r is not None and np.isfinite(r):
                        slot.append(float(r)); continue
                if fallback and etf in etf_m.columns:
                    er = _ret_delist(etf_m[etf], date, ndate)
                    if er is not None and np.isfinite(er):
                        slot.append(float(er))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        return _stats(rets, spies)

    results = {m: {"no_fallback": run(m, False), "fallback": run(m, True)} for m in METRICS}
    for m in METRICS:
        print(f"  {m:11} vsSPY {results[m]['no_fallback']['vs_spy']:>7}%  t={str(results[m]['no_fallback']['t_stat']):>5}  "
              f"Sh {results[m]['no_fallback']['sharpe']:>5}  DD {results[m]['no_fallback']['max_drawdown']:>6}%  "
              f"| fb {results[m]['fallback']['vs_spy']}%", flush=True)

    base = results["pb"]["no_fallback"]["vs_spy"]
    for m in METRICS:
        results[m]["vs_baseline"] = round(results[m]["no_fallback"]["vs_spy"] - base, 1)
    ranked = sorted(METRICS, key=lambda m: results[m]["no_fallback"]["vs_spy"], reverse=True)
    best = ranked[0]
    beats = results[best]["no_fallback"]["vs_spy"] > base

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_sectors": TOP_N, "benchmark": BENCH,
                   "months": int(len(midx)), "limit": limit,
                   "note": "rotation + guard + low_debt fixed; only the value metric that picks the stock varies"},
        "baseline": "pb", "best_metric": best, "beats_baseline": bool(beats), "ranking": ranked,
        "results": results,
        "legend": {"pb": "cheapest price/book (baseline)", "ev_ebit": "cheapest EV/EBIT",
                   "fcf_yield": "highest free-cash-flow yield (FCF/EV)", "earn_yield": "highest earnings yield (NI/mktcap)",
                   "ps": "cheapest price/sales", "composite": "best z-rank of cheap P/B + cheap EV/EBIT + high FCF-yield"},
        "verdict": (f"Best value metric = {best} ({results[best]['no_fallback']['vs_spy']}% vs SPY) "
                    f"{'BEATS' if beats else 'does NOT beat'} cheapest-P/B baseline ({base}%). "
                    + ("A cash-flow/earnings lens picks a better name than book." if beats
                       else "Cheapest P/B remains the best single value metric; the others don't add return.")),
        "caveat": ("PIT from FinancialReport; directional/no-fees; ~5y single regime. EV/EBIT needs positive "
                   "EBIT (excludes those names for that metric). Multiple metrics tested -> trust only if the "
                   "winner's edge is economically coherent + survives out-of-sample."),
    }
    return payload


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="value_ranking",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[value_ranking]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + payload["verdict"], flush=True)


if __name__ == "__main__":
    main()
