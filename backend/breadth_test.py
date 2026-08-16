#!/usr/bin/env python3
"""SHOULD WE HOLD MORE STOCKS PER SECTOR? Hold rotation+guard+low_debt fixed; instead of the single
cheapest-P/B name per strong sector, take the K CHEAPEST names (K in {1,2,3,5,all}), equal-weight
WITHIN the sector, equal-weight ACROSS the top-10 sectors (so sector allocation is held constant and
only within-sector breadth changes). K=1 must reproduce the +313%/+229% baseline (self-check).
Tells us the return/drawdown cost of diversifying away single-name risk.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/breadth_test.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

LOOKBACK, TOP_N = 6, 10
KS = [1, 2, 3, 5, 999]


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), t_stat=round(t, 2) if t is not None else None, periods=n)


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
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(R, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0

    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def run(K):
        rets, spies, nn = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            sector_rets, held = [], 0
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                if not use:
                    continue
                picks = pb.loc[date, use].nsmallest(K).index          # K cheapest P/B names
                rk = [_ret_delist(px[p], date, ndate) for p in picks]
                rk = [float(x) for x in rk if x is not None and np.isfinite(x)]
                if rk:
                    sector_rets.append(float(np.mean(rk))); held += len(rk)   # within-sector equal weight
            if sector_rets:
                rets.append(float(np.mean(sector_rets))); spies.append(float(sp)); nn.append(held)
        s = _stats(rets, spies)
        s["avg_total_names"] = round(float(np.mean(nn)), 1) if nn else 0
        return s

    res = {K: run(K) for K in KS}
    print("\n=== BREADTH: K cheapest-P/B names per sector (within-sector equal-wt) ===", flush=True)
    print(f"  {'K/sector':10} {'total':>8} {'vsSPY':>8} {'t':>6} {'Sharpe':>7} {'DD':>8} {'names':>6}", flush=True)
    for K in KS:
        s = res[K]; label = "all" if K == 999 else str(K)
        print(f"  {label:10} {s['total_return']:>7}% {s['vs_spy']:>7}% {str(s['t_stat']):>6} "
              f"{s['sharpe']:>7} {s['max_drawdown']:>7}% {s['avg_total_names']:>6}", flush=True)
    b = res[1]
    print(f"\n  [SELFCHECK] K=1 must ~= baseline 313%/+229%: tot {b['total_return']}% vsSPY {b['vs_spy']}%", flush=True)
    return res


if __name__ == "__main__":
    build()
