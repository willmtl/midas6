#!/usr/bin/env python3
"""HOW FRAGILE is the headline total return? A concentrated strategy (1 stock/sector) can have its total
return driven by a handful of months/picks -> small data changes flip a pick -> big total swing. Quantify:
compute the accel-value monthly return series, then show how much the TOTAL depends on the few best months
(leave-one-out / leave-top-3-out) and the single-month contribution spread. If dropping 1-3 months of 52
moves the total by tens of points, the headline is FRAGILE (trust Sharpe/relative instead).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fragility_check.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis

TOP_N = 10


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
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_m = _monthly_close(load_candles(all_holds)).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0

    rets, spies = [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranks = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        slot = []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            g = [c for c in cands if bool(low.loc[date, c])] or cands
            if not g:
                continue
            r = _ret_delist(px[pb.loc[date, g].idxmin()], date, ndate)
            if r is not None and np.isfinite(r):
                slot.append(float(r))
        if slot:
            rets.append(float(np.mean(slot))); spies.append(float(sp))
    r = np.array(rets); s = np.array(spies); n = len(r)
    tot = lambda x: (np.prod(1 + x) - 1) * 100
    full = tot(r) - tot(s)
    # leave-one-month-out spread
    loo = [tot(np.delete(r, i)) - tot(np.delete(s, i)) for i in range(n)]
    order = np.argsort(r)[::-1]                      # best months first
    ex_top1 = tot(np.delete(r, order[:1])) - tot(np.delete(s, order[:1]))
    ex_top3 = tot(np.delete(r, order[:3])) - tot(np.delete(s, order[:3]))
    print(f"\n=== FRAGILITY of the accel-value total return (n={n} months) ===", flush=True)
    print(f"  full vs-SPY:                 {full:.1f}%", flush=True)
    print(f"  best 3 monthly returns:      {sorted((r*100).round(1), reverse=True)[:3]}", flush=True)
    print(f"  drop single BEST month  ->   {ex_top1:.1f}%   (loses {full-ex_top1:.1f}pp from ONE month)", flush=True)
    print(f"  drop top-3 best months  ->   {ex_top3:.1f}%   (loses {full-ex_top3:.1f}pp from THREE)", flush=True)
    print(f"  leave-one-out range:         {min(loo):.1f}% .. {max(loo):.1f}%  (spread {max(loo)-min(loo):.1f}pp)", flush=True)
    print(f"  median monthly return:       {np.median(r)*100:.2f}%  | mean {r.mean()*100:.2f}%  (mean>>median = few big months carry it)", flush=True)
    print(f"\n  => the TOTAL is {'FRAGILE — driven by a few months; trust Sharpe/median/relative, not the headline %' if (full-ex_top3) > 60 else 'reasonably robust'}", flush=True)


if __name__ == "__main__":
    build()
