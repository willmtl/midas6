#!/usr/bin/env python3
"""HOW MUCH OF THE RETURN IS A FEW 'BANANA' PICKS? Each month holds ~9 equal-weight picks; over the
backtest that's ~470 individual pick-returns. Measure how concentrated the edge is in a handful of
monster winners: (1) list the biggest individual picks (ticker/month/return); (2) WINSORIZE — cap every
pick's monthly return at +25/50/100% and recompute the total, so the drop = the return that came from
picks running ABOVE the cap; (3) top-ticker contribution. If capping at +50% guts the total, the edge
rides on a few bananas.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/stock_concentration.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
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

    months, spies, picks = [], [], []          # months[i] = list of (ticker, ret) ; picks = flat (date,tk,ret)
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
            t = pb.loc[date, g].idxmin()
            r = _ret_delist(px[t], date, ndate)
            if r is not None and np.isfinite(r):
                slot.append((t, float(r))); picks.append((date, t, float(r)))
        if slot:
            months.append(slot); spies.append(float(sp))

    def total(cap=None):
        rs = []
        for slot in months:
            vals = [min(r, cap) if cap is not None else r for _, r in slot]
            rs.append(float(np.mean(vals)))
        return (np.prod(1 + np.array(rs)) - 1) * 100 - (np.prod(1 + np.array(spies)) - 1) * 100

    base = total()
    npick = len(picks)
    # biggest individual picks
    top = sorted(picks, key=lambda x: -x[2])[:12]
    # per-ticker: total contribution proxy = sum of that ticker's pick returns / n_months (rough attribution)
    tk_ret = defaultdict(list)
    for _, t, r in picks:
        tk_ret[t].append(r)
    tk_sum = sorted(((t, sum(rs), len(rs)) for t, rs in tk_ret.items()), key=lambda x: -x[1])[:10]

    print(f"\n=== STOCK CONCENTRATION ({npick} individual picks over {len(months)} months) ===", flush=True)
    print(f"  full vs-SPY: {base:.1f}%", flush=True)
    print("\n  WINSORIZE (cap each pick's monthly return):", flush=True)
    for c in (0.25, 0.50, 1.00):
        t = total(c)
        print(f"    cap +{int(c*100)}%  -> vs-SPY {t:.1f}%   ({base-t:.1f}pp of the edge came from picks that ran ABOVE +{int(c*100)}%)", flush=True)
    print(f"\n  biggest single picks (the bananas):", flush=True)
    for d, t, r in top:
        print(f"    {str(d.date())}  {t:9} {r*100:+.0f}%", flush=True)
    print(f"\n  top tickers by summed pick-return (times picked):", flush=True)
    for t, s, n in tk_sum:
        print(f"    {t:9} sum {s*100:+.0f}%  ({n} picks)", flush=True)
    nbanana = sum(1 for _, _, r in picks if r > 0.5)
    print(f"\n  picks that >50% in a month: {nbanana}/{npick} ({nbanana/npick*100:.1f}%)", flush=True)
    print(f"  => {'FRAGILE: a few monster picks drive it — cap +50% removes '+str(round(base-total(0.5),0))+'pp' if (base-total(0.5))>80 else 'reasonably spread across picks'}", flush=True)


if __name__ == "__main__":
    build()
