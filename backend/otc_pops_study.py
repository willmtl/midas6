#!/usr/bin/env python3
"""REMOVE OTC + profile the tradeable pops. No exchange field -> use a $-VOLUME floor as the OTC/
untradeable filter (AYRWF trades $0/day). (1) Re-run the accel-value engine excluding picks below a
$-vol floor (1/2/5M) -> honest tradeable return (removing bananas you couldn't actually buy). (2) Profile
the TRADEABLE big pops (pick ret > +40%, $vol > $2M): what do they share -> market cap, prior-12mo return
(beaten down?), P/B, sector. Answers 'what do the non-OTC pops have in common'.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/otc_pops_study.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter
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
    sector_map, all_holds, name_of = {}, set(), {}
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
        name_of[e] = n
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
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
    mktcap = px * sh
    prior12 = px.pct_change(12)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)

    def run(floor, etf_of=None):
        rets, spies, pops = [], [], []
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
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                         and (floor == 0 or (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= floor))]
                g = [c for c in cands if bool(low.loc[date, c])] or cands
                if not g:
                    continue
                t = pb.loc[date, g].idxmin()
                r = _ret_delist(px[t], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
                    if etf_of is not None and r > 0.4:
                        pops.append(dict(tk=t, ret=r, sector=name_of.get(etf, etf),
                                         mktcap=float(mktcap.loc[date, t]) if pd.notna(mktcap.loc[date, t]) else None,
                                         prior12=float(prior12.loc[date, t]) if pd.notna(prior12.loc[date, t]) else None,
                                         pb=float(pb.loc[date, t]), dvol=float(dvol.loc[date, t]) if pd.notna(dvol.loc[date, t]) else 0))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        vs = (np.prod(1 + np.array(rets)) - 1) * 100 - (np.prod(1 + np.array(spies)) - 1) * 100
        return round(vs, 1), pops

    base, _ = run(0)
    print(f"\n=== REMOVE OTC ($-volume floor) ===", flush=True)
    print(f"  no floor:        vsSPY {base}%", flush=True)
    for fl in (1e6, 2e6, 5e6, 10e6, 25e6, 50e6):
        vs, _ = run(fl)
        print(f"  min ${int(fl/1e6):>2}M/day:  vsSPY {vs:>6}%  ({'+' if vs-base>=0 else ''}{round(vs-base,1)}pp)", flush=True)

    # profile TRADEABLE pops (>$2M/day, ret>40%)
    _, pops = run(2e6, etf_of=True)
    pops = [p for p in pops if p["dvol"] >= 2e6]
    caps = [p["mktcap"] for p in pops if p["mktcap"]]
    p12 = [p["prior12"] for p in pops if p["prior12"] is not None]
    pbs = [p["pb"] for p in pops]
    print(f"\n=== WHAT THE TRADEABLE POPS SHARE (n={len(pops)}, ret>+40%, >$2M/day) ===", flush=True)
    print(f"  market cap: median ${np.median(caps)/1e9:.2f}B  ({sum(c<2e9 for c in caps)}/{len(caps)} are small-cap <$2B)", flush=True)
    print(f"  prior 12mo return: median {np.median(p12)*100:.0f}%  ({sum(x<0 for x in p12)}/{len(p12)} were DOWN/beaten-down going in)", flush=True)
    print(f"  P/B at entry: median {np.median(pbs):.2f}  ({sum(x<1 for x in pbs)}/{len(pbs)} below book value)", flush=True)
    print(f"  top sectors: {Counter(p['sector'] for p in pops).most_common(8)}", flush=True)
    print(f"\n  biggest tradeable pops:", flush=True)
    for p in sorted(pops, key=lambda x: -x["ret"])[:10]:
        print(f"    {p['tk']:9} {p['ret']*100:+.0f}%  {p['sector'][:18]:18} cap ${p['mktcap']/1e9:.2f}B  prior12 {p['prior12']*100 if p['prior12'] is not None else 0:+.0f}%  P/B {p['pb']:.2f}", flush=True)


if __name__ == "__main__":
    build()
