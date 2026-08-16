#!/usr/bin/env python3
"""PIN the 376-vs-422 gap. Run the SAME accel-month-end value strategy with 3 monthly-panel builders in
ONE process and isolate exactly where they diverge (deterministic — every point must be accountable):

  A  _monthly_close (resample('ME').last())            <- the validated harness (+422)
  B  to_monthly, label = p.to_timestamp('M')           <- the rebalance_timing harness (+376) — suspect: label is month-START
  C  to_monthly, label = p.to_timestamp('M', how='end')<- corrected month-END label
Prints each harness's first labels, month count, common-universe size, and vsSPY. If B's labels are
month-START while A/C are month-END, the gap = a PIT-misalignment BUG (fundamentals looked up ~30d early),
and the true number is A/C.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/reconcile_harness.py
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

# sanity: what does to_timestamp('M') actually return?
_p = pd.Period("2021-08", "M")
print(f"Period('2021-08').to_timestamp('M') = {_p.to_timestamp('M')}  | how='end' = {_p.to_timestamp('M', how='end')}", flush=True)


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
    stock_daily = load_candles(all_holds)
    spy_daily = etf_daily[BENCH]["Close"]
    reps = load_financial_reports(all_holds)

    def to_monthly(c, how):
        out = {}
        for p, s in c.groupby(c.index.to_period("M")):
            s = s.dropna()
            if len(s):
                out[p.to_timestamp("M", how=how)] = float(s.iloc[-1])
        return pd.Series(out).sort_index()

    def panels(kind):
        if kind == "A":                       # _monthly_close (resample ME)
            etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
            stk_m = _monthly_close(stock_daily).reindex(etf_m.index)
            spy_m = spy_daily.resample("ME").last().reindex(etf_m.index)
        else:
            how = "start" if kind == "B" else "end"
            etf_m = pd.DataFrame({t: to_monthly(etf_daily[t]["Close"], how) for t in etf_tk if t in etf_daily}).sort_index()
            stk_m = pd.DataFrame({t: to_monthly(stock_daily[t]["Close"], how) for t in all_holds
                                  if t in stock_daily and len(stock_daily[t]) > 60}).reindex(etf_m.index)
            spy_m = to_monthly(spy_daily, how).reindex(etf_m.index)
        return etf_m, stk_m, spy_m

    def run(etf_m, stk_m, spy_m):
        midx = etf_m.index
        accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
        sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                          ("shares_outstanding", "total_equity", "net_income", "total_debt"))
        common = stk_m.columns.intersection(sh.columns).intersection(eq.columns)
        R = lambda p: p.reindex(index=midx, columns=common)
        px = stk_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
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
        r = np.array(rets); s = np.array(spies)
        tot = float(np.prod(1 + r) - 1) * 100 if len(r) else 0
        sp = float(np.prod(1 + s) - 1) * 100 if len(s) else 0
        return round(tot - sp, 1), len(common), len(midx), midx

    print("\n=== HARNESS RECONCILE (accel month-end value) ===", flush=True)
    for kind in ("A", "B", "C"):
        etf_m, stk_m, spy_m = panels(kind)
        vs, nc, nm, midx = run(etf_m, stk_m, spy_m)
        lab = {"A": "_monthly_close (resample ME)", "B": "to_monthly how='start' (label=month-START)",
               "C": "to_monthly how='end'   (label=month-END)"}[kind]
        print(f"  {kind}  {lab:44} vsSPY {vs:>7}%  | common {nc}  months {nm}  first-label {str(midx[0].date())}", flush=True)


if __name__ == "__main__":
    build()
