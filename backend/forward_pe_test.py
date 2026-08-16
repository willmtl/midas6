#!/usr/bin/env python3
"""TEST FORWARD P/E as the value metric — is it worth sourcing an analyst-estimate feed?

We have NO history of analyst estimates, so real forward P/E is not PIT-backtestable. Instead run the
PERFECT-FORESIGHT upper bound: at each rebalance, 'forward earnings' = the ticker's ACTUAL net income
reported over the NEXT 12 months (sum of next 4 quarters). fwd_pe = mktcap / fwd_ni (positive only).
This is deliberately LOOKAHEAD-BIASED — it's the best forward P/E could ever do. Compared against:
  pb       cheapest P/B (validated baseline, ~+229% vs SPY)
  ttm_pe   cheapest TRAILING P/E (real, PIT-honest: only reports available at the date)
  fwd_pe   cheapest FORWARD P/E (perfect-foresight, lookahead -> UPPER BOUND)

Same selection harness as value_ranking: top-6mo-momentum sectors, guard (ex-trap), low-debt soft
preference, cheapest-by-metric pick, monthly equal-weight, vs SPY. If fwd_pe (even cheating) does NOT
clear pb, forward P/E is hopeless. If it crushes pb, a real (noisy) estimate feed MIGHT add value.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/forward_pe_test.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

LOOKBACK, TOP_N = 6, 10


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


def _ntm_panel(reps, midx, common, forward, days=365, respect_avail=True):
    """Per-ticker sum of net_income over a 12mo window. forward=True -> next 12mo (LOOKAHEAD);
    forward=False -> trailing 12mo, only reports available at the date (PIT honest)."""
    out = {}
    md = list(midx)
    his = [np.datetime64(d + pd.Timedelta(days=days)) for d in md]
    los = [np.datetime64(d - pd.Timedelta(days=days)) for d in md]
    d0 = [np.datetime64(d) for d in md]
    for tk in common:
        df = reps.get(tk)
        if df is None or len(df) == 0 or "period_end" not in df:
            continue
        d = df.dropna(subset=["period_end"]).copy()
        pe = pd.to_datetime(d["period_end"]).values
        ni = pd.to_numeric(d["net_income"], errors="coerce").values.astype(float)
        av = (pd.to_datetime(d["avail_date"]).values if "avail_date" in d else pe)
        col = np.full(len(md), np.nan)
        for j in range(len(md)):
            if forward:
                m = (pe > d0[j]) & (pe <= his[j])
            else:
                m = (pe > los[j]) & (pe <= d0[j])
                if respect_avail:
                    m = m & (av <= d0[j])
            if m.any() and np.isfinite(ni[m]).any():
                col[j] = np.nansum(ni[m])
        out[tk] = pd.Series(col, index=midx)
    return pd.DataFrame(out).reindex(index=midx, columns=common)


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
    mktcap = px * shares
    pb = mktcap / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0

    print("building forward/trailing 12mo earnings panels...", flush=True)
    fwd_ni = _ntm_panel(reps, midx, common, forward=True)
    ttm_ni = _ntm_panel(reps, midx, common, forward=False)
    fwd_pe = mktcap / fwd_ni.where(fwd_ni > 0)     # lower = cheaper (pick min); positive fwd earnings only
    ttm_pe = mktcap / ttm_ni.where(ttm_ni > 0)

    MP = {"pb": pb, "ttm_pe": ttm_pe, "fwd_pe": fwd_pe}
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def run(metric):
        mp = MP[metric]
        rets, spies, nn = [], [], []
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
                use = [c for c in use if pd.notna(mp.loc[date, c])]
                pick = mp.loc[date, use].idxmin() if use else None
                if pick is None:
                    continue
                r = _ret_delist(px[pick], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); nn.append(len(slot))
        s = _stats(rets, spies)
        s["avg_names"] = round(float(np.mean(nn)), 2) if nn else 0
        return s

    res = {m: run(m) for m in MP}
    print("\n=== FORWARD P/E TEST (same rotation+guard+low_debt harness) ===", flush=True)
    print(f"  {'metric':8} {'total':>8} {'vsSPY':>8} {'t':>6} {'Sharpe':>7} {'DD':>8} {'names':>6}", flush=True)
    for m in ("pb", "ttm_pe", "fwd_pe"):
        s = res[m]
        print(f"  {m:8} {s['total_return']:>7}% {s['vs_spy']:>7}% {str(s['t_stat']):>6} "
              f"{s['sharpe']:>7} {s['max_drawdown']:>7}% {s['avg_names']:>6}", flush=True)
    print("\n  pb=validated baseline | ttm_pe=REAL trailing P/E (PIT) | fwd_pe=PERFECT-FORESIGHT upper bound"
          " (lookahead)", flush=True)
    beats = res["fwd_pe"]["vs_spy"] > res["pb"]["vs_spy"]
    print(f"\n  VERDICT: even with perfect foresight, forward P/E {'BEATS' if beats else 'does NOT beat'} "
          f"cheapest-P/B ({res['fwd_pe']['vs_spy']}% vs {res['pb']['vs_spy']}%). "
          + ("A real estimate feed COULD be worth sourcing." if beats
             else "Real (noisy) forward estimates would be WORSE -> not worth sourcing; P/B stays."), flush=True)
    return res


if __name__ == "__main__":
    build()
