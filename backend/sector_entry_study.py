#!/usr/bin/env python3
"""CATCH THE SECTOR EARLIER? 6mo momentum feels late. Test EARLY-catch sector signals vs 6mo trailing,
each feeding the same value pick (cheapest-P/B guarded low-debt), reporting vsSPY / Sharpe / win rate:

  mom6      rank by 6mo return (baseline)
  mom1      rank by 1mo return (fast/early — expect noisy)
  mom3      rank by 3mo return
  accel     rank by ACCELERATION = 3mo-now minus 3mo-3ago (catch the inflection, not the level)
  accel_conf among sectors with 6mo>0, rank by acceleration (confirmed trend + still turning up)
  brk50     sectors that just crossed above their 50d MA (fresh breakout), ranked by 6mo mom
Does catching the inflection give better 'tenor' (return) than 6mo level, or just more noise?
-> BacktestResult[sector_entry] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sector_entry_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "sector_entry.json"
TOP_N = 10


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, win_pct=0, periods=0)
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
    m1, m3, m6 = etf_m.pct_change(1), etf_m.pct_change(3), etf_m.pct_change(6)
    accel = m3 - m3.shift(3)                       # acceleration of 3mo momentum
    ma50 = pd.DataFrame({e: (etf_daily[e]["Close"] / etf_daily[e]["Close"].rolling(50).mean() - 1)
                         .resample("ME").last() for e in etf_tk if e in etf_daily}).reindex(midx)
    brk = (ma50 > 0) & (ma50.shift(1) <= 0)        # just crossed above 50d MA (fresh breakout)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def Rf(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(Rf, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 9

    def pick_in(etf, date, ndate):
        _, holds = sector_map.get(etf, (etf, []))
        cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                 and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
        ld = [c for c in cands if bool(low_debt.loc[date, c])]
        use = ld or cands
        if not use:
            return None
        r = _ret_delist(px[pb.loc[date, use].idxmin()], date, ndate)
        return float(r) if (r is not None and np.isfinite(r)) else None

    def sel(sig, date):
        if sig == "mom6":
            return list(m6.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        if sig == "mom1":
            return list(m1.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        if sig == "mom3":
            return list(m3.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        if sig == "accel":
            return list(accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        if sig == "accel_conf":
            pos = m6.loc[date][m6.loc[date] > 0].index
            a = accel.loc[date, [e for e in pos if e in accel.columns]].dropna()
            return list(a.sort_values(ascending=False).head(TOP_N).index)
        if sig == "brk50":
            fresh = [e for e in brk.columns if bool(brk.loc[date, e])]
            a = m6.loc[date, [e for e in fresh if e in m6.columns]].dropna()
            return list(a.sort_values(ascending=False).head(TOP_N).index)
        return []

    SIGS = ["mom6", "mom1", "mom3", "accel", "accel_conf", "brk50"]

    def run(sig):
        rets, spies, pk = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            slot = [pick_in(e, date, ndate) for e in sel(sig, date)]
            slot = [x for x in slot if x is not None]
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); pk += slot
        s = _stats(rets, spies)
        s["pick_win_pct"] = round(float((np.array(pk) > 0).mean() * 100), 1) if pk else 0
        s["avg_names"] = round(len(pk) / max(len(rets), 1), 1) if rets else 0
        return s

    results = {s: run(s) for s in SIGS}
    base = results["mom6"]
    print("\n=== SECTOR ENTRY SIGNAL — catch early vs 6mo trailing ===", flush=True)
    for s in SIGS:
        v = results[s]
        d = "" if s == "mom6" else f"  ({'+' if v['vs_spy']-base['vs_spy']>=0 else ''}{round(v['vs_spy']-base['vs_spy'],1)}pp)"
        print(f"  {s:11} vsSPY {v['vs_spy']:>7}%  Sh {v['sharpe']}  DD {v['max_drawdown']}%  "
              f"pick-win {v['pick_win_pct']}%  names {v['avg_names']}{d}", flush=True)

    best = max(SIGS, key=lambda s: results[s]["vs_spy"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "results": results, "best": best,
        "verdict": (f"Best sector signal = {best} ({results[best]['vs_spy']}% vs mom6 {base['vs_spy']}%). "
                    + ("Catching the inflection/breakout early beats 6mo trailing." if results[best]["vs_spy"] > base["vs_spy"] + 10
                       else "6mo trailing (with the accel filter) is hard to beat — pure early/fast signals are "
                       "noisier; the level confirms the trend, acceleration times the freshness.")),
        "caveat": "In-sample, no fees, ~5y, 9mo warmup. Early signals (mom1/accel/brk) trade runway for false starts.",
    }
    return payload


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="sector_entry", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                           "computed_at": timezone.now()})
        print("Saved BacktestResult[sector_entry]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
