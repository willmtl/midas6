#!/usr/bin/env python3
"""VALUE-METRIC BAKE-OFF, QUARTERLY vs TTM (audit finding #4 re-test).

value_ranking_lab concluded "cheapest P/B beats P/E / EV-EBIT / FCF-yield". But those earnings/cashflow
metrics were built from SINGLE-QUARTER FinancialReport flows (`_pit_monthly_panel` forward-fills, never
sums 4 quarters), while P/B is a clean stock/stock ratio — so the comparison was UNFAIR (the earnings
metrics were ~4x mis-scaled and, worse, quarter-lumpy, biasing the ranking toward P/B). This re-runs the
EXACT same rotation+guard+low_debt harness twice: once with quarterly flows (reproduces the old result)
and once with trailing-12-month (rolling-4Q) flows (the fair comparison). Question: does "P/B is best"
survive the fix? -> BacktestResult[value_ttm_bakeoff] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/value_ttm_bakeoff.py  (--limit 300)
"""
import os, sys, json, warnings
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "value_ttm_bakeoff.json"
LOOKBACK, TOP_N = 6, 10
METRICS = ["pb", "ev_ebit", "fcf_yield", "earn_yield", "ps", "composite"]


def _pit_ttm_panel(reports_map, field, midx):
    """Trailing-12m panel: rolling sum of the last 4 QUARTERLY values (by period_end), ffill by avail_date."""
    out = {}
    for tk, r in reports_map.items():
        if field not in r.columns:
            continue
        d = r[["period_end", "avail_date", field]].dropna(subset=[field]).copy()
        if len(d) < 4:
            continue
        d = d.sort_values("period_end")
        d["ttm"] = d[field].rolling(4).sum()
        s = pd.Series(d["ttm"].values, index=pd.to_datetime(d["avail_date"])).dropna()
        if s.empty:
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[tk] = s.reindex(s.index.union(midx)).ffill().reindex(midx)
    return pd.DataFrame(out)


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


def build_metric_panels(reps, midx, px, shares, equity, debt, cash, flow_mode):
    """Return the metric panels (lower=better). flow_mode 'q'=single-quarter, 'ttm'=trailing-12m."""
    Pf = (lambda f: _pit_monthly_panel(reps, f, midx)) if flow_mode == "q" else (lambda f: _pit_ttm_panel(reps, f, midx))
    ni, opinc, rev, fcf = Pf("net_income"), Pf("operating_income"), Pf("revenue"), Pf("free_cash_flow")
    ni, opinc, rev, fcf = (p.reindex(index=midx, columns=px.columns) for p in (ni, opinc, rev, fcf))
    mktcap = px * shares
    ev = mktcap + debt.fillna(0) - cash.fillna(0)
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    ev_ebit = ev / opinc.where(opinc > 0)
    fcf_yield = -(fcf / ev.where(ev > 0))
    earn_yield = -(ni / mktcap.where(mktcap != 0))
    ps = mktcap / rev.where(rev > 0)

    def _z(p):
        return (p.sub(p.mean(axis=1), axis=0)).div(p.std(axis=1).replace(0, np.nan), axis=0)
    composite = _z(pb) + _z(ev_ebit) + _z(fcf_yield)
    mp = {"pb": pb, "ev_ebit": ev_ebit, "fcf_yield": fcf_yield, "earn_yield": earn_yield, "ps": ps, "composite": composite}
    return mp, pb, trap, low_debt


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
    stock_m = _monthly_close(load_candles(all_holds)).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, debt, cash = P("shares_outstanding"), P("total_equity"), P("total_debt"), P("cash_and_equivalents")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, debt, cash = R(shares), R(equity), R(debt), R(cash)
    warmup = max(LOOKBACK, 1)

    def run(mp, pb, trap, low_debt, metric, fallback=False, start_date=None, end_date=None):
        rets, spies = [], []
        _sd = pd.Timestamp(start_date) if start_date else None
        _ed = pd.Timestamp(end_date) if end_date else None
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            if (_sd is not None and date < _sd) or (_ed is not None and date > _ed):
                continue
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = [c for c in (ld or cands) if c in mp[metric].columns and pd.notna(mp[metric].loc[date, c])]
                pick = mp[metric].loc[date, use].idxmin() if use else None
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

    out = {"computed_at": pd.Timestamp.utcnow().isoformat(), "params": {"lookback": LOOKBACK, "top_n": TOP_N,
           "months": int(len(midx)), "stocks": int(len(common)), "limit": limit}, "modes": {}}
    print(f"months {len(midx)} | stocks {len(common)}\n", flush=True)
    for mode in ("q", "ttm"):
        mp, pb, trap, low_debt = build_metric_panels(reps, midx, px, shares, equity, debt, cash, mode)
        res = {m: run(mp, pb, trap, low_debt, m, False) for m in METRICS}
        base = res["pb"]["vs_spy"]
        ranked = sorted(METRICS, key=lambda m: res[m]["vs_spy"], reverse=True)
        out["modes"][mode] = {"results": res, "ranking": ranked, "best": ranked[0],
                              "pb_beats_all": all(res["pb"]["vs_spy"] >= res[m]["vs_spy"] for m in METRICS)}
        label = "QUARTERLY (old/buggy)" if mode == "q" else "TTM (fair)"
        print(f"=== {label} ===", flush=True)
        for m in ranked:
            d = res[m]
            print(f"  {m:11} vsSPY {d['vs_spy']:>8}%  t={str(d['t_stat']):>5}  Sh {d['sharpe']:>5}  "
                  f"DD {d['max_drawdown']:>6}%  (Δvs pb {d['vs_spy']-base:+.1f})", flush=True)
        print(f"  -> best={ranked[0]}  P/B-beats-all={out['modes'][mode]['pb_beats_all']}\n", flush=True)

    q, t = out["modes"]["q"], out["modes"]["ttm"]

    # WINDOW SWEEP (TTM fair panels): does 'P/B is worst / earnings win' survive EXCLUDING 2020? (user: the
    # earnings metrics may just be catching the 2020 crash+recovery). Same metrics, sliced by date.
    mp, pb, trap, low_debt = build_metric_panels(reps, midx, px, shares, equity, debt, cash, "ttm")
    wins = [("FULL 2019→now", None, None), ("ex-2020 (2021→now)", "2021-01-31", None),
            ("H1 2019→2022", None, "2022-12-31"), ("H2 2023→2026", "2023-01-31", None)]
    key = ["pb", "earn_yield", "fcf_yield", "ps", "composite"]
    print("\n=== WINDOW SWEEP (TTM): is P/B-worst a 2020 artifact? (vs-SPY %) ===", flush=True)
    print(f"  {'window':22}" + "".join(f"{m:>12}" for m in key), flush=True)
    out["window_sweep"] = {}
    for lab, sd, ed in wins:
        row = {m: run(mp, pb, trap, low_debt, m, False, sd, ed)["vs_spy"] for m in key}
        out["window_sweep"][lab] = row
        best = max(key, key=lambda m: row[m])
        print(f"  {lab:22}" + "".join(f"{row[m]:>+12.1f}" for m in key) + f"   best={best}", flush=True)

    out["verdict"] = (f"QUARTERLY: best={q['best']}, P/B-beats-all={q['pb_beats_all']}. "
                      f"TTM(fair) FULL: best={t['best']}, P/B-beats-all={t['pb_beats_all']}. "
                      "See window_sweep for whether earnings-beats-P/B survives ex-2020.")
    print("\n" + out["verdict"], flush=True)
    return out


def main():
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="value_ttm_bakeoff",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[value_ttm_bakeoff]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_TTM_BAKEOFF", flush=True)


if __name__ == "__main__":
    main()
