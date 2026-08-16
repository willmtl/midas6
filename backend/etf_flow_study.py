#!/usr/bin/env python3
"""ETF FUND-FLOW SIGNAL — test whether Polygon-derived sector-ETF fund flows (core.ETFFlow: daily creation/
redemption from share_class_shares_outstanding) add rotation edge. Monthly net flow % per sector = month-end
shares_out / prior month-end - 1. Arms (all: cheapest as-traded-P/B guard low-debt $5M pick, TR returns, monthly):

  baseline          rank sectors by momentum ACCELERATION (the deployed flagship) -> reference.
  flow_rank         rank sectors by trailing 3mo net inflow % (pure flow momentum).
  flow_confirm      accel top-10 but DROP any sector in net OUTFLOW over the trailing 3mo (flow as a veto).
  flow_tilt         accel top-10, div_2x-style: 2x conviction weight to sectors with strong positive inflow.
  flow_capitulation contrarian: rank by MOST-NEGATIVE 1mo flow (heavy redemptions) -> does it bounce?

If flow_confirm/flow_tilt beat baseline, flow is a real confirming overlay; if flow_rank rivals accel, flow
LEADS price; if capitulation works, outflow marks washouts. -> BacktestResult[etf_flow] + JSON.
Run (AFTER fetch_etf_flows.py backfill): MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/etf_flow_study.py
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
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_N = 10; CONV = 2.0; MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "etf_flow.json"


def load_flow_shares(tickers, midx):
    """month-end shares_out DataFrame [midx x ticker] from core.ETFFlow."""
    from core.models import ETFFlow
    rows = ETFFlow.objects.filter(ticker__in=list(tickers)).values("ticker", "date", "shares_out")
    if not rows:
        return pd.DataFrame(index=midx)
    df = pd.DataFrame(list(rows))
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(index="date", columns="ticker", values="shares_out", aggfunc="last").sort_index()
    return piv.resample("ME").last().reindex(midx).ffill(limit=2)


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)

    shares = load_flow_shares(etf_tk, midx)
    if shares.empty or shares.notna().sum().sum() == 0:
        print("No ETFFlow data yet — run fetch_etf_flows.py --run first.", flush=True)
        return None
    cov = shares.notna().any().sum()
    flow_1m = shares.pct_change()
    flow_3m = shares.pct_change(3)
    print(f"months {len(midx)} | ETFs with flow data {cov}/{len(etf_tk)} | "
          f"flow rows span {shares.dropna(how='all').index.min().date()}..{shares.dropna(how='all').index.max().date()}",
          flush=True)

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
    stock_daily = load_candles(all_holds)
    dvol = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def run(rank_fn, weight_fn=None, veto_fn=None):
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            order = rank_fn(date)
            if order is None:
                continue
            held = set(); wsum = rr = 0.0; n = 0
            for etf in order:
                if n >= TOP_N:
                    break
                if veto_fn is not None and veto_fn(etf, date):
                    continue
                p = pick(etf, date, held)
                if not p:
                    continue
                held.add(p); n += 1
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = weight_fn(etf, date) if weight_fn else 1.0
                wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return _perf(rets, spies)

    def accel_order(date):
        s = accel.loc[date].dropna()
        return list(s.sort_values(ascending=False).index) if len(s) else None

    def flow_order(date, col):
        s = col.loc[date].dropna()
        return list(s.sort_values(ascending=False).index) if len(s) else None

    def flow_veto(etf, date):
        f = flow_3m.loc[date].get(etf)
        return pd.notna(f) and f < 0                      # drop net-outflow sectors

    def flow_weight(etf, date):
        f = flow_3m.loc[date].get(etf)
        return CONV if (pd.notna(f) and f > 0.02) else 1.0  # 2x for >2% quarterly inflow

    results = {
        "baseline_accel": run(accel_order),
        "flow_rank": run(lambda d: flow_order(d, flow_3m)),
        "accel_flow_confirm": run(accel_order, veto_fn=flow_veto),
        "accel_flow_tilt": run(accel_order, weight_fn=flow_weight),
        "flow_capitulation": run(lambda d: (list(flow_1m.loc[d].dropna().sort_values().index)
                                            if flow_1m.loc[d].notna().any() else None)),
    }

    base = results["baseline_accel"]
    print(f"\n=== ETF FLOW SIGNAL (total / vsSPY / Sharpe / DD / t) ===", flush=True)
    for k in ("baseline_accel", "flow_rank", "accel_flow_confirm", "accel_flow_tilt", "flow_capitulation"):
        r = results[k]
        print(f"  {k:<20} {r['total']:>7}%  vsSPY {r['vs_spy']:>7}  Sh {r['sharpe']:>5}  DD {r['dd']:>6}%  "
              f"t {r['t_stat']}  ({r['months']}mo)", flush=True)

    conf, tilt = results["accel_flow_confirm"], results["accel_flow_tilt"]
    best_overlay = max(("accel_flow_confirm", "accel_flow_tilt"), key=lambda k: results[k]["sharpe"])
    verdict = (
        f"Baseline accel {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
        f"Flow-only rank {results['flow_rank']['total']}%/Sh{results['flow_rank']['sharpe']} "
        f"({'rivals' if results['flow_rank']['sharpe'] >= base['sharpe'] - 0.1 else 'trails'} accel -> flow "
        f"{'LEADS' if results['flow_rank']['sharpe'] >= base['sharpe'] - 0.1 else 'lags'} price). "
        f"Best flow overlay = {best_overlay} ({results[best_overlay]['total']}%/Sh{results[best_overlay]['sharpe']}/"
        f"DD{results[best_overlay]['dd']}%). "
        + ("Flow overlay BEATS baseline -> confirming inflow is a real edge, worth wiring."
           if results[best_overlay]["sharpe"] > base["sharpe"] + 0.05 else
           "Flow overlay does NOT beat baseline -> flow mostly echoes the price momentum we already rank on.")
        + f" Capitulation (buy heaviest outflow) {results['flow_capitulation']['total']}%/Sh"
          f"{results['flow_capitulation']['sharpe']}."
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "conviction_mult": CONV, "min_dvol": MIN_DVOL, "benchmark": BENCH,
                   "months": int(len(midx)), "etfs_with_flow": int(cov),
                   "flow_source": "Polygon share_class_shares_outstanding (core.ETFFlow), monthly net %"},
        "results": results, "verdict": verdict,
        "caveat": "Flow = month-end shares_out / prior month-end - 1 (net creation). PIT, no fees, present-day "
                  "holdings survivorship, TR returns, as-traded P/B. Recent months' shares can be provisional "
                  "(Polygon refines with a lag); ffill(limit=2) bridges sparse gaps.",
    }


def main():
    p = build()
    if p is None:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="etf_flow", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                       "computed_at": timezone.now()})
        print("Saved BacktestResult[etf_flow]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
