#!/usr/bin/env python3
"""LIQUIDITY as a TRADEABILITY filter — order-book/L2 is irrelevant for monthly holds; the real question
is capacity: does requiring a minimum daily $-volume (so you can actually buy the pick) cost return? Deep
value skews small-cap (strongest value premium there), so a liquidity floor likely trades return for
tradeability. Measure the cost at several thresholds on the accel-sector value pick.

  none / $2M / $5M / $10M / $25M / $50M minimum ~20d average dollar volume
vs SPY / Sharpe / DD / avg names / how often the pick gets bumped by the filter.
-> BacktestResult[liquidity] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/liquidity_study.py
"""
import os, json, warnings
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "liquidity.json"
LOOKBACK, TOP_N = 6, 10
THRESH = [0, 2e6, 5e6, 10e6, 25e6, 50e6]


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(vs_spy=0, sharpe=0, max_drawdown=0, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), max_drawdown=round(dd, 1), periods=n)


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
    etf_accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
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
    # dollar volume: 20d avg (Close*Volume), month-end
    dv = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 25:
            continue
        dv[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dv).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 12

    def run(thr):
        rets, spies, names, bumped, tot = [], [], [], 0, 0
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                guarded = [c for c in cands if bool(low_debt.loc[date, c])] or cands
                if not guarded:
                    continue
                tot += 1
                cheapest = pb.loc[date, guarded].idxmin()
                if thr > 0:
                    liq = [g for g in guarded if pd.notna(dvol.loc[date, g]) and dvol.loc[date, g] >= thr]
                    pick = pb.loc[date, liq].idxmin() if liq else None
                    if pick is None:
                        continue                      # no tradeable name -> skip sector
                    if pick != cheapest:
                        bumped += 1
                else:
                    pick = cheapest
                r = _ret_delist(px[pick], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); names.append(len(slot))
        s = _stats(rets, spies)
        s["avg_names"] = round(float(np.mean(names)), 1) if names else 0
        s["bump_pct"] = round(bumped / tot * 100, 1) if tot else 0
        return s

    results = {("none" if t == 0 else f"${int(t/1e6)}M"): run(t) for t in THRESH}
    base = results["none"]
    print("\n=== LIQUIDITY FILTER (min $-volume) — capacity vs return ===", flush=True)
    for k, s in results.items():
        d = "" if k == "none" else f"  ({'+' if s['vs_spy']-base['vs_spy']>=0 else ''}{round(s['vs_spy']-base['vs_spy'],1)}pp, {s['bump_pct']}% picks bumped)"
        print(f"  min {k:6} vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']}  DD {s['max_drawdown']}%  names {s['avg_names']}{d}", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx)), "thresholds_usd": THRESH,
                   "note": "no order-book/L2 data (irrelevant for monthly holds); dark-pool flow only ~10d history "
                   "(not backtestable yet). This is the tradeability/capacity filter."},
        "results": results,
        "verdict": (f"A $10M/day liquidity floor moves vs-SPY by {round(results['$10M']['vs_spy']-base['vs_spy'],1)}pp "
                    f"(bumps {results['$10M']['bump_pct']}% of picks). "
                    + ("Cheap — capacity filter costs little; add it for a larger book." if abs(results["$10M"]["vs_spy"]-base["vs_spy"]) < 30
                       else "Costly — the liquidity floor sacrifices real return (small-cap value premium); only add it "
                       "if your book size forces it.")),
        "caveat": "In-sample, no fees, ~5y. $-vol = 20d avg. A liquidity floor is a CAPACITY tool, not a return signal.",
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
            kind="liquidity", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[liquidity]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
