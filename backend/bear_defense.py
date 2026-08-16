#!/usr/bin/env python3
"""BEAR DEFENSE — absolute (time-series) momentum overlay, the fix the failed SPY-200MA filter wasn't.

The validated engine is ALWAYS-IN (top-10 relative-momentum sectors -> cheapest-P/B pick). Here we add a
DUAL-MOMENTUM gate: relative momentum still picks WHICH sectors, but each sector's OWN absolute momentum
decides IN or OUT — if a held sector is itself in a downtrend, that slot goes to CASH (risk-free) instead
of buying its pick. In a broad bear, most sectors fail the gate -> the book de-risks automatically, without
the whipsaw of a single market-wide SPY<200MA switch (which cut returns & worsened DD in the Return Lab).

Variants (vs the always-in baseline):
  abs6_cash    hold pick only if sector 6mo return > 0, else cash
  abs12_cash   ... 12mo return > 0
  sma10_cash   ... sector ETF above its 10-month SMA
  abs6_drop    fail -> drop the slot (concentrate in survivors) instead of cash
Reports total/vsSPY/t/Sharpe/DD + avg exposure + avg monthly return in BEAR months (SPY 6mo<0) vs BULL.
The win condition: cut drawdown / lift bear-month returns WITHOUT gutting total return.
-> BacktestResult[bear_defense] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/bear_defense.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "bear_defense.json"
LOOKBACK, TOP_N, RF_M = 6, 10, 0.0033
VARIANTS = ["baseline", "abs6_cash", "abs12_cash", "sma10_cash", "abs6_drop"]


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
    etf_t6 = etf_m.pct_change(LOOKBACK)
    etf_t12 = etf_m.pct_change(12)
    etf_sma10 = etf_m > etf_m.rolling(10).mean()
    spy_t6 = spy_m.pct_change(LOOKBACK)                    # for bear/bull month tagging
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
    warmup = 12

    def gate(variant, etf, date):
        if variant == "baseline":
            return True
        if variant in ("abs6_cash", "abs6_drop"):
            v = etf_t6.loc[date, etf]; return bool(pd.notna(v) and v > 0)
        if variant == "abs12_cash":
            v = etf_t12.loc[date, etf]; return bool(pd.notna(v) and v > 0)
        if variant == "sma10_cash":
            return bool(etf_sma10.loc[date, etf])
        return True

    def run(variant):
        rets, spies, exps, bear_r, bull_r = [], [], [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_t6.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slots, invested = [], 0
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                passed = gate(variant, etf, date)
                if passed and use:
                    pick = pb.loc[date, use].idxmin()
                    r = _ret_delist(px[pick], date, ndate)
                    if r is not None and np.isfinite(r):
                        slots.append(float(r)); invested += 1; continue
                # gate failed (or no pick): cash for this slot, unless a "drop" variant
                if variant.endswith("_drop"):
                    if use:                                  # had a pick but gate failed -> drop slot
                        continue
                    else:
                        continue
                slots.append(RF_M)                            # cash slot
            if not slots:
                continue
            port = float(np.mean(slots))
            rets.append(port); spies.append(float(sp))
            exps.append(invested / max(len(ranks), 1))
            (bear_r if (pd.notna(spy_t6.iloc[i]) and spy_t6.iloc[i] < 0) else bull_r).append(port)
        s = _stats(rets, spies)
        s["avg_exposure"] = round(float(np.mean(exps)) * 100, 1) if exps else 0
        s["bear_mo_avg_%"] = round(float(np.mean(bear_r)) * 100, 2) if bear_r else None
        s["bull_mo_avg_%"] = round(float(np.mean(bull_r)) * 100, 2) if bull_r else None
        s["bear_months"] = len(bear_r)
        return s

    results = {v: run(v) for v in VARIANTS}
    base = results["baseline"]
    print("\n=== BEAR DEFENSE (dual-momentum: relative picks sectors, absolute gates in/out) ===", flush=True)
    print(f"  {'variant':11} {'total':>7} {'vsSPY':>8} {'t':>5} {'Sh':>5} {'DD':>8} {'expo':>6} "
          f"{'bearMo':>7} {'bullMo':>7}", flush=True)
    for v in VARIANTS:
        s = results[v]
        print(f"  {v:11} {s['total_return']:>6}% {s['vs_spy']:>7}% {str(s['t_stat']):>5} {s['sharpe']:>5} "
              f"{s['max_drawdown']:>7}% {s['avg_exposure']:>5}% {str(s['bear_mo_avg_%']):>7} "
              f"{str(s['bull_mo_avg_%']):>7}", flush=True)
    print(f"\n  ({base['bear_months']} bear months where SPY 6mo<0)", flush=True)

    # win = cuts drawdown AND keeps most of the return AND improves bear-month return
    def better(v):
        s = results[v]
        return (s["max_drawdown"] > base["max_drawdown"] and s["total_return"] >= base["total_return"] * 0.85
                and (s["bear_mo_avg_%"] or -9) > (base["bear_mo_avg_%"] or -9))
    winners = [v for v in VARIANTS if v != "baseline" and better(v)]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "rf_monthly": RF_M, "benchmark": BENCH,
                   "months": int(len(midx)), "bear_def": "SPY 6mo return < 0"},
        "results": results, "baseline": base, "winners": winners,
        "verdict": ("Dual-momentum bear defense: " + (
            f"{', '.join(winners)} cut drawdown while keeping return & improving bear-month behavior."
            if winners else "no variant improved drawdown/bear-returns without gutting total return — the "
            "always-in engine's rotation is already its own defense (defensives rise into the top-10).")),
        "caveat": "In-sample, no fees, ~5y single regime (one real bear). Directional. Cash earns RF only.",
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
            kind="bear_defense", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                           "computed_at": timezone.now()})
        print("Saved BacktestResult[bear_defense]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
