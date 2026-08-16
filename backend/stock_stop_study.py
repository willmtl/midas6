#!/usr/bin/env python3
"""WHEN TO SELL LOSERS — does a price STOP-LOSS help the value-pick system, or does it hurt (as
mean-reversion theory predicts)? Baseline = no stop, hold each pick to the monthly rebalance (sell only
when it no longer qualifies). Variants add an intra-hold stop: if the pick's daily CLOSE falls to
S0*(1-stop) during the hold, exit at that close; else exit at the rebalance. Same engine (rotation top-10
-> cheapest-P/B guarded low-debt). Reports total/vsSPY/t/Sharpe/DD per stop level.
-> BacktestResult[stock_stop] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/stock_stop_study.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "stock_stop.json"
LOOKBACK, TOP_N = 6, 10
STOPS = [None, 0.10, 0.15, 0.20, 0.25]


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
    etf_trail = etf_m.pct_change(LOOKBACK)
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
    warmup = max(LOOKBACK, 1)

    def pick_ret(pick, date, ndate, stop):
        """Realized return: if stop set and daily close hits S0*(1-stop) during the hold, exit there;
        else the normal rebalance return."""
        base = _ret_delist(px[pick], date, ndate)
        if stop is None or pick not in stock_daily:
            return base
        c = stock_daily[pick]["Close"]
        seg = c[(c.index > date) & (c.index <= ndate)]
        if not len(seg):
            return base
        S0 = float(c[c.index <= date].iloc[-1]) if len(c[c.index <= date]) else float(seg.iloc[0])
        if S0 <= 0:
            return base
        lvl = S0 * (1 - stop)
        breached = seg[seg <= lvl]
        if len(breached):
            return float(breached.iloc[0] / S0 - 1)          # exit at the breaching close
        return base

    def run(stop):
        rets, spies = [], []
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
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                if not use:
                    continue
                pick = pb.loc[date, use].idxmin()
                r = pick_ret(pick, date, ndate, stop)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        return _stats(rets, spies)

    results = {}
    for s in STOPS:
        key = "no_stop" if s is None else f"stop_{int(s*100)}"
        results[key] = run(s)
    print("\n=== STOP-LOSS on the value pick (does cutting losers help?) ===", flush=True)
    base = results["no_stop"]
    for k, v in results.items():
        d = "" if k == "no_stop" else f"  ({'+' if v['vs_spy']-base['vs_spy']>=0 else ''}{round(v['vs_spy']-base['vs_spy'],1)}pp)"
        print(f"  {k:10} vsSPY {v['vs_spy']:>7}%  total {v['total_return']}%  t={v['t_stat']}  "
              f"Sh {v['sharpe']}  DD {v['max_drawdown']}%{d}", flush=True)

    best = max(results, key=lambda k: results[k]["vs_spy"])
    helped = best != "no_stop" and results[best]["vs_spy"] > base["vs_spy"] + 5
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx)),
                   "stops_tested": [s for s in STOPS if s is not None]},
        "results": results, "best": best,
        "verdict": ("A stop-loss HELPS the value pick." if helped else
                    "A stop-loss does NOT help — cutting losers on this mean-reversion value system sells into the "
                    "exact oversold dips it's built to exploit. HOLD to the rebalance; sell losers only when they "
                    "stop QUALIFYING (sector momentum fades / cheaper name replaces / guard trips), not on price."),
        "caveat": "Close-based stop, exit at breaching close (gaps can exit worse). In-sample, no fees, ~5y.",
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
            kind="stock_stop", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                         "computed_at": timezone.now()})
        print("Saved BacktestResult[stock_stop]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
