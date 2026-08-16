#!/usr/bin/env python3
"""ACCELERATION WINDOW SWEEP — is the 6-month lookback (two stacked 3mo blocks) too slow? Sweep the block
window w in accel = pct_change(w) - pct_change(w).shift(w) for w in {1,2,3,4} (total lookback 2w = 2..8 mo),
and ALSO test raw VELOCITY (simple momentum pct_change(w), no 2nd derivative) at fast windows w in {1,2,3} as
a reference — maybe fast velocity beats slow acceleration outright. Everything else held fixed to the validated
engine: top-10 sectors -> cheapest positive-P/B, profit-guard + low-debt + $5M dvol floor, equal-weight,
monthly rebalance. Report return/vsSPY/Sharpe/DD/win AND TURNOVER (faster = more churn = more cost), plus a
net-of-30bps column so the speed/cost tradeoff is explicit.
-> BacktestResult[accel_window]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/accel_window_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
COST = 0.003          # 30bps round-trip on the liquid flagship names


def _stats(r, spy, turn=None):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0, turn=0, net=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    tn = float(np.mean(turn)) * 100 if turn is not None and len(turn) else 0.0
    # net: subtract per-month turnover * cost from each month's return
    if turn is not None and len(turn) == n:
        rn = r - np.asarray(turn, float) * COST
        net = float(np.prod(1 + rn) - 1) * 100
    else:
        net = tot
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1), turn=round(tn, 1), net=round(net, 1))


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
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    # ranking signals: name -> monthly DataFrame of the sector score (higher = better)
    signals = {}
    for w in (1, 2, 3, 4):
        signals[f"accel_{w}mo(={2*w}mo)"] = etf_m.pct_change(w) - etf_m.pct_change(w).shift(w)
    for w in (1, 2, 3):
        signals[f"velocity_{w}mo"] = etf_m.pct_change(w)

    def qual(h, date):
        return (h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0
                and not bool(trap.loc[date, h]) and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6)

    def run(sig):
        rets, turns, spies, prev = [], [], [], set()
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            row = sig.loc[date].dropna()
            if row.empty:
                continue
            top = row.sort_values(ascending=False).head(TOP_N).index
            picks, cur = [], set()
            for etf in top:
                _, holds = sector_map.get(etf, (etf, []))
                c = [h for h in holds if qual(h, date)]
                g = [x for x in c if bool(low.loc[date, x])] or c
                if not g:
                    continue
                pick = min(g, key=lambda h: pb.loc[date, h])
                r = _ret_delist(px[pick], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                picks.append(float(r)); cur.add(pick)
            if not picks:
                continue
            rets.append(float(np.mean(picks)))
            # turnover = fraction of book replaced vs last month
            turns.append(1.0 - (len(prev & cur) / max(len(cur), 1)) if prev else 0.0)
            spies.append(float(sp)); prev = cur
        return rets, turns, spies

    print(f"\n=== ACCELERATION WINDOW SWEEP (faster vs slower sector signal) ===", flush=True)
    print(f"  {'signal':22} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6} {'turn/mo':>8} {'net-cost':>9}", flush=True)
    res = {}
    ref_spies = None
    for name, sig in signals.items():
        rets, turns, spies = run(sig)
        if ref_spies is None:
            ref_spies = spies
        s = _stats(rets, spies, turns); res[name] = s
        print(f"  {name:22} {s['total']:>7}% {s['vs_spy']:>8} {s['sharpe']:>5} {s['dd']:>7}% {s['win']:>5}% "
              f"{s['turn']:>7}% {s['net']:>8}%", flush=True)

    base = res.get("accel_3mo(=6mo)", {})
    faster = {k: v for k, v in res.items() if k in ("accel_1mo(=2mo)", "accel_2mo(=4mo)", "velocity_1mo", "velocity_2mo", "velocity_3mo")}
    best_fast = max(faster, key=lambda k: faster[k]["sharpe"]) if faster else None
    beats = bool(best_fast and faster[best_fast]["sharpe"] > base.get("sharpe", 0) + 0.03
                 and faster[best_fast]["net"] > base.get("net", 0))
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "cost_roundtrip": COST, "months": int(len(ref_spies or []))},
        "results": res, "baseline": "accel_3mo(=6mo)", "best_faster": best_fast,
        "verdict": (
            f"A FASTER signal wins: {best_fast} (Sh {faster[best_fast]['sharpe']} vs 3mo-accel {base.get('sharpe')}, "
            f"net-of-cost {faster[best_fast]['net']}% vs {base.get('net')}%). Speed pays even after the extra turnover."
            if beats else
            f"NO faster signal beats 3mo-acceleration risk-adjusted after turnover. Best faster = {best_fast} "
            f"(Sh {faster.get(best_fast, {}).get('sharpe')} net {faster.get(best_fast, {}).get('net')}% vs baseline "
            f"Sh {base.get('sharpe')} net {base.get('net')}%). The 6mo lookback is slow ON PURPOSE — shorter windows "
            f"add churn/whipsaw that costs more than the faster reaction earns. Keep 3mo blocks."),
        "caveat": "Monthly rebalance throughout (signal window varies, not rebalance freq). Turnover = fraction of "
                  "single-name book replaced/mo; net subtracts 30bps*turnover. In-sample ~5y, no slippage model.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/accel_window.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="accel_window", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[accel_window]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
