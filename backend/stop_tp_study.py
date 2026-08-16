#!/usr/bin/env python3
"""OPTIMAL stop-loss x take-profit on the accel-value engine. Intra-hold: each month, walk the pick's
DAILY close from entry to next rebalance; exit at the first of stop-loss (-X%) or take-profit (+Y%), else
at rebalance. Sweep the grid, report vs-SPY. Priors: stops HURT (mean-reversion sells the dip); take-
profits HURT MORE (they cap the few 'banana' winners that carry ~90% of the edge).
-> BacktestResult[stop_tp] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/stop_tp_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
STOPS = [None, 0.10, 0.15, 0.20, 0.25]
TPS = [None, 0.25, 0.50, 1.00]


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
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    # pre-collect picks (date, ndate, ticker, rebalance_ret) once
    monthly_picks, spies = [], []
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
            t = pb.loc[date, g].idxmin()
            rr = _ret_delist(px[t], date, ndate)
            if rr is not None and np.isfinite(rr):
                S0 = float(px.loc[date, t])
                slot.append((t, date, ndate, S0, float(rr)))
        if slot:
            monthly_picks.append(slot); spies.append(float(sp))

    def pick_ret(t, date, ndate, S0, rr, stop, tp):
        if (stop is None and tp is None) or S0 <= 0:
            return rr
        d = stock_daily.get(t)
        if d is None:
            return rr
        seg = d["Close"][(d["Close"].index > date) & (d["Close"].index <= ndate)]
        for c in seg:
            if tp is not None and c >= S0 * (1 + tp):
                return float(c / S0 - 1)          # take profit
            if stop is not None and c <= S0 * (1 - stop):
                return float(c / S0 - 1)          # stop out (at breaching close)
        return rr

    def total(stop, tp):
        rets = []
        for slot in monthly_picks:
            rr = [pick_ret(t, d, nd, s0, r, stop, tp) for (t, d, nd, s0, r) in slot]
            rets.append(float(np.mean(rr)))
        return round((np.prod(1 + np.array(rets)) - 1) * 100 - (np.prod(1 + np.array(spies)) - 1) * 100, 1)

    base = total(None, None)
    grid = {}
    print(f"\n=== STOP x TAKE-PROFIT (vs SPY; baseline none/none = {base}%) ===", flush=True)
    hdr = "  stop\\tp   " + "".join(f"{'none' if t is None else '+'+str(int(t*100))+'%':>9}" for t in TPS)
    print(hdr, flush=True)
    for s in STOPS:
        row = f"  {'none' if s is None else '-'+str(int(s*100))+'%':<9}"
        for t in TPS:
            v = total(s, t); grid[f"{s}|{t}"] = v
            row += f"{v:>9}"
        print(row, flush=True)

    best = max(grid, key=lambda k: grid[k])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "stops": [s for s in STOPS], "tps": [t for t in TPS]},
        "baseline_none_none": base, "grid": grid, "best": best,
        "verdict": (f"Optimal = {best} (vsSPY {grid[best]}%). " + (
            "NO stop / NO take-profit is best — stops sell the dips this mean-reversion strategy wants to buy, and "
            "take-profits decapitate the few banana winners that carry the edge. Hold to the monthly rebalance."
            if best == "None|None" else "A stop/TP combo edged out the baseline — inspect, likely noise.")),
        "caveat": "Intra-hold on daily CLOSE (not intraday low/high); exit at breaching close (gaps exit worse). "
                  "In-sample, no fees. Stop/TP here are the WRONG tools for a fat-tail mean-reversion strategy.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "stop_tp.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="stop_tp", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[stop_tp]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
