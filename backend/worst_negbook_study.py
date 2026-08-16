#!/usr/bin/env python3
"""DOES 'WORST SECTOR + NEGATIVE BOOK' GO DOWN? The most-bearish combo — a distressed neg-book name in a
decelerating (bottom-acceleration) sector — is the natural short candidate. Test forward returns in the
BOTTOM-N acceleration sectors by class; if even THIS doesn't reliably fall, nothing is a clean short (the
market's upward drift + mean-reversion of beaten-down names dominates). Report mean, win, and % that fell.
-> BacktestResult[worst_negbook] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/worst_negbook_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10


def _w(x):
    a = np.array(x, float)
    return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 2),
                                        win_pct=round((a > 0).mean() * 100, 1), fell_pct=round((a < 0).mean() * 100, 1))


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
    sh, eq, ni = (_pit_monthly_panel(reps, f, midx) for f in ("shares_outstanding", "total_equity", "net_income"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni = R(sh), R(eq), R(ni)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    grp = {"worst_negbook_unprof": [], "worst_negbook_all": [], "worst_expensive": [], "worst_all_stocks": [], "spy": []}
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        grp["spy"].append(float(sp))
        bot = accel.loc[date].dropna().sort_values(ascending=True).head(TOP_N).index    # most decelerating
        for etf in bot:
            _, holds = sector_map.get(etf, (etf, []))
            avail = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            for h in avail:
                r = _ret_delist(px[h], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                r = float(r); grp["worst_all_stocks"].append(r)
                e_, n_ = eq.loc[date, h], ni.loc[date, h]
                if pd.notna(e_) and e_ < 0:
                    grp["worst_negbook_all"].append(r)
                    if pd.notna(n_) and n_ <= 0:
                        grp["worst_negbook_unprof"].append(r)
            g = [h for h in avail if pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0]
            if g:
                rr = _ret_delist(px[pb.loc[date, g].idxmax()], date, ndate)   # most expensive in worst sector
                if rr is not None and np.isfinite(rr):
                    grp["worst_expensive"].append(float(rr))

    spy = _w(grp["spy"])
    print(f"\n=== WORST SECTOR (bottom-accel) forward returns — does anything go DOWN? ===", flush=True)
    print(f"  {'(SPY month avg)':28} mean {spy['mean_pct']:>6}%", flush=True)
    for k in ("worst_all_stocks", "worst_expensive", "worst_negbook_all", "worst_negbook_unprof"):
        s = _w(grp[k])
        print(f"  {k:28} mean {s['mean_pct']:>6}%  win {s['win_pct']}%  FELL {s['fell_pct']}%  (n{s['n']})" if s else f"  {k}: none", flush=True)

    wu = _w(grp["worst_negbook_unprof"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "groups": {k: _w(v) for k, v in grp.items() if k != "spy"}, "spy_month_mean_pct": spy["mean_pct"],
        "verdict": (f"Worst-sector + neg-book-unprofitable (the most-bearish combo) forward mean {wu['mean_pct']}%, "
                    f"fell only {wu['fell_pct']}% of the time. " + (
                    "It does NOT reliably go down — even the ugliest combo is roughly a coin-flip / drifts up, so "
                    "there's no clean short. The market's upward drift + mean-reversion of already-beaten-down names "
                    "dominates the 'bad fundamentals' story." if wu["mean_pct"] > -1 else
                    "It DOES tend to fall -> a potential short candidate.")),
        "caveat": "In-sample, ~5y bull-ish regime. Tradeable ($5M/day) only. No borrow/squeeze costs.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "worst_negbook.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="worst_negbook", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[worst_negbook]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
