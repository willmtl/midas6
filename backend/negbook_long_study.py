#!/usr/bin/env python3
"""BUY the negative-book names — is it a real strategy? They go up on average (+4.6% profitable, +7.7%
worst-sector-distressed), but win rate ~53% and wildly bimodal (many ->0, few 10x). Test as PORTFOLIOS
(monthly equal-weight, tradeable $5M/day) to see if the positive mean survives as usable risk-adjusted
return or is just lottery-ticket variance:
  nb_all         all negative-book stocks
  nb_profitable  neg-book + profitable (buyback machines)
  nb_distressed  neg-book + unprofitable
  nb_prof_accel  neg-book profitable in TOP-acceleration sectors (quality + rotation)
  nb_dist_worst  neg-book distressed in WORST-acceleration sectors (the +7.7% group)
vs SPY and the main long strategy. -> BacktestResult[negbook_long] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/negbook_long_study.py
"""
import os, json, warnings
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

TOP_N = 10


def _st(rets, spy):
    r = np.asarray(rets, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0, names=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


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
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    keys = ["nb_all", "nb_profitable", "nb_distressed", "nb_prof_accel", "nb_dist_worst"]
    port = {k: [] for k in keys}; spies = []; nnames = {k: [] for k in keys}
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        accd = accel.loc[date].dropna()
        top = set(accd.sort_values(ascending=False).head(TOP_N).index)
        bot = set(accd.sort_values(ascending=True).head(TOP_N).index)
        buckets = {k: [] for k in keys}
        for etf, (_, holds) in sector_map.items():
            for h in holds:
                if h not in px.columns or not _available_at(px[h], date):
                    continue
                if not (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6):
                    continue
                e_ = eq.loc[date, h]
                if pd.isna(e_) or e_ >= 0:
                    continue                             # negative-book only
                r = _ret_delist(px[h], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                r = float(r); prof = pd.notna(ni.loc[date, h]) and ni.loc[date, h] > 0
                buckets["nb_all"].append(r)
                (buckets["nb_profitable"] if prof else buckets["nb_distressed"]).append(r)
                if prof and etf in top:
                    buckets["nb_prof_accel"].append(r)
                if (not prof) and etf in bot:
                    buckets["nb_dist_worst"].append(r)
        any_ = False
        for k in keys:
            if buckets[k]:
                port[k].append(float(np.mean(buckets[k]))); nnames[k].append(len(buckets[k])); any_ = True
            else:
                port[k].append(0.0); nnames[k].append(0)      # flat month if no names (cash)
        if any_:
            spies.append(float(sp))

    print("\n=== BUY the negative-book names — portfolios ===", flush=True)
    res = {}
    for k in keys:
        s = _st(port[k][:len(spies)], spies); s["avg_names"] = round(float(np.mean([x for x in nnames[k] if True])), 1)
        res[k] = s
        print(f"  {k:14} total {s['total']:>7}%  vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']:>5}  DD {s['dd']:>7}%  "
              f"win {s['win']}%  ~{s['avg_names']} names/mo", flush=True)

    best_sh = max(keys, key=lambda k: res[k]["sharpe"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "results": res, "best_sharpe": best_sh,
        "verdict": (f"Best risk-adjusted neg-book long = {best_sh} (Sh {res[best_sh]['sharpe']}, vsSPY {res[best_sh]['vs_spy']}%, "
                    f"DD {res[best_sh]['dd']}%). Buying neg-book profitable (buyback machines) is a real long; distressed "
                    "neg-book is high-return but lottery variance/drawdown -> only as a tiny satellite."),
        "caveat": "In-sample, ~5y. Tiny baskets (few names/mo) = concentrated/high-variance. Neg-book un-rankable by "
                  "P/B; this is equal-weight-all. No fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "negbook_long.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="negbook_long", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[negbook_long]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
