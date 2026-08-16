#!/usr/bin/env python3
"""THE WHOLE THING: does blending the DISTRESSED sleeve into the FLAGSHIP improve the combined book? We have
two long books holding TOTALLY different names:
  FLAGSHIP   = rank sectors by acceleration -> top-10 -> cheapest positive-P/B, profit-guard (not trap),
               low-debt (D/E<1) -> equal-weight across sectors -> month-end. Sharpe ~1.7.
  DISTRESSED = every neg-book + unprofitable name, equal-weight, $5M floor. Sharpe ~1.4, higher return.
Measure their monthly-return CORRELATION, then blend w in {0..0.5} distressed: does any blend beat flagship-
alone on Sharpe AND/OR drawdown? Every prior blend failed because the sleeves were correlated/pro-cyclical
([[portfolio-blend]]) — distressed holds different names + was green in the 2022 bear, so maybe not.
-> BacktestResult[blend]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/blend_study.py
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


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
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

    flag, dist, spies, dates = [], [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        # FLAGSHIP
        fslot = []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if g:
                r = _ret_delist(px[pb.loc[date, g].idxmin()], date, ndate)
                if r is not None and np.isfinite(r):
                    fslot.append(float(r))
        # DISTRESSED (universe-wide, $5M floor)
        dslot = []
        for h in common:
            if not _available_at(px[h], date):
                continue
            if not (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6):
                continue
            e_ = eq.loc[date, h]
            if pd.isna(e_) or e_ >= 0 or not (pd.notna(ni.loc[date, h]) and ni.loc[date, h] <= 0):
                continue
            r = _ret_delist(px[h], date, ndate)
            if r is not None and np.isfinite(r):
                dslot.append(float(r))
        if fslot:
            flag.append(float(np.mean(fslot)))
            dist.append(float(np.mean(dslot)) if dslot else 0.0)
            spies.append(float(sp)); dates.append(ndate)

    flag, dist, spies = np.array(flag), np.array(dist), np.array(spies)
    corr = float(np.corrcoef(flag, dist)[0, 1])
    fb, db = _stats(flag, spies), _stats(dist, spies)
    # correlation in DOWN-flagship months (is distressed a hedge or pro-cyclical?)
    downs = flag < 0
    dist_when_flag_down = float(dist[downs].mean() * 100) if downs.any() else float("nan")
    print(f"\n=== THE WHOLE THING: flagship + distressed blend ===", flush=True)
    print(f"  FLAGSHIP   {fb['total']:>7}%  vsSPY {fb['vs_spy']:>7}  Sh {fb['sharpe']}  DD {fb['dd']}%  win {fb['win']}%", flush=True)
    print(f"  DISTRESSED {db['total']:>7}%  vsSPY {db['vs_spy']:>7}  Sh {db['sharpe']}  DD {db['dd']}%  win {db['win']}%", flush=True)
    print(f"  monthly-return CORRELATION = {corr:+.3f}   |  distressed avg in FLAGSHIP-DOWN months = {dist_when_flag_down:+.2f}%", flush=True)
    print(f"\n  blend (w = distressed weight):", flush=True)
    blends = {}
    for w in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        b = (1 - w) * flag + w * dist
        s = _stats(b, spies); blends[f"{w:.2f}"] = s
        tag = ""
        print(f"     w={w:.2f}  total {s['total']:>7}%  vsSPY {s['vs_spy']:>7}  Sh {s['sharpe']:>5}  DD {s['dd']:>7}%  win {s['win']}%{tag}", flush=True)

    best_sh = max(blends, key=lambda k: blends[k]["sharpe"])
    best_dd = max(blends, key=lambda k: blends[k]["dd"])   # least negative
    improves = blends[best_sh]["sharpe"] > blends["0.00"]["sharpe"] + 0.02
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(flag))},
        "flagship": fb, "distressed": db, "correlation": round(corr, 3),
        "distressed_in_flagship_down_months_pct": round(dist_when_flag_down, 2),
        "blends": blends, "best_sharpe_w": best_sh, "best_dd_w": best_dd,
        "verdict": (f"Sleeves correlate {corr:+.2f}. Best-Sharpe blend w={best_sh} (Sh {blends[best_sh]['sharpe']} vs "
                    f"flagship-alone {blends['0.00']['sharpe']}). " + (
                    f"BLEND HELPS — adding ~{best_sh} distressed raises risk-adjusted return; first real diversification "
                    "win. Distressed is a genuine satellite sleeve." if improves else
                    "Blend does NOT raise Sharpe — distressed adds return but proportional risk; hold it only if you "
                    "want more raw return and can take the deeper DD, not for diversification.")),
        "caveat": "In-sample ~5y non-crisis regime (distressed untested in a 2008-style deleveraging). No fees. "
                  "Unlevered convex sum; monthly rebalance to target weights.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "blend.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="blend", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[blend]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
