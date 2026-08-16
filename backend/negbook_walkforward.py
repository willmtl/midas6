#!/usr/bin/env python3
"""WALK-FORWARD the distressed sleeve — is +467% DISTRIBUTED or a mania artifact? The distressed-neg-book
basket is the most regime-suspect finding of the session: buying junk that pops is exactly what works in a
2021-25 speculative-mania regime and exactly what dies in a 2008-style deleveraging. Apply the SAME gate the
flagship passed: per-year vsSPY, both halves, and fragility (does dropping the best few months kill it?).
Reference vs posbook_cheap (the tame value basket). -> BacktestResult[negbook_walkforward].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/negbook_walkforward.py
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


def _tot(r):
    return float(np.prod(1 + np.asarray(r, float)) - 1) * 100 if len(r) else 0.0


def _sh(r):
    r = np.asarray(r, float)
    return float(r.mean() / r.std() * np.sqrt(12)) if len(r) > 1 and r.std() > 1e-9 else 0.0


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    all_holds = set()
    for n, e in etfs.items():
        for t in sector_holdings.get_holdings(n):
            if t not in (e, BENCH) and t not in CRYPTO:
                all_holds.add(t)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    midx = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk}).index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
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

    dates, dist, cheap, spies, ndist = [], [], [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        d_r, c_r = [], []
        for h in common:
            if not _available_at(px[h], date):
                continue
            if not (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6):
                continue
            e_ = eq.loc[date, h]
            if pd.isna(e_):
                continue
            r = _ret_delist(px[h], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            r = float(r)
            if e_ < 0 and pd.notna(ni.loc[date, h]) and ni.loc[date, h] <= 0:
                d_r.append(r)
            elif e_ > 0 and pd.notna(pb.loc[date, h]) and pb.loc[date, h] < 1.0:
                c_r.append(r)
        dates.append(ndate); spies.append(float(sp))
        dist.append(float(np.mean(d_r)) if d_r else 0.0); ndist.append(len(d_r))
        cheap.append(float(np.mean(c_r)) if c_r else 0.0)

    df = pd.DataFrame({"date": pd.to_datetime(dates), "dist": dist, "cheap": cheap, "spy": spies, "n": ndist})
    df["yr"] = df["date"].dt.year
    full_d, full_c, full_s = _tot(df.dist), _tot(df.cheap), _tot(df.spy)
    print(f"\n=== WALK-FORWARD: distressed-negbook vs SPY (reference: posbook_cheap) ===", flush=True)
    print(f"  FULL   distressed {full_d:>7.1f}%  (vsSPY {full_d-full_s:>+7.1f})   cheap {full_c:>7.1f}%  SPY {full_s:>6.1f}%", flush=True)

    # halves
    h = len(df) // 2
    h1d, h2d = _tot(df.dist[:h]), _tot(df.dist[h:]); h1s, h2s = _tot(df.spy[:h]), _tot(df.spy[h:])
    print(f"  1st-half distressed {h1d:>+7.1f}% (vsSPY {h1d-h1s:>+7.1f})   2nd-half {h2d:>+7.1f}% (vsSPY {h2d-h2s:>+7.1f})", flush=True)

    # per-year
    peryear = {}
    print("  per-year (distressed vsSPY | names/mo):", flush=True)
    beats = 0
    for yr, g in df.groupby("yr"):
        d_, s_ = _tot(g.dist), _tot(g.spy); vs = d_ - s_; peryear[int(yr)] = round(vs, 1)
        beats += vs > 0
        print(f"     {yr}: dist {d_:>+7.1f}%  SPY {s_:>+6.1f}%  vsSPY {vs:>+7.1f}%   (~{g.n.mean():.0f} names)", flush=True)

    # fragility: drop best-k months
    sd = np.sort(df.dist.values)[::-1]
    drop1 = _tot(df.dist.values[np.argsort(df.dist.values)][:-1])
    drop3 = _tot(df.dist.values[np.argsort(df.dist.values)][:-3])
    topmonth = df.loc[df.dist.idxmax()]
    print(f"  FRAGILITY: drop best month -> {drop1:.1f}% (from {full_d:.1f}); drop best 3 -> {drop3:.1f}%", flush=True)
    print(f"     biggest month = {topmonth['date'].date()} +{topmonth['dist']*100:.1f}% ({int(topmonth['n'])} names)", flush=True)

    robust = (h1d - h1s > 0) and (h2d - h2s > 0) and (beats >= len(peryear) - 1)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "full": {"distressed_total": round(full_d, 1), "vs_spy": round(full_d - full_s, 1),
                 "sharpe": round(_sh(df.dist), 2), "cheap_total": round(full_c, 1), "spy_total": round(full_s, 1)},
        "halves": {"h1_vs_spy": round(h1d - h1s, 1), "h2_vs_spy": round(h2d - h2s, 1)},
        "per_year_vs_spy": peryear, "years_beat_spy": f"{beats}/{len(peryear)}",
        "fragility": {"full": round(full_d, 1), "drop_best_1": round(drop1, 1), "drop_best_3": round(drop3, 1),
                      "biggest_month": f"{topmonth['date'].date()} +{topmonth['dist']*100:.1f}%"},
        "robust": bool(robust),
        "verdict": ("Distressed sleeve is ROBUST — edge distributed across halves and years, not one mania stretch. "
                    "Treat as a real (high-octane) satellite sleeve, sized small for the tail/DD."
                    if robust else
                    "Distressed sleeve is REGIME-CONCENTRATED — the return leans on a few months/years (mania artifact "
                    "risk). Do NOT size it as a core sleeve; it's a bet on the speculative regime repeating."),
        "caveat": "In-sample, ONE ~5y regime (2021-25 speculative mania) — subperiod split is NOT a true holdout and "
                  "canNOT test a 2008/2000 deleveraging where distressed -> 0. No fees. ~11 names = tail-dependent.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "negbook_walkforward.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="negbook_walkforward", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[negbook_walkforward]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
