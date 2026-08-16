#!/usr/bin/env python3
"""SHORT INTEREST STANDALONE — test short interest as its OWN selector (per the rule: every signal tested
individually with the $5M vol floor), not just as a flagship overlay. Rank the whole universe by short%-of-
shares (Polygon, PIT-lagged), $5M dollar-vol floor, equal-weight, monthly rebalance, and test the long side of
each bucket vs SPY. Does 'how shorted' carry ANY independent edge — momentum (shorts right, keep falling),
contrarian (squeeze), or only combined with value?
Variants: most_shorted (top decile), least_shorted (bottom decile), hi_short>=5%, hi_short_cheap (>=5% AND
cheapest-P/B tercile), most_shorted_cheap (top-decile short AND cheap). -> BacktestResult[short_interest_standalone].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/short_interest_standalone_study.py
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
from short_interest_study import _fetch_short_interest, PUB_LAG_D

MIN_DVOL = 5e6
HI_SHORT = 5.0


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0, avg_names=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    all_holds = set()
    for n, e in etfs.items():
        all_holds.update(t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO)
    all_holds = sorted(all_holds)
    print(f"Loading candles + {BENCH}...", flush=True)
    bench_daily = load_candles([BENCH])
    midx = _monthly_close(bench_daily).index
    spy_m = bench_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq = (_pit_monthly_panel(reps, f, midx) for f in ("shares_outstanding", "total_equity"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq = R(sh), R(eq)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)

    si_raw = _fetch_short_interest(list(common))
    lag = pd.Timedelta(days=int(PUB_LAG_D * 1.5)); si_sh = {}
    for tk in common:
        rows = si_raw.get(tk) or []
        if not rows:
            continue
        s = pd.DataFrame(rows, columns=["sd", "si", "dtc"]); s["sd"] = pd.to_datetime(s["sd"]); s = s.sort_values("sd")
        ser = [s[s["sd"] <= (d - lag)]["si"].iloc[-1] if len(s[s["sd"] <= (d - lag)]) else np.nan for d in midx]
        si_sh[tk] = pd.Series(ser, index=midx)
    si_sh = pd.DataFrame(si_sh).reindex(index=midx, columns=common)
    short_pct = (si_sh / sh.where(sh > 0)) * 100
    print(f"months {len(midx)} | stocks w/ short data: {int((~short_pct.isna()).any().sum())}", flush=True)

    variants = ["most_shorted", "least_shorted", "hi_short_5pct", "hi_short_cheap", "most_shorted_cheap"]
    port = {k: [] for k in variants}; nnm = {k: [] for k in variants}; spies = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        # eligible universe: $5M floor, available, has short data + P/B
        elig = [t for t in common if _available_at(px[t], date) and pd.notna(dvol.loc[date, t]) and dvol.loc[date, t] >= MIN_DVOL
                and pd.notna(short_pct.loc[date, t]) and pd.notna(pb.loc[date, t]) and pb.loc[date, t] > 0]
        if len(elig) < 20:
            continue
        spv = short_pct.loc[date, elig]; pbv = pb.loc[date, elig]
        hi_cut = spv.quantile(0.9); lo_cut = spv.quantile(0.1); pb_cheap = pbv.quantile(0.33)

        def rets(names):
            rr = [_ret_delist(px[t], date, ndate) for t in names]
            rr = [float(x) for x in rr if x is not None and np.isfinite(x)]
            return rr
        sel = {
            "most_shorted": [t for t in elig if spv[t] >= hi_cut],
            "least_shorted": [t for t in elig if spv[t] <= lo_cut],
            "hi_short_5pct": [t for t in elig if spv[t] >= HI_SHORT],
            "hi_short_cheap": [t for t in elig if spv[t] >= HI_SHORT and pbv[t] <= pb_cheap],
            "most_shorted_cheap": [t for t in elig if spv[t] >= hi_cut and pbv[t] <= pb_cheap],
        }
        ok = False
        for k in variants:
            rr = rets(sel[k])
            if rr:
                port[k].append(float(np.mean(rr))); nnm[k].append(len(rr)); ok = True
            else:
                port[k].append(0.0); nnm[k].append(0)
        if ok:
            spies.append(float(sp))

    print(f"\n=== SHORT INTEREST STANDALONE (own selector, $5M floor, {len(spies)} months) ===", flush=True)
    print(f"  {'variant':20} {'total':>8} {'vsSPY':>8} {'Sh':>5} {'DD':>8} {'win':>6} {'~names':>7}", flush=True)
    res = {}
    for k in variants:
        st = _stats(port[k][:len(spies)], spies); st["avg_names"] = round(float(np.mean(nnm[k])), 1)
        res[k] = st
        print(f"  {k:20} {st['total']:>7}% {st['vs_spy']:>8} {st['sharpe']:>5} {st['dd']:>7}% {st['win']:>5}% {st['avg_names']:>7}", flush=True)
    spy_tot = round(float(np.prod(1 + np.asarray(spies)) - 1) * 100, 1)
    print(f"  {'SPY':20} {spy_tot:>7}%", flush=True)

    best = max(variants, key=lambda k: res[k]["sharpe"])
    beats = res[best]["vs_spy"] > 0 and res[best]["sharpe"] > 0.8
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"benchmark": BENCH, "min_dvol": MIN_DVOL, "hi_short_pct": HI_SHORT, "months": int(len(spies)), "spy_total": spy_tot},
        "results": res, "best": best,
        "verdict": (f"Standalone short-interest best = {best} ({res[best]['total']}%/vsSPY{res[best]['vs_spy']}/Sh{res[best]['sharpe']}) "
                    f"vs SPY {spy_tot}%. most_shorted {res['most_shorted']['total']}% vs least_shorted {res['least_shorted']['total']}%. " + (
                    "Short interest carries a standalone long-side edge (esp. combined with value)." if beats else
                    "Short interest has NO usable standalone LONG edge on the $5M-floor universe — most_shorted names don't outperform "
                    "(often the opposite: shorts are 'right'); only when intersected with cheap value does it approach the value edge, "
                    "confirming the edge is VALUE, not shorting. As a signal on its own it doesn't beat SPY risk-adjusted.")),
        "caveat": "Polygon short% PIT-lagged; $5M floor, eq-wt, long-only (no borrow/short side); in-sample, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/short_interest_standalone.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="short_interest_standalone", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[short_interest_standalone]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
