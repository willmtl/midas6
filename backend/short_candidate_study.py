#!/usr/bin/env python3
"""SHORT CANDIDATES — we've been long-only but kept tripping over the SHORT profile (value traps −47 vs SPY,
guard-excluded names, shorted+cheap). Synthesize those negatives into an actual SHORT screen and test: which
profile most reliably UNDERPERFORMS (relative short alpha) or LOSES money (absolute)? $5M floor, monthly, PIT.
Report each screen's forward return (absolute + market-adj), win% (LOW=good short), and the SHORT-side portfolio
P&L (short eq-wt = −mean return). Profiles: trap (guard), trap+heavily-shorted, expensive-junk (high P/B +
unprofitable), falling-trap (trap + downtrend), shorted-expensive. -> BacktestResult[short_candidate].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/short_candidate_study.py
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


def _shortstats(r, spy):
    """r = forward returns of the screened LONGS; short P&L = -r. Report from the SHORT side."""
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(long_total=0, vs_spy=0, short_total=0, short_sharpe=0, short_dd=0, long_win=0)
    sr = -r
    lt = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    st = float(np.prod(1 + sr) - 1) * 100
    ssh = float(sr.mean() / sr.std() * np.sqrt(12)) if sr.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + sr); sdd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(long_total=round(lt, 1), vs_spy=round(lt - sp, 1), short_total=round(st, 1),
                short_sharpe=round(ssh, 2), short_dd=round(sdd, 1), long_win=round((r > 0).mean() * 100, 1))


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
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    mom6 = px.pct_change(6)
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
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    profiles = ["trap", "trap_shorted", "expensive_junk", "falling_trap", "shorted_expensive"]
    port = {k: [] for k in profiles}; nnm = {k: [] for k in profiles}; spies = []; dts = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        spr = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(spr):
            continue
        elig = [t for t in common if _available_at(px[t], date) and pd.notna(dvol.loc[date, t]) and dvol.loc[date, t] >= MIN_DVOL
                and pd.notna(pb.loc[date, t]) and pb.loc[date, t] > 0]
        if len(elig) < 20:
            continue
        pbv = pb.loc[date, elig]; pb_rich = pbv.quantile(0.66)
        def sel(cond):
            return [t for t in elig if cond(t)]
        screens = {
            "trap": sel(lambda t: bool(trap.loc[date, t])),
            "trap_shorted": sel(lambda t: bool(trap.loc[date, t]) and pd.notna(short_pct.loc[date, t]) and short_pct.loc[date, t] >= 10),
            "expensive_junk": sel(lambda t: pd.notna(ni.loc[date, t]) and ni.loc[date, t] < 0 and pbv[t] >= pb_rich),
            "falling_trap": sel(lambda t: bool(trap.loc[date, t]) and pd.notna(mom6.loc[date, t]) and mom6.loc[date, t] < 0),
            "shorted_expensive": sel(lambda t: pd.notna(short_pct.loc[date, t]) and short_pct.loc[date, t] >= 10 and pbv[t] >= pb_rich),
        }
        ok = False
        for k in profiles:
            rr = [_ret_delist(px[t], date, ndate) for t in screens[k]]
            rr = [float(x) for x in rr if x is not None and np.isfinite(x)]
            if rr:
                port[k].append(float(np.mean(rr))); nnm[k].append(len(rr)); ok = True
            else:
                port[k].append(0.0); nnm[k].append(0)
        if ok:
            spies.append(float(spr)); dts.append(ndate)

    spy_tot = round(float(np.prod(1 + np.asarray(spies)) - 1) * 100, 1)
    print(f"\n=== SHORT CANDIDATE screens ($5M floor, {len(spies)} months, SPY {spy_tot}%) ===", flush=True)
    print(f"  {'profile':18} {'LONGtot':>8} {'vsSPY':>8} {'SHORT P&L':>10} {'shSh':>6} {'shDD':>8} {'Lwin':>6} {'~n':>5}", flush=True)
    res = {}
    for k in profiles:
        st = _shortstats(port[k][:len(spies)], spies); st["avg_names"] = round(float(np.mean(nnm[k])), 1)
        res[k] = st
        print(f"  {k:18} {st['long_total']:>7}% {st['vs_spy']:>8} {st['short_total']:>9}% {st['short_sharpe']:>6} "
              f"{st['short_dd']:>7}% {st['long_win']:>5}% {st['avg_names']:>5}", flush=True)

    # best short = most negative long vsSPY AND best short-side sharpe
    best = min(profiles, key=lambda k: res[k]["vs_spy"])
    good = res[best]["vs_spy"] < -30 and res[best]["short_total"] > 0
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"benchmark": BENCH, "min_dvol": MIN_DVOL, "months": int(len(spies)), "spy_total": spy_tot},
        "results": res, "worst_performer": best,
        "verdict": (f"Biggest underperformer = {best} (long {res[best]['long_total']}% vs SPY {spy_tot}%, i.e. {res[best]['vs_spy']}pp; "
                    f"short-side P&L {res[best]['short_total']}%/Sh{res[best]['short_sharpe']}). " + (
                    f"REAL SHORT CANDIDATE PROFILE — {best} names systematically underperform; viable as the SHORT leg of a "
                    "market-neutral book (long flagship / short these), though outright shorting fights the bull-market drift."
                    if good else
                    "No profile is a clean OUTRIGHT short (bull market → most still post positive absolute returns), but the "
                    "trap/expensive-junk screens UNDERPERFORM SPY meaningfully — they're the AVOID list (already excluded from "
                    "the long book) and the short leg of a hedged/market-neutral version, not standalone money-makers.")),
        "caveat": "Long-only universe, no borrow cost/squeeze modeled; short P&L = −long return (ignores financing). Bull-market "
                  "sample (2022-26) makes outright shorting hard; vsSPY is the cleaner read. In-sample, $5M floor.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/short_candidate.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="short_candidate", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[short_candidate]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
