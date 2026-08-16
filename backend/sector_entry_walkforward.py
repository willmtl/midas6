#!/usr/bin/env python3
"""WALK-FORWARD the ACCELERATION sector signal vs 6mo momentum — is the +422% real or overfit? Runs
mom6 (baseline), accel, accel_conf; slices each monthly return series into first/second half, thirds,
per-year (vs SPY). Acceleration's +280pp edge must show in BOTH halves and most years to be trusted.
-> BacktestResult[sector_entry_wf] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sector_entry_walkforward.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "sector_entry_wf.json"
TOP_N = 10
SIGS = ["mom6", "accel", "accel_conf"]


def _seg(r, s):
    r = np.asarray(r, float); s = np.asarray(s, float); n = len(r)
    if n == 0:
        return dict(vs_spy=0, t_stat=None, win_pct=0, months=0)
    strat = float(np.prod(1 + r) - 1) * 100
    spy = float(np.prod(1 + s) - 1) * 100
    t = _tstat_from_returns(list(r - s))
    return dict(vs_spy=round(strat - spy, 1), t_stat=round(t, 2) if t is not None else None,
                win_pct=round((r > s).mean() * 100, 1), months=n)


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
    m3, m6 = etf_m.pct_change(3), etf_m.pct_change(6)
    accel = m3 - m3.shift(3)
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
    warmup = 9

    def pick_in(etf, date, ndate):
        _, holds = sector_map.get(etf, (etf, []))
        cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                 and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
        ld = [c for c in cands if bool(low_debt.loc[date, c])]
        use = ld or cands
        if not use:
            return None
        r = _ret_delist(px[pb.loc[date, use].idxmin()], date, ndate)
        return float(r) if (r is not None and np.isfinite(r)) else None

    def sel(sig, date):
        if sig == "mom6":
            return list(m6.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        if sig == "accel":
            return list(accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        if sig == "accel_conf":
            pos = m6.loc[date][m6.loc[date] > 0].index
            a = accel.loc[date, [e for e in pos if e in accel.columns]].dropna()
            return list(a.sort_values(ascending=False).head(TOP_N).index)
        return []

    def series(sig):
        rets, spies, dates = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            slot = [pick_in(e, date, ndate) for e in sel(sig, date)]
            slot = [x for x in slot if x is not None]
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); dates.append(ndate)
        return np.array(rets), np.array(spies), pd.to_datetime(dates)

    results = {}
    for sig in SIGS:
        r, s, d = series(sig)
        n = len(r); h = n // 2
        halves = {"first": _seg(r[:h], s[:h]), "second": _seg(r[h:], s[h:])}
        years = {str(y): _seg(r[d.year == y], s[d.year == y]) for y in sorted(set(d.year)) if (d.year == y).sum() >= 3}
        both = halves["first"]["vs_spy"] > 0 and halves["second"]["vs_spy"] > 0
        yb = sum(1 for v in years.values() if v["vs_spy"] > 0)
        results[sig] = {"full": _seg(r, s), "halves": halves, "years": years,
                        "both_halves_positive": bool(both), "years_beat": f"{yb}/{len(years)}",
                        "weaker_half": min(halves["first"]["vs_spy"], halves["second"]["vs_spy"]),
                        "robust": bool(both and yb >= 0.6 * len(years))}

    print("\n=== ACCELERATION WALK-FORWARD ===", flush=True)
    print(f"  {'signal':11} {'FULL':>7} {'1st':>8} {'2nd':>8} {'weak':>8} {'yrs':>5} robust", flush=True)
    for sig in SIGS:
        R = results[sig]
        print(f"  {sig:11} {R['full']['vs_spy']:>6}% {R['halves']['first']['vs_spy']:>7}% "
              f"{R['halves']['second']['vs_spy']:>7}% {R['weaker_half']:>7}% {R['years_beat']:>5} "
              f"{'YES' if R['robust'] else 'no'}", flush=True)
    print("\n  per-year vs SPY:", flush=True)
    for sig in SIGS:
        print(f"    {sig:11} " + "  ".join(f"{y}:{v['vs_spy']:+.0f}" for y, v in results[sig]["years"].items()), flush=True)

    ac, m6r = results["accel"], results["mom6"]
    survives = ac["both_halves_positive"] and ac["weaker_half"] > m6r["weaker_half"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "results": results, "accel_survives": bool(survives),
        "verdict": (f"Acceleration {'SURVIVES' if survives else 'does NOT survive'} walk-forward: full "
                    f"+{ac['full']['vs_spy']}% vs mom6 +{m6r['full']['vs_spy']}%; accel halves "
                    f"{ac['halves']['first']['vs_spy']}/{ac['halves']['second']['vs_spy']} (weaker {ac['weaker_half']}) "
                    f"vs mom6 weaker {m6r['weaker_half']}. "
                    + ("Both halves positive AND weaker-half beats mom6 -> the acceleration edge is REAL; promote it "
                       "to the sector signal." if survives else "Edge concentrated -> treat with caution.")),
        "caveat": "Subperiod split (not true holdout); rules-based so this is stability. 9mo warmup shortens window.",
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
            kind="sector_entry_wf", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                              "computed_at": timezone.now()})
        print("Saved BacktestResult[sector_entry_wf]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
