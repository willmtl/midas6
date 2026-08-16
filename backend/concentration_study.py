#!/usr/bin/env python3
"""CONCENTRATION SWEEP — return-additive per [[return-priority]] (user maximizes ABSOLUTE return, not Sharpe/DD).
The flagship holds the top-10 accel sectors' cheapest-P/B picks, A/D-divergence names 2x-weighted (div_2x).
Concentrating harder should raise return (at higher variance, which is fine): (a) fewer sectors = only the
STRONGEST-accel picks, (b) steeper conviction = lean harder into the A/D-divergence names.

Matrix: TOP_N in {3,5,8,10} x conviction mult in {1(equal),2(baseline),3,4}. Same pick logic (accel top-N ->
cheapest as-traded-P/B guard low-debt $5M), A/D-divergence names get the mult, renormalized. Monthly, TR.
Report LEADING WITH TOTAL RETURN (then vsSPY/Sharpe/DD/t as context). -> BacktestResult[concentration] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/concentration_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_NS = [3, 4, 5, 6, 8, 10]
CONVS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "concentration.json"


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), annual=round(ann, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                dd=round(dd, 1), t_stat=round(t, 2) if t is not None else None, months=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
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
    dvol, adl_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    def run(top_n, conv):
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(top_n).index
            held = set(); wsum = rr = 0.0
            for etf in top:
                p = pick(etf, date, held)
                if not p:
                    continue
                held.add(p)
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = conv if accumulating(p, date) else 1.0
                wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return _perf(rets, spies)

    results = {}
    for tn in TOP_NS:
        for cv in CONVS:
            results[f"top{tn}_div{int(cv)}x"] = run(tn, cv)

    base = results["top10_div2x"]
    order = sorted(results, key=lambda k: results[k]["total"], reverse=True)
    print(f"\n=== CONCENTRATION SWEEP (sorted by TOTAL RETURN; baseline = top10_div2x) ===", flush=True)
    print(f"  {'variant':<16}{'total':>9}{'annual':>8}{'vsSPY':>8}{'Sharpe':>8}{'DD':>8}{'t':>6}", flush=True)
    for k in order:
        r = results[k]
        star = "  <= baseline" if k == "top10_div2x" else ("  <= BEST RETURN" if k == order[0] else "")
        print(f"  {k:<16}{r['total']:>8}%{r['annual']:>7}%{r['vs_spy']:>8}{r['sharpe']:>8}{r['dd']:>7}%"
              f"{str(r['t_stat']):>6}{star}", flush=True)

    best = order[0]; b = results[best]
    verdict = (
        f"Baseline top10_div2x {base['total']}% ({base['annual']}%/yr, Sh{base['sharpe']}, DD{base['dd']}%). "
        f"BEST RETURN = {best}: {b['total']}% ({b['annual']}%/yr, Sh{b['sharpe']}, DD{b['dd']}%, t{b['t_stat']}) "
        f"= {b['total'] - base['total']:+.0f}pp vs baseline. "
        + ("Concentrating harder RAISES return" if b["total"] > base["total"] + 10 else
           "Concentrating does NOT materially raise return")
        + f" (best DD {b['dd']}% vs baseline {base['dd']}% — the return-priority trade-off)."
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_ns": TOP_NS, "convs": CONVS, "min_dvol": MIN_DVOL, "benchmark": BENCH,
                   "months": int(base["months"]), "pb_basis": "as-traded", "objective": "MAX TOTAL RETURN"},
        "results": results, "best_return": best, "verdict": verdict,
        "caveat": "Fewer sectors / steeper conviction = higher concentration = higher variance & drawdown (accepted "
                  "under return-priority). PIT, no fees, present-day-holdings survivorship, TR, as-traded P/B. "
                  "top3 can have <3 names some months. div1x = equal weight (no A/D tilt).",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="concentration", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                            "computed_at": timezone.now()})
        print("Saved BacktestResult[concentration]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
