#!/usr/bin/env python3
"""VALUE-METRIC SWEEP — the flagship picks the cheapest positive-P/B name in each top-accel sector. Is P/B the
best selection metric for RETURN ([[return-priority]])? Test alternatives on the SAME engine (accel top-10,
guard low-debt $5M, div_2x, monthly) — only the within-sector PICK metric changes:
  pb          cheapest Price/Book (baseline, as-traded)
  ev_ebit     cheapest EV/EBIT   (EV = mktcap + total_debt - cash; EBIT = operating_income)
  fcf_yield   highest FCF yield  (free_cash_flow / mktcap)
  earnings_y  highest earnings yield (eps_diluted*shares / mktcap = NI/mktcap)
  pb_x_mom    cheapest-P/B HALF of the sector, then highest 6mo price momentum (value x momentum)
Report LEADING WITH TOTAL RETURN. -> BacktestResult[value_metric] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/value_metric_study.py
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

TOP_N = 10; CONV = 2.0; MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "value_metric.json"


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
    panels = {f: _pit_monthly_panel(reps, f, midx) for f in
              ("shares_outstanding", "total_equity", "net_income", "total_debt", "operating_income",
               "free_cash_flow", "cash_and_equivalents", "eps_diluted", "revenue")}
    common = stock_m.columns
    for f in ("shares_outstanding", "total_equity"):
        common = common.intersection(panels[f].columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]
    P = {f: R(panels[f]) for f in panels}
    as_traded = price_basis.as_traded_close(px)
    mktcap = as_traded * P["shares_outstanding"]
    pb = mktcap / P["total_equity"].where(P["total_equity"] != 0)
    ev = mktcap + P["total_debt"].fillna(0) - P["cash_and_equivalents"].fillna(0)
    ev_ebit = ev / P["operating_income"].where(P["operating_income"] > 0)
    fcf_yield = P["free_cash_flow"] / mktcap.where(mktcap > 0)
    earnings_y = P["net_income"] / mktcap.where(mktcap > 0)
    mom6 = px.pct_change(6)
    trap = (P["net_income"] < 0) & (~(P["total_equity"] >= P["total_equity"].shift(12))) & (~(P["net_income"] > P["net_income"].shift(4)))
    low = (P["total_debt"] / P["total_equity"].where(P["total_equity"] != 0)) < 1.0
    adl_m, dvol = {}, {}
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

    def eligible(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        return [x for x in c if bool(low.loc[date, x])] or c

    def choose(metric, cands, date):
        if not cands:
            return None
        if metric == "pb":
            return min(cands, key=lambda h: pb.loc[date, h])
        if metric == "ev_ebit":
            v = [h for h in cands if pd.notna(ev_ebit.loc[date, h]) and ev_ebit.loc[date, h] > 0]
            return min(v, key=lambda h: ev_ebit.loc[date, h]) if v else min(cands, key=lambda h: pb.loc[date, h])
        if metric == "fcf_yield":
            v = [h for h in cands if pd.notna(fcf_yield.loc[date, h])]
            return max(v, key=lambda h: fcf_yield.loc[date, h]) if v else min(cands, key=lambda h: pb.loc[date, h])
        if metric == "earnings_y":
            v = [h for h in cands if pd.notna(earnings_y.loc[date, h])]
            return max(v, key=lambda h: earnings_y.loc[date, h]) if v else min(cands, key=lambda h: pb.loc[date, h])
        if metric == "pb_x_mom":
            ranked = sorted(cands, key=lambda h: pb.loc[date, h])
            half = ranked[: max(1, len(ranked) // 2)]
            m = [h for h in half if pd.notna(mom6.loc[date, h])]
            return max(m, key=lambda h: mom6.loc[date, h]) if m else half[0]
        return min(cands, key=lambda h: pb.loc[date, h])

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    def run(metric):
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0
            for etf in top:
                cands = eligible(etf, date, held)
                p = choose(metric, cands, date)
                if not p:
                    continue
                held.add(p)
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = CONV if accumulating(p, date) else 1.0
                wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return _perf(rets, spies)

    metrics = ["pb", "ev_ebit", "fcf_yield", "earnings_y", "pb_x_mom"]
    results = {m: run(m) for m in metrics}
    order = sorted(results, key=lambda k: results[k]["total"], reverse=True)
    print(f"\n=== VALUE-METRIC SWEEP (within-sector pick; top10 div2x; sorted by TOTAL RETURN) ===", flush=True)
    print(f"  {'metric':<12}{'total':>9}{'annual':>8}{'vsSPY':>8}{'Sharpe':>8}{'DD':>8}{'t':>6}", flush=True)
    for k in order:
        r = results[k]
        star = "  <= baseline (P/B)" if k == "pb" else ("  <= BEST" if k == order[0] else "")
        print(f"  {k:<12}{r['total']:>8}%{r['annual']:>7}%{r['vs_spy']:>8}{r['sharpe']:>8}{r['dd']:>7}%{str(r['t_stat']):>6}{star}", flush=True)
    best = order[0]; b = results[best]; base = results["pb"]
    verdict = (f"Baseline P/B {base['total']}% ({base['annual']}%/yr). BEST = {best}: {b['total']}% "
               f"({b['annual']}%/yr) = {b['total'] - base['total']:+.0f}pp. "
               + ("An alternative value metric beats P/B for return." if b["total"] > base["total"] + 10 and best != "pb"
                  else "P/B is as good or better than the alternatives — keep it."))
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "conv": CONV, "months": int(base["months"]), "objective": "MAX TOTAL RETURN"},
            "results": results, "best": best, "verdict": verdict,
            "caveat": "Only the within-sector PICK metric changes; accel top-10, guard, low-debt, div_2x, $5M vol "
                      "held constant. EV/EBIT & FCF-yield fall back to P/B when the fundamental is missing/negative. "
                      "PIT panels, as-traded mktcap, no fees, survivorship as base."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="value_metric", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[value_metric]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
