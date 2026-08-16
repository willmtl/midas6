#!/usr/bin/env python3
"""PER-SECTOR SEASONALITY TILT — user: use each sector's calendar-month seasonality as an EXTRA signal on top of
momentum (ts_diagnostics showed Cannabis-Aug, Oil-Mar, Semis/Solar-May, market Nov-strong/Sep-weak). Test it as a
return-additive overlay ([[return-priority]]). CRITICAL: seasonal score is POINT-IN-TIME — at each rebalance the
score for a sector = mean of its returns in the UPCOMING calendar month using ONLY prior years (no look-ahead).

Arms (flagship base: top-10 accel -> cheapest as-traded-P/B guard low-debt div_2x, monthly):
  baseline          momentum accel only
  seasonal_blend    rank sectors by (accel_rank + seasonal_rank), take top-10  (seasonality as co-equal factor)
  seasonal_tiltwt   accel top-10, but 2x conviction weight to picks whose sector is in a historically-strong
                    month (PIT seasonal score > 0)
  seasonal_only     rank sectors by PIT seasonal score alone (sanity — should be weak)
Report LEADING WITH TOTAL RETURN. -> BacktestResult[seasonality_tilt] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/seasonality_tilt_study.py
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
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "seasonality_tilt.json"


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
    ret_m = etf_m.pct_change()
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
    dvol = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)

    # PIT seasonal score: score[i][e] = mean of e's returns in month (midx[i+1].month) over rows STRICTLY before midx[i]
    months = ret_m.index.month.values
    seas = pd.DataFrame(index=midx, columns=etf_tk, dtype=float)
    for i in range(len(midx) - 1):
        hold_month = midx[i + 1].month
        prior = ret_m.iloc[:i]                                   # strictly before rebalance date -> no look-ahead
        if len(prior) < 12:
            continue
        mask = prior.index.month == hold_month
        if mask.sum() >= 2:
            seas.loc[midx[i]] = prior[mask].mean()
    rank_a = accel.rank(axis=1, ascending=False)
    rank_s = seas.rank(axis=1, ascending=False)

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    def run(order_fn, seasonal_wt=False):
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            order = order_fn(date)[:TOP_N]
            if not order:
                continue
            held = set(); wsum = rr = 0.0
            for etf in order:
                p = pick(etf, date, held)
                if not p:
                    continue
                held.add(p)
                r = _ret_delist(px[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = 1.0
                sc = seas.loc[date, etf] if etf in seas.columns else np.nan
                if seasonal_wt and pd.notna(sc) and sc > 0:
                    w *= CONV
                wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return _perf(rets, spies)

    def o_accel(d):
        s = accel.loc[d].dropna(); return list(s.sort_values(ascending=False).index)

    def o_blend(d):
        b = (rank_a.loc[d] + rank_s.loc[d]).dropna()
        return list(b.sort_values().index) if len(b) else list(accel.loc[d].dropna().sort_values(ascending=False).index)

    def o_seasonal(d):
        s = seas.loc[d].dropna(); return list(s.sort_values(ascending=False).index) if len(s) else []

    results = {
        "baseline": run(o_accel),
        "seasonal_blend": run(o_blend),
        "seasonal_tiltwt": run(o_accel, seasonal_wt=True),
        "seasonal_only": run(o_seasonal),
    }
    base = results["baseline"]; order = sorted(results, key=lambda k: results[k]["total"], reverse=True)
    print(f"\n=== SEASONALITY TILT (PIT; sorted by TOTAL RETURN) ===", flush=True)
    print(f"  {'mode':<17}{'total':>9}{'annual':>8}{'vsSPY':>8}{'Sharpe':>8}{'DD':>8}{'t':>6}", flush=True)
    for k in order:
        r = results[k]
        star = "  <= baseline" if k == "baseline" else ("  <= BEST" if k == order[0] else "")
        print(f"  {k:<17}{r['total']:>8}%{r['annual']:>7}%{r['vs_spy']:>8}{r['sharpe']:>8}{r['dd']:>7}%{str(r['t_stat']):>6}{star}", flush=True)
    best = order[0]; b = results[best]
    verdict = (f"Baseline {base['total']}% ({base['annual']}%/yr). BEST = {best}: {b['total']}% ({b['annual']}%/yr) "
               f"= {b['total'] - base['total']:+.0f}pp. "
               + ("Seasonality tilt ADDS return." if best != "baseline" and b["total"] > base["total"] + 10
                  else "Seasonality does NOT add return over momentum — PIT per-sector seasonality is too noisy on "
                       "~5y (only 2-4 samples per sector-month); the in-sample Nov/Sep pattern doesn't survive PIT."))
    print("\n" + verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "conv": CONV, "months": int(base["months"]),
                       "seasonal": "PIT mean of upcoming-month returns over prior years (>=2 samples)"},
            "results": results, "best": best, "verdict": verdict,
            "caveat": "PIT seasonality (no look-ahead) but only ~5y of data = 2-4 observations per sector-month, so "
                      "the seasonal score is very noisy. Blend = accel_rank + seasonal_rank co-equal. PIT/no-fees/"
                      "survivorship as base, div_2x, $5M vol."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="seasonality_tilt", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[seasonality_tilt]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
