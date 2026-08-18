#!/usr/bin/env python3
"""ANALYST PRICE-TARGET as a TRADEABLE PORTFOLIO SIGNAL. The event study (analyst_ratings_study) showed
IMPLIED-UPSIDE LEVEL (price_target / price - 1) predicts forward returns monotonically (Q0 +3.8% -> Q4
+8.7% fwd63). This tests whether that survives as an actual monthly LONG portfolio: each month rank every
name by its latest analyst implied-upside (target within the last 90d ÷ month-end close), form quintiles,
hold equal-weight one month, compare quintile portfolios + long-Q5 vs SPY. Also a Q5-Q1 long-short.
Answers: is high analyst-target-upside a tradeable monthly edge, or just an event-window blip / vol tail?
-> .data/studies/analyst_upside_portfolio.json + BacktestResult[analyst_upside_portfolio].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/analyst_upside_portfolio.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection
from studies import _tstat_from_returns

SRC = Path("/app/.data/analyst_ratings.jsonl")
OUT = Path("/app/.data/studies/analyst_upside_portfolio.json")
STALE_DAYS = 90       # a target older than this is dropped (no longer "current")


def _stats(r, spy):
    r = np.asarray(r, float)
    if len(r) == 0:
        return {"total": 0, "vs_spy": 0, "sharpe": 0, "t": None, "n": 0}
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    t = _tstat_from_returns(list(r))
    return {"total": round(tot, 1), "vs_spy": round(tot - sp, 1), "sharpe": round(sh, 2),
            "t": round(t, 2) if t is not None else None, "n": len(r)}


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, BacktestResult
    from django.utils import timezone

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("price_target") and r.get("date")]
    tickers = sorted({r["ticker"] for r in rows})
    print(f"price-target rows: {len(rows):,} | tickers: {len(tickers)}", flush=True)

    # monthly close panel (+ SPY)
    def monthly_close(tks):
        out = {}
        for tk in tks:
            q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
            if len(q) > 60:
                s = pd.Series({pd.Timestamp(d): float(c) for d, c in q}).sort_index()
                out[tk] = s.resample("ME").last()
        return pd.DataFrame(out)
    closes = monthly_close(tickers)
    spy = monthly_close(["SPY"])["SPY"]
    midx = closes.index
    print(f"monthly panel: {closes.shape[1]} tickers x {len(midx)} months", flush=True)

    # monthly latest-target panel: last target as-of each month-end, ffilled up to STALE_DAYS, else NaN
    tgt = {}
    for tk in tickers:
        pts = [(pd.Timestamp(r["date"]), float(r["price_target"])) for r in rows if r["ticker"] == tk]
        if not pts:
            continue
        s = pd.Series({d: v for d, v in pts}).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        # as-of each month-end: latest target on/before, and its age
        m = s.reindex(s.index.union(midx)).sort_index().ffill().reindex(midx)
        age = pd.Series(midx, index=midx).astype("int64")  # placeholder; compute age via last valid index
        last_dt = s.reindex(s.index.union(midx)).sort_index().ffill().reindex(midx)
        # recompute age: for each month, days since the last target date
        last_date = pd.Series(index=midx, dtype="datetime64[ns]")
        li = s.index
        for d in midx:
            prior = li[li <= d]
            last_date[d] = prior[-1] if len(prior) else pd.NaT
        age_days = (pd.Series(midx, index=midx) - last_date).dt.days
        m = m.where(age_days <= STALE_DAYS)
        tgt[tk] = m
    tgt = pd.DataFrame(tgt).reindex(index=midx, columns=closes.columns)

    upside = (tgt / closes.where(closes > 0)) - 1        # implied upside per name per month (PIT)

    # monthly quintile portfolios (equal-weight, next-month return)
    qret = {q: [] for q in range(5)}
    ls, spies = [], []
    coverage = []
    for i in range(len(midx) - 1):
        d, nd = midx[i], midx[i + 1]
        u = upside.loc[d].dropna()
        u = u[np.isfinite(u)]
        # next-month return for names present both months
        fwd = (closes.loc[nd] / closes.loc[d] - 1)
        u = u[[t for t in u.index if pd.notna(fwd.get(t))]]
        if len(u) < 25:
            continue
        coverage.append(len(u))
        ranks = pd.qcut(u.rank(method="first"), 5, labels=False)
        sp = spy.get(nd) / spy.get(d) - 1 if pd.notna(spy.get(nd)) and pd.notna(spy.get(d)) else np.nan
        if not np.isfinite(sp):
            continue
        qmean = {}
        for qi in range(5):
            names = ranks[ranks == qi].index
            rr = float(fwd[names].mean()) if len(names) else np.nan
            qmean[qi] = rr
            if np.isfinite(rr):
                qret[qi].append(rr)
        if np.isfinite(qmean.get(4, np.nan)) and np.isfinite(qmean.get(0, np.nan)):
            ls.append(qmean[4] - qmean[0]); spies.append(float(sp))

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "stale_days": STALE_DAYS,
               "n_rows": len(rows), "avg_names_ranked": round(float(np.mean(coverage)), 0) if coverage else 0,
               "months": len(spies)}
    print(f"\navg names ranked/month: {payload['avg_names_ranked']:.0f} | months: {payload['months']}\n", flush=True)
    print("=== implied-upside QUINTILE portfolios (Q0=lowest upside .. Q4=highest), long-only next-month ===", flush=True)
    # align each quintile's spy series (use the long-short spies as the common benchmark set length differs; recompute per-q vs SPY over same months is approximate -> report total + Sharpe)
    qstats = {}
    for qi in range(5):
        r = np.asarray(qret[qi], float)
        tot = float(np.prod(1 + r) - 1) * 100 if len(r) else 0
        sh = float(r.mean() / r.std() * np.sqrt(12)) if len(r) and r.std() > 1e-9 else 0
        t = _tstat_from_returns(list(r)) if len(r) else None
        qstats[f"Q{qi}"] = {"total": round(tot, 1), "sharpe": round(sh, 2),
                            "t": round(t, 2) if t is not None else None, "mean_mo": round(float(r.mean()) * 100, 2) if len(r) else 0}
        print(f"  Q{qi}  total {tot:>8.1f}%  mean/mo {r.mean()*100:>+5.2f}%  Sh {sh:>5.2f}  t {t}", flush=True)
    payload["quintiles"] = qstats
    payload["long_short_Q4_Q0"] = _stats(ls, spies)
    spy_tot = float(np.prod(1 + np.asarray(spies)) - 1) * 100 if spies else 0
    payload["spy_total_same_months"] = round(spy_tot, 1)
    lsx = payload["long_short_Q4_Q0"]
    print(f"\n  LONG-SHORT Q4-Q0: total {lsx['total']}%  Sharpe {lsx['sharpe']}  t {lsx['t']}  (n={lsx['n']} months)", flush=True)
    print(f"  SPY same months: {spy_tot:.1f}%  |  Q4 long-only vs SPY: {qstats['Q4']['total']-spy_tot:+.1f}pp", flush=True)

    q4, q0 = qstats["Q4"]["total"], qstats["Q0"]["total"]
    payload["verdict"] = (f"Q4 (highest analyst upside) {q4}% vs Q0 {q0}% vs SPY {spy_tot:.0f}%; "
                          f"long-short Q4-Q0 t={lsx['t']}. "
                          + ("Monotonic-ish, tradeable edge." if q4 > q0 and (lsx['t'] or 0) > 2 else
                             "Weak/again vol-tail — high-upside quintile is the beaten-down/high-vol tail, not a clean edge."))
    print("\nVERDICT:", payload["verdict"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="analyst_upside_portfolio",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[analyst_upside_portfolio]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_UPSIDE_PF", flush=True)


if __name__ == "__main__":
    main()
